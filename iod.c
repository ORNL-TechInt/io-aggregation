#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/queue.h>
#include <assert.h>
#include <inttypes.h>
#include <sys/types.h>
#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#endif
#include <poll.h>

#include "io.h"

cci_endpoint_t *ep = NULL;
int done = 0;
int connected_peers = 0;
int null_io = 0;
int blocking = 0;
cci_os_handle_t cci_fd = 0, *osh = NULL;

#define IOD_TX_FINI	((void*)((uintptr_t)0x1))

struct io_req;

typedef struct peer {
	TAILQ_ENTRY(peer)	entry;	/* Hang on peers */
	cci_connection_t	*conn;	/* CCI connection */
	void			*buffer;
	cci_rma_handle_t	*local;	/* Our CCI RMA handle */
	cci_rma_handle_t	*remote; /* Their CCI RMA handle */

	uint32_t		requests; /* Number of requests received */
	uint32_t		completed; /* Number of completed requests on ios list */
	TAILQ_HEAD(done, io_req) ios;	/* List of completed io_reqs */

	uint32_t		len;	/* RMA buffer length */
	uint32_t		rank;	/* Peer's MPI rank */
	int			fd;	/* File for peer */
	int			done;	/* client sent BYE message */

	pthread_mutex_t		lock;	/* Lock to protect done */
} peer_t;

TAILQ_HEAD(ps, peer)		peers;

typedef struct io_req {
	TAILQ_ENTRY(io_req)	entry;	/* To hang on io_q->reqs */
	peer_t			*peer;	/* Client peer */
	uint64_t		rx_us;	/* microsecs when received */
	uint64_t		cpy_us;	/* microsecs when copy completes */
	uint64_t		rma_us;	/* microsecs when RMA completes */
	uint64_t		deq_us;	/* microsecs when dequeued by io */
	uint64_t		io_us;	/* microsecs when write() completes */
	uint32_t		len;	/* Requesting write of len bytes */
} io_req_t;

TAILQ_HEAD(irq, io_req)		reqs;	/* List of io_reqs */
pthread_cond_t			cv;	/* Condition variable to wait on reqs */
pthread_mutex_t			lock;	/* To protect reqs and cv */

static uint64_t
get_us(void)
{
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return (uint64_t)(tv.tv_sec * 1000000) + (uint64_t)tv.tv_usec;
}

static void
free_peer(peer_t *p)
{
	if (!p)
		return;

	if (p->fd > 0)
		close(p->fd);

	free((void*)p->remote);

	if (p->local) {
		int ret = 0;

		ret = cci_rma_deregister(ep, p->local);
		if (ret) {
			fprintf(stderr, "%s: cci_rma_deregister() "
				"for rank %u failed with %s\n", __func__,
				p->rank, cci_strerror(ep, ret));
		}
	}

	if (p->conn)
		cci_disconnect(p->conn);

	free(p->buffer);
	free(p);

	return;
}

static void
print_results(peer_t *p)
{
	int ret = 0;
	io_msg_t msg;
	char *buf = NULL, name[32];
	size_t len = p->completed * 4 * 32, offset = 0, newlen = 0;

	close(p->fd);

	memset(name, 0, sizeof(name));
	snprintf(name, sizeof(name), "rank-%u-iod", p->rank);

	p->fd = open(name, O_RDWR|O_CREAT, 0644);
	if (p->fd == -1) {
		fprintf(stderr, "%s: open(%s) failed with %s\n", __func__, name,
				strerror(errno));
		return;
	}

	/* allocate buffer - reserve for 4 uint64_t (which use a max of 20 chars)
	 * for each completed IO request.
	 */
	len = p->completed * 4 * 32;

	buf = calloc(1, len);
	if (!buf) {
		fprintf(stderr, "%s: unable to allocate buffer for rank-%d\n",
			__func__, p->rank);
		return;
	}

	snprintf(buf, len, "iod\nrank %u num_requests %u max_len %u\n",
			p->rank, p->completed, p->len);
	newlen = strlen(buf);
	do {
		ret = write(p->fd, (void*)((uintptr_t)buf + offset), newlen - offset);
		if (ret > 0) {
			offset += ret;
		} else {
			fprintf(stderr, "%s: write() failed with %s\n",
				__func__, strerror(errno));
		}
	} while (offset < newlen);

	while (!TAILQ_EMPTY(&p->ios)) {
		io_req_t *io = NULL;

		io = TAILQ_FIRST(&p->ios);
		TAILQ_REMOVE(&p->ios, io, entry);

		memset(buf, 0, len);

		snprintf(buf, len, "len %u rx_us %"PRIu64" cpy_us %"PRIu64" rma_us %"PRIu64" "
			"deq_us %"PRIu64" io_us %"PRIu64" ", io->len,
			io->rx_us, io->cpy_us, io->rma_us, io->deq_us, io->io_us);

		newlen = strlen(buf);
		offset = 0;

		do {
			ret = write(p->fd, (void*)((uintptr_t)buf + offset), newlen - offset);
			if (ret > 0) {
				offset += ret;
			} else {
				fprintf(stderr, "%s: write() failed with %s\n",
					__func__, strerror(errno));
			}
		} while (offset < newlen);

		ret = write(p->fd, "\n", 1);
		if (ret != 1)
			fprintf(stderr, "%s: write() failed with %s\n",
				__func__, strerror(errno));
	}

	msg.fini.type = FINISHED;

	ret = cci_send(p->conn, &msg, sizeof(msg.fini), IOD_TX_FINI, 0);
	if (ret) {
		fprintf(stderr, "%s: failed to send FINISHED msg to rank %d\n",
			__func__, p->rank);
	}

	return;
}

static void
pin_to_core(int core)
{
#ifdef __linux__
	int ret = 0;
	cpu_set_t cpuset;
	pid_t tid;

	tid = syscall(SYS_gettid);

	CPU_ZERO(&cpuset);
	CPU_SET(core, &cpuset);

	fprintf(stderr, "%s: pinning tid %d to core %d\n", __func__, tid, core);

	ret = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
	if (ret) {
		fprintf(stderr, "%s: sched_setaffinity() failed with %s\n",
			__func__, strerror(errno));
	}
#endif
	return;
}

static void *
io(void *arg)
{
	pin_to_core(3);

	while (!done) {
		int ret = 0;
		io_req_t *io = NULL;
		peer_t *p = NULL;
		uint32_t offset = 0;
		io_msg_t reply;

		pthread_mutex_lock(&lock);
		if (TAILQ_EMPTY(&reqs)) {
			pthread_cond_wait(&cv, &lock);
		}

		io = TAILQ_FIRST(&reqs);
		TAILQ_REMOVE(&reqs, io, entry);
		pthread_mutex_unlock(&lock);

		io->deq_us = get_us();

		p = io->peer;

		if (!null_io) {
			do {
				int ret = write(p->fd,
					(void*)((uintptr_t)p->buffer + offset),
					io->len - offset);
				if (ret > 0) {
					offset += ret;
				} else if (errno != EINTR) {
					fprintf(stderr, "%s: write() of %u bytes for rank %u "
						" failed with %s\n", __func__,
						io->len - offset, p->rank, strerror(errno));
					assert(0);
				}
			} while (offset < io->len);
		}

		io->io_us = get_us();

		memset(&reply, 0, sizeof(reply.done));
		reply.done.type = WRITE_DONE;

		ret = cci_send(p->conn, &reply, sizeof(reply.done), NULL, 0);
		if (ret) {
			fprintf(stderr, "%s: cci_send() failed with %s\n",
					__func__, cci_strerror(ep, ret));
		}

		TAILQ_INSERT_TAIL(&p->ios, io, entry);

		pthread_mutex_lock(&p->lock);
		p->completed++;

		if (p->done) {
			if (p->requests == p->completed) {
				/* drop the lock, because print_results will
				 * free the peer */
				pthread_mutex_unlock(&p->lock);
				print_results(p);
				continue;
			}
		}
		pthread_mutex_unlock(&p->lock);
	}

	pthread_exit(NULL);
}

static void
handle_connect_request(cci_event_t *event)
{
	int ret = 0, i = 0;
	io_msg_t *msg = (void*) event->request.data_ptr;
	peer_t *p = NULL;
	char name[32];

	if (event->request.data_len != sizeof(msg->connect)) {
		fprintf(stderr, "%s: expected %zd bytes but received %u bytes\n",
			__func__, sizeof(msg->connect), event->request.data_len);
		ret = EINVAL;
		goto out;
	}

	p = calloc(1, sizeof(*p));
	if (!p) {
		fprintf(stderr, "%s: unable to allocate peer for rank %u\n",
			__func__, msg->connect.rank);
		ret = ENOMEM;
		goto out;
	}

	p->remote = calloc(1, sizeof(*p->remote));
	if (!p->remote) {
		fprintf(stderr, "%s: unable to allocate remote handle for rank %u\n",
			__func__, msg->connect.rank);
		ret = ENOMEM;
		goto out;
	}
	memcpy((void*)p->remote, &msg->connect.handle, sizeof(*p->remote));

	p->len = msg->connect.len;
	p->rank = msg->connect.rank;

	//p->buffer = malloc(msg->connect.len);
	ret = posix_memalign(&p->buffer, getpagesize(), p->len);
	if (!p->buffer) {
		fprintf(stderr, "%s: unable to allocate buffer for rank %u\n",
			__func__, p->rank);
		ret = ENOMEM;
		goto out;
	}
	fprintf(stderr, "%s: allocated %u bytes at %p\n", __func__, p->len, p->buffer);

	for (i = 0; i < p->len; i += 4096) {
		char *c = (void *)((uintptr_t)p->buffer + i);

		*c = 'a';
	}

	ret = cci_rma_register(ep, p->buffer, p->len, CCI_FLAG_WRITE, &p->local);
	if (ret) {
		fprintf(stderr, "%s: cci_rma_register() for rank %u failed with %s\n",
			__func__, p->rank, cci_strerror(ep, ret));
		goto out;
	}

	TAILQ_INIT(&p->ios);
	pthread_mutex_init(&p->lock, NULL);

	memset(name, 0, sizeof(name));
	snprintf(name, sizeof(name), "rank-%u-iod-data", p->rank);

	p->fd = open(name, O_RDWR|O_CREAT, 0644);
	if (p->fd == -1) {
		fprintf(stderr, "%s: open(%s) failed with %s\n", __func__, name,
				strerror(errno));
		goto out;
	}

	ret = cci_accept(event, p);
	if (ret) {
		fprintf(stderr, "%s: cci_accept() for rank %u failed with %s\n",
			__func__, p->rank, cci_strerror(ep, ret));
	}

	connected_peers++;

    out:
	if (ret)
		free_peer(p);
	return;
}

static void
handle_accept(cci_event_t *event)
{
	int ret = 0;
	peer_t *p = event->accept.context;
	io_msg_t ack;

	p->conn = event->accept.connection;

	if (!p->conn) {
		fprintf(stderr, "%s: accept failed for rank %u\n",
			__func__, p->rank);
		free_peer(p);
		done = 1;
	}

	memset(&ack, 0, sizeof(ack));
	ack.connect.type = CONNECT_ACK;
	memcpy((void*)&ack.connect.handle, (void*)p->local, sizeof(*p->local));

	ret = cci_send(p->conn, &ack, sizeof(ack.connect), NULL, 0);
	if (ret) {
		fprintf(stderr, "%s: failed to send ACK msg to rank %d\n",
			__func__, p->rank);
	}

	return;
}

static void
handle_write_req(cci_event_t *event)
{
	peer_t *p = event->recv.connection->context;
	io_msg_t *msg = (void*) event->recv.ptr;
	io_req_t *io = NULL;

	assert(msg);

	io = calloc(1, sizeof(*io));
	if (!io) {
		fprintf(stderr, "%s: unable to allocate io for rank %u "
			"len %u cookie %"PRIx64"\n", __func__, p->rank,
			msg->request.len, msg->request.cookie);
		goto out;
	}

	pthread_mutex_lock(&p->lock);
	p->requests++;
	pthread_mutex_unlock(&p->lock);

	io->peer = p;
	io->rx_us = get_us();
	io->len = msg->request.len;

	pthread_mutex_lock(&lock);
	TAILQ_INSERT_TAIL(&reqs, io, entry);
	pthread_cond_signal(&cv);
	pthread_mutex_unlock(&lock);

    out:
	return;
}

static void
handle_bye(cci_event_t *event)
{
	peer_t *p = event->recv.connection->context;

	pthread_mutex_lock(&p->lock);
	p->done = 1;

	if (p->requests == p->completed) {
		pthread_mutex_unlock(&p->lock);
		print_results(p);
		return;
	}
	pthread_mutex_unlock(&p->lock);

	return;
}

static void
handle_recv(cci_event_t *event)
{
	io_msg_t *msg = (void*) event->recv.ptr;

	switch (msg->type) {
	case WRITE_REQ:
		handle_write_req(event);
		break;
	case BYE:
		handle_bye(event);
		break;
	default:
		fprintf(stderr, "%s: ignoring %d event\n", __func__, msg->type);
		break;
	}

	return;
}

#if 0
static void
handle_rma(cci_event_t *event)
{
	io_req_t *io = event->send.context;

	io->rma_us = get_us();

	pthread_mutex_lock(&lock);
	TAILQ_INSERT_TAIL(&reqs, io, entry);
	pthread_cond_signal(&cv);
	pthread_mutex_unlock(&lock);

}
#endif

static void
handle_fini(cci_event_t *event)
{
	free_peer(event->send.connection->context);

	connected_peers--;

	fprintf(stderr, "%s: connected_peers=%d\n", __func__, connected_peers);

	if (connected_peers == 0)
		done = 1;

	return;
}

static void
handle_send(cci_event_t *event)
{
	void *ctx = event->send.context;

	if (ctx == IOD_TX_FINI) {
		handle_fini(event);
	}

	return;
}

static void
comm_loop(void)
{
	int ret = 0;
	cci_event_t *event = NULL;
	struct pollfd pfd;

	if (blocking) {
		pfd.fd = *osh;
		pfd.events = POLLIN;
	}

	TAILQ_INIT(&peers);

	while (!done) {
		if (blocking)
			poll(&pfd, 1, 0);

		ret = cci_get_event(ep, &event);
		if (ret) {
			if (ret != CCI_EAGAIN) {
				fprintf(stderr, "%s: cci_get_event() failed with %s\n",
						__func__, cci_strerror(ep, ret));
			}
			continue;
		}

		/* handle event */

		switch (event->type) {
		case CCI_EVENT_CONNECT_REQUEST:
			handle_connect_request(event);
			break;
		case CCI_EVENT_ACCEPT:
			handle_accept(event);
			break;
		case CCI_EVENT_RECV:
			handle_recv(event);
			break;
		case CCI_EVENT_SEND:
			handle_send(event);
			break;
		default:
			fprintf(stderr, "%s: ignoring %s\n",
				__func__, cci_event_type_str(event->type));
			break;
		}

		cci_return_event(event);
	}

	return;
}

void
print_usage(char *name)
{
	fprintf(stderr, "usage: %s [-b] [-n]\n", name);
	fprintf(stderr, "where:\n");
	fprintf(stderr, "\t-b\tUse CCI blocking mode\n");
	fprintf(stderr, "\t-n\tNULL IO - do not write data to file system\n");
	exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
	int ret = 0, i = 0, fd = -1, c = 0;
	uint32_t caps = 0;
	cci_device_t *const *devices, *device = NULL;
	char *uri = NULL;
	char hostname[64], name[128];
	pthread_t tid;

	while ((c = getopt(argc, argv, "bn")) != -1) {
		switch (c) {
		case 'n':
			null_io = 1;
			break;
		case 'b':
			blocking = 1;
			break;
		default:
			print_usage(argv[0]);
		}
	}

	ret = cci_init(CCI_ABI_VERSION, 0, &caps);
	if (ret) {
		fprintf(stderr, "cci_init() failed with %s\n",
				cci_strerror(NULL, ret));
		goto out;
	}

	ret = cci_get_devices(&devices);
	if (ret) {
		fprintf(stderr, "cci_get_devices() failed with %s\n",
				cci_strerror(NULL, ret));
		goto out;
	}

	for (i = 0; ; i++) {
		device = devices[i];

		if (!device)
			break;

		if (!strcmp(device->transport, "sm")) {
			if (!device->up) {
				fprintf(stderr, "sm device is down\n");
				goto out;
			}

			break;
		}
	}

	if (!device) {
		fprintf(stderr, "No sm device found\n");
		goto out;
	}

	if (blocking)
		osh = &cci_fd;

	ret = cci_create_endpoint(device, 0, &ep, osh);
	if (ret) {
		fprintf(stderr, "cci_create_endpoint() failed with %s\n",
				cci_strerror(NULL, ret));
		goto out;
	}

	ret = cci_get_opt(ep, CCI_OPT_ENDPT_URI, &uri);
	if (ret) {
		fprintf(stderr, "cci_get_opt() failed with %s\n",
				cci_strerror(ep, ret));
		goto out;
	}

	memset(hostname, 0, sizeof(hostname));
	gethostname(hostname, sizeof(hostname));

	memset(name, 0, sizeof(name));
	snprintf(name, sizeof(name), "%s-iod", hostname);

	fd = open(name, O_RDWR | O_CREAT | O_TRUNC, 0600);
	if (fd == -1) {
		fprintf(stderr, "open() failed with %s\n", strerror(errno));
		goto out;
	}

	ret = write(fd, uri, strlen(uri) + 1);
	if (ret != (strlen(uri) + 1)) {
		if (ret == -1) {
			fprintf(stderr, "write() failed with %s\n", strerror(errno));
		} else {
			fprintf(stderr, "write() returned %d\n", ret);
		}
		goto out;
	}

	close(fd);

	pin_to_core(1);

	TAILQ_INIT(&reqs);
	pthread_cond_init(&cv, NULL);
	pthread_mutex_init(&lock, NULL);

	ret = pthread_create(&tid, NULL, io, NULL);
	if (ret) {
		fprintf(stderr, "pthread_create() failed with %s\n", strerror(ret));
		goto out;
	}

	comm_loop();

	ret = unlink(name);
	if (ret) {
		perror("unlink()");
	}

	ret = pthread_join(tid, NULL);
	if (ret) {
		fprintf(stderr, "pthread_join() failed with %s\n", strerror(ret));
	}

    out:
	free(uri);

	if (ep) {
		ret = cci_destroy_endpoint(ep);
		if (ret) {
			fprintf(stderr, "cci_destroy_endpoint() failed with %s\n",
					cci_strerror(NULL, ret));
		}
	}

	ret = cci_finalize();
	if (ret) {
		fprintf(stderr, "cci_finalize() failed with %s\n",
				cci_strerror(NULL, ret));
	}

	return ret;
}
