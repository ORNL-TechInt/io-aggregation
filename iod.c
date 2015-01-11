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

#include "cci.h"
#include "io.h"

cci_endpoint_t *ep = NULL;
char *uri = NULL;
int done = 0;

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
print_results(peer_t *p)
{
	/* TODO */
	return;
}

static void *
io(void *arg)
{
	while (!done) {
		io_req_t *io = NULL;
		peer_t *p = NULL;
		uint32_t offset = 0;

		pthread_mutex_lock(&lock);
		if (TAILQ_EMPTY(&reqs)) {
			pthread_cond_wait(&cv, &lock);
		}

		io = TAILQ_FIRST(&reqs);
		TAILQ_REMOVE(&reqs, io, entry);
		pthread_mutex_unlock(&lock);

		io->deq_us = get_us();

		p = io->peer;

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

		io->io_us = get_us();

		TAILQ_INSERT_TAIL(&p->ios, io, entry);
		p->completed++;

		pthread_mutex_lock(&p->lock);
		if (p->done && p->requests == p->completed)
			print_results(p);
		pthread_mutex_unlock(&p->lock);
	}

	pthread_exit(NULL);
}

static void
free_peer(peer_t *p)
{
	if (!p)
		return;

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
	free(p->buffer);
	free(p);

	return;
}

static void
handle_connect_request(cci_event_t *event)
{
	int ret = 0;
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

	p->buffer = malloc(msg->connect.len);
	if (!p->buffer) {
		fprintf(stderr, "%s: unable to allocate buffer for rank %u\n",
			__func__, p->rank);
		ret = ENOMEM;
		goto out;
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
	snprintf(name, sizeof(name), "rank-%u", p->rank);

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

    out:
	if (ret)
		free_peer(p);
	return;
}

static void
handle_accept(cci_event_t *event)
{
	peer_t *p = event->accept.context;

	p->conn = event->accept.connection;

	if (!p->conn) {
		fprintf(stderr, "%s: accept failed for rank %u\n",
			__func__, p->rank);
		free_peer(p);
		done = 1;
	}

	return;
}

static void
handle_write_req(cci_event_t *event)
{
	int ret = 0;
	peer_t *p = event->recv.connection->context;
	io_msg_t *msg = (void*) event->recv.ptr, reply;
	io_req_t *io = NULL;

	assert(msg);

	io = calloc(1, sizeof(*io));
	if (!io) {
		fprintf(stderr, "%s: unable to allocate io for rank %u "
			"len %u cookie %"PRIx64"\n", __func__, p->rank,
			msg->request.len, msg->request.cookie);
		goto out;
	}

	p->requests++;

	io->peer = p;
	io->rx_us = get_us();
	io->len = msg->request.len;

	memset(&reply, 0, sizeof(reply.done));
	reply.done.type = WRITE_DONE;
	reply.done.cookie = msg->request.cookie;

	ret = cci_rma(p->conn, &reply, sizeof(reply.done), p->local, 0, p->remote, 0,
			io->len, io, CCI_FLAG_READ);

    out:
	return;
}

static void
handle_bye(cci_event_t *event)
{
	peer_t *p = event->recv.connection->context;

	pthread_mutex_lock(&p->lock);
	p->done = 1;
	pthread_mutex_unlock(&p->lock);

	if (p->requests == p->completed)
		print_results(p);

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

static void
handle_send(cci_event_t *event)
{
	io_req_t *io = event->send.context;

	io->rma_us = get_us();

	pthread_mutex_lock(&lock);
	TAILQ_INSERT_TAIL(&reqs, io, entry);
	pthread_cond_signal(&cv);
	pthread_mutex_unlock(&lock);

	return;
}

static void
comm_loop(void)
{
	int ret = 0;
	cci_event_t *event = NULL;

	TAILQ_INIT(&peers);

	while (!done) {
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

int
main(int argc, char *argv[])
{
	int ret = 0, i = 0, fd = -1;
	uint32_t caps = 0;
	cci_device_t *const *devices, *device = NULL;
	char hostname[64], name[128];
	pthread_t tid;

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

	ret = cci_create_endpoint(device, 0, &ep, NULL);
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

	TAILQ_INIT(&reqs);
	pthread_cond_init(&cv, NULL);
	pthread_mutex_init(&lock, NULL);

	ret = pthread_create(&tid, NULL, io, NULL);
	if (ret) {
		fprintf(stderr, "pthread_create() failed with %s\n", strerror(ret));
		goto out;
	}

	comm_loop();

    out:
	free(uri);

	ret = cci_destroy_endpoint(ep);
	if (ret) {
		fprintf(stderr, "cci_destroy_endpoint() failed with %s\n",
				cci_strerror(NULL, ret));
	}

	ret = cci_finalize();
	if (ret) {
		fprintf(stderr, "cci_finalize() failed with %s\n",
				cci_strerror(NULL, ret));
	}

	return ret;
}
