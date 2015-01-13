#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/queue.h>
#include <assert.h>
#include <inttypes.h>
#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#endif

#include "cci.h"
#include "io.h"

cci_endpoint_t *ep = NULL;
int done = 0;
int connected_peers = 0;

#define IOD_TX_FINI	((void*)((uintptr_t)0x1))

struct io_req;

#define RMA_BUF_SIZE (4 * 1024 * 1024)  /* 4MB RMA buffer */
typedef struct peer {
	TAILQ_ENTRY(peer)	entry;	/* Hang on peers */
	cci_connection_t	*conn;	/* CCI connection */
	void				*buffer;
	cci_rma_handle_t	*local;	/* Our CCI RMA handle */
	cci_rma_handle_t	*remote; /* Their CCI RMA handle */

	uint32_t		requests; /* Number of requests received */
	uint32_t		completed; /* Number of completed requests on ios list */
	TAILQ_HEAD(done, io_req) ios;	/* List of completed io_reqs */

	uint32_t		len;	/* RMA buffer length */
	uint32_t		rank;	/* Peer's MPI rank */
	int			fd;	/* File for peer */
	int			done;	/* client sent BYE message */

	pthread_mutex_t		lock;	/* Lock to protect buffer, buffer_in_use & done */
	pthread_cond_t		cv;		/* Condition variable to wait on the RMA to finish
                                 * writing to buffer */
} peer_t;

TAILQ_HEAD(ps, peer)		peers;


typedef struct io_req {
	TAILQ_ENTRY(io_req)	entry;	/* To hang on io_q->reqs */
	peer_t			*peer;	/* Client peer */
	io_msg_t 		*msg;
	uint64_t		rx_us;	/* microsecs when received */
	uint64_t		rma_us;	/* microsecs when RMA completes */
	uint64_t		deq_us;	/* microsecs when dequeued by io */
	uint64_t		io_us;	/* microsecs when write() completes */
	uint32_t		len;	/* Requesting write of len bytes */
	uint32_t		offset;	/* how may bytes we've written so far */
} io_req_t;

TAILQ_HEAD(irq, io_req)		reqs;	/* List of io_reqs */
sem_t					reqs_sem;  /* Semaphore for the reqs TAILQ */
pthread_mutex_t			reqs_lock;	/* To protect reqs and cv */


/* Struct for keeping track of cached data */
typedef struct cache_block {
	TAILQ_ENTRY(cache_block)	entry;
	peer_t			*peer;	/* client peer */
	io_req_t		*io_req; /* only valid if this is the last cache block for the request */
	uint64_t		offset; /* offset into the file where we should start writing */
	/* TODO: I don't think we need this offset value */
	uint64_t		len;	/* how much data to write */
	void			*cache;	/* pointer to the cached data. */
							/* NOTE: in the GPU implementation, this will be a
							 * pointer to GPU memory! */
} cache_block_t;
TAILQ_HEAD(cbq, cache_block)	cache_blocks;
pthread_mutex_t					cache_lock; /* protects both the TAILQ *and* cache_size */
sem_t							cache_sem;
uint64_t						cache_size; /* keep track of how much memory
                                             * we've allocated to cache */
#define MAX_CACHE_SIZE			(2L * 1024 * 1024 * 1024) /* 2GB */
/* TODO: This needs to be a command line parameter! */


uint64_t min64( uint64_t a, uint64_t b) { return (a < b)?a:b; }
uint64_t max64( uint64_t a, uint64_t b) { return (a > b)?a:b; }

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
	char *buf = NULL;
	size_t len = p->completed * 4 * 32, offset = 0, newlen = 0;

	ftruncate(p->fd, 0);
	lseek(p->fd, 0, SEEK_SET);

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

		snprintf(buf, len, "len %u rx_us %"PRIu64" rma_us %"PRIu64" "
			"deq_us %"PRIu64" io_us %"PRIu64" ", io->len,
			io->rx_us, io->rma_us, io->deq_us, io->io_us);

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

	free_peer(p);

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

/* ******************************************************************
 * These functions are called by the io thread.
 * They dequeue data from the cache and actually write it to disk
 * ****************************************************************** */
static void *
io(void *arg)
{
	pin_to_core(5);
	

	while (!done || !TAILQ_EMPTY(&cache_blocks)) {
		/* Don't exit until we've drained the cache, even if the rest
		 * of the program is shutting down. */
		cache_block_t * cb = NULL;
		io_req_t *io_req = NULL;
		peer_t *p = NULL;
		uint32_t offset = 0;
		
		if (sem_wait( &cache_sem))
		{
			if (errno == EINTR)
			{
				/* No big deal (seems to happen inside the debugger
				 * fairly often).  Go back to sleep */
				continue;
			}
		}
		
		if (done) continue;

		pthread_mutex_lock( &cache_lock);
		cb = TAILQ_FIRST( &cache_blocks);
		TAILQ_REMOVE(&cache_blocks, cb, entry);
		pthread_mutex_unlock( &cache_lock);
		
		p = cb->peer;
		if (cb->io_req != NULL) {
			io_req = cb->io_req;
			/* This must be the last cache block for this request */
			io_req->deq_us = get_us();
		}

		offset = 0;
		do {
			int ret = write(p->fd,
					(void*)((uintptr_t)cb->cache + offset),
					cb->len - offset);
			if (ret > 0) {
				offset += ret;
			} else if (errno != EINTR) {
				fprintf(stderr, "%s: write() of %lu bytes for rank %u "
						" failed with %s\n", __func__,
						cb->len - offset, p->rank, strerror(errno));
				assert(0);
			}
		} while (offset < cb->len);

		/* Free the cache block */
		free( cb->cache);
		pthread_mutex_lock(&cache_lock);
		cache_size -= cb->len;
		pthread_mutex_unlock(&cache_lock);
		free( cb);
		
		if (io_req) {
			/* This is the last cache block for this request.  Update the
			 * timers and push it on to the completed queue */
			io_req->io_us = get_us();

			TAILQ_INSERT_TAIL(&p->ios, io_req, entry);
			p->completed++;
		}
			
		pthread_mutex_lock(&p->lock);
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

/* ******************************************************************
 * These functions are called by the caching thread.
 * They perform the rma(s) for each request and add it to the cache.
 * ****************************************************************** */
static void
rma_and_cache(io_req_t *io)
{
	int ret = 0;
	peer_t *p = io->peer;
	
	io_msg_t reply; /* delcared on the stack so I don't have to malloc it */
	io_msg_t *reply_ptr = NULL;
	uint32_t reply_size = 0;
	uint64_t data_len;  /* amount of data to be transferred in a single RMA */

	while (io->offset < io->len) { /* while there's still data to be RMA'd... */
		/* if this is the last RMA for this request, then send back the completion
		* message */
		if ((io->len - io->offset) <= p->len) {
			memset(&reply, 0, sizeof(reply.done));
			reply.done.type = WRITE_DONE;
			reply.done.cookie = io->msg->request.cookie;
			reply_ptr = &reply;
			reply_size = sizeof( reply.done);
		}

		data_len = min64( (io->len - io->offset), p->len);
		pthread_mutex_lock( &p->lock);
		ret = cci_rma(p->conn, reply_ptr, reply_size, p->local, 0, p->remote, io->offset,
				data_len, io, CCI_FLAG_READ);
		if (ret) {
			fprintf(stderr, "%s: cci_rma() failed with %s\n",
					__func__, cci_strerror(ep, ret));
		}
		pthread_cond_wait( &p->cv, &p->lock);
		/* The main thread will signal us when it gets the SEND event that indicates
		* the RMA completed. */
		
		pthread_mutex_unlock( &p->lock);
		
		/* Set up the cache block */
		cache_block_t *cb = (cache_block_t *)malloc( sizeof (*cb));
		if (!cb) {
			fprintf(stderr, "%s: unable to allocate cache_block struct for rank %u\n",
					__func__, p->rank);
			/* TODO: return error message? */
			goto out;
		}
		cb->peer = p;
		cb->offset = io->offset;
		cb->len = data_len;
		
		if (reply_ptr) { /* Did we just do the last RMA for this request? */
			cb->io_req = io;
		} else {
			cb->io_req = NULL;
		}
		
		/* Make sure we have mem available to allocate for cache */
		pthread_mutex_lock( &cache_lock);
		if( (cache_size + data_len) > MAX_CACHE_SIZE) {
			fprintf(stderr, "Cache mem full! Waiting for IO thread to drain it.\n");
		}
		while ( (cache_size + data_len) > MAX_CACHE_SIZE) {
			/* Block until the IO thread frees up some cache */
			pthread_mutex_unlock( &cache_lock);
			usleep( 5 * 1000 ); /* 5 ms - at 700MB/s it takes 5.7 ms to write a
			                     * a 4MB block... */
			pthread_mutex_lock( &cache_lock);
		}
		pthread_mutex_unlock( &cache_lock);
		
		/* TODO: This malloc (and its corresponding free() ) are pretty inefficient.
		 * Better to implement some kind of block pool that we can allocate once
		 * and then use blocks from */
		/* TODO: For the GPU implementation, we'd use CUDAMalloc() here, and 
		 * CUDAMemCpy() down below. */
		cb->cache = malloc( data_len);
		if (!cb->cache) {
			fprintf(stderr, "%s: unable to allocate mem for cached data for rank %u\n",
					__func__, p->rank);
			/* TODO: return error message? */
			goto out;
		}
		
		pthread_mutex_lock( &cache_lock);
		cache_size += data_len;
		pthread_mutex_unlock( &cache_lock);
		
		memcpy( cb->cache, p->buffer, data_len);

		pthread_mutex_lock( &cache_lock);
		TAILQ_INSERT_TAIL(&cache_blocks, cb, entry);
		pthread_mutex_unlock( &cache_lock);
		sem_post( &cache_sem);  /* wake up the cache thread */
		
		io->offset += data_len;
	}

	io->rma_us = get_us();
	
	out:
	return;
}

/* The main function for the caching thread */
static void *
cache_thread(void *arg)
{
	io_req_t *io_req = NULL;
	
	pin_to_core(3);
	
	while (!done) {
		
		if (sem_wait( &reqs_sem))
		{
			if (errno == EINTR)
			{
				/* No big deal (seems to happen inside the debugger
				 * fairly often).  Go back to sleep */
				continue;
			}
		}
		
		if (done) continue;

		pthread_mutex_lock( &reqs_lock);
		io_req = TAILQ_FIRST( &reqs);
		TAILQ_REMOVE(&reqs, io_req, entry);
		pthread_mutex_unlock( &reqs_lock);
		
		rma_and_cache( io_req);
	}
	
	return NULL;
}

/* ******************************************************************
 * These functions are called by the main thread.
 * They handle all the CCI events.
 * ****************************************************************** */
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

	/* The client actually passes to us the size of its RMA buffer, but
	 * we're going to ignore that and use a small buffer on our side.
	 * We'll likely have to call cci_rma() multiple times per client
	 * request. */
	/* p->len = msg->connect.len; */
	p->len = RMA_BUF_SIZE;
	p->rank = msg->connect.rank;

	p->buffer = malloc(p->len);
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
	snprintf(name, sizeof(name), "rank-%u-iod", p->rank);

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

/* Create a new IO request and append it to the request queue */
static void
handle_write_req(cci_event_t *event)
{
	peer_t *p = event->recv.connection->context;
	io_msg_t *msg = (void*) event->recv.ptr;
	assert( msg);
	
	io_req_t *io = calloc(1, sizeof(*io));
	if (!io) {
		fprintf(stderr, "%s: unable to allocate io for rank %u "
			"len %u cookie %"PRIx64"\n", __func__, p->rank,
			msg->request.len, msg->request.cookie);
		goto out;
		/* this will cause the client to hang because cci_rma() will 
		 * never be called and will thus never send the completion event */
	}
	
	p->requests++;
	
	io->peer = p;
	io->msg = msg;
	io->rx_us = get_us();
	io->len = msg->request.len;
	io->offset = 0;
	
	pthread_mutex_lock(&reqs_lock);
	/* TODO: verify that this TAILQ isn't getting too large! */
	TAILQ_INSERT_TAIL(&reqs, io, entry);
	sem_post( &reqs_sem);
	pthread_mutex_unlock(&reqs_lock);
	
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
handle_fini(void)
{
	/* TODO */
	connected_peers--;

	if (connected_peers == 0)
		done = 1;

	return;
}

static void
handle_send(cci_event_t *event)
{
	void *ctx = event->send.context;

	if (ctx == IOD_TX_FINI) {
		handle_fini();
	} else {
		/* ths event means a cci_rma() call has completed and we need
		 * to signal the caching thread that it can do its memcpy */
		io_req_t *io = event->send.context;
		/* The caching thread will lock the mutex before initiaing the RMA
		 * request and will not unlock it until it's waiting on the CV.
		 * Therefore, if we can acquire the lock, we know the caching thread
		 * must be waiting on the cv. */
		pthread_mutex_lock(&io->peer->lock);
		pthread_cond_signal(&io->peer->cv);
		pthread_mutex_unlock(&io->peer->lock);
		/* NOTE: Would it be better to unlock *before* signalling the cv?
		 * That way, when the caching thread wakes, it can immediately acquire
		 * the lock, rather than waking and then sleeping on the mutex lock...*/
	}

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
	char *uri = NULL;
	char hostname[64], name[128];
	pthread_t io_tid, cache_tid;
	
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
	
	pin_to_core(1);

	TAILQ_INIT(&reqs);
	sem_init( &reqs_sem, 0, 0);
	pthread_mutex_init(&reqs_lock, NULL);

	TAILQ_INIT( &cache_blocks);
	sem_init( &cache_sem, 0, 0);
	pthread_mutex_init( &cache_lock, NULL);

#if 0
	/* wait for debugger to attach */
	int dbg_wait = 1;
	while (dbg_wait) {
		usleep( 100);
	}
#endif
	
	ret = pthread_create(&io_tid, NULL, io, NULL);
	if (ret) {
		fprintf(stderr, "pthread_create() failed with %s\n", strerror(ret));
		goto out;
	}
	
	ret = pthread_create(&cache_tid, NULL, cache_thread, NULL);
	if (ret) {
		fprintf(stderr, "pthread_create() failed with %s\n", strerror(ret));
		goto out;
	}
	
	comm_loop();

	ret = unlink(name);
	if (ret) {
		perror("unlink()");
	}
	
	ret = pthread_join(io_tid, NULL);
	if (ret) {
		fprintf(stderr, "pthread_join() failed with %s\n", strerror(ret));
	}
	
	ret = pthread_join(cache_tid, NULL);
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
