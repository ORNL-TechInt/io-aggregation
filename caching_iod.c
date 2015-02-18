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

#ifdef __NVCC__
#include <cuda_runtime.h>
#include <cuda.h>
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
	uint32_t		buff_used; /* number of bytes in buffer that contain valid
	                            * data (may be <= len)  Not valid if an RMA
	                            * is in progress! */
	/*TODO: doesn't look like we're using buff_used any more... */
	uint32_t		rank;	/* Peer's MPI rank */
	int			fd;	/* File for peer */
	int			done;	/* client sent BYE message */

	pthread_mutex_t		lock;	/* Lock to protect buffer, buffer_in_use & done */
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
	uint32_t		offset;	/* starting location in the output file for this io's data */
} io_req_t;

TAILQ_HEAD(irq, io_req)		reqs;	/* List of io_reqs */
uint32_t				reqs_len;	/* number of entries in the queue */
sem_t					reqs_sem;  /* Semaphore for the reqs TAILQ */
pthread_mutex_t			reqs_lock;	/* To protect reqs and reqs_len */
#define MAX_REQS_LEN	100		/* we'll want to play with this number a bit */


/* Struct for keeping track of cached data */
typedef struct cache_block {
	TAILQ_ENTRY(cache_block)	entry;
	peer_t			*peer;	/* client peer */
	io_req_t		*io_req; /* only valid if this is the last cache block for the request */
	/* Note: no longer using an explicit len member.  The length of the req
	 * (and therefore the size of the allocated cache) is assumed to be equal
	 * to io_req->len. */
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


#ifdef __NVCC__

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET 
#endif
#endif

void checkCudaErrImpl(cudaError_t result, char const *const func,
					  const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d -- %s -- \"%s\" \n",
				file, line, (unsigned int)(result), cudaGetErrorString(result), func);
				DEVICE_RESET
		// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}

#define checkCudaErrors(val)  checkCudaErrImpl( (val), #val, __FILE__, __LINE__ )

#endif  /* __NVCC__ */

/*  Don't actually need these right now...
static uint64_t min64( uint64_t a, uint64_t b) { return (a < b)?a:b; }
static uint64_t max64( uint64_t a, uint64_t b) { return (a > b)?a:b; }
*/

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
	
	void *bounce_buffer;
	
#ifdef __NVCC__
	/* We can't write to disk directly from the cuda memory (yet - CUDA 6 unified
	 * memory might fix this) so we need a buffer in system memory to copy
	 * the data into first. */
	bounce_buffer = malloc(RMA_BUF_SIZE);
	if (!bounce_buffer) {
		fprintf(stderr, "%s: failed to allocate gpu bounce buffer.  Aborting.\n",
			__func__);
		abort();
	}
#endif

	while (!done || !TAILQ_EMPTY(&cache_blocks)) {
		/* Don't exit until we've drained the cache, even if the rest
		 * of the program is shutting down. */
		cache_block_t * cb = NULL;
		io_req_t *io_req = NULL;
		peer_t *p = NULL;
		
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
		io_req = cb->io_req;
		io_req->deq_us = get_us();
		
#ifdef __NVCC__
		/* Copy the data out of the GPU memory */
		checkCudaErrors( cudaMemcpy( bounce_buffer, cb->cache, io_req->len, cudaMemcpyDeviceToHost));
#else
		bounce_buffer = cb->cache;
#endif
		
		int bytes_written = 0;
		/*TODO: Try to figure out a way to get rid of this lseek() */
		int ret = lseek(p->fd, io_req->offset, SEEK_SET);
		if (ret == (off_t)-1) {
			fprintf(stderr, "%s: lseek() to offset %u for rank %u "
					" failed with %s\n", __func__,
					io_req->offset, p->rank, strerror(errno));
			assert(0);
		}
		do {
			ret = write(p->fd, (void*)bounce_buffer, io_req->len - bytes_written);
			if (ret > 0) {
				bytes_written += ret;
			} else if (errno != EINTR) {
				fprintf(stderr, "%s: write() of %u bytes for rank %u "
					" failed with %s\n", __func__,
					io_req->len - bytes_written, p->rank,
					strerror(errno));
				assert(0);
			}
		} while (bytes_written < io_req->len);

		/* Free the cache block */
#ifdef __NVCC__
		checkCudaErrors( cudaFree(cb->cache));
#else
		free( cb->cache);
#endif
		pthread_mutex_lock(&cache_lock);
		cache_size -= io_req->len;
		pthread_mutex_unlock(&cache_lock);
		free( cb);
		
		/* Update the io_req timers and push it on to the completed queue */
		io_req->io_us = get_us();

		TAILQ_INSERT_TAIL(&p->ios, io_req, entry);
		p->completed++;
			
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

#ifdef __NVCC__
	free(bounce_buffer); /* Free the bounce buffer we allocated */
#endif
	
	pthread_exit(NULL);
}

/* ******************************************************************
 * These functions are called by the caching thread.
 * They perform the rma(s) for each request and add it to the cache.
 * ****************************************************************** */
/* Move the data from the peer's buffer into the cache and then cci_send() a
 * a "DONE" message to the peer knows it can do another RMA */
/* TODO: Change this function name.  The client handles the RMA's now... */
static void
rma_and_cache(io_req_t *io)
{
	int ret = 0;
	peer_t *p = io->peer;
	
	io_msg_t reply; /* delcared on the stack so I don't have to malloc it */
	
	/* Set up the cache block */
	cache_block_t *cb = (cache_block_t *)malloc( sizeof (*cb));
	if (!cb) {
		fprintf(stderr, "%s: unable to allocate cache_block struct for rank %u\n",
				__func__, p->rank);
		/* TODO: return error message? */
		goto out;
	}
	cb->peer = p;
	cb->io_req = io;
	
	/* Make sure we have mem available to allocate for cache */
	pthread_mutex_lock( &cache_lock);
	if( (cache_size + io->len) > MAX_CACHE_SIZE) {
		fprintf(stderr, "Cache mem full! Waiting for IO thread to drain it.\n");
	
		while ( (cache_size + io->len) > MAX_CACHE_SIZE) {
		/* Block until the IO thread frees up some cache */
		pthread_mutex_unlock( &cache_lock);
		usleep( 5 * 1000 ); /* 5 ms - at 700MB/s it takes 5.7 ms to write a
								* a 4MB block... */
		pthread_mutex_lock( &cache_lock);
		}
	}
	pthread_mutex_unlock( &cache_lock);
	
	/* TODO: This malloc (and its corresponding free() ) are pretty inefficient.
		* Better to implement some kind of block pool that we can allocate once
		* and then use blocks from */
	/* TODO: For the GPU implementation, we'd use CUDAMalloc() here, and 
		* CUDAMemCpy() down below.   Can probably use "#ifdef __NVCC__"*/
#ifdef __NVCC__
	checkCudaErrors(cudaMalloc((void **) &cb->cache, io->len));
	checkCudaErrors(cudaMemcpy(cb->cache, p->buffer, io->len, cudaMemcpyHostToDevice));
#else
	cb->cache = malloc( io->len);
	if (!cb->cache) {
		fprintf(stderr, "%s: unable to allocate mem for cached data for rank %u\n",
				__func__, p->rank);
		/* TODO: return error message? */
		goto out;
	}
		
	memcpy( cb->cache, p->buffer, io->len);
#endif

	pthread_mutex_lock( &cache_lock);
	cache_size += io->len;
	TAILQ_INSERT_TAIL(&cache_blocks, cb, entry);
	pthread_mutex_unlock( &cache_lock);
	sem_post( &cache_sem);  /* wake up the io thread */

	/* Done with copying to cache.  Send the WRITE_DONE message so the client
	 * knows it can do another RMA. */
	
	memset(&reply, 0, sizeof(reply.done));
	reply.done.type = WRITE_DONE;

	ret = cci_send(p->conn, &reply, sizeof(reply.done), NULL, 0);
	if (ret) {
		fprintf(stderr, "%s: cci_send() failed with %s\n",
			__func__, cci_strerror(ep, ret));
	}
	
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
		reqs_len--;
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

	/* The client actually passes to us the size of its RMA buffer, but
	 * we're going to ignore that and use a small buffer on our side.
	 * We'll send this value back to the client so it knows how big an
	 * RMA we can accept.  See handle_accept(). */
	/* p->len = msg->connect.len; */
	p->len = RMA_BUF_SIZE;
	p->rank = msg->connect.rank;

	//p->buffer = malloc(p->len);
	ret = posix_memalign( &p->buffer, sysconf(_SC_PAGESIZE), p->len);
	if (!p->buffer) {
		fprintf(stderr, "%s: unable to allocate buffer for rank %u\n",
			__func__, p->rank);
		ret = ENOMEM;
		goto out;
	}
	fprintf(stderr, "%s: allocated %zu bytes at %p\n", __func__, (size_t)p->len, p->buffer);
	
	i = 0;
	do {
                char *c = (void *)((uintptr_t)p->buffer + i);
                *c = 'a';
		i+= sysconf( _SC_PAGESIZE);
        } while (i < p->len);

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
	ack.connect.len = p->len;  // size of our local RMA buffer
	memcpy((void*)&ack.connect.handle, (void*)p->local, sizeof(*p->local));

	ret = cci_send(p->conn, &ack, sizeof(ack.connect), NULL, 0);
	if (ret) {
		fprintf(stderr, "%s: failed to send ACK msg to rank %d\n",
			__func__, p->rank);
	}

	return;
}

/* Calling this a write request is now a bit of a misnomer.  We receive
 * these messages AFTER the client as RMA'd the data to us.  All we need
 * to do is copy it into the cache as quickly as possible so that the client
 * can RMA another block */
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
	io->rma_us = get_us();
	io->len = msg->request.len;
	io->offset = msg->request.offset;
	
	pthread_mutex_lock(&reqs_lock);
	/* Make sure the request queue doesn't grow too large */
	if (reqs_len >= MAX_REQS_LEN) {
		fprintf(stderr, "%s: Cannot add request to queue. (Queue is at max size.) "
		        "Waiting for caching thread to catch up.\n", __func__);
		
		while (reqs_len >= MAX_REQS_LEN) {
			pthread_mutex_unlock(&reqs_lock);
			usleep( 1);	/* this is damn close to a spinlock, but I need to
						 * the caching thread a chance to acquire the lock */
			pthread_mutex_lock(&reqs_lock);
		}
	}
	
	TAILQ_INSERT_TAIL(&reqs, io, entry);
	reqs_len++;
	sem_post( &reqs_sem);  /* wake up the cache thread */
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
	
#if 1
	char *dbg_env = getenv( "WAIT_ON_GDB");
	if (dbg_env && *dbg_env) {
		/* wait for debugger to attach */
		int dbg_wait = 1;
		while (dbg_wait) {
			usleep( 100);
		}
	}
#endif
	
	
	
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
	reqs_len = 0;
	sem_init( &reqs_sem, 0, 0);
	pthread_mutex_init(&reqs_lock, NULL);

	TAILQ_INIT( &cache_blocks);
	sem_init( &cache_sem, 0, 0);
	pthread_mutex_init( &cache_lock, NULL);

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
