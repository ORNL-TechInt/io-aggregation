#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <inttypes.h>
#include <assert.h>
#include <sys/time.h>
#include <poll.h>

#include "io.h"

extern char **environ;

int ifd = -1;
pid_t pid = -1;
cci_endpoint_t *endpoint = NULL;
cci_connection_t *connection = NULL;
cci_rma_handle_t *local = NULL;
cci_rma_handle_t handle;
cci_rma_handle_t *remote = NULL;
int irank = -1;
cci_os_handle_t *cfd = NULL, ep_fd;
struct pollfd pfd;
int avail = 1;

uint32_t max_rma_len = 0;  // Maximum size we can send in one RMA call

static uint32_t min32( uint32_t a, uint32_t b) { return (a < b)?a:b; }

/* Don't need these yet...
static uint64_t min64( uint64_t a, uint64_t b) { return (a < b)?a:b; }
static uint32_t max32( uint32_t a, uint32_t b) { return (a > b)?a:b; }
static uint64_t max64( uint64_t a, uint64_t b) { return (a > b)?a:b; }
*/

static void
handle_sigchld(int sig)
{
	int status;

	waitpid(pid, &status, 0);
	if (WIFEXITED(status)) {
		fprintf(stderr, "%s: daemon exited, status=%d\n",
			__func__, WEXITSTATUS(status));
	} else if (WIFSIGNALED(status)) {
		fprintf(stderr, "%s: killed by signal %d\n",
			__func__, WTERMSIG(status));
		abort();
	}
	return;
}

static int
start_daemon(char **args)
{
	int ret = 0;

	signal(SIGCHLD, handle_sigchld);

	pid = fork();
	if (pid == -1) {
		ret = errno;
		fprintf(stderr, "%s: fork() failed with %s\n", __func__,
				strerror(ret));
	} else if (pid == 0) {
		execve(args[0], args, environ);
		/* if we return, exec() failed */
		ret = errno;
		fprintf(stderr, "%s: execve() failed with %s\n", __func__,
				strerror(ret));

		exit(ret);
	} else {
		fprintf(stderr, "%s daemon started with PID %d\n", args[0], pid);
	}

	return ret;
}

int io_init(void *buffer, uint32_t len, uint32_t rank, uint32_t ranks,
		char **daemon_args, int blocking)
{
	int ret = 0, fd_iod = -1, ready = 0;
	uint32_t caps = 0;
	io_msg_t msg;
	char hostname[256], server[256];

	if (!buffer || !len) {
		ret = EINVAL;
		goto out;
	}

	memset(hostname, 0, sizeof(hostname));
	gethostname(hostname, sizeof(hostname));

	ifd = open(hostname, O_WRONLY | O_CREAT | O_TRUNC | O_EXCL, 0600);
	if (ifd != -1) {
		ret = start_daemon( daemon_args);
		if (ret) {
			goto out;
		}
	}

	ret = cci_init(CCI_ABI_VERSION, 0, &caps);
	if (ret) {
		fprintf(stderr, "%s: cci_init() failed with %s\n",
				__func__, cci_strerror(NULL, ret));
		goto out;
	}

	if (blocking)
		cfd = &ep_fd;

	ret = cci_create_endpoint(NULL, 0, &endpoint, cfd);
	if (ret) {
		fprintf(stderr, "%s: cci_create_endpoint() failed with %s\n",
				__func__, cci_strerror(NULL, ret));
		goto out;
	}

	if (blocking) {
		pfd.fd = ep_fd;
		pfd.events = POLLIN;
	}

	ret = cci_rma_register(endpoint, buffer, (uint64_t)len,
			CCI_FLAG_WRITE|CCI_FLAG_READ, &local);
	if (ret) {
		fprintf(stderr, "%s: cci_rma_register() failed with %s\n",
				__func__, cci_strerror(endpoint, ret));
		goto out;
	}

	irank = rank;

	msg.connect.type = CONNECT;
	msg.connect.rank = rank;
	msg.connect.ranks = ranks;
	msg.connect.len = len;
	memcpy((void*)(uintptr_t)&msg.connect.handle, local, sizeof(*local));

	memset(server, 0, sizeof(server));
	snprintf(server, sizeof(server), "%s-iod", hostname);

	do {
		fd_iod = open(server, O_RDONLY);
	} while (fd_iod == -1);

    again:
	ret = read(fd_iod, server, sizeof(server));
	if (ret == -1) {
		ret = errno;
		if (ret == EINTR)
			goto again;

		fprintf(stderr, "%s: read() failed with %s\n",
				__func__, strerror(ret));
		goto out;
	} else if (ret == 0) {
		goto again;
	}

	ret = cci_connect(endpoint, server, &msg, sizeof(msg.connect),
			CCI_CONN_ATTR_RO, NULL, 0, NULL);
	if (ret) {
		fprintf(stderr, "%s: cci_connect() failed with %s\n",
				__func__, cci_strerror(endpoint, ret));
		goto out;
	}

	do {
		cci_event_t *event = NULL;
		io_msg_t *msg = NULL;

		ret = cci_get_event(endpoint, &event);
		if (!ret) {
			switch (event->type) {
			case CCI_EVENT_CONNECT:
				connection = event->connect.connection;
				ready++;
				break;
			case CCI_EVENT_RECV:
				msg = (void *)event->recv.ptr;
				assert(msg->connect.type == CONNECT_ACK);
				max_rma_len = msg->connect.len;
				memcpy((void*)&handle, &msg->connect.handle, sizeof(handle));
				remote = &handle;
				fprintf(stderr, "%s: received ack - remote = %p\n", __func__,
						(void*)remote);
				break;
			default:
				fprintf(stderr, "%s: ignoring %s\n", __func__,
						cci_event_type_str(event->type));
				break;
			}
			cci_return_event(event);
		}
	} while (!ready || remote == NULL);

	if (!connection) {
		ret = ENOTCONN;
		fprintf(stderr, "%s: CCI connect failed\n", __func__);
	}

    out:
	if (fd_iod != -1)
		close(fd_iod);

	if (ret)
		io_finalize();
	return ret;
}

static void
print_perf(uint32_t len, struct timeval start, struct timeval end)
{
	uint64_t usecs = 0;
	double lat = 0.0, bw = 0.0;

	usecs = (end.tv_sec - start.tv_sec) * 1000000 +
		end.tv_usec - start.tv_usec;

	lat = (double) usecs / 1000000.0;
	bw = (double) len / 1000000.0 / lat;

	fprintf(stderr, "rank %d: %u bytes  %10"PRIu64" usecs  %10.2lf MB/s\n",
			irank, len, usecs, bw);

	return;
}

static int
wait_for_avail(int i)
{
	int ret = 0;

	do {
		cci_event_t *event = NULL;

		if (cfd)
			poll(&pfd, 1, 0);

		ret = cci_get_event(endpoint, &event);
		if (!ret) {
			const io_msg_t *rx = NULL;

			switch (event->type) {
			case CCI_EVENT_RECV:
				rx = event->recv.ptr;
				assert(rx->type == WRITE_DONE);
				avail = 1;
				break;
			default:
				fprintf(stderr, "%s: ignoring %s\n",
					__func__, cci_event_type_str(event->type));
				break;
			}
			cci_return_event(event);
		}
	} while (!avail);

	return ret;
}

int io_write(uint32_t len)
{
	int ret = 0, done = 0;
	static int i = 0;
	io_msg_t msg;
	uint32_t bytes_transferred = 0;
	
	while (bytes_transferred < len)
	{
		if (!avail)
			wait_for_avail(i);

		avail = 0;

		i++;

		uint32_t bytes_remaining = len - bytes_transferred;
		uint32_t bytes_to_send = min32(bytes_remaining, max_rma_len);
		msg.request.type = WRITE_REQ;
		msg.request.len = bytes_to_send;
		msg.request.offset = bytes_transferred;

		ret = cci_rma(connection, &msg, sizeof(msg.request),
				local, bytes_transferred, remote, 0, bytes_to_send,
				(void*)(uintptr_t)i, CCI_FLAG_WRITE);
		if (ret) {
			fprintf(stderr, "%s: cci_rma() failed with %s\n",
					__func__, cci_strerror(endpoint, ret));
			goto out;
		}
		
		bytes_transferred += bytes_to_send;

		do {
			cci_event_t *event = NULL;

			if (cfd)
				poll(&pfd, 1, 0);

			ret = cci_get_event(endpoint, &event);
			if (!ret) {
				const io_msg_t *rx = NULL;

				switch (event->type) {
				case CCI_EVENT_SEND:
					assert(event->send.context == (void*)(uintptr_t)i);
					done = 1;
					break;
				case CCI_EVENT_RECV:
					rx = event->recv.ptr;
					assert(rx->type == WRITE_DONE);
					avail = 1;
					break;
				default:
					fprintf(stderr, "%s: ignoring %s\n",
						__func__, cci_event_type_str(event->type));
					break;
				}
				cci_return_event(event);
			}
		} while (!done);
	}

    out:
	return ret;
}

int io_finalize(void)
{
	int ret = 0;
	io_msg_t msg;
	char hostname[256];

	msg.type = BYE;

	if (connection) {
		ret = cci_send(connection, &msg, sizeof(msg.bye),
				(void*)(uintptr_t)0xdeadbeef, 0);
		if (!ret) {
			int done = 0, i = 0;
			cci_event_t *event = NULL;

			do {
				ret = cci_get_event(endpoint, &event);
				if (!ret) {
					const io_msg_t *rx = NULL;

					switch (event->type) {
					case CCI_EVENT_SEND:
						assert(event->send.context ==
								(void*)(uintptr_t)0xdeadbeef);
						done++;
						break;
					case CCI_EVENT_RECV:
						rx = event->recv.ptr;
						assert(rx->type == FINISHED || rx->type == WRITE_DONE);
						if (rx->type == FINISHED)
							done++;
						break;
					default:
						break;
					}
					cci_return_event(event);
				}
			} while (done < 2);

			/* to allow CCI to ack the send? */
			for (i = 0; i < 10; i++)
				cci_get_event(endpoint, &event);

		}

		ret = cci_disconnect(connection);
		if (ret) {
			fprintf(stderr, "%s: cci_disconnect() failed with %s\n",
					__func__, cci_strerror(endpoint, ret));
		}
	}

	if (local) {
		ret = cci_rma_deregister(endpoint, local);
		if (ret) {
			fprintf(stderr, "%s: cci_rma_deregister() failed with %s\n",
					__func__, cci_strerror(endpoint, ret));
		}
	}

	if (endpoint) {
		ret = cci_destroy_endpoint(endpoint);
		if (ret) {
			fprintf(stderr, "%s: cci_destroy_endpoint() failed with %s\n",
					__func__, cci_strerror(NULL, ret));
		}
	}

	ret = cci_finalize();
	if (ret) {
		fprintf(stderr, "%s: cci_destroy_endpoint() failed with %s\n", __func__,
				cci_strerror(NULL, ret));
	}

	sleep(1);

	if (pid != -1) {
		signal(SIGCHLD, SIG_IGN);
		kill(pid, SIGKILL);
		waitpid(pid, NULL, 0);
	}

	if (ifd != -1) {
		close(ifd);
	}

	memset(hostname, 0, sizeof(hostname));
	gethostname(hostname, sizeof(hostname));

	unlink(hostname);

	return ret;
}
