#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>

#include "mpi.h"

#include "io.h"

#define ITERS		(10)		/* Number of iterations */
#define SLEEP_SECS	(60)		/* Number of seconds to sleep */
#define MB		(1024*1024)	/* MB */
#define MIN_LENGTH	(1 * MB)	/* 1 MB */
#define MAX_LENGTH	(128 * MB)	/* 128 MB */
#define EXTRA_RAM   0           /* in MB */

int use_io_agg = 0;
int fd = -1, rank = -1;

static void
print_usage(char *name)
{
	fprintf(stderr, "usage: %s [-i <iterations>] [-s <sleep_seconds>] "
			"[-m <min_length>] [-M <max_length>] [-f]\n", name);
	fprintf(stderr, "where:\n");
	fprintf(stderr, "\t-i\tNumber of iterations (default %d)\n", ITERS);
	fprintf(stderr, "\t-s\tSeconds between writes (default %d)\n", SLEEP_SECS);
	fprintf(stderr, "\t-m\tMinimun length (default %d)\n", MIN_LENGTH);
	fprintf(stderr, "\t-M\tMaximum length (default %d)\n", MAX_LENGTH);
	fprintf(stderr, "\t-a\tStart the IO aggregation daemon\n");
	fprintf(stderr, "\t-r\tSize of the RMA buffer (for aggregation). "
	                "In MB (default %d)\n", MAX_LENGTH / (1024*1024));
    fprintf(stderr, "\t-e\tAllocate extra memory. In MB (default %d)\n", EXTRA_RAM);
	exit(EXIT_FAILURE);
}

static uint64_t
usecs(struct timeval start, struct timeval end)
{
	return (uint64_t)(end.tv_sec - start.tv_sec) * (uint64_t)1000000 +
		(uint64_t)(end.tv_usec - start.tv_usec);
}

static void
init_buffer(void *buf, size_t len, int seed)
{
	int i = 0, *ip = buf;

	for (i = 0; i < (len / sizeof(*ip)); i++)
		ip[i] = seed + i;

	return;
}

static void
write_it(void *buf, size_t left, int force_local)
{
	ssize_t rc = 0;
	size_t offset = 0;

	if (use_io_agg && !force_local) {
		io_write(left);
	} else {
		do {
			rc = write(fd, (void*)((uintptr_t)buf + offset), left);
			if (rc > 0) {
				left -= rc;
				offset += rc;
			} else if (rc == -1) {
				if (errno == EINTR) {
					continue;
				} else {
					perror("write()");
				}
			} else {
				fprintf(stderr, "rank %d: write() returned 0\n", rank);
			}
		} while (left);
	}
}

/* Allocate and then touch (to ensure the linux kernel has actually
 * allocated the memory) 'size' bytes of ram.
 */
void *allocate_and_touch( size_t size)
{
	fprintf(stderr, "Allocating %ld MB of ram...", size / (1024*1024));

	void *mem = malloc( size);

	long pgsize = sysconf(_SC_PAGESIZE);
	char *vals = (char *)mem;
	size_t i = 0;
	while (i < size) {
		vals[i] = 'A'; /* just write something - doesn't matter what */
		i+=pgsize;
	}

	fprintf(stderr, "Done.\n");
	return mem;
}


int main(int argc, char *argv[])
{
	int c = 0, iters = ITERS, secs = SLEEP_SECS;
	int rc = 0, ranks = 0, i = 0, j = 0;
	size_t len = MIN_LENGTH, max = MAX_LENGTH, tmp = 0;
	char fname[32];
	void *buf = NULL;
	uint64_t *latencies = NULL;
	int extra_ram_mb = EXTRA_RAM;
	char *extra_ram = NULL;

	while ((c = getopt(argc, argv, "i:s:m:M:a")) != -1) {
		switch (c) {
		case 'i':
			iters = strtol(optarg, NULL, 0);
			break;
		case 's':
			secs = strtol(optarg, NULL, 0);
			break;
		case 'm':
			len = strtoull(optarg, NULL, 0);
			break;
		case 'M':
			max = strtoull(optarg, NULL, 0);
			break;
		case 'a':
			use_io_agg = 1;
			break;
		default:
			print_usage(argv[0]);
		}
	}

	if (max < len)
		max = len;

	/* allocate a bunch of ram - just as if this was a real, useful program
	 * rather than a microbenchmark.
	 */
	if (extra_ram_mb > 0) {
		extra_ram = (char *)allocate_and_touch( extra_ram_mb * (size_t)(1024 * 1024));
	}

	/* determine how many sizes we will test given max and len.
	 * reuse c to count the sizes. */
	c = 0;
	tmp = max;

	do {
		c++;
		tmp = tmp >> 1;
	} while (tmp >= len);

	latencies = calloc(c*iters, sizeof(*latencies));
	if (!latencies) {
		perror("calloc():");
		exit(EXIT_FAILURE);
	}

	buf = malloc(max);
	if (!buf) {
		perror("malloc():");
		exit(EXIT_FAILURE);
	}

	init_buffer(buf, max, rank);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranks);

	if (use_io_agg) {
		rc = io_init(buf, max, rank, ranks);
		/* TODO handle error */
		if (rc)
			exit(EXIT_FAILURE);
	}

	memset(fname, 0, sizeof(fname));
	snprintf(fname, sizeof(fname), "rank-%d", rank);
	rc = open(fname, O_CREAT|O_TRUNC|O_RDWR, 0600);
	if (rc == -1) {
		perror("open():");
		exit(EXIT_FAILURE);
	}
	fd = rc;

	/* File is open, wait for everyone else */
	MPI_Barrier(MPI_COMM_WORLD);

	/* save starting length for later */
	tmp = len;

	for (j = 0; len <= max; j++) {
		if (rank == 0)
			fprintf(stderr, "Starting size %zu: ", len);

		for (i = 0; i < iters; i++) {
			size_t left = len;
			struct timeval start, end;

			init_buffer(buf, len, rank + i);

			/* Sync up */
			MPI_Barrier(MPI_COMM_WORLD);

			gettimeofday(&start, NULL);
			write_it(buf, left, 0);
			gettimeofday(&end, NULL);

			latencies[(j * iters) + i] = usecs(start, end);

			if (rank == 0)
				fprintf(stderr, "%d ", i);

			/* Sleep instead of doing work for now */
			sleep(secs);
		}
		/* truncate the file - we may be writing a lot of data */
		ftruncate(fd, 0);
		lseek(fd, 0, SEEK_SET);

		if (rank == 0)
			fprintf(stderr, "\nCompleted size %zu\n", len);

		len = len * (size_t)2;
	}

	/* store the results in the already truncated file to avoid
	 * all ranks dumping to stdout and getting intermingled. */

	len = tmp;

	for (j = 0; len <= max; j++) {
		char line[64];

		memset(line, 0, sizeof(line));
		snprintf(line, sizeof(line), "size: %10zu iters: %d latencies: ", len, iters);
		write_it(line, strlen(line), 1);

		for (i = 0; i < iters; i++) {
			memset(line, 0, sizeof(line));
			snprintf(line, sizeof(line), "%"PRIu64" ", latencies[(j * iters) + i]);
			write_it(line, strlen(line), 1);
		}

		memset(line, 0, sizeof(line));
		snprintf(line, sizeof(line), "\n");
		write_it(line, strlen(line), 1);

		len = len * (size_t)2;
	}

	/* We are done, clean up */
	MPI_Barrier(MPI_COMM_WORLD);

	if (use_io_agg) {
		io_finalize();
	} else {
		rc = close(fd);
		if (rc == -1) {
			perror("close():");
			exit(EXIT_FAILURE);
		}
	}

	MPI_Finalize();

	free( extra_ram);
	
	return 0;
}
