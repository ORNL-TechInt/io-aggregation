// Utilities for measuring time that both the client and daemon can use

#ifndef _TIMING_H_
#define _TIMING_H_

#include <stdint.h>
#include <stdlib.h>  // needed for definition of NULL
#include <sys/time.h>

// Note: for Linux systems, we might want to switch over to using clock_gettime()
// Apparently, it's not available on Macs, though.

static uint64_t getUs(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec * 1000000) + (uint64_t)tv.tv_usec;
}



#endif // _TIMING_H_