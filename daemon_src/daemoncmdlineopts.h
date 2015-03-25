// Command line options for new_test

#ifndef _DAEMON_CMDLINEOPTS_H_
#define _DAEMON_CMDLINEOPTS_H_

#include "gpumem.h"

#include <stdlib.h>

// Default vaules
#define MAX_SYS_RAM      0 // in MB (-1 means use all available)
#define MAX_GPU_RAM     -1 // in MB (-1 means use all available)
#define DEFAULT_ALLOC    allocate_fcfs  // "First come / first served" allocation
#define WRITE_THREADS    1 // one background write thread
#define CORE_PINNING     false 
#define BLOCKING_MODE    false

struct CommandLineOptions {
    // cache sizes
    int maxSysRam;      // in MB, 0 for no system ram cache, -1 for unlimited
    int maxGpuRam;      // in MB, 0 for no GPU ram cache, -1 for unlimited
    unsigned writeThreads;  // number of background write threads
    
    uint64_t (*gpuAlloc)( void **devPtr, uint64_t reqLen);
                        // Pointer to GPU mem allocation functions
    
    bool corePinning;  // Attempt to pin threads to odd numbered cores?
    bool blockingMode; // Run the comm loop in blocking mode    
    CommandLineOptions() : maxSysRam(MAX_SYS_RAM), maxGpuRam(MAX_GPU_RAM),
                           writeThreads( WRITE_THREADS), gpuAlloc( DEFAULT_ALLOC),
                           corePinning( CORE_PINNING), blockingMode(BLOCKING_MODE) { }
    
};

bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts); 
void printUsage( char *name);

#endif // _DAEMON_CMDLINEOPTS_H_
