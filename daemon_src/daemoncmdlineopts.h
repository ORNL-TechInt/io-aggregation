// Command line options for new_test

#ifndef _DAEMON_CMDLINEOPTS_H_
#define _DAEMON_CMDLINEOPTS_H_

#include <stdlib.h>

// Default vaules
#define MAX_SYS_RAM      0 // in MB (-1 means use all available)
#define MAX_GPU_RAM     -1 // in MB (-1 means use all available)
// TODO: Do we need an option for blocking mode?

struct CommandLineOptions {
    // cache sizes
    int maxSysRam;      // in MB, 0 for no system ram cache, -1 for unlimited
    int maxGpuRam;      // in MB, 0 for no GPU ram cache, -1 for unlimited
    
    CommandLineOptions() : maxSysRam(MAX_SYS_RAM), maxGpuRam(MAX_GPU_RAM)  { }
    
};

bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts); 
void printUsage( char *name);

#endif // _DAEMON_CMDLINEOPTS_H_