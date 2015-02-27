// Command line options for new_test

#ifndef _CMDLINEOPTS_H_
#define _CMDLINEOPTS_H_

#include <stdlib.h>

// Default vaules
#define ITERS       (10)        // Number of iterations 
#define SLEEP_SECS  (60)        // Number of seconds to sleep 
#define MB          (1024*1024) // MB 
#define MIN_LENGTH  (1 * MB)    // 1 MB 
#define MAX_LENGTH  (128 * MB)  // 128 MB 
#define EXTRA_RAM   0           // in MB 
#define NULL_IO     false       // use null io option
#define CLIENT_BLOCKING false   // use CCI blocking mode on the client
#define DAEMON_BLOCKING false   // use CCI blocking mode on the server
struct CommandLineOptions {

    int iters;
    int sleepSecs;
    size_t minLen;
    size_t maxLen;
    int extraRam;  // in megabytes
    bool nullIo;
    bool clientBlocking;
    bool daemonBlocking;
    size_t rmaBuf;  // in megabytes
    
    
    CommandLineOptions() :
     iters(ITERS), sleepSecs(SLEEP_SECS), minLen(MIN_LENGTH), maxLen(MAX_LENGTH),
     extraRam(EXTRA_RAM), nullIo(NULL_IO), clientBlocking(CLIENT_BLOCKING),
     daemonBlocking(DAEMON_BLOCKING), rmaBuf(MAX_LENGTH / (1024*1024))  { }
    
};

bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts); 
void printUsage( char *name);

#endif // _CMDLINEOPTS_H_