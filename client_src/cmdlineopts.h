// Command line options for new_test

#ifndef _CMDLINEOPTS_H_
#define _CMDLINEOPTS_H_

#include <stdlib.h>

#include <vector>
#include <string>

// Default vaules
#define ITERS            (10)        // Number of iterations 
#define SLEEP_SECS       (-1)        // Number of seconds to sleep (-1 is automatic mode)
#define MB               (1024*1024) // MB 
#define MIN_LENGTH       (1 * MB)    // 1 MB 
#define MAX_LENGTH       (128 * MB)  // 128 MB 
#define EXTRA_RAM        0           // in MB 
#define NULL_IO          false       // use null io option
#define CLIENT_BLOCKING  false       // use CCI blocking mode on the client
#define DAEMON_BLOCKING  false       // use CCI blocking mode on the server
#define USE_DAEMON       false       // whether or not to hand writes off to the daemon process
#define DAEMON_AUTOSTART true        // automatically start the daemon
struct CommandLineOptions {

    unsigned iters;
    int sleepSecs;
    size_t minLen;
    size_t maxLen;
    int extraRam;  // in megabytes
    bool nullIo;
    bool clientBlocking;
    bool daemonBlocking;
    size_t rmaBuf;  // in megabytes
    bool useDaemon;
    bool daemonAutostart;
    char **daemonArgs;   

    CommandLineOptions();
    ~CommandLineOptions();
    
    void addDaemonArg( const char*);
    char ** getDaemonArgs() { return daemonArgs; }
    // storing the daemon arguments in a string vector would be far more
    // convenient, but unfortunately execve() takes a char * const *. ie:
    // The actual argument values must be mutable.  Since string::c_str()
    // returns a const char *, we can't use strings and thus get to do
    // everything the hard way...
     
private:
    
    unsigned curDaemonArgs, maxDaemonArgs;
    
};

bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts); 
void printUsage( char *name);

#endif // _CMDLINEOPTS_H_