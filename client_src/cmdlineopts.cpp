// Command line options (implementation) for new_test


#include "cmdlineopts.h"

#include <getopt.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>
using namespace std;


static struct option long_options[] = {
    {"iters",            required_argument, 0, 'i'},
    {"sleep",            required_argument, 0, 's'},
    {"min_len",          required_argument, 0, 'm'},
    {"max_len",          required_argument, 0, 'M'},
    {"client_blocking",  no_argument,       0, 'b'},
    {"null_io",          no_argument,       0, 'n'},
    {"extra_ram",        required_argument, 0, 'e'},
    {"rma_buf",          required_argument, 0, 'r'},
    {"use_daemon",       no_argument,       0, 'd'},
    {"daemon_arg",       required_argument, 0, 'D'},
    {"no_auto_start",    no_argument,       0, 'N'},
    {0, 0, 0, 0} };


bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts)
{
    while (1)
    {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c = getopt_long( argc, argv, "i:s:m:M:bne:r:dD:N", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
                break;

        
        switch (c)
        {
            case 'i':
                opts.iters = atoi( optarg);
                break;
            case 's':
                opts.sleepSecs = atoi( optarg);
                break;
            case 'm':
                opts.minLen = atol(optarg);
                break;
            case 'M':
                opts.maxLen = atol(optarg);
                break;
            case 'b':
                opts.clientBlocking = true;
                break;
            case 'n':
                opts.nullIo = true;
                break;
            case 'e':
                opts.extraRam = atoi(optarg);
                break;
            case 'r':
                opts.rmaBuf = atol(optarg);
                break;
            case 'd':
                opts.useDaemon = true;
                break;
            case 'D':
                // We need to make sure the appropriate '-' char(s) are added
                // to the argument strings so that the daemon's calls to
                // getopt_long work correctly.
                // 
                opts.addDaemonArg( optarg);
                break;
            case 'N':
                opts.daemonAutostart = false;
                break;
            
            default:  // unrecognized option
                return false;
                
        }
    } // end while(1)
    
    // Check for any 'extra' positional arguments (ie: ones that didn't
    // start with '-' or '--'
    if (optind != argc) {
        cerr << "WARNING (" << argv[0] << "): Unexpected positional arguments: ";
        for (int i=optind; i < argc; i++) {
            cerr << argv[i] << " ";
        }
        cerr << endl;
        cerr << "Did you forget a '-' character?" << endl;
    }
// TODO: We ought to add some code to validate the options we've got - there's a numer
// of combinations that are invalid (ex: sleepSecs of -1 requires useDaemon)
    return true;  // no parse errors
}


void printUsage(char *name)
{
    cerr <<  "Usage: " << name << " [-i <iterations>] [-s <sleep_seconds>] "
         << "[-m <min_length>] [-M <max_length>] [-d] [-D <daemon_arg>] [-r] "
         << "[-e <megabytes>]" << endl;

    cerr << "where:" << endl;
    cerr << "\t-i\tNumber of iterations (default " << ITERS << ")" << endl;
    cerr << "\t-s\tSeconds between writes (default " << SLEEP_SECS << ")" << endl;
    cerr << "\t  \t  (-1 says to sleep until the daemon has flushed its cache.)" << endl;
    cerr << "\t-m\tMinimun length (default " << MIN_LENGTH << ")" << endl;
    cerr << "\t-M\tMaximum length (default " << MAX_LENGTH << ")" << endl;
    
    cerr << "** NOT IMPLEMENTED **" << "\t-r\tSize of the RMA buffer (for aggregation). "
         << "In MB (default " << MAX_LENGTH / (1024*1024) << ")" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-e\tAllocate extra memory. In MB (default " << EXTRA_RAM << ")" << endl;
    
    cerr << "\t-d\tSend writes over to the remote daemon" << endl;
    cerr << "\t-D\tPass the argument through to the daemon" << endl;
    cerr << "\t-N\tDisable automatic startup of the remote daemon" << endl
         << "\t\t(Useful if running the daemon in the debugger.)" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-b\tUse CCI blocking mode on client" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-N\tUse NULL IO on iod daemon" << endl;
    cerr << endl;
    cerr << "A note about passing parameters to the daemon:  be sure to include the " << endl
         << "appropriate '-' characters in the parameters to be passed.  For example:" << endl
         << "to pass \"-T2 -A 2 --gpu_ram=512\" (without the quotes) into the daemon, " << endl
         << "the command line would look like: " << endl
         << "\"" << name << " -D-T2 -D-A -D2 -D--gpu_ram=512\" (again, without the quotes)." << endl
         << "The extra dashes look weird, but that's how everything works." << endl;
}


CommandLineOptions::CommandLineOptions() :
     iters(ITERS), sleepSecs(SLEEP_SECS), minLen(MIN_LENGTH), maxLen(MAX_LENGTH),
     extraRam(EXTRA_RAM), nullIo(NULL_IO), clientBlocking(CLIENT_BLOCKING),
     rmaBuf(MAX_LENGTH / (1024*1024)),
     useDaemon( USE_DAEMON), daemonAutostart(DAEMON_AUTOSTART)
{ 
    // Initialize the daemonArgs array...
    maxDaemonArgs = 25;  // arbitrary initial value that is probably larger than
                         // we'll ever need...
    curDaemonArgs = 0;
    daemonArgs = (char **)calloc( maxDaemonArgs, sizeof( *daemonArgs));
    // Note: using calloc() instead of new so that addDaemonArg() can call
    // realloc() if necessary...
    
    addDaemonArg( "daemon");
    // Hard-coded name of the actual daemon executable
    // TODO: It'd be nice if this wasn't hard-coded.  Can we do something via the Makefile
    // since it knows what the name of the daemon executable is?
}

CommandLineOptions::~CommandLineOptions()
{
    for (unsigned i=0; i < curDaemonArgs; i++) {
        free( daemonArgs[i]);
    }
    
    free( daemonArgs);
}



// storing the daemon arguments in a string vector would be far more
// convenient, but unfortunately execve() takes a char * const *. ie:
// The actual argument values must be mutable.  Since string::c_str()
// returns a const char *, we can't use strings and thus get to do
// everything the hard way...
void CommandLineOptions::addDaemonArg( const char *arg)
{
    if (curDaemonArgs == maxDaemonArgs) {
        // need more pointers
        maxDaemonArgs *= 2;
        daemonArgs = (char **)realloc( daemonArgs, maxDaemonArgs * sizeof( char *));
    }

    char *temp = (char *)malloc(strlen( arg)+1);
    strcpy( temp, arg);
    daemonArgs[curDaemonArgs] = temp;
    curDaemonArgs++;
}   
    
 
