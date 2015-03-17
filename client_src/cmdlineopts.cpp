// Command line options (implementation) for new_test


#include "cmdlineopts.h"

#include <getopt.h>

#include <iostream>
using namespace std;


static struct option long_options[] = {
    {"iters",            required_argument, 0, 'i'},
    {"sleep",            required_argument, 0, 's'},
    {"min_len",          required_argument, 0, 'm'},
    {"max_len",          required_argument, 0, 'M'},
    {"client_blocking",  no_argument,       0, 'b'},
    {"daemon_blocking",  no_argument,       0, 'B'},
    {"null_io",          no_argument,       0, 'n'},
    {"extra_ram",        required_argument, 0, 'e'},
    {"rma_buf",          required_argument, 0, 'r'},
    {"use_daemon",       no_argument,       0, 'd'},
    {"no_auto_start",    no_argument,       0, 'N'},
    {0, 0, 0, 0} };


bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts)
{
    while (1)
    {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c = getopt_long( argc, argv, "i:s:m:M:bBne:r:dN", long_options, &option_index);

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
                opts.minLen = atoi(optarg);
                break;
            case 'M':
                opts.maxLen = atoi(optarg);
                break;
            case 'b':
                opts.clientBlocking = true;
                break;
            case 'B':
                opts.daemonBlocking = true;
                break;
            case 'n':
                opts.nullIo = true;
                break;
            case 'e':
                opts.extraRam = atoi(optarg);
                break;
            case 'r':
                opts.rmaBuf = atoi(optarg);
                break;
            case 'd':
                opts.useDaemon = true;
                break;
            case 'N':
                opts.daemonAutostart = false;
                break;
            
            default:  // unrecognized option
                return false;
                
        }
    } // end while(1)
        
    return true;  // no parse errors
}


void printUsage(char *name)
{
    cerr <<  "Usage: " << name << " [-i <iterations>] [-s <sleep_seconds>] "
         << "[-m <min_length>] [-M <max_length>] [-d] [-r] "
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
    cerr << "\t-N\tDisable automatic startup of the remote daemon" << endl
         << "\t\t(Useful if running the daemon in the debugger.)" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-b\tUse CCI blocking mode on client" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-B\tUse CCI blocking mode on iod daemon" << endl;
    cerr << "** NOT IMPLEMENTED **" << "\t-N\tUse NULL IO on iod daemon" << endl;
}

