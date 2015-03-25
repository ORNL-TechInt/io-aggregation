// Command line options (implementation) for the daemon


#include "daemoncmdlineopts.h"
#include "gpumem.h"

#include <getopt.h>

#include <iostream>
using namespace std;


static struct option long_options[] = {
    {"sys_ram",        required_argument, 0, 'S'},
    {"gpu_ram",        required_argument, 0, 'G'},
    {"gpu_allocator",  required_argument, 0, 'A'},
    {"write_threads",  required_argument, 0, 'T'},
    {"core_pinning",   no_argument,       0, 'P'},
    {"blocking_mode",  no_argument,       0, 'B'},
    {0, 0, 0, 0} };
    
    
#if 0
    {"min_len",          required_argument, 0, 'm'},
    {"max_len",          required_argument, 0, 'M'},
    {"client_blocking",  no_argument,       0, 'b'},
    {"daemon_blocking",  no_argument,       0, 'B'},
    {"null_io",          no_argument,       0, 'n'},
    {"extra_ram",        required_argument, 0, 'e'},
    {"rma_buf",          required_argument, 0, 'r'},
#endif


bool parseCmdLine( int argc, char **argv, CommandLineOptions &opts)
{
    while (1)
    {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c = getopt_long( argc, argv, "S:G:A:T:PB", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
                break;
        
        switch (c)
        {
            case 'S':
                opts.maxSysRam = atoi( optarg);
                break;
            case 'G':
                opts.maxGpuRam = atoi( optarg);
                break;
            case 'A':
                // TODO: Ideally, I should figure a way to map functions automatically,
                // both here and in printUsage().  Maybe with macros??
                switch (atoi( optarg)) {
                    case 1:
                        opts.gpuAlloc = allocate_fcfs;
                        break;
                    case 2:
                        opts.gpuAlloc = allocate_block;
                        break;
                    default:
                        // Note: deliberately NOT using 0 because that's what atoi()
                        // will return if someone gives it a non-numeric string
                        return false;  // invalid allocator choise
                }
                break;
            case 'T':
                opts.writeThreads = atoi(optarg);
                if (opts.writeThreads < 1) {
                    return false;
                }
                break;
            case 'P':
                opts.corePinning = true;
                break;
            case 'B':
                opts.blockingMode = true;
                break;
                
            default:  // unrecognized option
                return false;
                
        }
    } // end while(1)
        
    return true;  // no parse errors
}


void printUsage(char *name)
{
    cerr <<  "Usage: " << name << " [-S <MB>] [-G <MB>] [-T <threads>] [-A <1|2>]"
         << endl;

    cerr << "where:" << endl;
    cerr << "\t-S\tAmount of system ram to use for cache (in MB)." << endl
         << "\t\t0 for no system ram cache, -1 for unlimited. Default: "
         << MAX_SYS_RAM << endl;
    cerr << "\t-G\tAmount of GPU ram to use for cache (in MB)." << endl
         << "\t\t0 for no GPU ram cache, -1 for unlimited. Default: "
         << MAX_GPU_RAM << endl;
    cerr << "\t-P\tAttempt to pin threads to specific odd-numbered cores"
         << endl;
    cerr << "\t-B\tBlocking mode: the event loop blocks waiting for events, instead of polling" << endl;
    cerr << "\tWARNING: Blocking mode is untested!  It requires an experimental version of CCI!" << endl;
    cerr << "\t-T\tNumber of background write threads. Default: " << WRITE_THREADS << endl;
    cerr << "\t-A\tSelect allocator for GPU memory.  Valid options are:" << endl;
    cerr << "\t\t1 - First come, first served allocator" << endl;
    cerr << "\t\t2 - Fixed size block allocator" << endl;
}

