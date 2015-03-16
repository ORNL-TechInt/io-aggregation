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
        int c = getopt_long( argc, argv, "S:G:A:", long_options, &option_index);

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
                
            default:  // unrecognized option
                return false;
                
        }
    } // end while(1)
        
    return true;  // no parse errors
}


void printUsage(char *name)
{
    cerr <<  "Usage: " << name << " [-S <MB>] [-G <MB>] "
         << endl;

    cerr << "where:" << endl;
    cerr << "\t-S\tAmount of system ram to use for cache (in MB)." << endl
         << "\t\t0 for no system ram cache, -1 for unlimited. Default: "
         << MAX_SYS_RAM << endl;
    cerr << "\t-G\tAmount of GPU ram to use for cache (in MB)." << endl
         << "\t\t0 for no GPU ram cache, -1 for unlimited. Default: "
         << MAX_GPU_RAM << endl;
    cerr << "\t-A\tSelect allocator for GPU memory.  Valid options are:" << endl;
    cerr << "\t\t1 - First come, first served allocator" << endl;
    cerr << "\t\t2 - Fixed size block allocator" << endl;
}

