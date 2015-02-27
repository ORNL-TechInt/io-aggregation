// Command line options (implementation) for the daemon


#include "daemoncmdlineopts.h"

#include <getopt.h>

#include <iostream>
using namespace std;


static struct option long_options[] = {
    {"sys_ram",        required_argument, 0, 'S'},
    {"gpu_ram",        required_argument, 0, 'G'},
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
        int c = getopt_long( argc, argv, "S:G:", long_options, &option_index);

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
}

