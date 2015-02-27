// This is basically a complete re-write of the test.c
// It does the same basic tests and should write timing statistics that
// are directly comparable, but since I had to include CUDA code, I decided
// to start over from scratch.  (And since I was starting from scratch, I
// decided to use C++ instead of C
// - RGM - 23 Feb 2015



#include "cmdlineopts.h"
#include "utils.h"

#include <cci.h>
#include <mpi.h>
#include <stdint.h>
#include <string.h>

#include <iostream>
using namespace std;

int main( int argc, char **argv)
{
    CommandLineOptions cmdOpts;
    int rank;
    int ranks;
    
    // First up - parse the command line options
    if (! parseCmdLine( argc, argv, cmdOpts)) {
        printUsage(argv[0]);
        return -1;
    }

    // MPI initialization stuff
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    
    
    unsigned char* buf = new unsigned char[cmdOpts.maxLen];
    
    // Initialize CCI & start up the daemon
    char **daemonArgs = new char *[2];
    daemonArgs[0] = (char *)"daemon";
    daemonArgs[1] = NULL;
    
    int rc = initIo(buf, cmdOpts.maxLen, rank, ranks, daemonArgs);
    if (!rc) {
        if (rc > 0) {
            cerr << "CCI error during initialization: " 
                 << cci_strerror( NULL,(cci_status)rc) << endl;
        }
        else {
            cerr << "System error during initialization: " 
                 << strerror( -rc) << endl;
        }
        cerr << "Aborting" << endl;
        return -1;
    }
    
    delete[] daemonArgs;
    
 
 
    finalizeIo();
    return 0;
}




