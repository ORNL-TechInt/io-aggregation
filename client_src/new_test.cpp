// This is basically a complete re-write of the test.c
// It does the same basic tests and should write timing statistics that
// are directly comparable, but since I had to include CUDA code, I decided
// to start over from scratch.  (And since I was starting from scratch, I
// decided to use C++ instead of C
// - RGM - 23 Feb 2015



#include "cmdlineopts.h"
#include "timing.h"
#include "utils.h"

#include <cci.h>
#include <mpi.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

struct TimeStamp {
    uint64_t len;       // length of the write
    uint64_t start_us;  // start time (in microseconds)
    uint64_t end_us;    // end time (in microseconds)
    
    TimeStamp( uint64_t theLen) : len(theLen) { }
};


static void initBuffer(void *buf, size_t len, unsigned seed);

int main( int argc, char **argv)
{
    CommandLineOptions cmdOpts;
    int rank;
    int ranks;
    vector <TimeStamp> timeStamps;
    ostringstream fname;  // used for building up file names
    ofstream outf;  // stream for write the data to (if we're not using the daemon)
    
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
    
    // If we're not using the daemon, we'll need a file we can write to
    if (cmdOpts.useDaemon == false) {
        fname.str("");
        fname << "rank-" << rank << "-data";
        outf.open(fname.str().c_str());
        
        if (!outf) {
            cerr << "Rank " << rank << ": failed to open " << fname.str()
                << "!  Aborting!" << endl;
            return -1;
        }
    }
        
    // Initialize CCI & start up the daemon
    if (cmdOpts.useDaemon) {
        int rc = 0;
        if (cmdOpts.daemonAutostart) {
            char **daemonArgs = new char *[2];
            daemonArgs[0] = (char *)"daemon";
            daemonArgs[1] = NULL;
            rc = startOneDaemon( daemonArgs);
            if (rc) {
                cerr << "System error " << rc << " starting daemon: " 
                     << strerror( rc) << endl;
                cerr << "Aborting" << endl;
                return -1;
            }
            
            delete[] daemonArgs;
        }
        
        rc = initIo(buf, cmdOpts.maxLen, rank);
        if (rc) {         
            cerr << "CCI error " << rc << " during initialization: " 
                 << cci_strerror( NULL,(cci_status)rc) << endl;
            cerr << "Aborting" << endl;
            return -1;
        }
    }
    

    // Initialization steps are complete - wait for all ranks
    MPI_Barrier( MPI_COMM_WORLD);
    
    uint64_t len = cmdOpts.minLen;
    
    for (unsigned j = 0; len <= cmdOpts.maxLen; j++) {
        if (rank == 0)
            cerr << "Starting size " << len << ":" << endl;

        for (unsigned i = 0; i < cmdOpts.iters; i++) {           
            TimeStamp ts( len);

            initBuffer(buf, len, rank + i);

            /* Sync up */
            MPI_Barrier(MPI_COMM_WORLD);

            ts.start_us = getUs();
            if (cmdOpts.useDaemon) {
                writeRemote(buf, len);
            } else {
                writeLocal(buf, len, outf);
            }
            ts.end_us = getUs();
            
            timeStamps.push_back( ts);

            if (rank == 0)
                cerr << i << " ";

            // Sleep instead of doing work for now
            sleep( cmdOpts.sleepSecs);
        }

        if (rank == 0)
            cerr << endl << "Completed size " << len << endl;

        len = len * (size_t)2;
    }

    
    // Close the data file (if we were using it)
    if (outf) {
        outf.close();
    }
        

    // Write the time stamp data out to a file
    fname.str("");
    fname << "rank-" << rank;
    outf.open( fname.str().c_str());
    if (outf) {
        for (unsigned i=0; i < timeStamps.size(); i++) {
            outf << "len " << timeStamps[i].len
                << " start " << timeStamps[i].start_us
                << " end " << timeStamps[i].end_us << endl;
        }
        outf.close();
        
    } else { // failed to open the file!?!
        cerr << "Rank " << rank << ": failed to open results file! "
            << "No results will be saved!" << endl;
    }
    
    if (cmdOpts.useDaemon) {
        finalizeIo();
    }
    MPI_Finalize();

    return 0;
}



static void initBuffer(void *buf, size_t len, unsigned seed)
{
        unsigned *b = (unsigned *)buf;
        size_t num_vals = len / sizeof( unsigned);
        for (size_t i = 0; i < num_vals; i++) {
                b[i] = seed + i;
        }

        return;
}





