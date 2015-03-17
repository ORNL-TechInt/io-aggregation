// This is basically a complete re-write of the test.c
// It does the same basic tests and should write timing statistics that
// are directly comparable, but since I had to include CUDA code, I decided
// to start over from scratch.  (And since I was starting from scratch, I
// decided to use C++ instead of C
// - RGM - 23 Feb 2015



#include "cmdlineopts.h"
#include "cuda_util.h"
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

    
    // Make sure the GPU is in the proper compute mode, or this isn't going to work
    struct cudaDeviceProp prop;
    CUDA_CHECK_RETURN( cudaGetDeviceProperties ( &prop, 0));
    if (prop.computeMode != 0) {
        cerr << "Improper \"computeMode\" property: " << prop.computeMode 
             << " Property value must be 0 to run." << endl
             << "Aborting!" << endl;
        return -1;
    }

    // MPI initialization stuff
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    
    
    unsigned char* buf;
    bool bufIsPinned = false;
    
#ifndef DISABLE_DAEMON_PINNED_MEMORY
    cudaError_t cudaErr = cudaMallocHost( &buf, cmdOpts.maxLen);
    if (cudaErr == cudaSuccess) {
        bufIsPinned = true;
    } else {
        // print a warning, then fall back to using regular memory
        cerr << __FILE__ << ":" << __LINE__
             << ": WARNING: failed to allocate pinned memory. "
             << "Falling back to normal allocation." << endl;
    }
#endif

    if (! bufIsPinned) {
        // either the cudaMallocHost() call failed, or it wasn't compiled in
        buf = new unsigned char[cmdOpts.maxLen];
    }
    
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

            unsigned numWriteIterations = 0;
            ts.start_us = getUs();
            if (cmdOpts.useDaemon) {
                size_t totalBytesWritten = 0;
                size_t bytesWritten;
                while (totalBytesWritten < len) {
                    size_t bytesToWrite = len - totalBytesWritten;
                    writeRemote( &buf[totalBytesWritten], bytesToWrite, &bytesWritten);
                    if (bytesWritten == 0) {
                        // Out of space in the GPU memory - sleep briefly to
                        // keep from thrashing the system with write requests
                        // that can't be fullfilled
                        usleep( 10 * 1000);
                        // TODO: A better idea would be to write a small amount using
                        // writeLocal(), but in this test clients don't actually have
                        // a file open when they're running in daemon mode...
                    }
                    numWriteIterations++;
                    totalBytesWritten += bytesWritten;
                }
                
            } else {
                writeLocal(buf, len, outf);
                numWriteIterations++;
            }
            ts.end_us = getUs();
            
            timeStamps.push_back( ts);

            if (rank == 0)
                cerr << i << "(" << numWriteIterations << ") ";

            // Wait for the cache to drain (by sleeping)
            // If we're in automatic mode, check with the daemon for the
            // cache usage.  Otherwise, sleep for a fixed amount of time
            if (cmdOpts.sleepSecs == -1) {
                bool isEmpty = false;
                checkCacheUsage( &isEmpty);
                while ( ! isEmpty) {
                    usleep( 100 * 1000);
                    checkCacheUsage( &isEmpty);
                }
                
                sync();  // daemon's cache is empty - this flushes the OS pagecache
            } else if (cmdOpts.sleepSecs > 0) {
                sleep( cmdOpts.sleepSecs);
            }
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
            double secElapsed = (double)(timeStamps[i].end_us - timeStamps[i].start_us) / 1000000.0;
            outf << "len " << timeStamps[i].len
                << " start " << timeStamps[i].start_us
                << " end " << timeStamps[i].end_us
                << " elapsed " << timeStamps[i].end_us - timeStamps[i].start_us
                << " MB/s " <<  (timeStamps[i].len / (1024.0*1024.0)) / secElapsed
                << endl;
                
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
    
    if (bufIsPinned) {
        CUDA_CHECK_RETURN( cudaFreeHost( buf));
    }
    else {
        delete[] buf;
    }

    return 0;
}



static void initBuffer(void *buf, size_t len, unsigned seed)
{
        uint32_t *b = (uint32_t *)buf;
        size_t num_vals = len / sizeof( uint32_t);
        for (size_t i = 0; i < num_vals; i++) {
                b[i] = seed + i;
        }

        // Add checksums every 1K to make it easier to
        // verify the validity of the output files
        uint32_t checksum = 0;
        for (size_t i = 0; i < num_vals; i++) {
                if ((i+1) % (1024 / sizeof( uint32_t)) == 0) {
                    b[i] = checksum;
                    checksum = 0;
                } else {
                    checksum += b[i];
                }
        }
        
        return;
}





