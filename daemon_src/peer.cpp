// Implemention file for the Peer class

#include "peer.h"

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

Peer::Peer( uint32_t theRank)
 : rank(theRank), done(false)
{
    // Initialize the mutex
    pthread_mutex_init(&mut, NULL);

    // Open the output file
    ostringstream fname("");
    fname << "rank-" << rank << "-iod-data";
    outf.open( fname.str().c_str());
    if (!outf) {
        cerr << "Failed to open " << fname.str() << "!  Aborting!!" << endl;
        exit( -1);
    }
}


Peer::~Peer()
{
    // Close the output file
    if (outf) {
        outf.close();
    }
    
    // Destroy the mutex
    pthread_mutex_destroy( &mut);
    
}