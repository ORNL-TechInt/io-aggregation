// Implemention file for the Peer class

#include "peer.h"
#include "cci_util.h"

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

Peer::Peer( uint32_t rank)
 : m_conn( NULL), m_rank(rank), m_done(false)
{
    // Initialize the mutex
    pthread_mutex_init(&m_mut, NULL);

    // Open the output file
    ostringstream fname("");
    fname << "rank-" << rank << "-iod-data";
    m_outf.open( fname.str().c_str());
    if (!m_outf) {
        cerr << "Failed to open " << fname.str() << "!  Aborting!!" << endl;
        exit( -1);
    }
}


Peer::~Peer()
{
    int ret;
    // Close the CCI connection
    if (m_conn) {
        ret = cci_disconnect( m_conn);
        if (ret) {
            cciDbgMsg("cci_disconnect()", ret);
        }
    }
    
    // Close the output file
    if (m_outf) {
        m_outf.close();
    }
    
    // Destroy the mutex
    pthread_mutex_destroy( &m_mut);
    
    writeStatistics();  // Save the transfer time data we recorded...
    
}

void Peer::writeStatistics()
{
    // TODO: Implement me!
    cerr << "Peer::writeStatistics not implemented yet!" << endl;
}