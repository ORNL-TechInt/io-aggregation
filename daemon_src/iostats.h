// A class for handling IO Requests from the peers
// Mainly, it keeps track of timing information

#ifndef _IO_STATS_H_
#define _IO_STATS_H_

#include "timing.h"

#include <string.h>

#include <ostream>

class IoStats {
    
public:   
    // All times in microseconds (see timing.h)
    uint64_t    m_recv;     // write request message is first received 
    uint64_t    m_reply;    // reply sent back to the clien
    uint64_t    m_enqueue;  // done message received from client; block is moved to ready queue
    uint64_t    m_dequeue;  // background thread begins to work on the block
    uint64_t    m_done;     // I/O has been finished and GPU memory is freed
    

    uint64_t    m_requestedLen; // Size the client asked for (in bytes)
    uint64_t    m_actualLen;    // Size we actually granted (in bytes)

    
    // Default constructor initializes everything to invalid values.
    // (Default copy constructor & assignment operator are fine.)
    IoStats() { memset( this, 0, sizeof( *this)); }
    
    IoStats(uint64_t reqLen)
      : m_recv( getUs()), m_reply( 0), m_enqueue( 0), m_dequeue( 0),
        m_done( 0), m_requestedLen( reqLen), m_actualLen( 0) { }
    

    std::ostream &writeResults( std::ostream &outf);

private:

};

#endif // _IO_STATS_H_

