// A class for handling IO Requests from the peers
// Mainly, it keeps track of timing information

#ifndef _IO_REQUEST_H_
#define _IO_REQUEST_H_

#include "timing.h"

#include <ostream>

// TODO: should IoRequest be ref-counted?
class IoRequest {
    
public:
    // Peer *      peer; /* Client peer */
    uint64_t    rx_us;  // microsecs when write request is first received 
    uint64_t    cpy_us; // microsecs when copy completes 
    uint64_t    rma_us; // microsecs when RMA completes 
    uint64_t    deq_us; // microsecs when dequeued by the write thread
    uint64_t    io_us;  // microsecs when write() completes 
    uint32_t    len;    // Requesting write of len bytes
    
    IoRequest(uint32_t req_len)
      : rx_us( getUs()), cpy_us( 0), rma_us( 0), deq_us( 0), io_us( 0), len(req_len) {}

    std::ostream &writeResults( std::ostream &outf);
    
    
    // NOTE: Originally, cpy_us & rma_us might have been different time
    // values.  However, with the current scheme there is no extra copy.
    
    // Note: Because a single write operation (from the client's perspective)
    // might have to be split into multiple RMA's or cudaMemcpy's (because
    // there wasn't enough cache for the entire operation, for example), 
    // the user should be aware that cpy_us, rma_us, deq_us and io_us might
    // get updated multiple times.  (rx_us will only be set once.)
};

#endif // _IO_REQUEST_H_

