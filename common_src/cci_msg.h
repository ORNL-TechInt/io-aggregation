// Defines the messages that will be passed between the client and server


#ifndef _CCI_MSG_H_
#define _CCI_MSG_H_

#include "cci.h"

#include <cuda_runtime.h>
#include <cuda.h>

/* C = client - MPI rank
 * S = server - IO daemon
 * C->S = client to server message
 * C<-S = server to client message
 */

// Struct defining the messages we can send via CCI
enum IoMsgType {
    CONNECT = 0,       // C->S: rank
    CONNECT_REPLY,     // S->C: Connection established, Server's RMA handle
    WRITE_REQ,         // C->S: write request - length of write buffer
    WRITE_REQ_REPLY,   // S->C: request accepted, where & how much to write
    WRITE_DONE,        // C->S: RMA write or cudaMemCpy is done - server can now
                       //       start writing to disk
    BYE,               // C->S: client done
    CACHE_USAGE,       // C->S: How much data is currently stored in the cache
    CACHE_USAGE_REPLY, // S->C: Reply to above request
//    FINISHED         // C<-S: finished writing
};


union IoMsg {
    IoMsgType type; // always first in each msg

    struct ConnectMsg {
        IoMsgType type;     // CONNECT
        uint32_t rank;      // client's rank
    } connect;
    
    struct ConnectReplyMsg {
        IoMsgType type;     // CONNECT_REPLY
        struct cci_rma_handle handle;  // Daemon's RMA handle
        // can't use a cci_rma_handle_t here because it's
        // const and we have no initializer
    } connectReply;

    struct WriteRequestMsg {
        IoMsgType type;   // WRITE_REQ
        uint64_t len;     // Length of write
        uint64_t offset;  // Where in the file this data goes
    } writeRequest;
    
    // Daemon sends this back in response to a write request
    struct WriteRequestReplyMsg { 
        IoMsgType type;    // WRITE_REQ_REPLY
        uint32_t  reqId;   // Identify this request
        bool      gpuMem;  // write to GPU mem?
        uint64_t  len;     // how much data can be written (in bytes)
        
        uint32_t offset;  // where to RMA the data to (*if* we're using RMA)
        cudaIpcMemHandle_t   memHandle;  // if we're using CUDA
        // Note: We should probably use a union - we'll either
        // need the cuda handle or the offset, but not both.
    } writeReply;

    struct WriteDoneMsg {
        IoMsgType type;   // WRITE_DONE
        uint32_t  reqId;  // The request ID from writeReply
    } writeDone;

    struct ByeMsg {
        IoMsgType type; // BYE 
    } bye;

    struct CacheUsageMsg {
        IoMsgType type; // CACHE_USAGE
    } cacheUsage;
    
    struct CacheUsageReplyMsg {
        IoMsgType type; // CACHE_USAGE
        bool isEmpty;
        // For now, all I need is the isEmpty bool.  In the future, it
        // may be useful to add actual MB values... 
    } cacheUsageReply;
    
// Not use - will probably be deleted
//    struct FiniMsg {
//        IoMsgType type; // FINISHED
//    } fini;
    
};


// The sentinal that the daemon will write after the URI.  Once we
// read this, we know everything before it is the complete URI.
// Not sure this really belongs in this file, but it needs to be some place
// where it can be read by both the client and the daemon...
#define END_OF_URI "||URI_END||"



#endif // _CCI_MSG_H_