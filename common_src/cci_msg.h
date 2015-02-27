// Defines the messages that will be passed between the client and server


#ifndef _CCI_MSG_H_
#define _CCI_MSG_H_

#include "cci.h"

#ifdef __NVCC__
#include <cuda_runtime.h>
#include <cuda.h>
#endif

/* C = client - MPI rank
 * S = server - IO daemon
 * C->S = client to server message
 * C<-S = server to client message
 */

// Struct defining the messages we can send via CCI
enum IoMsgType {
    CONNECT = 0,     // C->S: rank
    CONNECT_REPLY,   // S->C: Connection established, Server's RMA handle
    WRITE_REQ,       // C->S: write request - length of write buffer
    WRITE_REQ_REPLY, // S->C: request accepted, where & how much to write
    WRITE_DONE,      // C<-S: RMA Read is done - client can reuse buffer
    BYE,             // C->S: client done
    FINISHED         // C<-S: finished writing
};


union IoMsg {
    IoMsgType type; // always first in each msg

    struct ConnectMsg {
        IoMsgType type;     // CONNECT
        uint32_t rank;      // client's rank
        uint32_t ranks;     // number of local ranks
    } connect;
    
    struct ConnectReplyMsg {
        IoMsgType type;     // CONNECT_REPLY
        struct cci_rma_handle handle;  // CCI RMA handle
        // can't use a cci_rma_handle_t here because it's
        // const and we have no initializer
    } connectReply;

    struct WriteRequestMsg {
        IoMsgType type;   // WRITE_REQ
        uint32_t len;     // Length of write
        uint64_t cookie;  // IO request opaque pointer
    } writeRequest;
    
    // Daemon sends this back in response to a write request
    struct WriteRequestReplyMsg { 
        IoMsgType type;   // WRITE_REQ_REPLY
        uint32_t  reqId;
        bool      gpuMem;      // write to GPU mem?
        
        uint32_t  len;  // how much data can be written
        
        uint32_t offset;  // where to RMA the data to (*if* we're using RMA)
        
#ifdef __NVCC__
        cudaIpcMemHandle_t   memHandle;
        cudaIpcEventHandle_t eventHandle;
        // Note: not sure if we'll use the event handle or not..
#endif
        // Note: We should probably use a union - we'll either
        // need the cuda handles or the offset, but not both.
    } writeReply;

    struct WriteDoneMsg {
        IoMsgType type;   // WRITE_DONE
        uint32_t  reqId;  // The request ID from writeReply
        uint32_t pad;
        uint64_t cookie;  // IO request opaque pointer
    } writeDone;

    struct ByeMsg {
        IoMsgType type; // BYE 
    } bye;

    struct FiniMsg {
        IoMsgType type; // FINISHED
    } fini;
    
};


// The sentinal that the daemon will write after the URI.  Once we
// read this, we know everything before it is the complete URI.
// Not sure this really belongs in this file, but it needs to be some place
// where it can be read by both the client and the daemon...
#define END_OF_URI "||URI_END||"



#endif // _CCI_MSG_H_