// The main code for the (caching) daemon



#include "cci_msg.h"  // client/daemon message definitions
#include "cci_util.h"
#include "cacheblock.h"
#include "daemoncmdlineopts.h"
#include "iorequest.h"
#include "peer.h"

#include <assert.h>
#include <cci.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <unistd.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
using namespace std;


#include <cuda_runtime.h>
#include <cuda.h>

// This macro checks return value of the CUDA runtime call and exits
// the application if the call failed.
#define CUDA_CHECK_RETURN(value) {                                           \
    cudaError_t _m_cudaStat = value;                                         \
    if (_m_cudaStat != cudaSuccess) {                                        \
        fprintf(stderr, "Error %d - %s - at line %d in file %s\n",           \
         _m_cudaStat, cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);  \
        exit(1);                                                             \
        } }


// Does a bunch of CCI initialization stuff.
// Returns a cci_status
// If uri is non-null, a pointer to the char string for the selected URI
// will be written to it.
static int cciSetup( char **uri);

// write the results for each rank
static void printResults();

// Handle all the CCI events
static void commLoop();

// thread function for copying data out of cache and writing it to disk
void *writeThread( void *);
sem_t writeThreadSem;  // the write thread will wait on this.  Every
                       // we move a block from the incoming list to the
                       // ready list (see below) we'll post it.

cci_endpoint_t *endpoint = NULL;
cci_os_handle_t *endpointFd = NULL;  // file descriptor that can block waiting for
                                     // progress on the endpoint
                                    
struct cci_rma_handle localRmaHandle; 
// can't use a cci_rma_handle_t here because it's
// const and we have no initializer
                                    
                                    

map <cci_connection_t *, Peer *> peerList;
// Keep track of our connections.  The key is the client's connection
// pointer.
// NOTE: Container of pointers! Don't forget to delete the object when
// you remove it from the container!
// TODO: do we need a mutex for this??

// Two lists of cache blocks - the first is for blocks that have been allocated
// and clients are in the process of copying data to.  The second is for blocks
// whose the clients have finished their copy and can now be written to disk
// The key is the reqId value (see handleWriteRequest())
map <uint32_t, CacheBlock *> incomingBlockList;
deque <CacheBlock *> readyBlockList;
// NOTE: containers of pointers! Don't forget to call delete after popping
// one off the list
pthread_mutex_t blockListMut; // Protect access to the 2 block lists

// Set to true by the main thread to indicate that background thread(s)
// need to exit
bool shuttingDown = false;

int main(int argc, char *argv[])
{
    int ret = 0; //, i = 0, fd = -1, c = 0;
    
    // The maximum amount of GPU memory we will attempt to allocate.
    // Note: initializing to 0 is important.  We use a cuda call to set it,
    // and if we're compiled without cuda, the value must remain 0.
    uint64_t maxGpuRam = 0;  // in bytes
    uint64_t usedGpuRam = 0;  // amount (in bytes) that we've currently allocated   
    // TODO: we're not making use of these parameters!
 
    char *uri = NULL;
    char hostname[64], uriFileName[128];
    pthread_t tid;  // thread ID for the write thread

    CommandLineOptions cmdOpts;
    
    // First up - parse the command line options
    if (! parseCmdLine( argc, argv, cmdOpts)) {
        printUsage(argv[0]);
        return -1;
    }
    
    ret = cciSetup( &uri);

    memset(hostname, 0, sizeof(hostname));
    gethostname(hostname, sizeof(hostname));

    memset(uriFileName, 0, sizeof(uriFileName));
    snprintf(uriFileName, sizeof(uriFileName), "%s-iod", hostname);

    ofstream uriFile( uriFileName, ios::trunc);
    if (!uriFile) {
        cerr << "Failed to open " << uriFileName << endl;
        goto out;
    }
    
    uriFile << uri << END_OF_URI;
    uriFile.close();
    
    
    
    // TODO: To start with, we're not going to use any system memory
    // as cache.  That means we also don't need to worry about
    // transfering data via RMA...
    
    // FUTURE: For now, we'll manage GPU ram by calling cudaMalloc and
    // cudaFree for each request.  In the future, we'll probably want
    // to allocate all of it at once and then manage it in small (4MB?)
    // blocks.
    
    // Figure out how much GPU memory we have to work with.
    size_t freeMem, totalMem;
    CUDA_CHECK_RETURN( cudaMemGetInfo( &freeMem, &totalMem));
   
    // Debug message
    cerr << "CUDADBG: Free: " << freeMem / (1024*1024) << "MB  Total: "
         << totalMem / (1024*1024) << "MB" << endl;
    
    // Use the lesser of the 2 values
    // TODO: Is using the lesser the right idea?
    maxGpuRam = (freeMem < totalMem) ? freeMem : totalMem;
    
    if (cmdOpts.maxGpuRam != -1) { // If the user specified an actual value
        if ( ((uint64_t)cmdOpts.maxGpuRam * (1024 * 1024)) > maxGpuRam) {
            // Uh-oh.  The user wants more ram than we actually have
            cerr << "Requested " << cmdOpts.maxGpuRam << "MB of GPU ram but "
                 << "only " << maxGpuRam / (1024*1024) << "MB are available."
                 << endl;
            cerr << "Aborting." << endl;
            goto out;
        }
        else {
            maxGpuRam = cmdOpts.maxGpuRam * 1024 * 1024;
        }   
    }
 
    
    ret = pthread_mutex_init(&blockListMut, NULL);
 
    ret = sem_init( &writeThreadSem, 0, 0);
    if (ret) {
        cerr << "sem_init() failed with " << strerror(ret) << endl;;
        goto out;
    }
    
    ret = pthread_create(&tid, NULL, writeThread, NULL);
    if (ret) {
        cerr << "pthread_create() failed with " << strerror(ret) << endl;;
        goto out;
    }
 
    
    //pin_to_core(1);

    // Handle all the CCI events...
    commLoop();
    
    // If we've returned from commLoop(), then it's time to shut down

    shuttingDown = true;
    sem_post( &writeThreadSem);  // wake up the thread so it sees the shutdown flag
    
    ret = pthread_join(tid, NULL);
    if (ret) {
        cerr << "pthread_join() failed with " << strerror(ret) << endl;
    }

    // Write out the stats for the completed I/O requests
    printResults();
    
    out:
    
    pthread_mutex_destroy(&blockListMut);
    sem_destroy( &writeThreadSem);
    
    // Remove the 'hostname'-iod file
    ret = unlink(uriFileName);
    if (ret) {
        perror("unlink()");
    }
    
    
    free(uri);  // Note: the official CCI docs don't say if this is necessary or not

    if (endpoint) {
        ret = cci_destroy_endpoint(endpoint);
        if (ret) {
            cciDbgMsg( "cci_destroy_endpoint()", ret);
        }
    }

    ret = cci_finalize();
    if (ret) {
        cciDbgMsg( "cci_finalize()", ret);
    }

    return ret;
}



// Does a bunch of CCI initialization stuff.
// Returns a cci_status or a negated errno
// If uri is non-null, a pointer to the char string for the selected URI
// will be written to it.
static int cciSetup( char **uri)
{
    int ret;
    uint32_t caps = 0;
    cci_device_t *const *devices, *device = NULL;
    
    ret = cci_init(CCI_ABI_VERSION, 0, &caps);
    if (ret) {
        cciDbgMsg( "cci_init()", ret);
        goto out;
    }

    ret = cci_get_devices(&devices);
    if (ret) {
        cciDbgMsg( "cci_get_devices()", ret);
        goto out;
    }

    for (unsigned i = 0; ; i++) {
        device = devices[i];

        if (!device)
            break;

        if (!strcmp(device->transport, "sm")) {
            if (!device->up) {
                cerr << __func__ << ": sm device is down" << endl;
                ret = -ENODEV;
                goto out;
            }

            break;
        }
    }

    if (!device) {
        cerr << __func__ << ": No sm device found" << endl;
        ret = -ENODEV;
        goto out;
    }

    ret = cci_create_endpoint(device, 0, &endpoint, endpointFd);
    if (ret) {
        cciDbgMsg( "cci_create_endpoint()", ret);
        goto out;
    }

    if (uri) {
        ret = cci_get_opt(endpoint, CCI_OPT_ENDPT_URI, uri);
        if (ret) {
            cciDbgMsg( "cci_get_opt()", ret);
            goto out;
        }
    }
    
    
#if 0
TODO: uncomment this once we have set up buffer & len
    ret = cci_rma_register(endpoint, buffer, len, CCI_FLAG_READ, &localRmaHandle);
    if (ret) {
        cciDbgMsg( "cci_rma_register()", ret);
        goto out;
    }
#endif
    
out:
    return ret;
}


// thread function for copying data out of cache and writing it to disk
// also handles other tasks that could take a while, such as cleaning
// the finished peers out of the peer list
void *writeThread( void *)
{
    while (!shuttingDown) {
        if (sem_wait( &writeThreadSem)) {
            if (errno == EINTR) {
            // No big deal (seems to happen inside the debugger
            // fairly often).  Go back to sleep.
            continue;
            } else {  // log the error (should never happen)
                int err = errno;
                cerr << __func__ << "semaphore error " << err << ": "
                    << strerror( err) << endl;   
            }
        }
        
        // Time to exit?
        if (shuttingDown){
            continue;
        }
        
        // lock the mutex, pop the first block off the ready list and then
        // release the mutex
        bool cacheBlocksDeleted = false;
        CacheBlock *cb = NULL;
        pthread_mutex_lock( &blockListMut);
        while (readyBlockList.size() > 0) {
            cb = readyBlockList.front();
            readyBlockList.pop_front();
            pthread_mutex_unlock( &blockListMut);
        
            cb->write();
            delete cb;    
            cacheBlocksDeleted = true;
            pthread_mutex_lock( &blockListMut);
        }
        pthread_mutex_unlock( &blockListMut);
        
        
        if (! cacheBlocksDeleted) {
            // We woke up for some reason other than writing out
            // ready cache blocks...
        
            // Check the peer list for any peers that are marked done
            auto it = peerList.begin();
            while (it != peerList.end()) {
                Peer *peer = it->second;
                if (peer->isDone()) {
                    peerList.erase( it);
                    delete peer;  // Peer destructor will write out statistics
                    // NOTE: CacheBlock instances all have Peer pointers,
                    // so we need to ensure don't delete Peer objects while there
                    // are still cache blocks pointing to them.  We manage this
                    // with the WRITE_DONE and FINISHED messages.  Peer get
                    // marked "done" when the FINISHED message comes in.  Once we
                    // receive that message, we cannot accept any more messages
                    // on that connection.
                }
                it++;
            }
        }
        
    }
        
        
    return NULL;
}

// Specific event handlers...
static void handleConnectRequest( cci_event_t * event)
{
    int ret = 0;
    IoMsg *msg = (IoMsg *) event->request.data_ptr;
    Peer *peer = NULL;

    if (event->request.data_len != sizeof(msg->connect)) {
        cerr << __func__ << "%s: expected " << sizeof(msg->connect)
             << " bytes but received " << event->request.data_len
             << " bytes" << endl;
        ret = EINVAL;
        goto out;
    }
    
    peer = new Peer( msg->connect.rank);

    // Note: peer isn't complete yet.  It still needs a connection
    // pointer which we don't have.  So, attach the peer pointer as
    // the context to cci_accept() and finish setting it up down in
    // handleAccept()
    ret = cci_accept(event, peer);
    if (ret) {
        cciDbgMsg( "cci_accept()", ret);
        goto out;
    }


    out:
    if (ret)
        delete peer;
    
    return;
}

static void handleAccept( cci_event_t * event)
{
    // Fetch the connection pointer from the event struct
    if (event->accept.status) {
        cerr << __func__ << ": Accept event status "
             << event->accept.status << "!  Connection not accepted!"
             << endl;
    } else {
        cci_connection_t *conn = event->accept.connection;
        Peer *peer = (Peer *)event->accept.context;
        peerList[conn] = peer;
    }
}

// The client has asked for a location to write some data...
static void handleWriteRequest( const IoMsg *rx, cci_connection_t *conn)
{
    // Note: I expect this function to get more complex as we start
    // doing things like supporting main memory cache or managing
    // the GPU ram ourselves (instead of calling cudaMalloc/cudaFree)
    
    static uint32_t maxReqId = 0;  // used for assigning request ID's
       
    // Allocate some memory
    uint32_t len = rx->writeRequest.len;
    void *memPtr;
    CUDA_CHECK_RETURN( cudaMalloc( &memPtr, len));
    // TODO: if the length is too big, cudaMalloc will fail!!
    // We'll need to do a better job than this in production!
       
    // Set up the rest of the reply message
    IoMsg sendMsg;
    sendMsg.writeReply.type = WRITE_REQ_REPLY;
    sendMsg.writeReply.reqId =  maxReqId++;
    sendMsg.writeReply.gpuMem = true;
    sendMsg.writeReply.len = len;
    sendMsg.writeReply.offset = 0;  // not used for writes to GPU mem
    
    CUDA_CHECK_RETURN( cudaIpcGetMemHandle( &sendMsg.writeReply.memHandle, memPtr));  
    
    // Sending the reply just as quickly as we can (even though we've still
    // got some work to do on this end (such as creating the cache block)
    int ret = cci_send( conn, &sendMsg.writeReply,
                        sizeof(sendMsg.writeReply), NULL, 0);
    if (ret) {
        cciDbgMsg( "cci_send()", ret);
    }
    
    // Create the cache block
    Peer *peer = peerList[conn];
    assert( peer != NULL);
    GPURamCacheBlock *cb = new GPURamCacheBlock(
                                memPtr, len, rx->writeRequest.offset, peer);
    
    pthread_mutex_lock( &blockListMut);
    incomingBlockList[sendMsg.writeReply.reqId] = cb;
    pthread_mutex_unlock( &blockListMut);

}

// The client has just told us it's done copying data
static void handleWriteDone( const IoMsg *rx)
{
    // Move the associated CacheBlock from the incoming list
    // to the ready list
    uint32_t reqId = rx->writeDone.reqId;
    pthread_mutex_lock( &blockListMut);
    CacheBlock *cb = incomingBlockList[reqId];
    assert( cb);
    incomingBlockList.erase( reqId);
    readyBlockList.push_back( cb);
    pthread_mutex_unlock( &blockListMut);
    
    // Wake up the background thread
    sem_post( &writeThreadSem);
}

static void handleBye( const IoMsg *rx, cci_connection_t *conn)
{
    // Mark the appropriate peer object as done and then wake the
    // background thread.  (Don't want to tie up the comm loop
    // handling tasks that might take a little time.)
    Peer *peer = peerList[conn];
    peer->setDone( true);
    sem_post( &writeThreadSem);
}

static void handleRecv( cci_event_t * event)
{
    IoMsg *rx = (IoMsg *)event->recv.ptr;
    cci_connection_t *conn = event->recv.connection;
    
    // Validate the connection (once we've received the BYE
    // message, we can't accept any more message from the peer)
    Peer *peer = peerList[conn];
    assert( peer->isDone() == false);
    
    switch (rx->type) {
        case WRITE_REQ:
            handleWriteRequest( rx, conn);
            break;
        case WRITE_DONE:
            handleWriteDone( rx);
            break;
        case BYE:
            handleBye( rx, conn);
            break;
 
            
        default:
            cerr << __func__ << ": Ignoring unexpected message type "
                 << rx->type << endl;
    }
}

static void handleSend( cci_event_t * event)
{
    //TODO: do we need to do anything for send events?
}



// Handle all the CCI events
static void commLoop()
{
    int ret = 0;
    cci_event_t *event = NULL;
        
    while (!shuttingDown) {
        ret = cci_get_event(endpoint, &event);
        if (ret) { 
            if (ret != CCI_EAGAIN) {
                cciDbgMsg( "cci_get_event()", ret);
            }
            continue;
        }
                
        /* handle event */
        
        switch (event->type) {
            case CCI_EVENT_CONNECT_REQUEST:
                handleConnectRequest(event);
                break;
            case CCI_EVENT_ACCEPT:
                handleAccept(event);
                break;
            case CCI_EVENT_RECV:
                handleRecv(event);
                break;
            case CCI_EVENT_SEND:
                handleSend(event);
                break;
            default:
                cerr << __func__ << ": ignoring "
                     << cci_event_type_str(event->type) << endl;
                break;
        }
        
        cci_return_event(event);
    }
        
    return;
}

static void printResults()
{
    
    ofstream out;
    
    // Each peer gets its own output file
    auto it = peerList.begin();
    while (it != peerList.end()) {
        Peer *peer = it->second;
        ostringstream fname("");
        fname << "rank-" << peer->m_rank << "-iod";
        out.open( fname.str().c_str());
        if (! out) {
            cerr << "Failed to open " << fname.str()
                 << ".  Skipping results for rank " << peer->m_rank
                 << endl;
            continue;
        }
        
        // First line is the name of the program that wrote the file
        // (This is a leftover from when we had different executables.)
        out << "daemon" << endl;
        
        // Second line is some overall stats for this rank
        out << "rank " << peer->m_rank
            << " num_requests " << peer->m_completedReqs.size()
            << " max_len " << "?????" << endl;
            // TODO: implement the max length stuff
        
        // one line for each completed request
        for (unsigned j=0; j < peer->m_completedReqs.size(); j++) {
            peer->m_completedReqs[j].writeResults( out);
        }
        
        out.close();
        it++;
    }
        
    return;
}



