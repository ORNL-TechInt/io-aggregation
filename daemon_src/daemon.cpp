// The main code for the (caching) daemon



#include "cci_msg.h"  // client/daemon message definitions
#include "cci_util.h"
#include "cacheblock.h"
#include "daemoncmdlineopts.h"
#include "iorequest.h"
#include "peer.h"

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


#ifdef __NVCC__
    #include <cuda_runtime.h>
    #include <cuda.h>

    // This macro checks return value of the CUDA runtime call and exits
    // the application if the call failed.
    #define CUDA_CHECK_RETURN(value) {                              \
        cudaError_t _m_cudaStat = value;                            \
        if (_m_cudaStat != cudaSuccess) {                           \
            fprintf(stderr, "Error %s at line %d in file %s\n",     \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
            exit(1);                                                \
            } }
#endif // defined __NVCC__

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
cci_os_handle_t *endpointFd = NULL; // file descriptor that can block waiting for
                                    // progress on the endpoint
                                    
struct cci_rma_handle localRmaHandle; 
// can't use a cci_rma_handle_t here because it's
// const and we have no initializer
                                    
                                    

map <unsigned, Peer *> peerList;
// Keep track of our connections.  The key is the client's MPI rank
// NOTE: Container of pointers! Don't forget to delete the object when
// you remove it from the container!
// TODO: do we need a mutex for this??

// Two lists of cache blocks - the first is for blocks that have been allocated
// and clients are in the process of copying data to.  The second is for blocks
// whose the clients have finished their copy and can now be written to disk
deque <CacheBlock *> incomingBlockList;
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
#ifdef __NVCC__
    size_t freeMem, totalMem;
    CUDA_CHECK_RETURN( cudaMemGetInfo( &freeMem, &totalMem));
   
    // Debug message
    cerr << "CUDADBG: Free: " << freeMem / (1024*1024) << "MB  Total: "
         << totalMem / (1024*1024) << "MB" << endl;
    
    // Use the lesser of the 2 values
    // TODO: Is using the lesser the right idea?
    maxGpuRam = (freeMem < totalMem) ? freeMem : totalMem;
#endif
    
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
        CacheBlock *cb;
        pthread_mutex_lock( &blockListMut);
        if (readyBlockList.size() > 0) {
            cb = readyBlockList.front();
            readyBlockList.pop_front();
        } else {
            // The only reason I can think of that the list would be empty
            // is if the sem_wait() returned an error...
            cerr << __func__
                 << ": empty readyBlockList.  Why did the thread wake up??"
                 << endl;
        }
        pthread_mutex_unlock( &blockListMut);

        
        //cb->write(); TODO: Implement!!
        delete cb;
    }
        
        
    return NULL;
}

// Specific event handlers...
static void handle_connect_request( cci_event_t * event)
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

    ret = cci_accept(event, peer);
    if (ret) {
        cciDbgMsg( "cci_accept()", ret);
        goto out;
    }

    peerList[peer->rank] = peer;

    out:
    if (ret)
        delete peer;
    
    return;
}

static void handle_accept( cci_event_t * event)
{
    
}
static void handle_recv( cci_event_t * event)
{
}
static void handle_send( cci_event_t * event)
{
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
                handle_connect_request(event);
                break;
            case CCI_EVENT_ACCEPT:
                handle_accept(event);
                break;
            case CCI_EVENT_RECV:
                handle_recv(event);
                break;
            case CCI_EVENT_SEND:
                handle_send(event);
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
    for (unsigned i=0; i < peerList.size(); i++) {
        ostringstream fname("");
        fname << "rank-" << peerList[i]->rank << "-iod";
        out.open( fname.str().c_str());
        if (! out) {
            cerr << "Failed to open " << fname.str()
                 << ".  Skipping results for rank " << peerList[i]->rank
                 << endl;
            continue;
        }
        
        // First line is the name of the program that wrote the file
        // (This is a leftover from when we had different executables.)
        out << "daemon" << endl;
        
        // Second line is some overall stats for this rank
        out << "rank " << peerList[i]->rank
            << " num_requests " << peerList[i]->completedReqs.size()
            << " max_len " << "?????" << endl;
            // TODO: implement the max length stuff
        
        // one line for each completed request
        for (unsigned j=0; j < peerList[i]->completedReqs.size(); j++) {
            peerList[i]->completedReqs[j].writeResults( out);
        }
        
        out.close();
    }
        
    return;
}



