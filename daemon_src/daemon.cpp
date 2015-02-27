// The main code for the (caching) daemon



#include "cci_msg.h"  // client/daemon message definitions
#include "cci_util.h"
#include "daemoncmdlineopts.h"

#include <cci.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>
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
int cciSetup( char **uri);


cci_endpoint_t *endpoint = NULL;
cci_os_handle_t *endpointFd = NULL; // file descriptor that can block waiting for
                                    // progress on the endpoint
                                    
struct cci_rma_handle localRmaHandle; 
// can't use a cci_rma_handle_t here because it's
// const and we have no initializer
                                    
                                    
                                    
                                    
struct IoRequest;
                                    
struct Peer {
    cci_connection_t    *conn; // CCI connection
//    void    *buffer;
    
//    cci_rma_handle_t    *remote; /* Their CCI RMA handle */
// don't think we need this - client will initiate RMA's

//    uint32_t    requests; /* Number of requests received */
//    uint32_t    completed; /* Number of completed requests on ios list */
// These 2 are handled the size() function on the vectors

    vector <IoRequest> receivedReqs;
    vector <IoRequest> completedReqs;
    // TODO: should these be maps instead?
    // TODO: Should these be containers of pointers, rather than actual structs?
    //       Would make it easier to associate cache blocks...
    
//    uint32_t    len; /* RMA buffer length */
    uint32_t    rank; // Peer's MPI rank
    int fd; // File for peer
    // FUTURE:  For the "real" version, we'll want clients to be able
    // to specify the file descriptor per request, so this value will
    // have to move to the IoRequest struc.  We'll also need open & 
    // close message so the clients can tell the daemon which files to
    // write to.
    int done; // client sent BYE message
    pthread_mutex_t lock; // Lock to protect everything in this struct
};



// TODO: should IoRequest be ref-counted?
struct IoRequest {
    Peer *      peer; /* Client peer */
    uint64_t    rx_us; /* microsecs when received */
    uint64_t    cpy_us; /* microsecs when copy completes */
    uint64_t    rma_us; /* microsecs when RMA completes */
    uint64_t    deq_us; /* microsecs when dequeued by io */
    uint64_t    io_us; /* microsecs when write() completes */
    uint32_t    len; /* Requesting write of len bytes */
};


class CacheBlock {
    
protected:
    IoRequest *req;  // cache blocks are all associated with a request
                     // (more than 1 cache block may be associated with the
                     // same request)
    
};

class SysRamCacheBlock : public CacheBlock {

    
};

class GPURamCacheBlock : public CacheBlock {
};

vector <CacheBlock *> blockList;
// vector of pointers! Don't forget to call delete after popping one off the list

vector <Peer> peerList;
                                    
                                    
                                    
                                    

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
    //pthread_t tid;

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
    
#if 0
    pin_to_core(1);

    TAILQ_INIT(&reqs);
    pthread_cond_init(&cv, NULL);
    pthread_mutex_init(&lock, NULL);

    ret = pthread_create(&tid, NULL, io, NULL);
    if (ret) {
        fprintf(stderr, "pthread_create() failed with %s\n", strerror(ret));
        goto out;
    }

    comm_loop();

    ret = pthread_join(tid, NULL);
    if (ret) {
        fprintf(stderr, "pthread_join() failed with %s\n", strerror(ret));
    }
#endif

    out:
    
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
int cciSetup( char **uri)
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
    
out:
    return ret;
}