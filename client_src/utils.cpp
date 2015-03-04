
#include "cci_msg.h"
#include "cci_util.h"
#include "cuda_util.h"
#include "utils.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <fstream>
using namespace std;


static void handleSigchld( int sig);

// returns 0 for success, errno on failure
static int startDaemon( char **args);


// variables that will need to be shared by multiple functions in this .cpp file
pid_t daemonPid;  // Process ID of the daemon
cci_endpoint_t *endpoint = NULL;
cci_os_handle_t *endpointFd = NULL; // file descriptor that can block waiting for
                                    // progress on the endpoint
cci_connection_t *connection = NULL;
cci_rma_handle_t *local = NULL;
unsigned connection_context = 1; // can be anything.  It's actually the address
                                 // that gets sent back and forth
                                 
// Start up the daemon and set up the CCI connection
// Returns a cci_status value, or a negated errno value (ie: -22 for EINVAL)
int initIo(void *buffer, uint32_t len, uint32_t rank, char **daemon_args)
{
    int ret = CCI_SUCCESS;
    uint32_t caps = 0;
    IoMsg msg;
    char hostname[256], server[256];
    
    int ifd;  // file descriptor used to determine which 
              // rank starts the daemon
    
    // used for reading the URI from the file that the daemon writes
    ifstream iodFile;
    bool uriFound = false;
    string uri;

    bool connectEvent = false; // have we received the connect event yet?
    
    if (!buffer || !len) {
        ret = -EINVAL;
        goto out;
    }

    // We want one daemon per node.  There's no guaranteed mapping between
    // nodes and ranks.  So, all ranks attemtp to create a file exclusively.
    // The ranks that succeed, start up daemons.
    memset(hostname, 0, sizeof(hostname));
    gethostname(hostname, sizeof(hostname));

    ifd = open(hostname, O_WRONLY | O_CREAT | O_TRUNC | O_EXCL, 0600);
    if (ifd != -1) {
        ret = -startDaemon( daemon_args);
        // note the negation of ret.  Positive return values are interpretted
        // as cci_status values
        if (ret) {
            goto out;
        }
    }

    // Do all the CCI initialization stuff: init, create an endpoint, register
    // the RMA buffer
    ret = cci_init(CCI_ABI_VERSION, 0, &caps);
    if (ret) {
        cciDbgMsg( "cci_init()", ret);
        goto out;
    }

    ret = cci_create_endpoint(NULL, 0, &endpoint, endpointFd);
    if (ret) {
        cciDbgMsg( "cci_create_endpoint()", ret);
        goto out;
    }

    ret = cci_rma_register(endpoint, buffer, (uint64_t)len,
            CCI_FLAG_WRITE|CCI_FLAG_READ, &local);
    if (ret) {
        cciDbgMsg( "cci_rma_register()", ret);
        goto out;
    }

    // The daemon will write the CCI connection URI to a
    // well-known file.
    memset(server, 0, sizeof(server));
    snprintf(server, sizeof(server), "%s-iod", hostname);

    do {
        iodFile.open( server);
    } while (! iodFile);
    
    do {
        getline( iodFile, uri);
        size_t pos = uri.find( END_OF_URI);
        if (pos != std::string::npos) {
            uri = uri.substr( 0, pos);
            uriFound = true;
        }
        else {
            // Complete URI hadn't been written yet.
            // Reset the stream and try again
            iodFile.seekg( 0);
        }
    } while (! uriFound);
    
    iodFile.close();

    // Set up a CCI message to 
    msg.connect.type = CONNECT;
    msg.connect.rank = rank;
   
    ret = cci_connect(endpoint, uri.c_str(), &msg, sizeof(msg.connect),
            CCI_CONN_ATTR_RO, &connection_context, 0, NULL);
    if (ret) {
        cciDbgMsg("cci_connect()", ret);
        goto out;
    }   
    
    // Poll for the event signifying a successful connection
    do {
        cci_event_t *event = NULL;

        ret = cci_get_event(endpoint, &event);
        if (!ret) {
            switch (event->type) {
            case CCI_EVENT_CONNECT:
                connection = event->connect.connection;
                connectEvent = true;
                break;
            default:
                cerr << __func__ << ": ignoring "
                     << cci_event_type_str(event->type) << endl;
                break;
            }
            cci_return_event(event);
        }
    } while (!connectEvent);

    if (!connection) {
        // Strictly speaking, I don't think there's any way to
        // get into this section...
        ret = -ENOTCONN;
        cerr << __func__ << ": CCI connect failed!" << endl;
    }
    out:
    if (ret)
        finalizeIo();

    return ret;
}



// Shut down the CCI connection and stop the daemon
// Returns cci_status or negated errno (ie -22 for EINVAL)
int finalizeIo(void)
{
    int ret = 0;
    IoMsg msg;
    char hostname[256];

    msg.type = BYE;

    if (connection) {
        ret = cci_send(connection, &msg, sizeof(msg.bye),
                (void*)(uintptr_t)0xdeadbeef, 0);
        if (!ret) {
            // Check for the expected events
            bool sendEventProcessed = false;
            bool recvEventProcessed = false;
            cci_event_t *event = NULL;

            do {
                ret = cci_get_event(endpoint, &event);
                if (!ret) {
                    const IoMsg *rx = NULL;

                    switch (event->type) {
                    case CCI_EVENT_SEND:
                        assert(event->send.context ==
                                (void*)(uintptr_t)0xdeadbeef);
                        sendEventProcessed = true;
                        break;
                    case CCI_EVENT_RECV:
                        rx = (const IoMsg *)event->recv.ptr;
                        assert(rx->type == FINISHED);
                        recvEventProcessed = true;
                        break;
                    default:
                        cerr << __func__ << ": ignoring "
                             << cci_event_type_str(event->type) << endl;
                        break;
                    }
                    cci_return_event(event);
                }
            } while (!sendEventProcessed && !recvEventProcessed);

            // to allow CCI to ack the send?
            for (unsigned i = 0; i < 10; i++)
                cci_get_event(endpoint, &event);

        }

        ret = cci_disconnect(connection);
        if (ret) {
            cciDbgMsg("cci_disconnect()", ret);
        }
    }

    if (local) {
        ret = cci_rma_deregister(endpoint, local);
        if (ret) {
            cciDbgMsg("cci_rma_deregister()", ret);
        }
    }

    if (endpoint) {
        ret = cci_destroy_endpoint(endpoint);
        if (ret) {
            cciDbgMsg("cci_destroy_endpoint()", ret);
        }
    }

    ret = cci_finalize();
    if (ret) {
        cciDbgMsg("cci_finalize()", ret);
    }

    sleep(1);

    // Stop the daemon
    if (daemonPid != -1) {
        signal(SIGCHLD, SIG_IGN);
        kill(daemonPid, SIGKILL);
        waitpid(daemonPid, NULL, 0);
        
        // remove the file we exclusively created in order to
        // determine which rank started the daemon (see initIo())
        memset(hostname, 0, sizeof(hostname));
        gethostname(hostname, sizeof(hostname));
        unlink(hostname);
    }

    return ret;
}


// returns 0 for success, or errno
static int startDaemon(char **args)
{
    int ret = 0;

    signal(SIGCHLD, handleSigchld);

    daemonPid = fork();
    if (daemonPid == -1) {
        ret = errno;
        cerr << __func__ << ": fork() failed with " << strerror( ret) << endl;
    } else if (daemonPid == 0) {
        // HACK!! Don't Check in!
        // We'll start the daemon manually in the debugger
        execve("/usr/bin/true", args, environ);
        //execve(args[0], args, environ);
        // if we actually return, it means exec() failed
        ret = errno;
        cerr << __func__ << ": execve() failed with " << strerror(ret) << endl;
        exit(ret);
    } else {
        cerr << args[0] << " daemon started with PID " << daemonPid << endl;
    }

    return ret;
}



// If the daemon is terminated, we'll get a SIGCHLD
static void handleSigchld( int sig)
{
    int status;

    waitpid( daemonPid, &status, 0);
    if (WIFEXITED(status)) {
        cerr << __func__ << ": daemon exited, status=" << WEXITSTATUS(status) << endl;
    } else if (WIFSIGNALED(status)) {
        cerr << __func__ << ": daemon killed by signal " << WTERMSIG(status) << endl;
        abort();
    }
    return;
}

// write to a local file
int writeLocal(void *buf, streamsize len, ofstream &outf)
{   
    outf.seekp( 0);
    outf.write( (const char *)buf, len);
    return 0;
}



// Write to the remote daemon (either the GPU or system ram depending
// on what the daemon tells us).
// returns a CCI_STATUS
int writeRemote(void *buf, size_t len)
{   

    int ret = 0;
    unsigned done = 0; // used for checking cci events
    
    // First up - send a write request message
    IoMsg sndMsg, replyMsg;
    sndMsg.writeRequest.type = WRITE_REQ;
    sndMsg.writeRequest.len = len;
    sndMsg.writeRequest.offset = 0;
       
    ret = cci_send( connection, &sndMsg, sizeof( sndMsg.writeRequest), &sndMsg, CCI_FLAG_NO_COPY);
    // Note: using the address of the buffer as the context...
    if (ret) {
        cciDbgMsg( "cci_send()", ret);
        goto out;
    }
    
    done = 0;
    do {
        cci_event_t *event = NULL;

        ret = cci_get_event(endpoint, &event);
        if (!ret) {

            switch (event->type) {
                case CCI_EVENT_SEND:
                assert(event->send.context == &sndMsg);
                done++;
                break;
                case CCI_EVENT_RECV:
                    
                assert( ((IoMsg *)(event->recv.ptr))->type == WRITE_REQ_REPLY);
                memcpy(&replyMsg.writeReply, event->recv.ptr, event->recv.len);
                done++;
                break;
                default:
                cerr << __func__ << ": ignoring cci_event_type_str(event->type)" << endl;
                break;
            }
            cci_return_event(event);
        }
    } while (done < 2);

    // OK, the reply will tell us where to write the data (and also how much
    // we can write)
    if (replyMsg.writeReply.gpuMem) {
        // write to the GPU memory
        void *cudaMem;
        
        if (replyMsg.writeReply.len != len) {
            cerr << __func__ << ": Haven't implemented multi-copy messages yet!" << endl;
            abort();
        }
        
        // unpack the mem handle and copy the data
        CUDA_CHECK_RETURN( cudaIpcOpenMemHandle(
                            &cudaMem, replyMsg.writeReply.memHandle,
                            cudaIpcMemLazyEnablePeerAccess));
        CUDA_CHECK_RETURN( cudaMemcpy( cudaMem, buf, len, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN( cudaIpcCloseMemHandle( cudaMem));
        
        // Send the writeDone message
        sndMsg.writeDone.type = WRITE_DONE;
        sndMsg.writeDone.reqId = replyMsg.writeReply.reqId;
        ret = cci_send( connection, &sndMsg, sizeof( sndMsg.writeRequest), &sndMsg, CCI_FLAG_NO_COPY);
        // Note: using the address of the buffer as the context...
        if (ret) {
            cciDbgMsg( "cci_send()", ret);
            goto out;
        }
        
        done = 0;
        do {
            cci_event_t *event = NULL;

            ret = cci_get_event(endpoint, &event);
            if (!ret) {

                switch (event->type) {
                    case CCI_EVENT_SEND:
                    assert(event->send.context == &sndMsg);
                    done++;
                    break;
                    
                    default:
                    cerr << __func__ << ": ignoring cci_event_type_str(event->type)" << endl;
                    break;
                }
                cci_return_event(event);
            }
        } while (done < 1);
        
        // OK, the WRITE_DONE message is on its way
        
    } else { 
        //write to system ram
        
        // TODO: implement me!!
        cerr << __func__ << ": Haven't implemented caching to system ram yet!" << endl;
        abort();
    }
    
    
    out:
    return ret;
}


