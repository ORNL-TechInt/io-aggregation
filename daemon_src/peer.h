// Class for keeping track of the connections to the daemon

#ifndef _PEER_H_
#define _PEER_H_


#include "cci.h"
#include "iorequest.h"

#include <pthread.h>

#include <fstream>
#include <vector>  
                                                                       
class Peer {
public:
    Peer( uint32_t theRank);
    ~Peer();
    
    
    bool isDone() { return done; }
    
    // TODO: should the data all be private?
    
    cci_connection_t    *conn; // CCI connection
//    void    *buffer;
    
//    cci_rma_handle_t    *remote; /* Their CCI RMA handle */
// don't think we need this - client will initiate RMA's

//    uint32_t    requests; /* Number of requests received */
//    uint32_t    completed; /* Number of completed requests on ios list */
// These 2 are handled the size() function on the vectors

    std::vector <IoRequest> receivedReqs;
    std::vector <IoRequest> completedReqs;
    // TODO: should these be maps instead?
    // TODO: Should these be containers of pointers, rather than actual structs?
    //       Would make it easier to associate cache blocks...
    
//    uint32_t    len; /* RMA buffer length */
    uint32_t    rank; // Peer's MPI rank
    std::ofstream outf; // File for peer
    // FUTURE:  For the "real" version, we'll want clients to be able
    // to specify the file descriptor per request, so this value will
    // have to move to the IoRequest struc.  We'll also need open & 
    // close message so the clients can tell the daemon which files to
    // write to.
    bool done; // client sent BYE message
    pthread_mutex_t mut; // Lock to protect everything in this struct
    
    
private:
    // No default constructor, copy constructor or assignment operator
    Peer();
    Peer( const Peer &p);
    Peer &operator=( const Peer &p);
};





#endif // _PEER_H_