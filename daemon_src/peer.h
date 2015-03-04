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
    Peer( uint32_t rank);
    ~Peer();
    
    void setConnection( cci_connection_t * conn) { m_conn = conn; }
    void setDone( bool done) { m_done = done; }
    bool isDone() { return m_done; }
    void writeStatistics();
    
    // TODO: should the data all be private?
    
    cci_connection_t    *m_conn; // CCI connection

    std::vector <IoRequest> m_receivedReqs;
    std::vector <IoRequest> m_completedReqs;
    // TODO: should these be maps instead?
    // TODO: Should these be containers of pointers, rather than actual structs?
    //       Would make it easier to associate cache blocks...
    

    uint32_t    m_rank; // Peer's MPI rank (mainly for identification when
                      // printing statistics)
                      
    std::ofstream m_outf; // File for peer
    // FUTURE:  For the "real" version, we'll want clients to be able
    // to specify the file descriptor per request, so this value will
    // have to move to the IoRequest struc.  We'll also need open & 
    // close message so the clients can tell the daemon which files to
    // write to.
    bool m_done; // client sent BYE message
    pthread_mutex_t m_mut; // Lock to protect everything in this struct
    
    
private:
    // No default constructor, copy constructor or assignment operator
    Peer();
    Peer( const Peer &p);
    Peer &operator=( const Peer &p);
};





#endif // _PEER_H_