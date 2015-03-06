// Implementation file for CacheBlock and its decendents

#include "cacheblock.h"
#include "cuda_util.h"

// iostream needed for debug cerr messages
#include <iostream>
using namespace std;

CacheBlock::CacheBlock( void *addr, uint64_t length, off_t offset, Peer *peer)
 : m_length(length), m_addr( addr), m_offset(offset), m_peer(peer)
{ /* no-op */ }

CacheBlock::~CacheBlock()
{
    // TODO: Implement me!
}

GPURamCacheBlock::GPURamCacheBlock( void *addr, uint64_t length, off_t offset, Peer *peer)
    : CacheBlock( addr, length, offset, peer)
{ /* No-op */ }

GPURamCacheBlock::~GPURamCacheBlock()
{
    if (m_addr) {
        CUDA_CHECK_RETURN( cudaFree( m_addr));
    }
}

// TODO: It'd be nice to only allocate one bounce buffer for all
// instances of GPURamCacheBlock.  Possibly use a static member?
bool GPURamCacheBlock::write()
{
    bool ret = false;  
    const uint32_t BOUNCE_SIZE = (4 * 1024 * 1024); // 4MB
    
    // Allocate a buffer in system memory we can use to hold data we've
    // moved out of GPU ram
    char *bounceBuf;
    bool pinnedMem = false;
#ifndef DISABLE_DAEMON_PINNED_MEMORY
    cudaError_t cudaErr = cudaMallocHost( &bounceBuf, BOUNCE_SIZE);
    if (cudaErr == cudaSuccess) {
        pinnedMem = true;
    } else {
        // print a warning, then fall back to using regular memory
        cerr << __FILE__ << ":" << __LINE__
             << ": WARNING: failed to allocate pinned memory. "
             << "Falling back to normal allocation." << endl;
    }
#endif
    
    if (! pinnedMem) {
        // either the cudaMallocHost() call failed, or it wasn't compiled in
        bounceBuf = new char[BOUNCE_SIZE];
    }
    
    char *deviceMem = (char *)m_addr;
    // Treat the device memory as an array of characters - makes it easier
    // to handle offsets   
        
    uint64_t bytesWritten = 0;
    uint64_t bytesRemaining = m_length;
    
    // Check on the status of the output file
    if (!m_peer->m_outf.good()) {
        cerr << __FILE__ << ":" << __LINE__
             <<  ": Output file in a bad state!  Can't write to it!" << endl;
        goto out;
    }
    
    // Seek to the position we're supposed to start writing
    m_peer->m_outf.seekp( m_offset);

    while (bytesRemaining) {
        // Copy the data from GPU mem to system mem
        uint64_t bytesToCopy = 
            (BOUNCE_SIZE<bytesRemaining) ? BOUNCE_SIZE : bytesRemaining;
        CUDA_CHECK_RETURN( cudaMemcpy( bounceBuf, &deviceMem[bytesWritten],
                                       bytesToCopy, cudaMemcpyDeviceToHost));
            
        m_peer->m_outf.write( bounceBuf, bytesToCopy);
        // Check on the status of the output file
        if (!m_peer->m_outf.good()) {
            cerr << __FILE__ << ":" << __LINE__
                 << ": Output file when bad after attempting to write "
                 << bytesToCopy << " bytes at offset " << m_offset + bytesWritten
                 << ".  Aborting write() function!" << endl;
                 goto out;
        }
        
        bytesWritten += bytesToCopy;
        bytesRemaining -= bytesToCopy;
    }
    
    ret = true;
out:
    if (pinnedMem) {
        CUDA_CHECK_RETURN( cudaFreeHost( bounceBuf));
    }
    else {
        delete[] bounceBuf;
    }
    
    return ret;
}