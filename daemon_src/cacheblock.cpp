// Implementation file for CacheBlock and its decendents

#include "cacheblock.h"


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
    // TODO: Implement me!
}

// TODO: It'd be nice to only allocate one bounce buffer for all
// instances of GPURamCacheBlock.  Possibly use a static member?
bool GPURamCacheBlock::write()
{
    // Allocate a buffer in system memory we can use to hold data we've
    // moved out of GPU ram
    const uint32_t BOUNCE_SIZE = (4 * 1024 * 1024); // 4MB
    unsigned char *bounceBuf = new unsigned char[BOUNCE_SIZE];
    
    
    
    
    
    
    delete[] bounceBuf;
    
    return false;
}