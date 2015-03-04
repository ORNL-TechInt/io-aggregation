// Classes for managing the cache memory - whether it's system ram or GPU ram


#ifndef _CACHE_BLOCK_H_
#define _CACHE_BLOCK_H_
#include "peer.h"

class CacheBlock {
    
public:
    
    CacheBlock( void *addr, uint64_t length, off_t offset, Peer *peer);
    virtual ~CacheBlock();
    
    // write this cache block to the specified file descriptor at the
    // specified offset
    virtual bool write() = 0;
    
protected:
//    IoRequest * req;     // cache blocks are all associated with a request
                         // (more than 1 cache block may be associated with
                         // the same request)
    uint64_t    m_length;  // length of the cache block - in bytes
    void *      m_addr;    // Address of the block (might be system ram, or it
                         // might be GPU ram depending on the child
                         // implementation)
    off_t   m_offset;  // offset into the file where the data goes
    Peer * m_peer; // peer that requested this write
    
    
private:
    CacheBlock( );
    CacheBlock( const CacheBlock &cb);
    CacheBlock &operator= (const CacheBlock &cb);
    
};

class SysRamCacheBlock : public CacheBlock {
public:
    ~SysRamCacheBlock();
    
};

class GPURamCacheBlock : public CacheBlock {
public:
    GPURamCacheBlock( void *addr, uint64_t length, off_t offset, Peer *peer);
    ~GPURamCacheBlock();
    
    bool write();
};


#endif // _CACHE_BLOCK_H_
