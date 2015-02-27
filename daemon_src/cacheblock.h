// Classes for managing the cache memory - whether it's system ram or GPU ram


#ifndef _CACHE_BLOCK_H_
#define _CACHE_BLOCK_H_


class CacheBlock {
    
public:
    
    virtual ~CacheBlock();
    // write this cache block to the specified file descriptor at the
    // specified offset
    virtual bool write() = 0;
    
protected:
//    IoRequest * req;     // cache blocks are all associated with a request
                         // (more than 1 cache block may be associated with
                         // the same request)
    uint64_t    length;  // length of the cache block - in bytes
    void *      addr;    // Address of the block (might be system ram, or it
                         // might be GPU ram depending on the child
                         // implementation)
    int     fd;      // file descriptor the data should be written to
    off_t   offset;  // offset into the file where the data goes
    
};

class SysRamCacheBlock : public CacheBlock {
public:
    ~SysRamCacheBlock();
    
};

class GPURamCacheBlock : public CacheBlock {
public:
    ~GPURamCacheBlock();
    
};


#endif // _CACHE_BLOCK_H_
