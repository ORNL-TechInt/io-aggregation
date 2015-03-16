// Various functions for allocating GPU memory (using different schemes
// for load-balancing all the ranks)

#include "gpumem.h"
#include "cuda_util.h"


#include <iostream>
using namespace std;

// We don't ever seem to be able to allocate all of what's reported as free
// by cudaMemGetInfo(), so count some as inaccessible
// (The 1MB value was determined experimentally.)
#define INACCESSIBLE_GPU_MEM (1024*1024)


// Attempts to allocate up to reqLen bytes of GPU memory.  
// (First come; first served - ie: no load balancing)
// Returns the actual length of allocated memory (in bytes)
// NOTE: This is likely to be less than the requested amount!
uint64_t allocate_fcfs( void **devPtr, uint64_t reqLen)
{
    uint64_t ret = 0;
    size_t freeMem, totalMem;
    CUDA_CHECK_RETURN( cudaMemGetInfo( &freeMem, &totalMem));

// Commented out - too much output    
//    cerr << __func__ << ": Requesting " << (reqLen / (1024*1024))
//             << "MB.  (" << (freeMem / (1024*1024)) << "MB free.)" << endl;
    
    if (freeMem < INACCESSIBLE_GPU_MEM) {
        cerr << __func__ << ": Not enough free GPU mem to bother with a "
             << "malloc attempt."  << endl;
             return 0;
    }
    freeMem -= INACCESSIBLE_GPU_MEM;
    
    // If we can't handle the entire request, then it's more efficient to have
    // the length match the size of the bounce buffer (ie: 16MB).  This also
    // helps with the CCI efficiency.  So, we use a mask on freeMem.
    freeMem = freeMem & 0xFFFFFFFFFF000000;
    
    uint64_t attemptedLen = (freeMem < reqLen) ? freeMem : reqLen;
    
    cudaError_t mallocRet = cudaMalloc( devPtr, attemptedLen);
    while (mallocRet != cudaSuccess && attemptedLen > (2*1024*1024)) {
        // This code is really an emergency fallback.  I don't think
        // we should ever get here. And don't bother even trying if
        // the request gets too small
        cerr << __func__ << ": cudaMalloc() of " << (float)(attemptedLen / (1024*1024))
             << "MB failed.  Trying again with smaller request." << endl;
        attemptedLen /= 2;
        mallocRet = cudaMalloc( devPtr, attemptedLen);
    }
     
     
    if (mallocRet == cudaSuccess) {
        ret = attemptedLen;
    } else {
        cerr << __func__ << ": Failed to allocate any memory!" << endl;
        ret = 0;
    }
        
    return ret;
}


// Attempts to allocate a single, fixed-size block
// (Hopefully, this will result in allocations being more-or-less
// fairly shared among all ranks)
// 
// Returns the actual length of allocated memory (in bytes)
// NOTE: This is likely to be less than the requested amount!
uint64_t allocate_block( void **devPtr, uint64_t reqLen)
{
#define BLOCK_SIZE (16 * 1024 * 1024)  // 16MB
// 16MB was chosen because it matches the size of the bounce buffer and
// also because CCI and transfer data efficiently if it's page-aligned
// TODO: Double-check that statement w/Scott

    size_t freeMem, totalMem;
    CUDA_CHECK_RETURN( cudaMemGetInfo( &freeMem, &totalMem));

// Commented out - too much output    
//    cerr << __func__ << ": Requesting " << (reqLen / (1024*1024))
//             << "MB.  (" << (freeMem / (1024*1024)) << "MB free.)" << endl;
    

    freeMem -= INACCESSIBLE_GPU_MEM;
    if (freeMem < BLOCK_SIZE) {
// Too much output
//        cerr << __func__ << ": Not enough free GPU mem to bother with a "
//             << "malloc attempt."  << endl;
             return 0;
    }
        
    uint64_t attemptedLen = (reqLen < BLOCK_SIZE) ? reqLen : BLOCK_SIZE;
    
    CUDA_CHECK_RETURN( cudaMalloc( devPtr, attemptedLen));
    return attemptedLen;
}

