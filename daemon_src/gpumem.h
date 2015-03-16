// Various functions for allocating GPU memory (using different schemes
// for load-balancing all the ranks)

#ifndef _GPUMEM_H_
#define _GPUMEM_H_

#include <stdint.h>

uint64_t allocate_fcfs( void **devPtr, uint64_t reqLen);
uint64_t allocate_block( void **devPtr, uint64_t reqLen);

#endif // _GPUMEM_H_



