// Some useful CUDA related utils

#include <cuda_runtime.h>
#include <cuda.h>

// This macro checks return value of the CUDA runtime call and exits
// the application if the call failed.
#define CUDA_CHECK_RETURN(value) {                                           \
    cudaError_t _m_cudaStat = value;                                         \
    if (_m_cudaStat != cudaSuccess) {                                        \
        fprintf(stderr, "Error %d - %s - at line %d in file %s\n",           \
         _m_cudaStat, cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);  \
        exit(1);                                                             \
        } }
