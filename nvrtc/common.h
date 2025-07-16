#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    // print_mtx.lock();
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    // print_mtx.unlock();
    if (abort) exit(code);
  }
}

#define BLOCK_SIZE 128

#endif // COMMON_H
