#ifndef MGPU_KERNEL_CUH
#define MGPU_KERNEL_CUH

#include <cuda_runtime.h>

void sum_cog(
  const double3* data,
  double3* cog_out,
  double3* peer_cogs,
  unsigned int* d_count,
  const int device_index,
  const int size,
  const int num_devices,
  cudaStream_t stream);

void sum_cog_from_devices(
  double3* peer_cogs,
  double3* cog_out,
  const int device_index,
  const int num_devices,
  cudaStream_t stream);

#endif // MGPU_KERNEL_CUH
