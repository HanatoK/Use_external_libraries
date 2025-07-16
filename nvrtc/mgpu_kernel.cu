// #include "mgpu_kernel.cuh"
// #include "common.h"
#include <cub/block/block_reduce.cuh>

#define BLOCK_SIZE 128

// template <int BLOCK_SIZE>
__global__ void sum_cog_kernel(
  const double3* __restrict data,
  double3* __restrict cog_out,
  double3* __restrict peer_cogs,
  unsigned int* __restrict count,
  const int device_index,
  const int size,
  const int num_devices) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  double3 p{0, 0, 0};
  if (i < size) {
    p.x = data[i].x;
    p.y = data[i].y;
    p.z = data[i].z;
  }
  typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  p.x = BlockReduce(temp_storage).Sum(p.x); __syncthreads();
  p.y = BlockReduce(temp_storage).Sum(p.y); __syncthreads();
  p.z = BlockReduce(temp_storage).Sum(p.z); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&(cog_out->x), p.x);
    atomicAdd(&(cog_out->y), p.y);
    atomicAdd(&(cog_out->z), p.z);
    __threadfence();
    unsigned int value = atomicInc(count, gridDim.x);
    // printf("value = %u, gridDim.x = %d, blockIdx.x = %d\n", value, gridDim.x, blockIdx.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  __syncthreads();
  if (isLastBlockDone) {
    if (threadIdx.x == 0) {
      peer_cogs[device_index].x = cog_out->x;
      peer_cogs[device_index].y = cog_out->y;
      peer_cogs[device_index].z = cog_out->z;
      (*count) = 0;
      __threadfence();
    }
  }
  // __syncthreads();
}

/*
void sum_cog(
  const double3* data,
  double3* cog_out,
  double3* peer_cogs,
  unsigned int* d_count,
  const int device_index,
  const int size,
  const int num_devices,
  cudaStream_t stream) {
  if (size <= 0) return;
  const int block_size = 128;
  const int grid = (size + block_size - 1) / block_size;
  sum_cog_kernel<block_size><<<grid, block_size, 0, stream>>>(
    data, cog_out, peer_cogs, d_count, device_index, size, num_devices);
}*/

// template <int BLOCK_SIZE>
__global__ void sum_cog_devices_kernel(
  double3* __restrict peer_cogs,
  double3* __restrict cog_out,
  const int device_index,
  const int num_devices) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  double3 p{0, 0, 0};
  if (i < num_devices) {
    p.x = peer_cogs[i].x;
    p.y = peer_cogs[i].y;
    p.z = peer_cogs[i].z;
  }
  typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  p.x = BlockReduce(temp_storage).Sum(p.x); __syncthreads();
  p.y = BlockReduce(temp_storage).Sum(p.y); __syncthreads();
  p.z = BlockReduce(temp_storage).Sum(p.z); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&(cog_out->x), p.x);
    atomicAdd(&(cog_out->y), p.y);
    atomicAdd(&(cog_out->z), p.z);
    __threadfence();
  }
  __syncthreads();
}

/*
void sum_cog_from_devices(
  double3* peer_cogs,
  double3* cog_out,
  const int device_index,
  const int num_devices,
  cudaStream_t stream) {
  if (num_devices <= 0) return;
  const int block_size = 16;
  const int grid = (num_devices + block_size - 1) / block_size;
  sum_cog_devices_kernel<block_size><<<grid, block_size, 0, stream>>>(
    peer_cogs, cog_out, device_index, num_devices);
}*/
