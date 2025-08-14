#include <vector>
#include <random>
// #include <algorithm>
#include <iostream>
#include <fmt/format.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/barrier>
#include <chrono>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 128

struct rmatrix {
  double xx, xy, xz;
  double yx, yy, yz;
  double zx, zy, zz;
};

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

std::vector<double> genData(size_t N, int seed = 0) {
  std::vector<double> result;
  // std::random_device rd;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> distrib(-2, 2);
  for (size_t i = 0; i < 3 * N; ++i) {
    result.push_back(distrib(gen));
  }
  return result;
}

// double3 reduceCpu(const std::vector<double3>& data) {
//   return std::accumulate(data.begin(), data.end(), double3{0, 0, 0}, [](const double3& a, const double3& b){return double3{a.x+b.x, a.y+b.y, a.z+b.z};});
// }

// Performs a reduction step and updates numTotal with how many are remaining
template <typename T, typename Group> __device__ T cg_reduce_n(T in, Group &threads)
{
  return cg::reduce(threads, in, cg::plus<T>());
}

template <unsigned int blockSize>
__device__ __forceinline__ double reduceBlock(volatile double *sdata, double mySum, const unsigned int tid, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid]                       = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    const int vid = tid & (VEC - 1);

    double beta = mySum;
    double temp;

    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (vid < i) {
            temp = sdata[tid + i];
            beta += temp;
            sdata[tid] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
    return sdata[0];
}

__global__ void reduce1(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  rmatrix* __restrict out,
  unsigned int* __restrict tbcount,
  size_t num_pos) {
  cg::thread_block cta = cg::this_thread_block();
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
  unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;
  rmatrix elem = {0};
  while (i < num_pos) {
    elem.xx += pos1x[i] * pos2x[i];
    elem.xy += pos1x[i] * pos2y[i];
    elem.xz += pos1x[i] * pos2z[i];
    elem.yx += pos1y[i] * pos2x[i];
    elem.yy += pos1y[i] * pos2y[i];
    elem.yz += pos1y[i] * pos2z[i];
    elem.zx += pos1z[i] * pos2x[i];
    elem.zy += pos1z[i] * pos2y[i];
    elem.zz += pos1z[i] * pos2z[i];
    if (i + BLOCK_SIZE < num_pos) {
      elem.xx += pos1x[i+BLOCK_SIZE] * pos2x[i+BLOCK_SIZE];
      elem.xy += pos1x[i+BLOCK_SIZE] * pos2y[i+BLOCK_SIZE];
      elem.xz += pos1x[i+BLOCK_SIZE] * pos2z[i+BLOCK_SIZE];
      elem.yx += pos1y[i+BLOCK_SIZE] * pos2x[i+BLOCK_SIZE];
      elem.yy += pos1y[i+BLOCK_SIZE] * pos2y[i+BLOCK_SIZE];
      elem.yz += pos1y[i+BLOCK_SIZE] * pos2z[i+BLOCK_SIZE];
      elem.zx += pos1z[i+BLOCK_SIZE] * pos2x[i+BLOCK_SIZE];
      elem.zy += pos1z[i+BLOCK_SIZE] * pos2y[i+BLOCK_SIZE];
      elem.zz += pos1z[i+BLOCK_SIZE] * pos2z[i+BLOCK_SIZE];
    }
    i += gridSize;
  }
  __shared__ double sdata[BLOCK_SIZE];
  elem.xx = reduceBlock<BLOCK_SIZE>(sdata, elem.xx, tid, cta);
  elem.xy = reduceBlock<BLOCK_SIZE>(sdata, elem.xy, tid, cta);
  elem.xz = reduceBlock<BLOCK_SIZE>(sdata, elem.xz, tid, cta);
  elem.yx = reduceBlock<BLOCK_SIZE>(sdata, elem.yx, tid, cta);
  elem.yy = reduceBlock<BLOCK_SIZE>(sdata, elem.yy, tid, cta);
  elem.yz = reduceBlock<BLOCK_SIZE>(sdata, elem.yz, tid, cta);
  elem.zx = reduceBlock<BLOCK_SIZE>(sdata, elem.zx, tid, cta);
  elem.zy = reduceBlock<BLOCK_SIZE>(sdata, elem.zy, tid, cta);
  elem.zz = reduceBlock<BLOCK_SIZE>(sdata, elem.zz, tid, cta);
  if (threadIdx.x == 0) {
    atomicAdd(&(out->xx), elem.xx);
    atomicAdd(&(out->xy), elem.xy);
    atomicAdd(&(out->xz), elem.xz);
    atomicAdd(&(out->yx), elem.yx);
    atomicAdd(&(out->yy), elem.yy);
    atomicAdd(&(out->yz), elem.yz);
    atomicAdd(&(out->zx), elem.zx);
    atomicAdd(&(out->zy), elem.zy);
    atomicAdd(&(out->zz), elem.zz);
    __threadfence();
    unsigned int value = atomicInc(tbcount, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  __syncthreads();
  if (isLastBlockDone) {
    if (threadIdx.x == 0) {
      tbcount[0] = 0;
    }
  }
}

__global__ void reduce2(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  double* __restrict buffer_xx,
  double* __restrict buffer_xy,
  double* __restrict buffer_xz,
  double* __restrict buffer_yx,
  double* __restrict buffer_yy,
  double* __restrict buffer_yz,
  double* __restrict buffer_zx,
  double* __restrict buffer_zy,
  double* __restrict buffer_zz,
  rmatrix* __restrict out,
  unsigned int* __restrict tbcount,
  size_t num_pos) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  // const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int gridSize = blockDim.x * gridDim.x;
  rmatrix elem = {0};
  while (i < num_pos) {
    elem.xx += pos1x[i] * pos2x[i];
    elem.xy += pos1x[i] * pos2y[i];
    elem.xz += pos1x[i] * pos2z[i];
    elem.yx += pos1y[i] * pos2x[i];
    elem.yy += pos1y[i] * pos2y[i];
    elem.yz += pos1y[i] * pos2z[i];
    elem.zx += pos1z[i] * pos2x[i];
    elem.zy += pos1z[i] * pos2y[i];
    elem.zz += pos1z[i] * pos2z[i];
    i += gridSize;
  }
  extern __shared__ double sdata[];
  elem.xx = reduceBlock<BLOCK_SIZE>(sdata, elem.xx, tid, cta);
  elem.xy = reduceBlock<BLOCK_SIZE>(sdata, elem.xy, tid, cta);
  elem.xz = reduceBlock<BLOCK_SIZE>(sdata, elem.xz, tid, cta);
  elem.yx = reduceBlock<BLOCK_SIZE>(sdata, elem.yx, tid, cta);
  elem.yy = reduceBlock<BLOCK_SIZE>(sdata, elem.yy, tid, cta);
  elem.yz = reduceBlock<BLOCK_SIZE>(sdata, elem.yz, tid, cta);
  elem.zx = reduceBlock<BLOCK_SIZE>(sdata, elem.zx, tid, cta);
  elem.zy = reduceBlock<BLOCK_SIZE>(sdata, elem.zy, tid, cta);
  elem.zz = reduceBlock<BLOCK_SIZE>(sdata, elem.zz, tid, cta);
  // Thread 0 takes a ticket
  if (tid == 0) {
    buffer_xx[blockIdx.x] = elem.xx;
    buffer_xy[blockIdx.x] = elem.xy;
    buffer_xz[blockIdx.x] = elem.xz;
    buffer_yx[blockIdx.x] = elem.yx;
    buffer_yy[blockIdx.x] = elem.yy;
    buffer_yz[blockIdx.x] = elem.yz;
    buffer_zx[blockIdx.x] = elem.zx;
    buffer_zy[blockIdx.x] = elem.zy;
    buffer_zz[blockIdx.x] = elem.zz;
    __threadfence();
    unsigned int value = atomicInc(tbcount, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  cg::sync(cta);
  if (isLastBlockDone) {
    // extern __shared__ double smem[];
    int i = tid;
    elem.xx = 0;
    elem.xy = 0;
    elem.xz = 0;
    elem.yx = 0;
    elem.yy = 0;
    elem.yz = 0;
    elem.zx = 0;
    elem.zy = 0;
    elem.zz = 0;
    while (i < gridDim.x) {
      elem.xx += buffer_xx[i];
      elem.xy += buffer_xy[i];
      elem.xz += buffer_xz[i];
      elem.yx += buffer_yx[i];
      elem.yy += buffer_yy[i];
      elem.yz += buffer_yz[i];
      elem.zx += buffer_zx[i];
      elem.zy += buffer_zy[i];
      elem.zz += buffer_zz[i];
      i += blockDim.x;
    }
    elem.xx = reduceBlock<BLOCK_SIZE>(sdata, elem.xx, tid, cta);
    elem.xy = reduceBlock<BLOCK_SIZE>(sdata, elem.xy, tid, cta);
    elem.xz = reduceBlock<BLOCK_SIZE>(sdata, elem.xz, tid, cta);
    elem.yx = reduceBlock<BLOCK_SIZE>(sdata, elem.yx, tid, cta);
    elem.yy = reduceBlock<BLOCK_SIZE>(sdata, elem.yy, tid, cta);
    elem.yz = reduceBlock<BLOCK_SIZE>(sdata, elem.yz, tid, cta);
    elem.zx = reduceBlock<BLOCK_SIZE>(sdata, elem.zx, tid, cta);
    elem.zy = reduceBlock<BLOCK_SIZE>(sdata, elem.zy, tid, cta);
    elem.zz = reduceBlock<BLOCK_SIZE>(sdata, elem.zz, tid, cta);
    if (tid == 0) {
      out->xx = elem.xx;
      out->xy = elem.xy;
      out->xz = elem.xz;
      out->yx = elem.yx;
      out->yy = elem.yy;
      out->yz = elem.yz;
      out->zx = elem.zx;
      out->zy = elem.zy;
      out->zz = elem.zz;
      tbcount[0] = 0;
    }
  }
}

template <int block_size>
__global__ void reduce3(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  rmatrix* __restrict out,
  unsigned int* __restrict tbcount,
  size_t num_pos) {
  __shared__ rmatrix sdata[block_size];
  cg::thread_block cta = cg::this_thread_block();
  // Handle to tile in thread block
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  unsigned int ctaSize     = cta.size();
  unsigned int numCtas     = gridDim.x;
  unsigned int threadRank  = cta.thread_rank();
  unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;
  __shared__ bool isLastBlockDone;
  if (threadRank == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  rmatrix elem = {0};
  {
    unsigned int i           = threadIndex;
    unsigned int indexStride = (numCtas * ctaSize);
    while (i < num_pos) {
      elem.xx += pos1x[i] * pos2x[i];
      elem.xy += pos1x[i] * pos2y[i];
      elem.xz += pos1x[i] * pos2z[i];
      elem.yx += pos1y[i] * pos2x[i];
      elem.yy += pos1y[i] * pos2y[i];
      elem.yz += pos1y[i] * pos2z[i];
      elem.zx += pos1z[i] * pos2x[i];
      elem.zy += pos1z[i] * pos2y[i];
      elem.zz += pos1z[i] * pos2z[i];
      i += indexStride;
    }
    sdata[threadRank].xx = elem.xx;
    sdata[threadRank].xy = elem.xy;
    sdata[threadRank].xz = elem.xz;
    sdata[threadRank].yx = elem.yx;
    sdata[threadRank].yy = elem.yy;
    sdata[threadRank].yz = elem.yz;
    sdata[threadRank].zx = elem.zx;
    sdata[threadRank].zy = elem.zy;
    sdata[threadRank].zz = elem.zz;
  }

  {
    // unsigned int ctaSteps = tile.meta_group_size();
    unsigned int ctaIndex = ctaSize >> 1;
    while (ctaIndex >= 32) {
      cta.sync();
      if (threadRank < ctaIndex) {
        elem.xx += sdata[threadRank + ctaIndex].xx;
        elem.xy += sdata[threadRank + ctaIndex].xy;
        elem.xz += sdata[threadRank + ctaIndex].xz;
        elem.yx += sdata[threadRank + ctaIndex].yx;
        elem.yy += sdata[threadRank + ctaIndex].yy;
        elem.yz += sdata[threadRank + ctaIndex].yz;
        elem.zx += sdata[threadRank + ctaIndex].zx;
        elem.zy += sdata[threadRank + ctaIndex].zy;
        elem.zz += sdata[threadRank + ctaIndex].zz;
        sdata[threadRank].xx = elem.xx;
        sdata[threadRank].xy = elem.xy;
        sdata[threadRank].xz = elem.xz;
        sdata[threadRank].yx = elem.yx;
        sdata[threadRank].yy = elem.yy;
        sdata[threadRank].yz = elem.yz;
        sdata[threadRank].zx = elem.zx;
        sdata[threadRank].zy = elem.zy;
        sdata[threadRank].zz = elem.zz;
      }
      // ctaSteps >>= 1;
      ctaIndex >>= 1;
    }
  }

  {
    cta.sync();
    if (tile.meta_group_rank() == 0) {
      elem.xx = cg_reduce_n(elem.xx, tile);
      elem.xy = cg_reduce_n(elem.xy, tile);
      elem.xz = cg_reduce_n(elem.xz, tile);
      elem.yx = cg_reduce_n(elem.yx, tile);
      elem.yy = cg_reduce_n(elem.yy, tile);
      elem.yz = cg_reduce_n(elem.yz, tile);
      elem.zx = cg_reduce_n(elem.zx, tile);
      elem.zy = cg_reduce_n(elem.zy, tile);
      elem.zz = cg_reduce_n(elem.zz, tile);
    }
  }
  if (threadRank == 0) {
    atomicAdd(&(out->xx), elem.xx);
    atomicAdd(&(out->xy), elem.xy);
    atomicAdd(&(out->xz), elem.xz);
    atomicAdd(&(out->yx), elem.yx);
    atomicAdd(&(out->yy), elem.yy);
    atomicAdd(&(out->yz), elem.yz);
    atomicAdd(&(out->zx), elem.zx);
    atomicAdd(&(out->zy), elem.zy);
    atomicAdd(&(out->zz), elem.zz);
    __threadfence();
    unsigned int value = atomicInc(tbcount, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  cta.sync();
  if (isLastBlockDone) {
    if (threadRank == 0) {
      tbcount[0] = 0;
    }
  }
}

template <int block_size>
__global__ void reduce4(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  rmatrix* __restrict out,
  unsigned int* __restrict tbcount,
  size_t num_pos) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  rmatrix elem = {0};
  unsigned int gridSize = block_size * gridDim.x;
  while (i < num_pos) {
    elem.xx += pos1x[i] * pos2x[i];
    elem.xy += pos1x[i] * pos2y[i];
    elem.xz += pos1x[i] * pos2z[i];
    elem.yx += pos1y[i] * pos2x[i];
    elem.yy += pos1y[i] * pos2y[i];
    elem.yz += pos1y[i] * pos2z[i];
    elem.zx += pos1z[i] * pos2x[i];
    elem.zy += pos1z[i] * pos2y[i];
    elem.zz += pos1z[i] * pos2z[i];
    i += gridSize;
  }
  __syncthreads();
  using BlockReduce = cub::BlockReduce<double, block_size, cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
  __shared__ typename BlockReduce::TempStorage temp_storage_xx;
  __shared__ typename BlockReduce::TempStorage temp_storage_xy;
  __shared__ typename BlockReduce::TempStorage temp_storage_xz;
  __shared__ typename BlockReduce::TempStorage temp_storage_yx;
  __shared__ typename BlockReduce::TempStorage temp_storage_yy;
  __shared__ typename BlockReduce::TempStorage temp_storage_yz;
  __shared__ typename BlockReduce::TempStorage temp_storage_zx;
  __shared__ typename BlockReduce::TempStorage temp_storage_zy;
  __shared__ typename BlockReduce::TempStorage temp_storage_zz;
  elem.xx = BlockReduce(temp_storage_xx).Sum(elem.xx);
  elem.xy = BlockReduce(temp_storage_xy).Sum(elem.xy);
  elem.xz = BlockReduce(temp_storage_xz).Sum(elem.xz);
  elem.yx = BlockReduce(temp_storage_yx).Sum(elem.yx);
  elem.yy = BlockReduce(temp_storage_yy).Sum(elem.yy);
  elem.yz = BlockReduce(temp_storage_yz).Sum(elem.yz);
  elem.zx = BlockReduce(temp_storage_zx).Sum(elem.zx);
  elem.zy = BlockReduce(temp_storage_zy).Sum(elem.zy);
  elem.zz = BlockReduce(temp_storage_zz).Sum(elem.zz);
  if (threadIdx.x == 0) {
    atomicAdd(&(out->xx), elem.xx);
    atomicAdd(&(out->xy), elem.xy);
    atomicAdd(&(out->xz), elem.xz);
    atomicAdd(&(out->yx), elem.yx);
    atomicAdd(&(out->yy), elem.yy);
    atomicAdd(&(out->yz), elem.yz);
    atomicAdd(&(out->zx), elem.zx);
    atomicAdd(&(out->zy), elem.zy);
    atomicAdd(&(out->zz), elem.zz);
    __threadfence();
    unsigned int value = atomicInc(tbcount, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  if (isLastBlockDone) {
    if (threadIdx.x == 0) {
      tbcount[0] = 0;
    }
  }
}

template <int block_size>
__global__ void reduce5(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  rmatrix* __restrict out,
  unsigned int* __restrict tbcount,
  size_t num_pos) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ bool isLastBlockDone;
  if (threadIdx.x == 0) {
    isLastBlockDone = false;
  }
  __syncthreads();
  rmatrix elem = {0};
  unsigned int gridSize = block_size * gridDim.x;
  while (i < num_pos) {
    elem.xx += pos1x[i] * pos2x[i];
    elem.xy += pos1x[i] * pos2y[i];
    elem.xz += pos1x[i] * pos2z[i];
    elem.yx += pos1y[i] * pos2x[i];
    elem.yy += pos1y[i] * pos2y[i];
    elem.yz += pos1y[i] * pos2z[i];
    elem.zx += pos1z[i] * pos2x[i];
    elem.zy += pos1z[i] * pos2y[i];
    elem.zz += pos1z[i] * pos2z[i];
    i += gridSize;
  }
  __syncthreads();
  using BlockReduce = cub::BlockReduce<double, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  elem.xx = BlockReduce(temp_storage).Sum(elem.xx); __syncthreads();
  elem.xy = BlockReduce(temp_storage).Sum(elem.xy); __syncthreads();
  elem.xz = BlockReduce(temp_storage).Sum(elem.xz); __syncthreads();
  elem.yx = BlockReduce(temp_storage).Sum(elem.yx); __syncthreads();
  elem.yy = BlockReduce(temp_storage).Sum(elem.yy); __syncthreads();
  elem.yz = BlockReduce(temp_storage).Sum(elem.yz); __syncthreads();
  elem.zx = BlockReduce(temp_storage).Sum(elem.zx); __syncthreads();
  elem.zy = BlockReduce(temp_storage).Sum(elem.zy); __syncthreads();
  elem.zz = BlockReduce(temp_storage).Sum(elem.zz); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&(out->xx), elem.xx);
    atomicAdd(&(out->xy), elem.xy);
    atomicAdd(&(out->xz), elem.xz);
    atomicAdd(&(out->yx), elem.yx);
    atomicAdd(&(out->yy), elem.yy);
    atomicAdd(&(out->yz), elem.yz);
    atomicAdd(&(out->zx), elem.zx);
    atomicAdd(&(out->zy), elem.zy);
    atomicAdd(&(out->zz), elem.zz);
    __threadfence();
    unsigned int value = atomicInc(tbcount, gridDim.x);
    isLastBlockDone = (value == (gridDim.x - 1));
  }
  if (isLastBlockDone) {
    if (threadIdx.x == 0) {
      tbcount[0] = 0;
    }
  }
}

// template <int num_warps>
// __global__ void reduce6(
//   const double* __restrict pos1x,
//   const double* __restrict pos1y,
//   const double* __restrict pos1z,
//   const double* __restrict pos2x,
//   const double* __restrict pos2y,
//   const double* __restrict pos2z,
//   rmatrix* __restrict out,
//   unsigned int* __restrict tbcount,
//   size_t num_pos) {
//   auto block = cg::this_thread_block();
//   cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
//   const unsigned int buffer_size = (num_warps * warpSize) / 2;
//   __shared__ rmatrix buffer[buffer_size];
//   __shared__ double buffer_summed[num_warps / 2];
//   __shared__ cuda::barrier<cuda::thread_scope_block> bar[num_warps];
//   __shared__ bool isLastBlockDone;
//   buffer[block.thread_rank()] = {0};
//   if (block.thread_rank() < num_warps) {
//     init(bar + block.thread_rank(), block.size());
//   }
//   if (block.thread_rank() < num_warps / 2) {
//     buffer_summed[block.thread_rank()] = 0;
//   }
//   if (block.thread_rank() == 0) {
//     isLastBlockDone = false;
//   }
//   block.sync();
//   if (tile.meta_group_rank() < num_warps / 2) {
//     // producer
//     unsigned int ctaSize     = block.size();
//     unsigned int numCtas     = gridDim.x;
//     unsigned int threadRank  = block.thread_rank();
//     unsigned int threadIndex = (blockIdx.x * ctaSize) + threadRank;
//     unsigned int i           = threadIndex;
//     unsigned int indexStride = (numCtas * ctaSize);
//     while (i < num_pos) {
//       for (int j = 0; j < num_warps / 2; ++j) {
//         // TODO
//       }
//       // buffer[threadRank].xx = pos1x[i] * pos2x[i];
//       buffer[threadRank].xx = pos1x[i] * pos2x[i];
//       buffer[threadRank].xy = pos1x[i] * pos2y[i];
//       buffer[threadRank].xz = pos1x[i] * pos2z[i];
//       buffer[threadRank].yx = pos1y[i] * pos2x[i];
//       buffer[threadRank].yy = pos1y[i] * pos2y[i];
//       buffer[threadRank].yz = pos1y[i] * pos2z[i];
//       buffer[threadRank].zx = pos1z[i] * pos2x[i];
//       buffer[threadRank].zy = pos1z[i] * pos2y[i];
//       buffer[threadRank].zz = pos1z[i] * pos2z[i];
//     }
//   } else {
//     // consumer
//   }
// }

rmatrix reduceCuda1(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  double* d_pos1;
  double* d_pos2;
  rmatrix* d_out;
  rmatrix* h_out;
  unsigned int* tbcount;
  checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
  checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  const int num_blocks = std::min(64, ((int)N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  const bool profiling = (elapsed_time != nullptr);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
  }
  reduce1<<<num_blocks, BLOCK_SIZE, 0>>>(
    d_pos1, d_pos1 + N, d_pos1 + 2 * N,
    d_pos2, d_pos2 + N, d_pos2 + 2 * N,
    d_out, tbcount , N);
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
  // std::cout << fmt::format("reduceCuda1: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  rmatrix res;
  std::memcpy(&res, h_out, sizeof(rmatrix));
  checkCudaErrors(cudaFree(d_pos1));
  checkCudaErrors(cudaFree(d_pos2));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(tbcount));
  checkCudaErrors(cudaFreeHost(h_out));
  return res;
}

rmatrix reduceCuda2(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  double* d_pos1;
  double* d_pos2;
  rmatrix* d_out;
  rmatrix* h_out;
  unsigned int* tbcount;
  double* d_buffer;
  checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
  checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  int minGridSize, suggesteBlockSize;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &suggesteBlockSize, (void*)reduce2, [](const unsigned int bsize){return bsize*sizeof(double);}, 512));
  // std::cout << "minGridSize = " << minGridSize << "\n";
  // std::cout << "suggesteBlockSize = " << suggesteBlockSize << std::endl;
  const int num_blocks = std::min(minGridSize, ((int)N + suggesteBlockSize - 1) / suggesteBlockSize);
  checkCudaErrors(cudaMalloc(&d_buffer, 9 * num_blocks * sizeof(double)));
  const bool profiling = (elapsed_time != nullptr);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
  }
  reduce2<<<num_blocks, suggesteBlockSize, suggesteBlockSize*sizeof(double)>>>(
    d_pos1, d_pos1 + N, d_pos1 + 2 * N,
    d_pos2, d_pos2 + N, d_pos2 + 2 * N,
    d_buffer,
    d_buffer + num_blocks * 1,
    d_buffer + num_blocks * 2,
    d_buffer + num_blocks * 3,
    d_buffer + num_blocks * 4,
    d_buffer + num_blocks * 5,
    d_buffer + num_blocks * 6,
    d_buffer + num_blocks * 7,
    d_buffer + num_blocks * 8,
    d_out, tbcount , N);
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
  // std::cout << fmt::format("reduceCuda2: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  rmatrix res;
  std::memcpy(&res, h_out, sizeof(rmatrix));
  checkCudaErrors(cudaFree(d_pos1));
  checkCudaErrors(cudaFree(d_pos2));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_buffer));
  checkCudaErrors(cudaFree(tbcount));
  checkCudaErrors(cudaFreeHost(h_out));
  return res;
}

rmatrix reduceCuda3(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  double* d_pos1;
  double* d_pos2;
  rmatrix* d_out;
  rmatrix* h_out;
  unsigned int* tbcount;
  checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
  checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  const int block_size = 32;
  const int num_blocks = std::min(256, ((int)N + block_size - 1) / block_size);
  const bool profiling = (elapsed_time != nullptr);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
  }
  reduce3<block_size><<<num_blocks, block_size, 0>>>(
    d_pos1, d_pos1 + N, d_pos1 + 2 * N,
    d_pos2, d_pos2 + N, d_pos2 + 2 * N,
    d_out, tbcount , N);
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
  // std::cout << fmt::format("reduceCuda3: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  rmatrix res;
  std::memcpy(&res, h_out, sizeof(rmatrix));
  checkCudaErrors(cudaFree(d_pos1));
  checkCudaErrors(cudaFree(d_pos2));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(tbcount));
  checkCudaErrors(cudaFreeHost(h_out));
  return res;
}

rmatrix reduceCuda4(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  double* d_pos1;
  double* d_pos2;
  rmatrix* d_out;
  rmatrix* h_out;
  unsigned int* tbcount;
  checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
  checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  const int block_size = 128;
  const int num_blocks = std::min(64, ((int)N + block_size - 1) / block_size);
  const bool profiling = (elapsed_time != nullptr);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
  }
  reduce4<block_size><<<num_blocks, block_size, 0>>>(
    d_pos1, d_pos1 + N, d_pos1 + 2 * N,
    d_pos2, d_pos2 + N, d_pos2 + 2 * N,
    d_out, tbcount , N);
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
  // std::cout << fmt::format("reduceCuda3: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  rmatrix res;
  std::memcpy(&res, h_out, sizeof(rmatrix));
  checkCudaErrors(cudaFree(d_pos1));
  checkCudaErrors(cudaFree(d_pos2));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(tbcount));
  checkCudaErrors(cudaFreeHost(h_out));
  return res;
}

rmatrix reduceCuda5(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  double* d_pos1;
  double* d_pos2;
  rmatrix* d_out;
  rmatrix* h_out;
  unsigned int* tbcount;
  checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
  checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
  checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
  checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
  const int block_size = 128;
  const int num_blocks = std::min(64, ((int)N + block_size - 1) / block_size);
  const bool profiling = (elapsed_time != nullptr);
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
  }
  reduce5<block_size><<<num_blocks, block_size>>>(
    d_pos1, d_pos1 + N, d_pos1 + 2 * N,
    d_pos2, d_pos2 + N, d_pos2 + 2 * N,
    d_out, tbcount , N);
  if (profiling) {
    checkCudaErrors(cudaDeviceSynchronize());
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
  // std::cout << fmt::format("reduceCuda3: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  rmatrix res;
  std::memcpy(&res, h_out, sizeof(rmatrix));
  checkCudaErrors(cudaFree(d_pos1));
  checkCudaErrors(cudaFree(d_pos2));
  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(tbcount));
  checkCudaErrors(cudaFreeHost(h_out));
  return res;
}

// rmatrix reduceCuda6(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
//   double* d_pos1;
//   double* d_pos2;
//   rmatrix* d_out;
//   rmatrix* h_out;
//   unsigned int* tbcount;
//   checkCudaErrors(cudaMalloc(&d_pos1, 3 * N * sizeof(double)));
//   checkCudaErrors(cudaMalloc(&d_pos2, 3 * N * sizeof(double)));
//   checkCudaErrors(cudaMalloc(&d_out, sizeof(rmatrix)));
//   checkCudaErrors(cudaMalloc(&tbcount, sizeof(unsigned int)));
//   checkCudaErrors(cudaMemset(d_out, 0, 1 * sizeof(rmatrix)));
//   checkCudaErrors(cudaMemset(tbcount, 0, 1 * sizeof(unsigned int)));
//   checkCudaErrors(cudaMallocHost(&h_out, sizeof(rmatrix)));
//   checkCudaErrors(cudaMemcpy(d_pos1, pos1.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
//   checkCudaErrors(cudaMemcpy(d_pos2, pos2.data(), 3 * N * sizeof(double), cudaMemcpyHostToDevice));
//   const int block_size = 128;
//   int devID = 0;
//   int numSMs;
//   checkCudaErrors(cudaGetDevice(&devID));
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numSMs, (void*)reduce5<block_size>, block_size, 0);
//   // std::cout << "num_blocks = " << numSMs << std::endl;
//   const int num_blocks = numSMs;
//   const bool profiling = (elapsed_time != nullptr);
//   std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
//   if (profiling) {
//     checkCudaErrors(cudaDeviceSynchronize());
//     start = std::chrono::high_resolution_clock::now();
//   }
//   reduce5<block_size><<<num_blocks, block_size>>>(
//     d_pos1, d_pos1 + N, d_pos1 + 2 * N,
//     d_pos2, d_pos2 + N, d_pos2 + 2 * N,
//     d_out, tbcount , N);
//   if (profiling) {
//     checkCudaErrors(cudaDeviceSynchronize());
//     end= std::chrono::high_resolution_clock::now();
//     *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//   }
//   checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(rmatrix), cudaMemcpyDeviceToHost));
//   // std::cout << fmt::format("reduceCuda3: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
//   // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
//   // std::cout << fmt::format("             ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
//   rmatrix res;
//   std::memcpy(&res, h_out, sizeof(rmatrix));
//   checkCudaErrors(cudaFree(d_pos1));
//   checkCudaErrors(cudaFree(d_pos2));
//   checkCudaErrors(cudaFree(d_out));
//   checkCudaErrors(cudaFree(tbcount));
//   checkCudaErrors(cudaFreeHost(h_out));
//   return res;
// }

rmatrix reduceCPU(const std::vector<double>& pos1, const std::vector<double> pos2, const size_t N, double *elapsed_time = nullptr) {
  const bool profiling = (elapsed_time != nullptr);
  rmatrix elem = {0};
  const auto* pos1x = pos1.data();
  const auto* pos1y = pos1.data() + N;
  const auto* pos1z = pos1.data() + 2 * N;
  const auto* pos2x = pos2.data();
  const auto* pos2y = pos2.data() + N;
  const auto* pos2z = pos2.data() + 2 * N;
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  if (profiling) {
    start = std::chrono::high_resolution_clock::now();
  }
  for (size_t i = 0; i < N; ++i) {
    elem.xx += pos1x[i] * pos2x[i];
    elem.xy += pos1x[i] * pos2y[i];
    elem.xz += pos1x[i] * pos2z[i];
    elem.yx += pos1y[i] * pos2x[i];
    elem.yy += pos1y[i] * pos2y[i];
    elem.yz += pos1y[i] * pos2z[i];
    elem.zx += pos1z[i] * pos2x[i];
    elem.zy += pos1z[i] * pos2y[i];
    elem.zz += pos1z[i] * pos2z[i];
  }
  if (profiling) {
    end= std::chrono::high_resolution_clock::now();
    *elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
  // rmatrix* h_out = &elem;
  // std::cout << fmt::format("reduceCPU: ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->xx, h_out->xy, h_out->xz);
  // std::cout << fmt::format("           ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->yx, h_out->yy, h_out->yz);
  // std::cout << fmt::format("           ({:12.5f}, {:12.5f}, {:12.5f})\n", h_out->zx, h_out->zy, h_out->zz);
  return elem;
}

double compute_rmsd(const rmatrix& mat1, const rmatrix& mat2){
  double sum = 0;
  sum += (mat1.xx - mat2.xx) * (mat1.xx - mat2.xx);
  sum += (mat1.xy - mat2.xy) * (mat1.xy - mat2.xy);
  sum += (mat1.xz - mat2.xz) * (mat1.xz - mat2.xz);
  sum += (mat1.yx - mat2.yx) * (mat1.yx - mat2.yx);
  sum += (mat1.yy - mat2.yy) * (mat1.yy - mat2.yy);
  sum += (mat1.yz - mat2.yz) * (mat1.yz - mat2.yz);
  sum += (mat1.zx - mat2.zx) * (mat1.zx - mat2.zx);
  sum += (mat1.zy - mat2.zy) * (mat1.zy - mat2.zy);
  sum += (mat1.zz - mat2.zz) * (mat1.zz - mat2.zz);
  return std::sqrt(sum);
}

int main() {
  const size_t N = 100000;
  double sum_times[6] = {0};
  int count = 0;
  for (int j = 0; j < 50; ++j) {
    double times[6] = {0};
    const auto pos1 = genData(N, 0+j);
    const auto pos2 = genData(N, 1000+j);
    const auto cpu_result = reduceCPU(pos1, pos2, N, j > 0 ? &times[0] : nullptr);
    const auto cuda1_result = reduceCuda1(pos1, pos2, N, j > 0 ? &times[1] : nullptr);
    const auto cuda2_result = reduceCuda2(pos1, pos2, N, j > 0 ? &times[2] : nullptr);
    const auto cuda3_result = reduceCuda3(pos1, pos2, N, j > 0 ? &times[3] : nullptr);
    const auto cuda4_result = reduceCuda4(pos1, pos2, N, j > 0 ? &times[4] : nullptr);
    const auto cuda5_result = reduceCuda5(pos1, pos2, N, j > 0 ? &times[5] : nullptr);
    const double error1 = compute_rmsd(cuda1_result, cpu_result);
    const double error2 = compute_rmsd(cuda2_result, cpu_result);
    const double error3 = compute_rmsd(cuda3_result, cpu_result);
    const double error4 = compute_rmsd(cuda4_result, cpu_result);
    const double error5 = compute_rmsd(cuda5_result, cpu_result);
    if (j > 0) {
      for (int k = 0; k < 6; ++k) {
        sum_times[k] += times[k];
      }
      count += 1;
    }
    std::cout << fmt::format(
      "Error1 = {}, Error2 = {}, Error3 = {}, Error4 = {}, Error5 = {}\n",
      error1, error2, error3, error4, error5);
  }
  std::cout << std::string(80, '=') + "\n" + "Average time\n" + std::string(80, '=') << std::endl;
  std::cout << fmt::format("CPU: {}\n", sum_times[0] / count);
  for (int k = 1; k < 6; ++k) {
    std::cout << fmt::format("CUDA Kernel{}: {}\n", k, sum_times[k] / count);
  }
  return 0;
}
