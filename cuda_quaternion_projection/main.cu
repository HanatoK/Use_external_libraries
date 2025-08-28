#include "main.h"
#include "cub/block/block_reduce.cuh"

template <int block_size>
__global__ void project1_kernel(
  const rvector* pos, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int gridSize = gridDim.x * blockDim.x;
  double4 sum_dxdq{0, 0, 0, 0};
  while (i < num_pos) {
    const auto tmp_q = q->position_derivative_inner(pos[i], f[i]);
    sum_dxdq.w += tmp_q.q0;
    sum_dxdq.x += tmp_q.q1;
    sum_dxdq.y += tmp_q.q2;
    sum_dxdq.z += tmp_q.q3;
    i += gridSize;
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum_dxdq.w = BlockReduce(temp_storage).Sum(sum_dxdq.w); __syncthreads();
  sum_dxdq.x = BlockReduce(temp_storage).Sum(sum_dxdq.x); __syncthreads();
  sum_dxdq.y = BlockReduce(temp_storage).Sum(sum_dxdq.y); __syncthreads();
  sum_dxdq.z = BlockReduce(temp_storage).Sum(sum_dxdq.z); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&(out->w), sum_dxdq.w);
    atomicAdd(&(out->x), sum_dxdq.x);
    atomicAdd(&(out->y), sum_dxdq.y);
    atomicAdd(&(out->z), sum_dxdq.z);
  }
}

void project1_cuda(const rvector* pos1, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = (num_pos + block_size - 1) / block_size;
  const int num_blocks_used = num_blocks > max_blocks ? max_blocks : num_blocks;
  project1_kernel<block_size><<<num_blocks_used, block_size, 0, stream>>>(
    pos1, f, q, out, num_pos);
}

template <int block_size>
__global__ void project2_kernel(
  const rvector* pos, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int gridSize = gridDim.x * blockDim.x;
  // double4 sum_dxdq{0, 0, 0, 0};
  double C[3][3] = {{0}};
  while (i < num_pos) {
    C[0][0] += f[i].x * pos[i].x;
    C[0][1] += f[i].x * pos[i].y;
    C[0][2] += f[i].x * pos[i].z;
    C[1][0] += f[i].y * pos[i].x;
    C[1][1] += f[i].y * pos[i].y;
    C[1][2] += f[i].y * pos[i].z;
    C[2][0] += f[i].z * pos[i].x;
    C[2][1] += f[i].z * pos[i].y;
    C[2][2] += f[i].z * pos[i].z;
    i += gridSize;
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  C[0][0] = BlockReduce(temp_storage).Sum(C[0][0]); __syncthreads();
  C[0][1] = BlockReduce(temp_storage).Sum(C[0][1]); __syncthreads();
  C[0][2] = BlockReduce(temp_storage).Sum(C[0][2]); __syncthreads();
  C[1][0] = BlockReduce(temp_storage).Sum(C[1][0]); __syncthreads();
  C[1][1] = BlockReduce(temp_storage).Sum(C[1][1]); __syncthreads();
  C[1][2] = BlockReduce(temp_storage).Sum(C[1][2]); __syncthreads();
  C[2][0] = BlockReduce(temp_storage).Sum(C[2][0]); __syncthreads();
  C[2][1] = BlockReduce(temp_storage).Sum(C[2][1]); __syncthreads();
  C[2][2] = BlockReduce(temp_storage).Sum(C[2][2]); __syncthreads();
  if (threadIdx.x == 0) {
    const auto x = q->derivative_element_wise_product_sum(C);
    atomicAdd(&(out->w), x[0]);
    atomicAdd(&(out->x), x[1]);
    atomicAdd(&(out->y), x[2]);
    atomicAdd(&(out->z), x[3]);
  }
}

void project2_cuda(const rvector* pos1, const rvector* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = (num_pos + block_size - 1) / block_size;
  const int num_blocks_used = num_blocks > max_blocks ? max_blocks : num_blocks;
  project2_kernel<block_size><<<num_blocks_used, block_size, 0, stream>>>(
    pos1, f, q, out, num_pos);
}

template <int block_size>
__global__ void project1_soa_kernel(
  const double* pos_x,
  const double* pos_y,
  const double* pos_z,
  const double* f_x,
  const double* f_y,
  const double* f_z,
  const quaternion* q, double4* out, unsigned int num_pos) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int gridSize = gridDim.x * blockDim.x;
  double4 sum_dxdq{0, 0, 0, 0};
  while (i < num_pos) {
    const rvector pos{pos_x[i], pos_y[i], pos_z[i]};
    const rvector f{f_x[i], f_y[i], f_z[i]};
    const auto tmp_q = q->position_derivative_inner(pos, f);
    sum_dxdq.w += tmp_q.q0;
    sum_dxdq.x += tmp_q.q1;
    sum_dxdq.y += tmp_q.q2;
    sum_dxdq.z += tmp_q.q3;
    i += gridSize;
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  sum_dxdq.w = BlockReduce(temp_storage).Sum(sum_dxdq.w); __syncthreads();
  sum_dxdq.x = BlockReduce(temp_storage).Sum(sum_dxdq.x); __syncthreads();
  sum_dxdq.y = BlockReduce(temp_storage).Sum(sum_dxdq.y); __syncthreads();
  sum_dxdq.z = BlockReduce(temp_storage).Sum(sum_dxdq.z); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(&(out->w), sum_dxdq.w);
    atomicAdd(&(out->x), sum_dxdq.x);
    atomicAdd(&(out->y), sum_dxdq.y);
    atomicAdd(&(out->z), sum_dxdq.z);
  }
}

void project1_soa_cuda(const double* pos, const double* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = (num_pos + block_size - 1) / block_size;
  const int num_blocks_used = num_blocks > max_blocks ? max_blocks : num_blocks;
  const double* pos_x = pos;
  const double* pos_y = pos_x + num_pos;
  const double* pos_z = pos_y + num_pos;
  const double* f_x = f;
  const double* f_y = f_x + num_pos;
  const double* f_z = f_y + num_pos;
  project1_soa_kernel<block_size><<<num_blocks_used, block_size, 0, stream>>>(
    pos_x, pos_y, pos_z, f_x, f_y, f_z, q, out, num_pos);
}

template <int block_size>
__global__ void project2_soa_kernel(
  const double* pos_x,
  const double* pos_y,
  const double* pos_z,
  const double* f_x,
  const double* f_y,
  const double* f_z,
  const quaternion* q, double4* out, unsigned int num_pos) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int gridSize = gridDim.x * blockDim.x;
  // double4 sum_dxdq{0, 0, 0, 0};
  double C[3][3] = {{0}};
  while (i < num_pos) {
    C[0][0] += f_x[i] * pos_x[i];
    C[0][1] += f_x[i] * pos_y[i];
    C[0][2] += f_x[i] * pos_z[i];
    C[1][0] += f_y[i] * pos_x[i];
    C[1][1] += f_y[i] * pos_y[i];
    C[1][2] += f_y[i] * pos_z[i];
    C[2][0] += f_z[i] * pos_x[i];
    C[2][1] += f_z[i] * pos_y[i];
    C[2][2] += f_z[i] * pos_z[i];
    i += gridSize;
  }
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  C[0][0] = BlockReduce(temp_storage).Sum(C[0][0]); __syncthreads();
  C[0][1] = BlockReduce(temp_storage).Sum(C[0][1]); __syncthreads();
  C[0][2] = BlockReduce(temp_storage).Sum(C[0][2]); __syncthreads();
  C[1][0] = BlockReduce(temp_storage).Sum(C[1][0]); __syncthreads();
  C[1][1] = BlockReduce(temp_storage).Sum(C[1][1]); __syncthreads();
  C[1][2] = BlockReduce(temp_storage).Sum(C[1][2]); __syncthreads();
  C[2][0] = BlockReduce(temp_storage).Sum(C[2][0]); __syncthreads();
  C[2][1] = BlockReduce(temp_storage).Sum(C[2][1]); __syncthreads();
  C[2][2] = BlockReduce(temp_storage).Sum(C[2][2]); __syncthreads();
  if (threadIdx.x == 0) {
    const auto x = q->derivative_element_wise_product_sum(C);
    atomicAdd(&(out->w), x[0]);
    atomicAdd(&(out->x), x[1]);
    atomicAdd(&(out->y), x[2]);
    atomicAdd(&(out->z), x[3]);
  }
}

void project2_soa_cuda(const double* pos, const double* f,
  const quaternion* q, double4* out, unsigned int num_pos, cudaStream_t stream) {
  const int block_size = 128;
  const int num_blocks = (num_pos + block_size - 1) / block_size;
  const int num_blocks_used = num_blocks > max_blocks ? max_blocks : num_blocks;
  const double* pos_x = pos;
  const double* pos_y = pos_x + num_pos;
  const double* pos_z = pos_y + num_pos;
  const double* f_x = f;
  const double* f_y = f_x + num_pos;
  const double* f_z = f_y + num_pos;
  project2_soa_kernel<block_size><<<num_blocks_used, block_size, 0, stream>>>(
    pos_x, pos_y, pos_z, f_x, f_y, f_z, q, out, num_pos);
}
