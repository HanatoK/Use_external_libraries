#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "mgpu_kernel.dp.hpp"
#include <dpct/dpl_utils.hpp>

template <int BLOCK_SIZE>
void sum_cog_kernel(const sycl::double3 *__restrict data,
                    sycl::double3 *__restrict cog_out,
                    sycl::double3 *__restrict peer_cogs,
                    unsigned int *__restrict count, const int device_index,
                    const int size, const int num_devices,
                    bool &isLastBlockDone) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  const int i = item_ct1.get_local_id(0) +
                item_ct1.get_group(0) * item_ct1.get_local_range(0);

  if (item_ct1.get_local_id(0) == 0) {
    isLastBlockDone = false;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  sycl::double3 p{0, 0, 0};
  if (i < size) {
    p.x() = data[i].x();
    p.y() = data[i].y();
    p.z() = data[i].z();
  }
  // typedef sycl::group<1> BlockReduce;

  p.x() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.x(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  p.y() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.y(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  p.z() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.z(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  if (item_ct1.get_local_id(0) == 0) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->x()), p.x());
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->y()), p.y());
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->z()), p.z());
    /*
    DPCT1078:0: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    unsigned int value = dpct::atomic_fetch_compare_inc<
        sycl::access::address_space::generic_space>(
        count, item_ct1.get_group_range(0));
    // printf("value = %u, gridDim.x = %d, blockIdx.x = %d\n", value, gridDim.x, blockIdx.x);
    isLastBlockDone = (value == (item_ct1.get_group_range(0) - 1));
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (isLastBlockDone) {
    if (item_ct1.get_local_id(0) == 0) {
      peer_cogs[device_index].x() = cog_out->x();
      peer_cogs[device_index].y() = cog_out->y();
      peer_cogs[device_index].z() = cog_out->z();
      (*count) = 0;
      /*
      DPCT1078:1: Consider replacing memory_order::acq_rel with
      memory_order::seq_cst for correctness if strong memory order restrictions
      are needed.
      */
      sycl::atomic_fence(sycl::memory_order::acq_rel,
                         sycl::memory_scope::device);
    }
  }
  // __syncthreads();
}

void sum_cog(const sycl::double3 *data, sycl::double3 *cog_out,
             sycl::double3 *peer_cogs, unsigned int *d_count,
             const int device_index, const int size, const int num_devices,
             dpct::queue_ptr stream) {
  if (size <= 0) return;
  const int block_size = 128;
  const int grid = (size + block_size - 1) / block_size;
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<bool, 0> isLastBlockDone_acc_ct1(cgh);

      cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid) * sycl::range<1>(block_size),
                                         sycl::range<1>(block_size)),
                       [=](sycl::nd_item<1> item_ct1) {
                         sum_cog_kernel<block_size>(
                             data, cog_out, peer_cogs, d_count, device_index,
                             size, num_devices, isLastBlockDone_acc_ct1);
                       });
    });
  }
}

template <int BLOCK_SIZE>
void sum_cog_devices_kernel(sycl::double3 *__restrict peer_cogs,
                            sycl::double3 *__restrict cog_out,
                            const int device_index, const int num_devices) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  const int i = item_ct1.get_local_id(0) +
                item_ct1.get_group(0) * item_ct1.get_local_range(0);
  sycl::double3 p{0, 0, 0};
  if (i < num_devices) {
    p.x() = peer_cogs[i].x();
    p.y() = peer_cogs[i].y();
    p.z() = peer_cogs[i].z();
  }
  typedef sycl::group<1> BlockReduce;

  /*
  DPCT1113:3: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "sum_cog_devices_kernel" is called in a
  multidimensional kernel.
  */
  p.x() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.x(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  /*
  DPCT1113:4: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "sum_cog_devices_kernel" is called in a
  multidimensional kernel.
  */
  p.y() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.y(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  /*
  DPCT1113:5: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "sum_cog_devices_kernel" is called in a
  multidimensional kernel.
  */
  p.z() = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_work_group<1>(), p.z(),
      sycl::plus<>());
      item_ct1.barrier(sycl::access::fence_space::local_space);
  if (item_ct1.get_local_id(0) == 0) {
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->x()), p.x());
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->y()), p.y());
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &(cog_out->z()), p.z());
    /*
    DPCT1078:2: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
  }
  /*
  DPCT1113:6: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "sum_cog_devices_kernel" is called in a
  multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

void sum_cog_from_devices(sycl::double3 *peer_cogs, sycl::double3 *cog_out,
                          const int device_index, const int num_devices,
                          dpct::queue_ptr stream) {
  if (num_devices <= 0) return;
  const int block_size = 16;
  const int grid = (num_devices + block_size - 1) / block_size;
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});

    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(grid) * sycl::range<1>(block_size),
                                           sycl::range<1>(block_size)),
                         [=](sycl::nd_item<1> item_ct1) {
                           sum_cog_devices_kernel<block_size>(
                               peer_cogs, cog_out, device_index, num_devices);
                         });
  }
}
