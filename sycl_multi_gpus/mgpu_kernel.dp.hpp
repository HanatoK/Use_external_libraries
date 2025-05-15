#ifndef MGPU_KERNEL_CUH
#define MGPU_KERNEL_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

void sum_cog(const sycl::double3 *data, sycl::double3 *cog_out,
             sycl::double3 *peer_cogs, unsigned int *d_count,
             const int device_index, const int size, const int num_devices,
             dpct::queue_ptr stream);

void sum_cog_from_devices(sycl::double3 *peer_cogs, sycl::double3 *cog_out,
                          const int device_index, const int num_devices,
                          dpct::queue_ptr stream);

#endif // MGPU_KERNEL_CUH
