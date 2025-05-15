#include <mutex>
#include <iostream>
#include <condition_variable>
#include <memory>
#include <random>
#include <vector>
#include <thread>
#include <sstream>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpl_utils.hpp>
#include "mgpu_kernel.dp.hpp"


std::mutex print_mtx;

class barrier {
public:
  barrier(int count):thread_count(count), counter(0), waiting(0) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m);
    ++counter;
    ++waiting;
    cv.wait(lk, [&]{return counter >= thread_count;});
    cv.notify_one();
    --waiting;
    if (waiting == 0) {
      counter = 0;
    }
    lk.unlock();
  }
private:
  std::mutex m;
  std::condition_variable cv;
  int thread_count;
  int counter;
  int waiting;
};

std::unique_ptr<barrier> sync_host_threads;

class my_data {
public:
  using sycl_host_allocator = sycl::usm_allocator<sycl::double3, sycl::usm::alloc::host>;
  my_data(int num_points, sycl::queue& master_queue):
    q(master_queue), points(sycl_host_allocator(q)) {
    std::mt19937 gen(0);
    std::normal_distribution<> dis(1.0, 6.0);
    points.resize(num_points);
    q.wait();
    ref_cog.x() = 0;
    ref_cog.y() = 0;
    ref_cog.z() = 0;
    for (int i = 0; i < num_points; ++i) {
      points[i].x() = dis(gen);
      points[i].y() = dis(gen);
      points[i].z() = dis(gen);
    }
    for (int i = 0; i < num_points; ++i) {
      ref_cog.x() += points[i].x();
      ref_cog.y() += points[i].y();
      ref_cog.z() += points[i].z();
    }
    ref_cog.x() /= num_points;
    ref_cog.y() /= num_points;
    ref_cog.z() /= num_points;
    h_thread_cogs = nullptr;
    std::cout << "Ref cog: (" << std::to_string(ref_cog.x()) << ", " << std::to_string(ref_cog.y()) << ", " << std::to_string(ref_cog.z()) << ")" << std::endl;
  }
  void set_host_com_buffers(int num_threads) {
    if (h_thread_cogs) sycl::free(h_thread_cogs, q);
    h_thread_cogs = (sycl::double3*)sycl::malloc_device(sycl::double3::byte_size()*num_threads, q);
    q.wait();
  }
  ~my_data() {
    if (h_thread_cogs) sycl::free(h_thread_cogs, q);
    q.wait();
  }
public:
  sycl::queue& q;
  std::vector<sycl::double3, sycl_host_allocator> points;
  sycl::double3 ref_cog;
  sycl::double3* h_thread_cogs;
};

void print_with_lock(const std::string& str, const bool flush = true) {
  print_mtx.lock();
  std::cout << str;
  if (flush) std::cout << std::flush;
  print_mtx.unlock();
}

void call_from_thread(int tid, int num_threads, my_data& data_in, sycl::device dev) {
  // const auto backend = dev.get_backend();
  sycl::queue thread_queue(dev, sycl::property_list{sycl::property::queue::in_order{}});
  int work_size = data_in.points.size() / num_threads;
  const int my_work_start = tid * work_size;
  if (tid == num_threads - 1) {
    work_size = data_in.points.size() - (num_threads - 1) * work_size;
  }
  sycl::double3* d_points;
  sycl::double3* d_cog;
  sycl::double3* h_cog;
  unsigned int* d_count;
  h_cog = (sycl::double3*)sycl::malloc_host(sycl::double3::byte_size(), thread_queue);
  d_points = (sycl::double3*)sycl::malloc_device(sycl::double3::byte_size()*work_size, thread_queue);
  d_cog = (sycl::double3*)sycl::malloc_device(sycl::double3::byte_size(), thread_queue);
  d_count = (unsigned int*)sycl::malloc_device(sizeof(unsigned int), thread_queue);
  sycl::double3* h_points = data_in.points.data() + my_work_start;
  thread_queue.memcpy(d_points, h_points, sycl::double3::byte_size()*work_size);
  thread_queue.memset(d_cog, 0, sycl::double3::byte_size());
  thread_queue.memset(d_count, 0, sizeof(unsigned int));
  // thread_queue.wait();
  sum_cog(d_points, d_cog, data_in.h_thread_cogs, d_count, tid, work_size, num_threads, &thread_queue);
#if 0
  {
#define ATOMIC_FETCH_ADD(T, data, val)  sycl::atomic_ref< T, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(data).fetch_add(val);
    const int block_size = 128;
    const int grid = (work_size + block_size - 1) / block_size;
    auto* peer_cogs = data_in.h_thread_cogs;
    thread_queue.submit([grid, work_size, &d_points, &d_cog, &d_count, &peer_cogs, tid](sycl::handler &cgh) {
      sycl::local_accessor<bool, 0> isLastBlockDone_acc_ct1(cgh);
      isLastBlockDone_acc_ct1 = false;
      // sycl::stream out(1024, 256, cgh); //output buffer
      cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(grid * block_size),
                          sycl::range<1>(block_size)),
        [isLastBlockDone_acc_ct1, work_size, d_points, d_cog, d_count, peer_cogs, tid/*, out*/](sycl::nd_item<1> item_ct1) {
          const int i = item_ct1.get_local_id(0) +
            item_ct1.get_group(0) * item_ct1.get_local_range(0);
          const int totaltb = item_ct1.get_group_range(0);
          sycl::double3 p{0, 0, 0};
          if (item_ct1.get_local_id(0) == 0) {
            isLastBlockDone_acc_ct1 = false;
          }
          item_ct1.barrier();
          if (i < work_size) {
            p.x() = d_points[i].x();
            p.y() = d_points[i].y();
            p.z() = d_points[i].z();
          }
          p.x() = sycl::reduce_over_group(item_ct1.get_group(), p.x(), sycl::plus<>());
          item_ct1.barrier();
          p.y() = sycl::reduce_over_group(item_ct1.get_group(), p.y(), sycl::plus<>());
          item_ct1.barrier();
          p.z() = sycl::reduce_over_group(item_ct1.get_group(), p.z(), sycl::plus<>());
          item_ct1.barrier();
          if (item_ct1.get_local_id(0) == 0) {
            ATOMIC_FETCH_ADD(double, d_cog->x(), p.x());
            ATOMIC_FETCH_ADD(double, d_cog->y(), p.y());
            ATOMIC_FETCH_ADD(double, d_cog->z(), p.z());
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
            // unsigned int value = ATOMIC_FETCH_ADD(unsigned int, *d_count, totaltb);
            // auto atomic_count = sycl::atomic_ref<unsigned int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space>(*d_count);
            // const unsigned int old_value = atomic_count.load(sycl::memory_order::acquire, sycl::memory_scope::device);
            // ATOMIC_FETCH_ADD(unsigned int, *d_count, (old_value >= totaltb) ? 0 : (old_value + 1));
            // atomic_count.store((old_value >= totaltb) ? 0 : (old_value + 1), sycl::memory_order::release, sycl::memory_scope::device);
            // sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
            // const unsigned int value = atomic_count.load();
            // out << "value = " << old_value << ", totaltb = " << totaltb << ", blockIdx = " << item_ct1.get_group_linear_id() << ", local_range = " << item_ct1.get_local_range(0) << "\n";
            unsigned int value = dpct::atomic_fetch_compare_inc<
              sycl::access::address_space::generic_space>(
              d_count, item_ct1.get_group_range(0));
            isLastBlockDone_acc_ct1 = (value == totaltb - 1);
            // sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
          }
          item_ct1.barrier();
          // if (item_ct1.get_local_id(0) == 0) {
          //   // TODO
          //   unsigned int atomic_count = sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*d_count).load();
          //   isLastBlockDone_acc_ct1 = (atomic_count == totaltb);
          // }
          if (isLastBlockDone_acc_ct1) {
            // out << isLastBlockDone_acc_ct1 << "\n";
            if (item_ct1.get_local_id(0) == 0) {
              peer_cogs[tid] = *d_cog;
              (*d_count) = 0;
              sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
            }
          }
          item_ct1.barrier();
        });
    });
  }
#endif
  if (num_threads > 1) {
    thread_queue.wait();
    sync_host_threads->wait();
  }
  // if (true /*backend == sycl::backend::ext_oneapi_cuda*/) {
  //   // TODO: I don't know why
  //   thread_queue.wait();
  // }
  thread_queue.memset(d_cog, 0, sycl::double3::byte_size());
  sum_cog_from_devices(data_in.h_thread_cogs, d_cog, tid, num_threads, &thread_queue);
  // if (true /*backend == sycl::backend::ext_oneapi_cuda*/) {
  //   // TODO: I don't know why
  //   thread_queue.wait();
  // }
  thread_queue.memcpy(h_cog, d_cog, sycl::double3::byte_size());
  thread_queue.wait();
  h_cog->x() /= data_in.points.size();
  h_cog->y() /= data_in.points.size();
  h_cog->z() /= data_in.points.size();
  const std::string result = "Thread " + std::to_string(tid) + ": cog = (" + std::to_string(h_cog->x()) + ", " + std::to_string(h_cog->y()) + ", " + std::to_string(h_cog->z()) + ")\n";
  print_with_lock(result);
  sycl::free(h_cog, thread_queue);
  sycl::free(d_points, thread_queue);
  sycl::free(d_cog, thread_queue);
  sycl::free(d_count, thread_queue);
  thread_queue.wait();
  sync_host_threads->wait();
}

int main(int argc, char* argv[]) {
  if (argc != 3) return 1;
  try {
    const int num_threads = std::stoi(argv[1]);
    const int num_points = std::stoi(argv[2]);
    if (num_threads < 1) throw;
    int num_devices = 0;
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    num_devices = devices.size();
    if (num_devices < num_threads) {
      print_with_lock("The number of available GPUs is less than the number of threads!\n");
      return 1;
    } else {
      num_devices = num_threads;
    }
    // TODO: How can I setup Intel GPU memory peer access?
    const auto& master_device = devices[0];
    sycl::queue master_queue(master_device);
    my_data data(num_points, master_queue);
    data.set_host_com_buffers(num_devices);
    std::vector<std::thread> threads;
    sync_host_threads = std::make_unique<barrier>(num_threads);
    print_with_lock("Launch from the main thread\n");
    master_queue.wait();
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(call_from_thread, i, num_threads, std::ref(data), devices[i]);
    }
    for (int i = 0; i < num_threads; ++i) {
      threads[i].join();
    }
  } catch (...) {
    std::cerr << "Failed to get the number of threads.\n";
    return 1;
  }
  return 0;
}
