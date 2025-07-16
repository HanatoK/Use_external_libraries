#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <string>
#include <random>
#include <condition_variable>
#include <memory>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <sstream>
#include <format>
#include <unordered_map>
#include "common.h"
// #include "mgpu_kernel.cuh"
#include "nvrtc_helper.h"
#include "common.h"

std::mutex print_mtx;

// #define checkCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//   if (code != cudaSuccess) {
//     // print_mtx.lock();
//     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//     // print_mtx.unlock();
//     if (abort) exit(code);
//   }
// }

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
  my_data(int num_points) {
    std::mt19937 gen(0);
    std::normal_distribution<> dis(1.0, 6.0);
    points.resize(num_points);
    ref_cog.x = 0;
    ref_cog.y = 0;
    ref_cog.z = 0;
    for (int i = 0; i < num_points; ++i) {
      points[i].x = dis(gen);
      points[i].y = dis(gen);
      points[i].z = dis(gen);
    }
    for (int i = 0; i < num_points; ++i) {
      ref_cog.x += points[i].x;
      ref_cog.y += points[i].y;
      ref_cog.z += points[i].z;
    }
    ref_cog.x /= num_points;
    ref_cog.y /= num_points;
    ref_cog.z /= num_points;
    h_thread_cogs = nullptr;
    std::cout << "Ref cog: (" << std::to_string(ref_cog.x) << ", " << std::to_string(ref_cog.y) << ", " << std::to_string(ref_cog.z) << ")" << std::endl;
  }
  ~my_data() {
    if (h_thread_cogs) checkCudaError(cudaFree(h_thread_cogs));
  }
  void set_host_com_buffers(int num_threads) {
    if (h_thread_cogs) checkCudaError(cudaFree(h_thread_cogs));
    checkCudaError(cudaMalloc(&h_thread_cogs, sizeof(double3)*num_threads));
  }
  std::vector<double3> points;
  double3 ref_cog;
  double3* h_thread_cogs;
};

void print_with_lock(const std::string& str, const bool flush = true) {
  print_mtx.lock();
  std::cout << str;
  if (flush) std::cout << std::flush;
  print_mtx.unlock();
}

template <class T>
std::string to_string(T const& val, ...)
{
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

void call_from_thread(
  const int tid, int num_threads, my_data& data_in,
  cudaLibrary_t library, const std::unordered_map<std::string, std::string>& kernel_name_map) {
  const int my_device_id = tid;
  const int num_devices = num_threads;
  checkCudaError(cudaSetDevice(my_device_id));
  cudaStream_t stream;
  checkCudaError(cudaStreamCreate(&stream));
  // Print kernel counts
  unsigned int kernel_counts = 0;
  checkCudaError(cudaLibraryGetKernelCount(&kernel_counts, library));
  print_with_lock(std::format("Number of kernels: {}\n", kernel_counts));
  std::vector<cudaKernel_t> tmp_kernels(kernel_counts);
  checkCudaError(cudaLibraryEnumerateKernels(tmp_kernels.data(), kernel_counts, library));
  // Load kernels from the library
  cudaKernel_t sum_cog_kernel, sum_cog_from_devices_kernel;
  checkCudaError(
    cudaLibraryGetKernel(&sum_cog_kernel, library, kernel_name_map.at("&sum_cog_kernel").c_str()));
  checkCudaError(
    cudaLibraryGetKernel(&sum_cog_from_devices_kernel, library, kernel_name_map.at("&sum_cog_devices_kernel").c_str()));
  int work_size = data_in.points.size() / num_threads;
  const int my_work_start = tid * work_size;
  if (tid == num_threads - 1) {
    work_size = data_in.points.size() - (num_threads - 1) * work_size;
  }
  double3* d_points;
  double3* d_cog;
  double3* h_cog;
  unsigned int* d_count;
  checkCudaError(cudaHostAlloc(&h_cog, sizeof(double3), cudaHostAllocPortable));
  checkCudaError(cudaMallocAsync(&d_points, sizeof(double3) * work_size, stream));
  checkCudaError(cudaMallocAsync(&d_cog, sizeof(double3), stream));
  checkCudaError(cudaMallocAsync(&d_count, sizeof(unsigned int), stream));
  checkCudaError(cudaMemsetAsync(d_cog, 0, sizeof(double3), stream));
  checkCudaError(cudaMemsetAsync(d_count, 0, sizeof(unsigned int), stream));
  double3* h_points = data_in.points.data() + my_work_start;
  checkCudaError(cudaMemcpyAsync(d_points, h_points, sizeof(double3) *work_size, cudaMemcpyHostToDevice, stream));
  // sum_cog(d_points, d_cog, data_in.h_thread_cogs, d_count,
  //         my_device_id, work_size, num_devices, stream);
  {
    const int grid = (work_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const void* args[] = {&d_points, &d_cog, &(data_in.h_thread_cogs), &d_count, &tid, &work_size, &num_threads};
    checkCudaError(cudaLaunchKernel(sum_cog_kernel, grid, BLOCK_SIZE, const_cast<void**>(args), 0, stream));
  }
  checkCudaError(cudaPeekAtLastError());
  if (num_threads > 1) {
    // checkCudaError(cudaStreamSynchronize(stream));
    checkCudaError(cudaDeviceSynchronize());
    sync_host_threads->wait();
  }
  checkCudaError(cudaMemsetAsync(d_cog, 0, sizeof(double3), stream));
  // sum_cog_from_devices(data_in.h_thread_cogs, d_cog, my_device_id, num_devices, stream);
  {
    const int grid = (num_devices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const void *args[] = {&(data_in.h_thread_cogs), &d_cog, &tid, &num_threads};
    checkCudaError(cudaLaunchKernel(sum_cog_from_devices_kernel, grid, BLOCK_SIZE, const_cast<void**>(args), 0, stream));
  }
  checkCudaError(cudaMemcpyAsync(h_cog, d_cog, sizeof(double3), cudaMemcpyDeviceToHost, stream));
  checkCudaError(cudaStreamSynchronize(stream));
  h_cog->x /= data_in.points.size();
  h_cog->y /= data_in.points.size();
  h_cog->z /= data_in.points.size();
  const std::string result = "Thread " + std::to_string(tid) + ": cog = (" + std::to_string(h_cog->x) + ", " + std::to_string(h_cog->y) + ", " + std::to_string(h_cog->z) + ")\n";
  print_with_lock(result);
  checkCudaError(cudaFreeHost(h_cog));
  checkCudaError(cudaFree(d_points));
  checkCudaError(cudaFree(d_cog));
  checkCudaError(cudaFree(d_count));
  checkCudaError(cudaStreamDestroy(stream));
  sync_host_threads->wait();
}

int main(int argc, char* argv[]) {
  if (argc != 3) return 1;
  try {
    const int num_threads = std::stoi(argv[1]);
    const int num_points = std::stoi(argv[2]);
    if (num_threads < 1) throw;
    int num_devices = 0;
    checkCudaError(cudaGetDeviceCount(&num_devices));
    if (num_devices < num_threads) {
      print_with_lock("The number of available GPUs is less than the number of threads!\n");
      return 1;
    }
    num_devices = num_threads;
    const int master_device = 0;
    for (int i = 0; i < num_devices; ++i) {
      checkCudaError(cudaSetDevice(i));
      for (int j = 0; j < num_devices; ++j) {
        if (i != j)
          checkCudaError(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
    checkCudaError(cudaSetDevice(master_device));
    my_data data(num_points);
    data.set_host_com_buffers(num_devices);
    std::vector<std::thread> threads;
    sync_host_threads = std::make_unique<barrier>(num_threads);
    print_with_lock("Compile CUDA kernels\n");
    char* cubinResult;
    size_t cubinResultSize;
    const std::string kernel_filename{"../mgpu_kernel.cu"};
    std::vector<std::string> kernel_name_vec, kernel_mangled_name_vec;
    kernel_name_vec.push_back("&sum_cog_kernel");
    kernel_name_vec.push_back("&sum_cog_devices_kernel");
    compileFileToCUBIN(kernel_filename.c_str(),
                       &cubinResult,
                       &cubinResultSize, 0,
                       kernel_name_vec, kernel_mangled_name_vec);
    std::unordered_map<std::string, std::string> kernel_name_map;
    for (size_t i = 0; i < kernel_name_vec.size(); ++i) {
      kernel_name_map[kernel_name_vec[i]] = kernel_mangled_name_vec[i];
    }
    cudaLibrary_t library = loadCUBIN(cubinResult);
    print_with_lock("Launch from the main thread\n");
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(call_from_thread, i, num_threads, std::ref(data), library, std::ref(kernel_name_map));
    }
    for (int i = 0; i < num_threads; ++i) {
      threads[i].join();
    }
    checkCudaError(cudaLibraryUnload(library));
  } catch (...) {
    std::cerr << "Failed to get the number of threads.\n";
    return 1;
  }
  return 0;
}
