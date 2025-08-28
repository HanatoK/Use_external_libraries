#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <array>
#include <cstring>
#include <fmt/format.h>

#include "main.h"

std::array<double, 4> project1(const quaternion& q, const std::vector<rvector>& pos, const std::vector<rvector>& f) {
  std::array<double, 4> qf = {0, 0, 0, 0};
  for (size_t i = 0; i < pos.size(); ++i) {
    const quaternion dq = q.position_derivative_inner(pos[i], f[i]);
    qf[0] += dq.q0;
    qf[1] += dq.q1;
    qf[2] += dq.q2;
    qf[3] += dq.q3;
  }
  return qf;
}

auto project2(const quaternion& q, const std::vector<rvector>& pos, const std::vector<rvector>& f) {
  double C[3][3] = {{0}};
  for (size_t i = 0; i < pos.size(); ++i) {
    C[0][0] += f[i].x * pos[i].x;
    C[0][1] += f[i].x * pos[i].y;
    C[0][2] += f[i].x * pos[i].z;
    C[1][0] += f[i].y * pos[i].x;
    C[1][1] += f[i].y * pos[i].y;
    C[1][2] += f[i].y * pos[i].z;
    C[2][0] += f[i].z * pos[i].x;
    C[2][1] += f[i].z * pos[i].y;
    C[2][2] += f[i].z * pos[i].z;
  }
  return q.derivative_element_wise_product_sum(C);
}

void run_test_derivatie_cpu() {
  quaternion q;
  q.set_from_euler_angles(0.7653981633974483, 0.3, 1.2);
  const size_t N = 100000;
  const int M = 20;
  std::cout << fmt::format("Running CPU test ({} points, {} iterations):\n", N, M);
  std::cout << fmt::format("q = ({}, {}, {}, {})\n", q.q0, q.q1, q.q2, q.q3);
  std::vector<rvector> points(N), forces(N);
  // std::random_device rd;
  double project1_time = 0;
  double project2_time = 0;
  double max_error = 0;
  for (size_t j = 0; j < M; ++j) {
    std::mt19937 gen(123+j);
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    for (size_t i = 0; i < points.size(); ++i) {
      points[i].x = dis(gen);
      points[i].y = dis(gen);
      points[i].z = dis(gen);
      forces[i].x = dis(gen);
      forces[i].y = dis(gen);
      forces[i].z = dis(gen);
    }
    auto start = std::chrono::high_resolution_clock::now();
    const auto dq1 = project1(q, points, forces);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> diff1 = end - start;
    // std::cout << fmt::format("Iteration {}: project1: {} {} {} {}\n", j, dq1[0], dq1[1], dq1[2], dq1[3]);
    project1_time += diff1.count();

    start = std::chrono::high_resolution_clock::now();
    const auto dq2 = project2(q, points, forces);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> diff2 = end - start;
    // std::cout << fmt::format("Iteration {}: Project2: {} {} {} {}\n", j, dq2[0], dq2[1], dq2[2], dq2[3]);
    project2_time += diff2.count();

    const double error = std::sqrt(
      (dq1[0] - dq2[0]) * (dq1[0] - dq2[0]) +
      (dq1[1] - dq2[1]) * (dq1[1] - dq2[1]) +
      (dq1[2] - dq2[2]) * (dq1[2] - dq2[2]) +
      (dq1[3] - dq2[3]) * (dq1[3] - dq2[3]));
    if (error > max_error) {
      max_error = error;
    }
  }
  std::cout << "Project1 average running time: " << project1_time / M << std::endl;
  std::cout << "Project2 average running time: " << project2_time / M << std::endl;
  std::cout << "Max errror = " << max_error << std::endl;
}

void run_test_derivatie_cuda() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  quaternion q;
  q.set_from_euler_angles(0.7653981633974483, 0.3, 1.2);
  const size_t N = 100000;
  const int M = 20;
  std::cout << fmt::format("Running GPU test ({} points, {} iterations):\n", N, M);
  std::cout << fmt::format("q = ({}, {}, {}, {})\n", q.q0, q.q1, q.q2, q.q3);
  std::vector<rvector> points(N), forces(N);
  double project1_time = 0;
  double project2_time = 0;
  rvector* d_points;
  rvector* d_forces;
  quaternion* d_q;
  double4* d_project1_out;
  double4* d_project2_out;
  double4* h_project1_out;
  double4* h_project2_out;
  cudaMalloc(&d_q, sizeof(quaternion));
  cudaMemcpy(d_q, &q, 1*sizeof(quaternion), cudaMemcpyHostToDevice);
  cudaMalloc(&d_points, sizeof(rvector) * N);
  cudaMalloc(&d_forces, sizeof(rvector) * N);
  cudaMalloc(&d_project1_out, sizeof(double4));
  cudaMalloc(&d_project2_out, sizeof(double4));
  cudaMallocHost(&h_project1_out, sizeof(double4));
  cudaMallocHost(&h_project2_out, sizeof(double4));
  double max_error = 0;
  for (size_t j = 0; j < M; ++j) {
    cudaMemset(d_project1_out, 0, sizeof(double4));
    cudaMemset(d_project2_out, 0, sizeof(double4));
    std::mt19937 gen(123+j);
    std::uniform_real_distribution<> dis(-100.0, 100.0);
    for (size_t i = 0; i < points.size(); ++i) {
      points[i].x = dis(gen);
      points[i].y = dis(gen);
      points[i].z = dis(gen);
      forces[i].x = dis(gen);
      forces[i].y = dis(gen);
      forces[i].z = dis(gen);
    }
    cudaMemcpy(d_points, points.data(), sizeof(rvector)*points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, forces.data(), sizeof(rvector)*points.size(), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    project1_cuda(d_points, d_forces, d_q, d_project1_out, points.size(), stream);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_project1_out, d_project1_out, sizeof(double4), cudaMemcpyDeviceToHost);
    std::chrono::duration<double, std::micro> diff1 = end - start;
    // std::cout << fmt::format("Iteration {}: project1: {} {} {} {}\n", j, h_project1_out->w, h_project1_out->x, h_project1_out->y, h_project1_out->z);
    project1_time += diff1.count();

    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();
    project2_cuda(d_points, d_forces, d_q, d_project2_out, points.size(), stream);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_project2_out, d_project2_out, sizeof(double4), cudaMemcpyDeviceToHost);
    std::chrono::duration<double, std::micro> diff2 = end - start;
    // std::cout << fmt::format("Iteration {}: project2: {} {} {} {}\n", j, h_project2_out->w, h_project2_out->x, h_project2_out->y, h_project2_out->z);
    project2_time += diff2.count();

    const double error = std::sqrt(
      (h_project2_out->w - h_project1_out->w) * (h_project2_out->w - h_project1_out->w) +
      (h_project2_out->x - h_project1_out->x) * (h_project2_out->x - h_project1_out->x) +
      (h_project2_out->y - h_project1_out->y) * (h_project2_out->y - h_project1_out->y) +
      (h_project2_out->z - h_project1_out->z) * (h_project2_out->z - h_project1_out->z));
    if (error > max_error) {
      max_error = error;
    }
  }
  cudaFree(d_q);
  cudaFree(d_forces);
  cudaFree(d_points);
  cudaFree(d_project1_out);
  cudaFree(d_project2_out);
  cudaFreeHost(h_project1_out);
  cudaFreeHost(h_project2_out);
  std::cout << "Project1 average running time: " << project1_time / M << std::endl;
  std::cout << "Project2 average running time: " << project2_time / M << std::endl;
  std::cout << "Max errror = " << max_error << std::endl;
}

int main() {
  run_test_derivatie_cpu();
  run_test_derivatie_cuda();
  return 0;
}
