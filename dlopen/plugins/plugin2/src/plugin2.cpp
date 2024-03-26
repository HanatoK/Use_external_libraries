#include "plugin2.h"
#include <cmath>
#include <iostream>

#if defined (__linux__) || defined (__APPLE__)
extern "C" {
  Compute2* allocator() {
    return new Compute2();
  }
  void deleter(Compute2* ptr) {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
}
#endif

#ifdef WIN32
extern "C" {
  __declspec (dllexport) Compute2* allocator() {
    return new Compute2();
  }
  __declspec (dllexport) void deleter(Compute2* ptr) {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
}
#endif

float Compute2::compute(float x) {
  return std::cos(2.0f * x);
}

Compute2::~Compute2() {
  std::cout << "Compute2::~Compute2\n";
}
