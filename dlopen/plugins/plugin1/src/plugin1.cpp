#include "plugin1.h"
#include <cmath>
#include <iostream>

#if defined (__linux__) || defined (__APPLE__)
extern "C" {
  Compute1* allocator() {
    return new Compute1();
  }
  void deleter(Compute1* ptr) {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
}
#endif

#ifdef WIN32
extern "C" {
  __declspec (dllexport) Compute1* allocator() {
    return new Compute1();
  }
  __declspec (dllexport) void deleter(Compute1* ptr) {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
}
#endif

float Compute1::compute(float x) {
  return std::sin(2.0f * x);
}

Compute1::~Compute1() {
  std::cout << "Compute1::~Compute1\n";
}
