/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_NVRTC_HELPER_H_

#define COMMON_NVRTC_HELPER_H_ 1

#include <cuda_runtime.h>
// #include <helper_cuda_drvapi.h>
#include <nvrtc.h>
#include <cstring>
#include <vector>
#include <string>


#define NVRTC_SAFE_CALL(Name, x)                                \
  do {                                                          \
    nvrtcResult result = x;                                     \
    if (result != NVRTC_SUCCESS) {                              \
      std::cerr << "\nerror: " << Name << " failed with error " \
                << nvrtcGetErrorString(result);                 \
      exit(1);                                                  \
    }                                                           \
  } while (0)

void compileFileToCUBIN(
  const char *filename, /*int argc, char **argv,*/ char **cubinResult,
  size_t *cubinResultSize, int requiresCGheaders,
  const std::vector<std::string>& kernel_name_vec,
  std::vector<std::string>& kernel_lowered_name_vec);

// CUmodule loadCUBIN(char *cubin, int argc, char **argv) {
//   CUmodule module;
//   CUcontext context;
//   int major = 0, minor = 0;
//   char deviceName[256];
//
//   // Picks the best CUDA device available
//   CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);
//
//   // get compute capabilities and the devicename
//   checkCudaError(cuDeviceGetAttribute(
//       &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
//   checkCudaError(cuDeviceGetAttribute(
//       &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
//   checkCudaError(cuDeviceGetName(deviceName, 256, cuDevice));
//   printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
//
//   checkCudaError(cuInit(0));
//   checkCudaError(cuCtxCreate(&context, 0, cuDevice));
//
//   checkCudaError(cuModuleLoadData(&module, cubin));
//   free(cubin);
//
//   return module;
// }

cudaLibrary_t loadCUBIN(const char* cubin);

#endif  // COMMON_NVRTC_HELPER_H_
