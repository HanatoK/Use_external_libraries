#include "nvrtc_helper.h"
#include "common.h"
#include <fstream>
#include <iostream>
// #include <sstream>
#include <string>
#include <vector>

void compileFileToCUBIN(
  const char *filename, /*int argc, char **argv,*/ char **cubinResult,
  size_t *cubinResultSize, int requiresCGheaders,
  const std::vector<std::string>& kernel_name_vec,
  std::vector<std::string>& kernel_lowered_name_vec) {
  if (!filename) {
    std::cerr << "\nerror: filename is empty for compileFileToCUBIN()!\n";
    exit(1);
  }

  std::ifstream inputFile(filename,
                          std::ios::in | std::ios::binary | std::ios::ate);

  if (!inputFile.is_open()) {
    std::cerr << "\nerror: unable to open " << filename << " for reading!\n";
    exit(1);
  }

  std::streampos pos = inputFile.tellg();
  size_t inputSize = (size_t)pos;
  char *memBlock = new char[inputSize + 1];

  inputFile.seekg(0, std::ios::beg);
  inputFile.read(memBlock, inputSize);
  inputFile.close();
  memBlock[inputSize] = '\x0';

  int numCompileOptions = 0;

  char *compileParams[7];

  int major = 0, minor = 0;
  // char deviceName[256];

  // Picks the best CUDA device available
  // CUdevice cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

  // get compute capabilities and the devicename
  // checkCudaErrors(cuDeviceGetAttribute(
  //     &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  // checkCudaErrors(cuDeviceGetAttribute(
  //     &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  int currentDevice;
  checkCudaError(cudaGetDevice(&currentDevice));
  // cudaDeviceProp prop;
  // checkCudaError(cudaGetDeviceProperties(&prop, currentDevice));
  checkCudaError(cudaDeviceGetAttribute(
    &major, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, currentDevice));
  checkCudaError(
    cudaDeviceGetAttribute(&minor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, currentDevice));

  {
    // Compile cubin for the GPU arch on which are going to run cuda kernel.
    std::string compileOptions;
    compileOptions = "--gpu-architecture=sm_";

    compileParams[numCompileOptions] = reinterpret_cast<char *>(
                    malloc(sizeof(char) * (compileOptions.length() + 10)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 10),
              "%s%d%d", compileOptions.c_str(), major, minor);
#else
    snprintf(compileParams[numCompileOptions], compileOptions.size() + 10, "%s%d%d",
            compileOptions.c_str(), major, minor);
#endif
  }

  numCompileOptions++;

  {
    std::string compileOptions = "--include-path=../";
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
          malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  {
    std::string compileOptions = "--include-path=/usr/local/cuda/include";
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
          malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  {
    std::string compileOptions = "--include-path=/usr/local/cuda/include/cuda/std";
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
          malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

//   {
//     std::string compileOptions = "--include-path=/usr/include/c++/14";
//     compileParams[numCompileOptions] = reinterpret_cast<char *>(
//           malloc(sizeof(char) * (compileOptions.length() + 1)));
// #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//     sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
//               "%s", compileOptions.c_str());
// #else
//     snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
//              compileOptions.c_str());
// #endif
//     numCompileOptions++;
//   }

//   {
//     std::string compileOptions = "--include-path=/usr/lib64/gcc/x86_64-suse-linux/14/include";
//     compileParams[numCompileOptions] = reinterpret_cast<char *>(
//           malloc(sizeof(char) * (compileOptions.length() + 1)));
// #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//     sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
//               "%s", compileOptions.c_str());
// #else
//     snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
//              compileOptions.c_str());
// #endif
//     numCompileOptions++;
//   }

  {
    std::string compileOptions = "-default-device";
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
          malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.length() + 1, "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  if (requiresCGheaders) {
    std::string compileOptions;
    char HeaderNames[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#else
    snprintf(HeaderNames, sizeof(HeaderNames), "%s", "cooperative_groups.h");
#endif

    compileOptions = "--include-path=";

    // char *strPath = sdkFindFilePath(HeaderNames, argv[0]);
    char *strPath = nullptr;
    if (!strPath) {
      std::cerr << "\nerror: header file " << HeaderNames << " not found!\n";
      exit(1);
    }
    std::string path = strPath;
    if (!path.empty()) {
      std::size_t found = path.find(HeaderNames);
      path.erase(found);
    } else {
      // printf(
      //     "\nCooperativeGroups headers not found, please install it in %s "
      //     "sample directory..\n Exiting..\n",
      //     argv[0]);
      exit(1);
    }
    compileOptions += path.c_str();
    compileParams[numCompileOptions] = reinterpret_cast<char *>(
        malloc(sizeof(char) * (compileOptions.length() + 1)));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(compileParams[numCompileOptions], sizeof(char) * (compileOptions.length() + 1),
              "%s", compileOptions.c_str());
#else
    snprintf(compileParams[numCompileOptions], compileOptions.size(), "%s",
             compileOptions.c_str());
#endif
    numCompileOptions++;
  }

  // compile
  nvrtcProgram prog;
  NVRTC_SAFE_CALL("nvrtcCreateProgram",
                  nvrtcCreateProgram(&prog, memBlock, filename, 0, NULL, NULL));
  // std::vector<std::string> kernel_name_vec;
  // kernel_name_vec.push_back("&sum_cog_kernel");
  // kernel_name_vec.push_back("&sum_cog_devices_kernel");
  for (size_t i = 0; i < kernel_name_vec.size(); ++i) {
    NVRTC_SAFE_CALL("nvrtcAddNameExpression",
                    nvrtcAddNameExpression(prog, kernel_name_vec[i].c_str()));
  }

  for (int i = 0; i < numCompileOptions; ++i) {
    std::cout << compileParams[i] << std::endl;
  }
  nvrtcResult res = nvrtcCompileProgram(prog, numCompileOptions, compileParams);
  kernel_lowered_name_vec.resize(kernel_name_vec.size());
  for (size_t i = 0; i < kernel_lowered_name_vec.size(); ++i) {
    const char* name;
    NVRTC_SAFE_CALL("nvrtcGetLoweredName", nvrtcGetLoweredName(prog, kernel_name_vec[i].c_str(), &name));
    kernel_lowered_name_vec[i] = std::string(name);
    std::cout << "kernel_lowered_name_vec[i] = " << kernel_lowered_name_vec[i] << std::endl;
  }

  // dump log
  size_t logSize;
  NVRTC_SAFE_CALL("nvrtcGetProgramLogSize",
                  nvrtcGetProgramLogSize(prog, &logSize));
  char *log = reinterpret_cast<char *>(malloc(sizeof(char) * logSize + 1));
  NVRTC_SAFE_CALL("nvrtcGetProgramLog", nvrtcGetProgramLog(prog, log));
  log[logSize] = '\x0';

  if (std::strlen(log) >= 2) {
    std::cerr << "\n compilation log ---\n";
    std::cerr << log;
    std::cerr << "\n end log ---\n";
  }

  free(log);

  NVRTC_SAFE_CALL("nvrtcCompileProgram", res);

  size_t codeSize;
  NVRTC_SAFE_CALL("nvrtcGetCUBINSize", nvrtcGetCUBINSize(prog, &codeSize));
  char *code = new char[codeSize];
  NVRTC_SAFE_CALL("nvrtcGetCUBIN", nvrtcGetCUBIN(prog, code));
  *cubinResult = code;
  *cubinResultSize = codeSize;

  for (int i = 0; i < numCompileOptions; i++) {
    free(compileParams[i]);
  }
  // return prog;
}

cudaLibrary_t loadCUBIN(const char* cubin) {
  int currentDevice;
  checkCudaError(cudaGetDevice(&currentDevice));
  int major = 0, minor = 0;
  checkCudaError(cudaDeviceGetAttribute(
    &major, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, currentDevice));
  checkCudaError(
    cudaDeviceGetAttribute(&minor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, currentDevice));
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);
  cudaLibrary_t library;
  checkCudaError(cudaLibraryLoadData(
    &library, cubin, nullptr, nullptr, 0, nullptr, nullptr, 0));
  return library;
}
