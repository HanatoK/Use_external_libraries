#include "main.h"
#include <iostream>

#if defined (__linux__) || defined (__APPLE__)
#include "dlloader_posix.h"
#elif defined WIN32
#include "dlloader_win32.h"
#endif

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Please specify a library to load.\n";
    return 1;
  }
  std::string pluginPath{argv[1]};
  dlloader::DLLoader<Compute> loader(pluginPath);
  try {
    loader.DLOpenLib();
    {
      auto computeObj = loader.DLGetInstance();
      std::cout << computeObj->compute(0.6) << std::endl;
    }
    loader.DLCloseLib();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
