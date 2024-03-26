#ifndef DLLOADER_POSIX_H
#define DLLOADER_POSIX_H

#include "dlloader.h"
#include <dlfcn.h>
#include <string>

// https://github.com/theo-pnv/Dynamic-Loading

namespace dlloader {
  template <typename T>
  class DLLoader: public DLLoaderInterface<T> {
  private:
    void* mHandle;
    std::string mPathToLib;
    std::string mAllocClassSymbol;
    std::string mDeleteClassSymbol;
  public:
    DLLoader(
      std::string pathToLib,
      std::string allocClassSymbol = "allocator",
      std::string deleteClassSymbol = "deleter"):
      mPathToLib(std::move(pathToLib)),
      mAllocClassSymbol(std::move(allocClassSymbol)),
      mDeleteClassSymbol(std::move(deleteClassSymbol)) {}
    ~DLLoader() = default;
    void DLOpenLib() override {
      if (!(mHandle = dlopen(mPathToLib.c_str(), RTLD_NOW|RTLD_LAZY))) {
        throw std::runtime_error(std::string{dlerror()});
      }
    }
    void DLCloseLib() override {
      if (dlclose(mHandle) != 0) {
        throw std::runtime_error(std::string{dlerror()});
      }
    }
    std::shared_ptr<T> DLGetInstance() override {
      using allocClass = T *(*)();
      using deleteClass = void (*)(T *);
      auto allocFunc = reinterpret_cast<allocClass>(dlsym(mHandle, mAllocClassSymbol.c_str()));
      auto deleteFunc = reinterpret_cast<deleteClass>(dlsym(mHandle, mDeleteClassSymbol.c_str()));
      if (!allocFunc || !deleteFunc) {
        try {
          DLCloseLib();
          throw std::runtime_error(std::string{dlerror()});
        } catch (...) {
          throw;
        }
        throw std::runtime_error(std::string{dlerror()});
      }
      return std::shared_ptr<T>(allocFunc(), [deleteFunc](T* p){deleteFunc(p);});
    }
  };
}

#endif // DLLOADER_POSIX_H
