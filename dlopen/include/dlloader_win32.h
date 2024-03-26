#ifndef DLLOADER_WIN32_H
#define DLLOADER_WIN32_H

#include "dlloader.h"
#include "windows.h"
#include <string>

// https://github.com/theo-pnv/Dynamic-Loading

namespace dlloader {
  template <typename T>
  class DLLoader: public DLLoaderInterface<T> {
  private:
    HMODULE mHandle;
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
      if (!(mHandle = LoadLibrary(mPathToLib.c_str()))) {
        throw std::runtime_error(std::string{"Cannot open and load "} + mPathToLib);
      }
    }
    void DLCloseLib() override {
      if (FreeLibrary(mHandle) == 0) {
        throw std::runtime_error(std::string{"Cannot close "} + mPathToLib);
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
        } catch (...) {
          throw;
        }
        throw std::runtime_error(std::string{"Cannot find allocator or deleter symbol in "} + mPathToLib);
      }
      return std::shared_ptr<T>(allocFunc(), [deleteFunc](T* p){deleteFunc(p);});
    }
  };
}

#endif
