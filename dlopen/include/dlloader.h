#ifndef DLLOADER_H
#define DLLOADER_H

#include <memory>

namespace dlloader {
  template <typename T>
  class DLLoaderInterface {
  public:
    virtual ~DLLoaderInterface() = default;
    virtual void DLOpenLib() = 0;
    virtual std::shared_ptr<T> DLGetInstance() = 0;
    virtual void DLCloseLib() = 0;
  };
}

#endif // DLLOADER_H
