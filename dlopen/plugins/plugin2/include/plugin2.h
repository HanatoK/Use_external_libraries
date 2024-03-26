#ifndef PLUGIN1_H
#define PLUGIN1_H

#include "main.h"

class Compute2: public Compute {
public:
  Compute2() = default;
  float compute(float x) override;
  virtual ~Compute2();
};

#endif // PLUGIN1_H
