#ifndef PLUGIN1_H
#define PLUGIN1_H

#include "main.h"

class Compute1: public Compute {
public:
  Compute1() = default;
  float compute(float x) override;
  virtual ~Compute1();
};

#endif // PLUGIN1_H
