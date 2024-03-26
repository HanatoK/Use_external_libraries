#ifndef MAIN_H
#define MAIN_H

class Compute {
public:
  virtual ~Compute() = default;
  virtual float compute(float x) = 0;
};

#endif // MAIN_H
