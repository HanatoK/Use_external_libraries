#include <iostream>
#include <boost/histogram.hpp>
#include <boost/format.hpp>
#include <vector>
#include <string>

int main() {
  auto axis_x = boost::histogram::axis::circular<double>(72, -180.0, 180.0, "psi");
  auto axis_y = boost::histogram::axis::circular<double>(72, -180.0, 180.0, "phi");
  std::vector<decltype(axis_x)> axes{axis_x, axis_y};
  auto hist = boost::histogram::make_histogram(axes);
  hist(61.0, 178.0, boost::histogram::weight(0.6));
  for (const auto& x : boost::histogram::indexed(hist, boost::histogram::coverage::all)) {
    std::cout << boost::format("%12.5f %12.5f %12.7f\n")
      % x.bin(1).center()
      % x.bin(0).center() % *x;
  }
  return 0;
}
