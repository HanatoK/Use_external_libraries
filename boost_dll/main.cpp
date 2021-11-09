#include <vector>
#include <tuple>
#include <iostream>
#include <boost/dll.hpp>
#include <boost/function.hpp>

int main(int argc, char* argv[]) {
  if (argc > 1) {
    boost::dll::shared_library lib(argv[1]);
    boost::function<std::tuple<double, double, double>(const std::vector<std::vector<double>>&)>
      fit_plane = lib.get<std::tuple<double, double, double>(const std::vector<std::vector<double>>&)>("fit_plane");
    std::vector<std::vector<double>> points{
      {-1.0, 1.5, 2.0},
      {-6.2, -6.5, -4.3},
      {2.3, 1.9, 0.5},
      {-1.0, 1.0, -1.4},
      {7.6, 1.8, -6.9}
    };
    const auto k = fit_plane(points);
    std::cout << "k0 = " << std::get<0>(k) << std::endl;
    std::cout << "k1 = " << std::get<1>(k) << std::endl;
    std::cout << "k2 = " << std::get<2>(k) << std::endl;
  } else {
    std::cerr << "Please specify the library.\n";
  }
  return 0;
}
