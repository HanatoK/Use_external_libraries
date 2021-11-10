#include <iostream>
#include <boost/histogram.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
  typedef boost::histogram::axis::regular<double> axis_type;
  std::vector<axis_type> axes;
  axes.push_back(axis_type(100, -1.0, 1.0, "encoded"));
  auto hist = boost::histogram::make_histogram(axes);
  if (argc > 1) {
    typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
    boost::char_separator<char> sep{" "};
    std::ifstream ifs(argv[1]);
    std::string line;
    double data;
    while (std::getline(ifs, line)) {
      tokenizer tok{line, sep};
      for (auto it = tok.begin(); it != tok.end(); ++it) {
        data = std::stod(*it);
      }
      hist(data);
    }
  }
  std::ofstream ofs_count("round0_count.dat");
  double sum = 0;
  for (const auto& x : boost::histogram::indexed(hist)) {
    ofs_count << boost::format("%12.5f %12.5f\n")
      % x.bin(0).center() % *x;
    sum += *x;
  }
  std::ofstream ofs_pdf("round0_pdf.dat");
  for (const auto& x : boost::histogram::indexed(hist)) {
    const double pdf = 1.0 / x.bin(0).width() * (*x) / sum;
    ofs_pdf << boost::format("%12.5f %12.5f\n")
      % x.bin(0).center() % pdf;
  }
}
