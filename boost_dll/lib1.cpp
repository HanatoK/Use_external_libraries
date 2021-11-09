#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <boost/dll.hpp>

using namespace boost;
#define API extern "C" BOOST_SYMBOL_EXPORT

API std::tuple<double, double, double> fit_plane(const std::vector<std::vector<double>>& points);

std::tuple<double, double, double> fit_plane(const std::vector<std::vector<double>>& points) {
  Eigen::MatrixXd A(points.size(), 3);
  Eigen::VectorXd b(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    A(i, 0) = points[i][0];
    A(i, 1) = points[i][1];
    A(i, 2) = 1.0;
    b(i) = points[i][2];
  }
  const auto L = (A.transpose() * A).ldlt();
  const auto solution_linalg = L.solve(A.transpose() * b);
  std::tuple<double, double, double> k{solution_linalg(0), solution_linalg(1), solution_linalg(2)};
  return k;
}
