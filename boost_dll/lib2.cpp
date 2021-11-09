#include <vector>
#include <tuple>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <boost/dll.hpp>

using namespace boost;
#define API extern "C" BOOST_SYMBOL_EXPORT

API std::tuple<double, double, double> fit_plane(const std::vector<std::vector<double>>& points);

std::tuple<double, double, double> fit_plane(const std::vector<std::vector<double>>& points) {
  gsl_matrix *A = gsl_matrix_alloc(points.size(), 3);
  gsl_matrix *ATA = gsl_matrix_alloc(3, 3);
  gsl_vector *b = gsl_vector_alloc(points.size());
  gsl_vector *ATb = gsl_vector_alloc(3);
  gsl_vector *x = gsl_vector_alloc(3);
  for (size_t i = 0; i < points.size(); ++i) {
    gsl_matrix_set(A, i, 0, points[i][0]);
    gsl_matrix_set(A, i, 1, points[i][1]);
    gsl_matrix_set(A, i, 2, 1.0);
    gsl_vector_set(b, i, points[i][2]);
  }
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, A, 0.0, ATA);
  gsl_blas_dgemv(CblasTrans, 1.0, A, b, 0.0, ATb);
  gsl_linalg_ldlt_decomp(ATA);
  gsl_linalg_ldlt_solve(ATA, ATb, x);
  std::tuple<double, double, double> k{gsl_vector_get(x, 0), gsl_vector_get(x, 1), gsl_vector_get(x, 2)};
  gsl_vector_free(x);
  gsl_vector_free(ATb);
  gsl_vector_free(b);
  gsl_matrix_free(ATA);
  gsl_matrix_free(A);
  return k;
}
