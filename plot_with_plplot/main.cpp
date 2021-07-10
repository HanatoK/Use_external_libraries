#include <plplot/plstream.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

class Plot {
public:
  Plot(int argc, char* argv[]) {
    m_num_points = 100;
    m_data_x = new PLFLT[m_num_points];
    m_data_y = new PLFLT[m_num_points];
    for (size_t i = 0; i < m_num_points; ++i) {
      m_data_x[i] = i / static_cast<double>(m_num_points);
      m_data_y[i] = m_data_x[i] * m_data_x[i];
    }
    // initialize plstream
    m_pls = new plstream();
    m_pls->parseopts(&argc, argv, PL_PARSE_FULL);
    m_pls->init();
    m_pls->env(0.0, 1.0, 0.0, 1.0, 0, 0);
    m_pls->lab("X", "Y", "Demo");
    m_pls->line(m_num_points, m_data_x, m_data_y);
  }
  ~Plot() {
    if (m_pls != nullptr) delete m_pls;
    if (m_data_x != nullptr) delete[] m_data_x;
    if (m_data_y != nullptr) delete[] m_data_y;
  }
private:
  plstream* m_pls;
  size_t m_num_points;
  PLFLT* m_data_x;
  PLFLT* m_data_y;
};

int main(int argc, char* argv[]) {
  Plot* plot = new Plot(argc, argv);
  delete plot;
  return 0;
}