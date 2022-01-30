#include <string>
#include <vector>
#include <cmath>
#include <boost/python.hpp>

class World {
public:
  World(std::string msg): m_msg(msg) {}
  void set(std::string msg) {
    m_msg = msg;
  }
  std::string greet() {
    return m_msg;
  }
  double func(boost::python::list& vec) {
    using namespace boost::python;
    double s = 0;
    for (int i = 0; i < len(vec); ++i) {
      s += std::sin(boost::python::extract<double>(vec[i]));
    }
    return s;
  }
private:
  std::string m_msg;
};

const char* greet() {
  return "hello, world";
}

BOOST_PYTHON_MODULE(hello_ext)
{
  using namespace boost::python;
  def("greet", greet);
  class_<World>("World", init<std::string>())
    .def("greet", &World::greet)
    .def("set", &World::set)
    .def("func", &World::func);
}