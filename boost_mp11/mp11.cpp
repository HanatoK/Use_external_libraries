#include <any>
#include <map>
#include <string>
#include <vector>
#include <type_traits>
#include <iostream>
#include <boost/mp11/map.hpp>
#include <boost/mp11/list.hpp>

enum class Foo {
  A1, A2
};

enum class Bar {
  B1, B2, B3
};

// This does not work in gcc:
// see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104867
// also https://github.com/boostorg/mp11/issues/72

// template <auto EnumVal>
// struct enum_ {};

template <auto EnumVal> using enum_ = std::integral_constant<decltype(EnumVal), EnumVal>;

using enum_type_map = boost::mp11::mp_list<
  boost::mp11::mp_list<enum_<Foo::A1>, int>,
  boost::mp11::mp_list<enum_<Foo::A2>, double>,
  boost::mp11::mp_list<enum_<Bar::B1>, std::string>,
  boost::mp11::mp_list<enum_<Bar::B2>, std::vector<int>>
>;

template <auto EnumVal>
using enum_to_type = boost::mp11::mp_at_c<boost::mp11::mp_map_find<enum_type_map, enum_<EnumVal>>, 1>;

template <auto EnumVal>
enum_to_type<EnumVal> get_attribute(
  const std::map<decltype(EnumVal), std::any>& m) {
  return std::any_cast<enum_to_type<EnumVal>>(m.at(EnumVal));
}

int main() {
  std::map<Bar, std::any> m;
  m[Bar::B1] = std::string{"hello"};
  m[Bar::B2] = std::vector<int>{1,2,3,4};
  std::map<Foo, std::any> n;
  n[Foo::A1] = 13;
  n[Foo::A2] = 0.5;
  const auto s = get_attribute<Bar::B1>(m);
  std::cout << s << std::endl;
  return 0;
}