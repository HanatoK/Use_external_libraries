#include <iostream>
#include <boost/signals2.hpp>
#include <functional>

template <typename T>
class Sender {
public:
  boost::signals2::signal<void(T)> m_foo; // not copy-constructible
  Sender() {}
  Sender(const Sender& rhs) {}
  void send(int x) {
    m_foo(x);
  }
};

template <typename T>
class Receiver {
private:
  std::ostream& m_os;
public:
  Receiver(std::ostream& os): m_os(os) {}
  void receive(T x) {
    m_os << x << std::endl;
  }
};

int main() {
  Sender<int> sender;
  Receiver<int> receiver(std::cout);
  boost::signals2::connection c;
  c = sender.m_foo.connect(
      std::bind(&decltype(receiver)::receive, &receiver, std::placeholders::_1));
  sender.send(100);
  sender.send(20);
  decltype(sender) sender2(sender);
  // no output
  sender2.send(30);
  return 0;
}
