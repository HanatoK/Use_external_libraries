#include <iostream>
#include <boost/signals2.hpp>
#include <functional>

class Sender {
public:
  boost::signals2::signal<void(int)> m_foo; // not copy-constructible
  Sender() {}
  Sender(const Sender& rhs) {}
  void send(int x) {
    m_foo(x);
  }
};

class Receiver {
private:
  std::ostream& m_os;
public:
  Receiver(std::ostream& os): m_os(os) {}
  void receive(int x) {
    m_os << x << std::endl;
  }
};

int main() {
  Sender sender;
  Receiver receiver(std::cout);
  boost::signals2::connection c = sender.m_foo.connect(
      std::bind(&Receiver::receive, &receiver, std::placeholders::_1));
  sender.send(100);
  sender.send(20);
  Sender sender2(sender);
  // no output
  sender2.send(30);
  return 0;
}
