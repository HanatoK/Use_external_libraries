#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    mat a = { {1, 2, 3} };
    mat b = { {3, 4, 5} };
    mat c = a + b;
    double x = dot(a, b);
    mat d = cross(a, c);
    c.print();
    d.print();
    cout << "X is " << x << endl;
    cout << a << b << endl;
    return 0;
}
