#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class ForceEstimation{
private:
    int y = 1;
public:
    ForceEstimation(int _x=5);
};
