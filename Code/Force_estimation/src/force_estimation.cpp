#include "force_estimation.h"
#include <iostream>
#include <stdio.h>

ForceEstimation::ForceEstimation(int _x)
{
    y = 9;
    std::cout << "test" << std::endl;
}

PYBIND11_MODULE(force_estimation, m) {
    py::class_<ForceEstimation>(m, "ForceEstimation")
        .def(py::init<int>(), py::arg("_x") = 8);
        // .def("run", &Matching::run)
}