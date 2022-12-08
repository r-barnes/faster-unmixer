#include "faster-unmixer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace fastunmixer;

PYBIND11_MODULE(pyfastunmix, m) {
  m.doc() = "fastunmix module"; // optional module docstring

  py::class_<SampleNode>(m, "SampleNode", py::dynamic_attr())
      .def_readonly("name", &SampleNode::name)
      .def_readonly("x", &SampleNode::x)
      .def_readonly("y", &SampleNode::y)
      .def_readonly("downstream_node", &SampleNode::downstream_node)
      .def_readonly("upstream_nodes", &SampleNode::upstream_nodes)
      .def_readonly("area", &SampleNode::area)
      .def_readonly("total_upstream_area", &SampleNode::total_upstream_area)
      .def_readonly("label", &SampleNode::label);

  m.def("fastunmix", &faster_unmixer, "Get a graph for fast unmixing", py::arg("flowdirs_filename"), py::arg("sample_filename"));

  m.attr("root_node_name") = root_node_name;
}
