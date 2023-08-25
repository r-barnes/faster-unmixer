#include "faster-unmixer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace fastunmixer;

PYBIND11_MODULE(_funmixer_native, m) {
  m.doc() = "fastunmix module"; // optional module docstring

  py::class_<NativeSampleNode>(m, "NativeSampleNode")
      .def_readonly("name", &NativeSampleNode::name)
      .def_readonly("x", &NativeSampleNode::x)
      .def_readonly("y", &NativeSampleNode::y)
      .def_readonly("downstream_node", &NativeSampleNode::downstream_node)
      .def_readonly("upstream_nodes", &NativeSampleNode::upstream_nodes)
      .def_readonly("area", &NativeSampleNode::area)
      .def_readonly("total_upstream_area", &NativeSampleNode::total_upstream_area)
      .def_readonly("label", &NativeSampleNode::label);

  m.def("fastunmix", &faster_unmixer, "Get a graph for fast unmixing", py::arg("flowdirs_filename"), py::arg("sample_filename"));

  m.attr("root_node_name") = root_node_name;
}
