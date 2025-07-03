#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "graph.h"

namespace py = pybind11;


PYBIND11_MODULE(kcore, m) {
    pybind11::class_<Graph>(m, "Graph")
        .def(pybind11::init())
        .def("loadUndirGraph", &Graph::loadUndirGraph)
        .def("RemoveNode", &Graph::RemoveNode)
        .def("GetFollowerNum", &Graph::GetFollowerNum)
        .def("Get_follower_exact", &Graph::Get_follower_exact)
        .def("FindBest", &Graph::FindBest)
        .def("Greedy", &Graph::Greedy)
        .def("Compute_un_dominated", &Graph::Compute_un_dominated)
        .def("Get_un_dominated", &Graph::Get_un_dominated);

}
