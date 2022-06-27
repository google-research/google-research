// NOLINTBEGIN

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "knowledge_graph.h"
#include "sampler.h"
#include "query.h"
#include "alias_method.h"


template<typename Dtype>
void declare_sampler(py::module &mod, std::string typestr_suffix)
{
    std::string naive_sampler_name = "NaiveSampler" + typestr_suffix;
    py::class_<NaiveSampler<Dtype> >(mod, naive_sampler_name.c_str())
        .def(py::init<KG<Dtype>*, py::list, py::list, bool, bool, bool, bool, Dtype, Dtype, Dtype, Dtype, int, py::list>())
        .def("set_seed", &NaiveSampler<Dtype>::set_seed)
        .def("prefetch", &NaiveSampler<Dtype>::prefetch)
        .def("next_batch", &NaiveSampler<Dtype>::next_batch)
        .def("print_queries", &NaiveSampler<Dtype>::print_queries);

    std::string no_search_sampler_name = "NoSearchSampler" + typestr_suffix;
    py::class_<NoSearchSampler<Dtype> >(mod, no_search_sampler_name.c_str())
        .def(py::init<KG<Dtype>*, py::list, py::list, bool, bool, bool, bool, Dtype, Dtype, Dtype, Dtype, int, py::list>())
        .def("set_seed", &NaiveSampler<Dtype>::set_seed)
        .def("prefetch", &NaiveSampler<Dtype>::prefetch)
        .def("next_batch", &NaiveSampler<Dtype>::next_batch)
        .def("print_queries", &NaiveSampler<Dtype>::print_queries);

    std::string rejection_sampler_name = "RejectionSampler" + typestr_suffix;
    py::class_<RejectionSampler<Dtype> >(mod, rejection_sampler_name.c_str())
        .def(py::init<KG<Dtype>*, py::list, py::list, bool, bool, bool, bool, Dtype, Dtype, Dtype, Dtype, int, py::list>())
        .def("set_seed", &RejectionSampler<Dtype>::set_seed)
        .def("prefetch", &RejectionSampler<Dtype>::prefetch)
        .def("next_batch", &RejectionSampler<Dtype>::next_batch)
        .def("print_queries", &RejectionSampler<Dtype>::print_queries);

    std::string test_sampler_name = "TestSampler" + typestr_suffix;
    py::class_<TestSampler<Dtype> >(mod, test_sampler_name.c_str())
        .def(py::init<NaiveSampler<Dtype>*, NaiveSampler<Dtype>*, Dtype, Dtype, int>())
        .def("launch_sampling", &TestSampler<Dtype>::launch_sampling)
        .def("fetch_query", &TestSampler<Dtype>::fetch_query);
}


template<typename Dtype>
void declare_kg(py::module &mod, std::string typestr_suffix)
{
    std::string kg_name = "KG" + typestr_suffix;
    py::class_<KG<Dtype>>(mod, kg_name.c_str())
        .def(py::init<>())
        .def(py::init<Dtype, Dtype>())
        .def("load_triplets", &KG<Dtype>::load_triplets, "load triplets from txt data", py::arg("fname"), py::arg("has_reverse_edges"))
        .def("load_triplets_from_files", &KG<Dtype>::load_triplets_from_files)
        .def("has_forward_edge", &KG<Dtype>::has_forward_edge)
        .def("has_backward_edge", &KG<Dtype>::has_backward_edge)
        .def("max_degree", &KG<Dtype>::max_degree)
        .def("ptr", &KG<Dtype>::ptr)
        .def("load", &KG<Dtype>::load)
        .def("dump", &KG<Dtype>::dump)
        .def("dump_nt", &KG<Dtype>::dump_nt)
        .def_readonly("dtype", &KG<Dtype>::dtype)
        .def_readonly("num_ent", &KG<Dtype>::num_ent)
        .def_readonly("num_rel", &KG<Dtype>::num_rel)
        .def_readonly("num_edges", &KG<Dtype>::num_edges);
}

template<typename Dtype>
void declare_qtree(py::module &mod, std::string typestr_suffix)
{
    std::string qt_name = "QueryTree" + typestr_suffix;
    py::class_<QueryTree<Dtype> > qtree(mod, qt_name.c_str());

    qtree.def("add_child", &QueryTree<Dtype>::add_child)
         .def("get_children", &QueryTree<Dtype>::get_children)
         .def_readwrite("is_inverse", &QueryTree<Dtype>::is_inverse)
         .def_readonly("sqrt_middle", &QueryTree<Dtype>::sqrt_middle)
         .def_readonly("node_type", &QueryTree<Dtype>::node_type)
         .def_readonly("parent_edge", &QueryTree<Dtype>::parent_edge)
         .def("__repr__", [](const QueryTree<Dtype> &t) {
            return t.str_bracket(true);
         });
}


PYBIND11_MODULE(libsampler, mod) {

    declare_sampler<unsigned>(mod, "32");
    declare_sampler<uint64_t>(mod, "64");

    declare_kg<unsigned>(mod, "32");
    declare_kg<uint64_t>(mod, "64");

    declare_qtree<unsigned>(mod, "32");
    declare_qtree<uint64_t>(mod, "64");

    py::class_<AliasMethod>(mod, "AliasMethod")
        .def(py::init<>())
        .def("setup_from_numpy", &AliasMethod::setup_from_numpy)
        .def("draw_sample", &AliasMethod::draw_sample);

    py::enum_<QueryNodeType>(mod, "QueryNodeType")
        .value("entity", QueryNodeType::entity)
        .value("intersect", QueryNodeType::intersect)
        .value("union", QueryNodeType::union_set)
        .value("entity_set", QueryNodeType::entity_set)
        .export_values();

    py::enum_<QueryEdgeType>(mod, "QueryEdgeType")
        .value("no_op", QueryEdgeType::no_op)
        .value("relation", QueryEdgeType::relation)
        .value("negation", QueryEdgeType::negation)
        .export_values();

    mod.def("create_qt32", &create_qt32, py::return_value_policy::reference);
    mod.def("create_qt64", &create_qt64, py::return_value_policy::reference);
}
// NOLINTEND
