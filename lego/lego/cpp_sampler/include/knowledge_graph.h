// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KNOWLEDGE_GRAPH_H
#define KNOWLEDGE_GRAPH_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <string>
#include <vector>
#include <cstdio>
#include <unordered_map>

template<typename Dtype>
class SortedList
{
 public:
    SortedList(Dtype _num_ent, Dtype _num_rel);
    ~SortedList();

    template<int src_idx>
    void setup_edges(std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges);

    bool has_edge(Dtype src, Dtype r, Dtype dst);

    Dtype num_ent, num_rel;
    Dtype num_frag_rels;
    Dtype* src_rel_map, *rel_dst_map;
    Dtype* dst_list;
    void dump(FILE* fid, Dtype num_edges);
    void dump_nt(FILE* fid, Dtype num_edges);
    void load(FILE* fid, Dtype num_edges);
    size_t load(Dtype* buffer, Dtype num_edges);

    Dtype* locate_rel_dst(Dtype node, Dtype rel);
    Dtype src_degree(Dtype node);
    Dtype src_num_unique_rel(Dtype node);

 private:
    void try_free();
    bool mem_shared;
};

template<typename Dtype>
class KG
{
 public:
    KG();
    KG(Dtype _num_ent, Dtype _num_rel);
    ~KG();
    size_t ptr();
    void load(const std::string& fname);
    void load_from_binary(Dtype* buf_ptr, size_t n_ints);
    void load_from_numpy(void* triplets,
                         size_t n_triplets,
                         const bool has_reverse_edges);

    void dump(const std::string& fname);
    void dump_nt(const std::string& fname);

    void load_triplets(const std::string& fname, const bool has_reverse_edges);
    void load_triplets_from_files(py::list list_files,
                                  const bool has_reverse_edges);

    bool has_forward_edge(Dtype src, Dtype r, Dtype dst);
    bool has_backward_edge(Dtype src, Dtype r, Dtype dst);

    Dtype in_degree(Dtype node);
    Dtype out_degree(Dtype node);
    Dtype unique_num_in_rel(Dtype node);
    Dtype unique_num_out_rel(Dtype node);

    Dtype max_degree();

    Dtype num_ent, num_rel;
    Dtype num_edges;
    SortedList<Dtype>* ent_out, *ent_in;
    std::string dtype;
 private:
    void try_free();

    void load_edges_from_file(
        const std::string& fname,
        const bool has_reverse_edges,
        std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges);
    void build_kg_from_edges(
        std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges);
};

template<typename Dtype>
void sample_rand_neighbor(SortedList<Dtype>* ent_list,
                          Dtype node, Dtype& r, Dtype& other_node);

#endif
