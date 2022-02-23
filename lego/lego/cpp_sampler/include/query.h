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

#ifndef QUERY_H
#define QUERY_H
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <vector>
#include <string>
#include <unordered_set>

enum QueryNodeType {
    entity = 0,
    intersect = 1,
    union_set = 2,
    entity_set = 3,
};

enum QueryEdgeType {
    no_op = 0,
    relation = 1,
    negation = 2,
};


template<typename Dtype>
class QueryTree
{
 public:
    QueryTree(QueryNodeType node_type);
    ~QueryTree();

    QueryTree<Dtype>* copy_backbone();
    QueryTree<Dtype>* copy_instantiated_query();
    bool has_negation();

    void get_query_args(std::vector<int>& query_args);

    std::vector< std::pair<QueryEdgeType, QueryTree<Dtype>*> > children;
    void add_child(QueryEdgeType edge_type, QueryTree<Dtype>* child);
    std::string str_bracket(bool backbone_only) const;

    QueryEdgeType parent_edge;
    QueryNodeType node_type;

    Dtype parent_r;
    Dtype answer;
    long long hash_code;  // NOLINT
    bool sqrt_middle, lazy_negation, is_inverse;

    Dtype num_intermediate_answers();
    void reset_answers();
    void set_answers();
    void set_answers(Dtype answer);
    void set_answers(const Dtype* ptr_begin, const Dtype* ptr_end);
    std::vector<Dtype> intermediate_answers;

    const Dtype* ans_ptr_begin() const;
    const Dtype* ans_ptr_end() const;

    void get_children(py::list ch_list);

 private:
    const Dtype* _ans_ptr_begin;
    const Dtype* _ans_ptr_end;
};

template<typename Dtype>
class QuerySample
{
 public:
    QuerySample(int _q_type);

    virtual ~QuerySample();
    int q_type;
    Dtype positive_answer;
    float sample_weight;
    std::vector<int> query_args;
    std::vector<Dtype> negative_samples;
};

template<typename Dtype>
class TestQuerySample : public QuerySample<Dtype>
{
 public:
    TestQuerySample(int _q_type);
    virtual ~TestQuerySample();

    std::vector<Dtype> true_positive, false_positive, false_negative;
};


QueryTree<unsigned>* create_qt32(QueryNodeType node_type);
QueryTree<uint64_t>* create_qt64(QueryNodeType node_type);


#endif
