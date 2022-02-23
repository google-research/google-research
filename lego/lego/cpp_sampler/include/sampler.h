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

// NOLINTBEGIN

#ifndef SAMPLER_H
#define SAMPLER_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <string>
#include <future>
#include <random>
#include <set>
#include <unordered_set>
#include "knowledge_graph.h"
#include "query.h"

class ThreadPool;
class AliasMethod;

template<typename Dtype>
class ISampler {
 public:
    ISampler(KG<Dtype>* _kg, py::list _query_prob, bool share_negative,
             bool _same_in_batch, bool weighted_answer_sampling,
             bool weighted_negative_sampling, Dtype negative_sample_size,
             Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
             int num_threads);
    virtual ~ISampler();
    virtual QuerySample<Dtype>* gen_sample(int query_type, const Dtype* list_neg_candidates) = 0;

    void set_seed(Dtype seed);
    void prefetch(int _batch_size, int num_batches);

    void next_batch(
        py::array_t<long long, py::array::c_style | py::array::forcecast> _positive_samples,
                    py::array_t<long long, py::array::c_style | py::array::forcecast> _negative_samples, 
                    py::array_t<float, py::array::c_style | py::array::forcecast> _sample_weights, 
                    py::array_t<float, py::array::c_style | py::array::forcecast> _is_negative,
                    py::list list_query, py::list list_q_idx);
    Dtype sample_entity(bool weighted, AliasMethod* am);

    KG<Dtype>* kg;
    ThreadPool* thread_pool;
    std::default_random_engine query_rand_engine;

    bool share_negative, same_in_batch;
    bool weighted_answer_sampling, weighted_negative_sampling;
    Dtype negative_sample_size, rel_bandwidth, max_to_keep, max_n_partial_answers;

    std::vector< std::future<QuerySample<Dtype>*> > sample_buf;
    std::vector< Dtype > negsample_buf;
    std::discrete_distribution<int> query_dist;
    std::vector<float> query_prob;

    Dtype sample_pos;
protected:
    int batch_size;
    void place_batch_job(int b_idx);
    AliasMethod *am_in, *am_out;
    std::future<QuerySample<Dtype>*> enqueue_job(int q_type, const Dtype* neg_ptr);
};

template<typename Dtype>
class Sampler : public ISampler<Dtype>
{
public:
    Sampler(KG<Dtype>* _kg, py::list query_trees, py::list _query_prob, 
            bool share_negative, bool _same_in_batch,
            bool weighted_answer_sampling, bool weighted_negative_sampling,
            Dtype negative_sample_size, Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
            int num_threads);
    virtual ~Sampler();

    QueryTree<Dtype>* instantiate_query(QueryTree<Dtype>* query_template);

    bool exec_junction(QueryTree<Dtype>* cur_root);
    bool exec_relation(QueryTree<Dtype>* cur_root, bool inverse);
    bool dfs_answer(QueryTree<Dtype>* cur_root, bool inverse);

    std::vector<QueryTree<Dtype>*> query_trees;
    virtual void print_queries();

protected:
    bool sample_actual_query(QueryTree<Dtype>* qt, Dtype answer, bool inverse);
    bool verify_sampled_query(QueryTree<Dtype>* qt, Dtype answer, bool inverse);
};

template<typename Dtype>
class NoSearchSampler : public Sampler<Dtype>
{
public:
    NoSearchSampler(KG<Dtype>* _kg, py::list query_trees, py::list _query_prob, bool share_negative, bool _same_in_batch,
            bool weighted_answer_sampling, bool weighted_negative_sampling,
            Dtype negative_sample_size, Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
            int num_threads, py::list no_search_list);
    virtual QuerySample<Dtype>* gen_sample(int query_type, const Dtype* list_neg_candidates) override;
    void negative_sampling(QuerySample<Dtype>* sample, QueryTree<Dtype>* qt, const Dtype* list_neg_candidates);
};

template<typename Dtype>
class NaiveSampler : public Sampler<Dtype>
{
public:
    NaiveSampler(KG<Dtype>* _kg, py::list query_trees, py::list _query_prob, bool share_negative, bool _same_in_batch,
            bool weighted_answer_sampling, bool weighted_negative_sampling,
            Dtype negative_sample_size, Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
            int num_threads, py::list no_search_list);
    QueryTree<Dtype>* gen_query(int query_type);

    virtual QuerySample<Dtype>* gen_sample(int query_type, const Dtype* list_neg_candidates) override;
    void negative_sampling(QuerySample<Dtype>* sample, QueryTree<Dtype>* qt, const Dtype* list_neg_candidates);
};

template<typename Dtype>
class RejectionSampler: public Sampler<Dtype>
{
public:
    RejectionSampler(KG<Dtype>* _kg, py::list query_trees, py::list _query_prob, bool share_negative, bool _same_in_batch,
            bool weighted_answer_sampling, bool weighted_negative_sampling,
            Dtype negative_sample_size, Dtype rel_bandwidth, Dtype max_to_keep, Dtype max_n_partial_answers,
            int num_threads, py::list no_search_list);
    virtual QuerySample<Dtype>* gen_sample(int query_type, const Dtype* list_neg_candidates) override;
    Dtype negative_sampling(QuerySample<Dtype>* sample, QueryTree<Dtype>* qt, const Dtype* list_neg_candidates);
    bool bottomup_search(QueryTree<Dtype>* qt, bool inverse);
    bool is_positive(QueryTree<Dtype>* qt, bool inverse);
    bool is_positive(QueryTree<Dtype>* qt, Dtype candidate);

    virtual void print_queries() override;
    std::vector<bool> has_false_pos;
    std::set<int> no_search_set;
    double avg_sample_weight;
protected:
    int check_neg_depth(QueryTree<Dtype>* qt, bool& false_pos_flag);
};

template<typename Dtype>
class TestSampler
{
public:
    TestSampler(NaiveSampler<Dtype>* _small_sampler, NaiveSampler<Dtype>* _large_sampler, Dtype _negative_sample_size, Dtype _max_missing_ans, int num_threads);
    ~TestSampler();

    void launch_sampling(int query_type);
    TestQuerySample<Dtype>* gen_sample(int query_type);
    int fetch_query(py::list query_args, py::list list_tp, py::list list_fp, py::list list_fn, py::list list_neg_samples);

    NaiveSampler<Dtype>* small_sampler, *large_sampler;
    Dtype negative_sample_size, max_missing_ans;
    int num_threads;
    Dtype sample_pos;
    ThreadPool* thread_pool;
    std::vector< std::future<TestQuerySample<Dtype>*> > sample_buf;
};

template<typename Dtype>
std::pair<Dtype*, Dtype*> get_ch_range(SortedList<Dtype>* sl, Dtype node, Dtype r, Dtype rel_bandwidth);

template<typename T1, typename T2, typename Dtype>
void merge_sorted_list(T1 first1, T1 last1, T2 first2, T2 last2,
                       std::vector<Dtype>& result, QueryNodeType node_type)
{
    if (node_type == QueryNodeType::intersect)
        std::set_intersection(first1, last1, first2, last2, std::back_inserter(result));
    else {
        assert(node_type == QueryNodeType::union_set);
        std::set_union(first1, last1, first2, last2, std::back_inserter(result));
    }
}


template<typename T1, typename T2, typename Dtype>
void intersect_lazy_negation(T1 first1, T1 last1, T2 nfirst2, T2 nlast2, std::vector<Dtype>& result)
{
    result.clear();
    for (auto p = first1; p != last1; ++p)
    {
        while (nfirst2 != nlast2 && *nfirst2 < *p)
            nfirst2++;
        if (nfirst2 != nlast2 && *p == *nfirst2)
            continue;
        result.push_back(*p);
    }
}

template<typename T1, typename T2>
bool has_set_intersect(T1* first1, T1* last1, T2* first2, T2* last2)
{
    if (first1 == last1 || first2 == last2)
        return false;
    size_t n1 = last1 - first1, n2 = last2 - first2;
    if (n1 == 1u)
        return std::binary_search(first2, last2, *first1);
    if (n2 == 1u)
        return std::binary_search(first1, last1, *first2);
    T1* rbegin1 = first1 + n1 - 1u;
    T2* rbegin2 = first2 + n2 - 1u;
    if (*rbegin1 < *first2)
        return false;
    if (*rbegin2 < *first1)
        return false;

    while (first1!=last1 && first2!=last2)
    {
        if (*first1<*first2)
            ++first1;
        else if (*first2<*first1)
            ++first2;
        else
            return true;
    }
    return false;
}

#endif

// NOLINTEND
