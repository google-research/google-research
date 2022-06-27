// NOLINTBEGIN

#include <iostream>
#include <unordered_set>
#include <set>
#include <functional>
#include <algorithm>
#include <ctime>

#include "sampler.h"
#include "utils.h"
#include "query.h"
#include "knowledge_graph.h"
#include "ThreadPool.h"


template<typename Dtype>
TestSampler<Dtype>::TestSampler(NaiveSampler<Dtype>* _small_sampler, NaiveSampler<Dtype>* _large_sampler, Dtype _negative_sample_size, Dtype _max_missing_ans, int _num_threads)
            : small_sampler(_small_sampler), large_sampler(_large_sampler), negative_sample_size(_negative_sample_size), max_missing_ans(_max_missing_ans), num_threads(_num_threads)
{
    thread_pool = new ThreadPool(num_threads);
    sample_buf.clear();
    sample_pos = 0;
}

template<typename Dtype>
TestSampler<Dtype>::~TestSampler()
{
    if (thread_pool != nullptr)
    {
        for (size_t i = 0; i < sample_buf.size(); ++i)
        {
            auto* sample = sample_buf[i].get();
            delete sample;
        }
        delete thread_pool;
    }
}

template<typename Dtype>
void TestSampler<Dtype>::launch_sampling(int query_type)
{
    // clean obsolete jobs
    for (size_t i = 0; i < sample_buf.size(); ++i)
    {
        auto* sample = sample_buf[i].get();
        delete sample;
    }
    sample_buf.clear();
    for (int i = 0; i < num_threads; ++i)
    {
        sample_buf.emplace_back(
            thread_pool->enqueue([=]{
                return this->gen_sample(query_type);
            })
        );
    }
}

template<typename Dtype>
TestQuerySample<Dtype>* TestSampler<Dtype>::gen_sample(int query_type)
{
    TestQuerySample<Dtype>* sample = new TestQuerySample<Dtype>(query_type);
    while (true)
    {
        QueryTree<Dtype>* qt = large_sampler->gen_query(query_type);
        bool has_negation = qt->has_negation();
        if (!qt->num_intermediate_answers()) {
            delete qt;
            continue;
        }
        assert(qt->num_intermediate_answers());
        QueryTree<Dtype>* qt_small = qt->copy_instantiated_query();
        small_sampler->dfs_answer(qt_small, qt->is_inverse);

        sample->false_positive.clear();
        sample->false_negative.clear();
        sample->true_positive.clear();
        // fp = large - small
        intersect_lazy_negation(qt->ans_ptr_begin(), qt->ans_ptr_end(), qt_small->ans_ptr_begin(), qt_small->ans_ptr_end(), sample->false_negative);

        if (has_negation)
            intersect_lazy_negation(qt_small->ans_ptr_begin(), qt_small->ans_ptr_end(), qt->ans_ptr_begin(), qt->ans_ptr_end(), sample->false_positive);

        if (sample->false_negative.size() == 0 || sample->false_negative.size() > max_missing_ans
            || (has_negation && sample->false_positive.size() == 0) || (has_negation && sample->false_positive.size() > max_missing_ans))
        {
            delete qt; 
            delete qt_small;
            continue;
        }

        merge_sorted_list(qt->ans_ptr_begin(), qt->ans_ptr_end(), qt_small->ans_ptr_begin(), qt_small->ans_ptr_end(), 
                          sample->true_positive,
                          QueryNodeType::intersect);
        sample->query_args.clear();
        sample->negative_samples.clear();
        qt->get_query_args(sample->query_args);
        if (negative_sample_size > 0) {
            std::unordered_set<Dtype> neg_answers;
            while (neg_answers.size() < negative_sample_size)
            {
                Dtype neg_ans = rand_int((Dtype)0, large_sampler->kg->num_ent);
                if (std::binary_search(sample->true_positive.begin(), sample->true_positive.end(), neg_ans))
                    continue;
                if (std::binary_search(sample->false_negative.begin(), sample->false_negative.end(), neg_ans))
                    continue;
                if (has_negation && std::binary_search(sample->false_positive.begin(), sample->false_positive.end(), neg_ans))
                    continue;
                if (neg_answers.count(neg_ans))
                    continue;
                neg_answers.insert(neg_ans);
            }
            for (auto neg_ans : neg_answers)
                sample->negative_samples.push_back(neg_ans);
        }
        delete qt;
        delete qt_small;
        break;
    }
    return sample;
}

template<typename Dtype>
int TestSampler<Dtype>::fetch_query(py::list query_args, py::list list_tp, py::list list_fp, py::list list_fn, py::list list_neg_samples)
{
    assert(sample_pos < this->sample_buf.size());
    TestQuerySample<Dtype>* sample = nullptr;
    int query_type;
    while (true)
    {
        for (size_t i = 0; i < this->sample_buf.size(); ++i)
            if (this->sample_buf[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                sample = this->sample_buf[i].get();
                query_type = sample->q_type;
                this->sample_buf[i] = thread_pool->enqueue([=]{
                    return this->gen_sample(query_type);
                });
                break;
            }
        if (sample != nullptr)
            break;
    }

    for (auto&& a : sample->query_args)
        query_args.append(a);
    for (auto x : sample->false_positive)
        list_fp.append(x);
    for (auto x : sample->false_negative)
        list_fn.append(x);
    if (negative_sample_size > 0)
    {
        for (auto x : sample->negative_samples)
            list_neg_samples.append(x);
    } else {
        for (auto x : sample->true_positive)
            list_tp.append(x);
    }
    delete sample;
    return query_type;
}


template class TestSampler<unsigned>;
template class TestSampler<uint64_t>;
// NOLINTEND
