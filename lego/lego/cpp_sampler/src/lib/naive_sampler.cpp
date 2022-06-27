// NOLINTBEGIN

#include "utils.h"
#include "sampler.h"
#include "knowledge_graph.h"

template<typename Dtype>
NaiveSampler<Dtype>::NaiveSampler(KG<Dtype>* _kg, py::list _query_trees, py::list _query_prob, bool _share_negative, bool _same_in_batch,
                           bool _weighted_answer_sampling, bool _weighted_negative_sampling,
                           Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, Dtype _max_n_partial_answers,
                           int num_threads, py::list no_search_list) : Sampler<Dtype>(_kg, _query_trees, _query_prob, _share_negative, _same_in_batch, 
                                                      _weighted_answer_sampling, _weighted_negative_sampling,
                                                      _negative_sample_size, _rel_bandwidth, _max_to_keep,
                                                      _max_n_partial_answers, num_threads) {}

template<typename Dtype>
void NaiveSampler<Dtype>::negative_sampling(QuerySample<Dtype>* sample, QueryTree<Dtype>* qt, const Dtype* list_neg_candidates)
{
    sample->negative_samples.clear();
    Dtype neg_ans;
    for (Dtype i = 0; i < this->negative_sample_size; ++i)
    {
        if (list_neg_candidates != nullptr)
        {
            bool is_positive = std::binary_search(qt->ans_ptr_begin(), qt->ans_ptr_end(), list_neg_candidates[i]);
            sample->negative_samples.push_back(!is_positive);
            continue;
        }
        while (true) {
            neg_ans = this->sample_entity(this->weighted_negative_sampling, 
                                          qt->is_inverse ? this->am_out : this->am_in);
            if (std::binary_search(qt->ans_ptr_begin(), qt->ans_ptr_end(), neg_ans))
                continue;
            break;
        }
        sample->negative_samples.push_back(neg_ans);
    }
}

template<typename Dtype>
QueryTree<Dtype>* NaiveSampler<Dtype>::gen_query(int query_type)
{
    QueryTree<Dtype>* qt = nullptr;
    while (true)
    {
        qt = this->instantiate_query(this->query_trees[query_type]);
        if (!this->dfs_answer(qt, qt->is_inverse)) // oversize
        {
            delete qt;
            continue;
        }
        assert(!qt->lazy_negation);
        if (qt->num_intermediate_answers() > this->max_n_partial_answers)
            delete qt;
        else
            break;
    }
    return qt;
}

template<typename Dtype>
QuerySample<Dtype>* NaiveSampler<Dtype>::gen_sample(int query_type, const Dtype* list_neg_candidates)
{
    QueryTree<Dtype>* qt = this->gen_query(query_type);
    QuerySample<Dtype>* sample = new QuerySample<Dtype>(query_type);
    sample->query_args.clear();
    qt->get_query_args(sample->query_args);
    sample->positive_answer = qt->answer;
    sample->sample_weight = sqrt(1.0 / (qt->num_intermediate_answers() + 1));
    negative_sampling(sample, qt, list_neg_candidates);
    delete qt;
    return sample;
}

template class NaiveSampler<unsigned>;
template class NaiveSampler<uint64_t>;
// NOLINTEND
