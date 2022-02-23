// NOLINTBEGIN

#include <iostream>
#include <unordered_set>
#include <set>
#include <functional>
#include <algorithm>
#include <ctime>

#include "alias_method.h"
#include "sampler.h"
#include "utils.h"
#include "query.h"
#include "knowledge_graph.h"
#include "ThreadPool.h"

template<typename Dtype>
ISampler<Dtype>::ISampler(KG<Dtype>* _kg, py::list _query_prob, bool _share_negative, bool _same_in_batch,
                bool _weighted_answer_sampling, bool _weighted_negative_sampling,
                Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, Dtype _max_n_partial_answers,
                int num_threads) : kg(_kg), share_negative(_share_negative), same_in_batch(_same_in_batch), 
                                weighted_answer_sampling(_weighted_answer_sampling), weighted_negative_sampling(_weighted_negative_sampling),
                                negative_sample_size(_negative_sample_size),
                                rel_bandwidth(_rel_bandwidth), max_to_keep(_max_to_keep), max_n_partial_answers(_max_n_partial_answers)
{
    if (num_threads)
        thread_pool = new ThreadPool(num_threads);
    else
        thread_pool = nullptr;
    query_prob.clear();
    for (size_t i = 0; i < py::len(_query_prob); ++i)
        query_prob.push_back(py::cast<float>(_query_prob[i]));
    query_dist = std::discrete_distribution<int>(query_prob.begin(), query_prob.end());
    set_seed(time(NULL));
    sample_buf.clear();
    negsample_buf.clear();
    sample_pos = 0;
    this->am_in = new AliasMethod();
    this->am_out = new AliasMethod();
    if (this->weighted_answer_sampling || this->weighted_negative_sampling)
    {
        std::vector<double> in_degrees, out_degrees;
        in_degrees.resize(this->kg->num_ent);
        out_degrees.resize(this->kg->num_ent);
        for (Dtype i = 0; i < this->kg->num_ent; ++i)
        {
            in_degrees[i] = this->kg->in_degree(i);
            out_degrees[i] = this->kg->out_degree(i);
        }
        this->am_in->setup(in_degrees);
        this->am_out->setup(out_degrees);
    }
}

template<typename Dtype>
ISampler<Dtype>::~ISampler()
{
    if (thread_pool != nullptr)
        delete thread_pool;
    delete am_in;
    delete am_out;
}

template<typename Dtype>
void ISampler<Dtype>::set_seed(Dtype seed)
{
    query_rand_engine.seed(seed);
}

template<typename Dtype>
std::future<QuerySample<Dtype>*> ISampler<Dtype>::enqueue_job(int q_type, const Dtype* neg_ptr)
{
    return thread_pool->enqueue([=]{
        return this->gen_sample(q_type, neg_ptr);
    });
}

template<typename Dtype>
void ISampler<Dtype>::place_batch_job(int b_idx)
{
    int q_type = query_dist(query_rand_engine);
    const Dtype* neg_ptr = nullptr;
    if (share_negative) {
        for (Dtype i = b_idx * negative_sample_size; i < (b_idx + 1ULL) * negative_sample_size; ++i)
            negsample_buf[i] = this->sample_entity(this->weighted_negative_sampling, this->am_out);
        neg_ptr = negsample_buf.data() + b_idx * negative_sample_size;
    }
    size_t cur_pos = b_idx * batch_size;
    for (int i = 0; i < batch_size; ++i)
    {
        if (!same_in_batch)
            q_type = query_dist(query_rand_engine);
        if (sample_buf.size() <= cur_pos)
            sample_buf.emplace_back(enqueue_job(q_type, neg_ptr));
        else
            sample_buf[cur_pos] = enqueue_job(q_type, neg_ptr);
        cur_pos++;
    }
}

template<typename Dtype>
Dtype ISampler<Dtype>::sample_entity(bool weighted, AliasMethod* am)
{
    if (!weighted)
        return rand_int((Dtype)0, this->kg->num_ent);
    return am->draw_sample();
}

template<typename Dtype>
void ISampler<Dtype>::prefetch(int _batch_size, int num_batches)
{
    if (sample_buf.size() || negsample_buf.size())
    {
        std::cerr << "only prefetch once!" << std::endl;
        return;
    }
    if (share_negative)
        negsample_buf.resize(num_batches * negative_sample_size);

    batch_size = _batch_size;
    for (int b_idx = 0; b_idx < num_batches; ++b_idx)
        place_batch_job(b_idx);
}

template<typename Dtype>
void ISampler<Dtype>::next_batch(py::array_t<long long, py::array::c_style | py::array::forcecast> _positive_samples,
                                 py::array_t<long long, py::array::c_style | py::array::forcecast> _negative_samples,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> _sample_weights,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> _is_negative,
                                 py::list list_query, py::list list_q_idx)
{
    assert(sample_pos < sample_buf.size());
    assert(sample_pos % batch_size == 0);
    long long* positive_samples = _positive_samples.mutable_unchecked<1>().mutable_data(0);
    long long* negative_samples = nullptr;
    float* is_negative = nullptr;
    if (share_negative)
        is_negative = _is_negative.mutable_unchecked<2>().mutable_data(0, 0);
    else
        negative_samples = _negative_samples.mutable_unchecked<2>().mutable_data(0, 0);

    float* sample_weights = _sample_weights.mutable_unchecked<1>().mutable_data(0);

    int cur_batch_idx = sample_pos / batch_size;
    for (int i = 0; i < batch_size; ++i)
    {
        QuerySample<Dtype>* cur_sample = sample_buf[sample_pos].get();
        sample_pos = (sample_pos + 1ULL) % sample_buf.size();
        positive_samples[i] = cur_sample->positive_answer;

        if (!share_negative)
        {
            auto* cur_negs = negative_samples + i * negative_sample_size;
            for (Dtype j = 0; j < negative_sample_size; ++j)
                cur_negs[j] = cur_sample->negative_samples[j];
        } else {
            auto* cur_is_neg = is_negative + i * negative_sample_size;
            for (Dtype j = 0; j < negative_sample_size; ++j)
                cur_is_neg[j] = cur_sample->negative_samples[j];
        }

        sample_weights[i] = cur_sample->sample_weight;

        py::list cur_args;
        for (auto&& a : cur_sample->query_args) {
            cur_args.append(a);
        }
        list_query.append(cur_args);
        list_q_idx.append(cur_sample->q_type);
	    delete cur_sample;
    }
    if (share_negative) {
        negative_samples = _negative_samples.mutable_unchecked<2>().mutable_data(0, 0);
        auto* cur_negs = negsample_buf.data() + cur_batch_idx * negative_sample_size;
        for (Dtype i = 0; i < negative_sample_size; ++i)
            negative_samples[i] = cur_negs[i];
    }
    place_batch_job(cur_batch_idx);
}

template<typename Dtype>
Sampler<Dtype>::Sampler(KG<Dtype>* _kg, py::list _query_trees, py::list _query_prob, bool _share_negative, bool _same_in_batch,
                bool _weighted_answer_sampling, bool _weighted_negative_sampling,
                Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, Dtype _max_n_partial_answers,
                int num_threads) : ISampler<Dtype>(_kg, _query_prob, _share_negative, _same_in_batch,
                                                   _weighted_answer_sampling, _weighted_negative_sampling,
                                                   _negative_sample_size, _rel_bandwidth, _max_to_keep, 
                                                   _max_n_partial_answers, num_threads)
{
    query_trees.clear();
    for (size_t i = 0; i < py::len(_query_trees); ++i)
    {
        QueryTree<Dtype>* qt = py::cast<QueryTree<Dtype>*>(_query_trees[i]);
        query_trees.push_back(qt);
    }
}

template<typename Dtype>
void Sampler<Dtype>::print_queries()
{
    for (auto* qt : query_trees)
        std::cerr << qt->str_bracket(true) << std::endl;
}

template<typename Dtype>
Sampler<Dtype>::~Sampler()
{
    for (size_t i = 0; i < query_trees.size(); ++i)
        delete query_trees[i];
}

template<typename Dtype>
bool Sampler<Dtype>::sample_actual_query(QueryTree<Dtype>* qt, Dtype answer, bool inverse)
{
    qt->answer = answer;
    if (qt->node_type == QueryNodeType::intersect || qt->node_type == QueryNodeType::union_set)
    {
        qt->hash_code = qt->node_type;
        assert(qt->children.size() > 1u); // it is a non-trivial interesect/join
        for (auto& ch : qt->children)
        {
            assert(ch.first == QueryEdgeType::no_op);
            auto* subtree = ch.second;
            if (!sample_actual_query(subtree, answer, inverse))
                return false;
            hash_combine(qt->hash_code, subtree->hash_code);
        }

        for (size_t i = 0; i + 1 < qt->children.size(); ++i)  // assume the number of branches is small
        {
            auto& code_first = qt->children[i].second->hash_code;
            for (size_t j = i + 1; j < qt->children.size(); ++j)
                if (code_first == qt->children[j].second->hash_code)
                    return false;
        }
        return true;
    } else {
        if (qt->node_type == QueryNodeType::entity) // we have successfully instantiated the query at this branch
        {
            qt->hash_code = answer;
            return true;
        }
        assert(qt->children.size() == 1u); // it should have a single relation/negation child.
        auto& ch = qt->children[0];
        auto e_type = ch.first;
        assert(e_type != QueryEdgeType::no_op); // doesn't make sense to have no-op here

        Dtype r, prev;
        qt->hash_code = 0;
        if (e_type == QueryEdgeType::relation)
        {
            auto* edge_set = inverse ? this->kg->ent_out : this->kg->ent_in;
            if (!inverse && this->kg->in_degree(answer) == 0)
                return false;
            if (inverse && this->kg->out_degree(answer) == 0)
                return false;
            sample_rand_neighbor(edge_set, answer, r, prev);
            ch.second->parent_r = r;
            qt->hash_code = r;
        } else { // negation
            assert(e_type == QueryEdgeType::negation);
            prev = this->sample_entity(this->weighted_negative_sampling, inverse ? this->am_out : this->am_in);
            ch.second->parent_r = this->kg->num_rel;
            qt->hash_code = this->kg->num_rel;
        }
        bool ch_result = sample_actual_query(ch.second, prev, inverse);
        hash_combine(qt->hash_code, ch.second->hash_code);
        return ch_result;
    }
}

template<typename Dtype>
bool Sampler<Dtype>::verify_sampled_query(QueryTree<Dtype>* qt, Dtype answer, bool inverse)
{
    if (qt->node_type == QueryNodeType::intersect || qt->node_type == QueryNodeType::union_set)
    {
        bool junction_result = true;
        for (auto& ch : qt->children)
        {
            auto* subtree = ch.second;
            bool ch_result = verify_sampled_query(subtree, answer, inverse);
            if (qt->node_type == QueryNodeType::intersect)
                junction_result &= ch_result;
            else
                junction_result |= ch_result;
        }
        return junction_result;
    } else {
        if (qt->node_type == QueryNodeType::entity)
            return answer == qt->answer;
        auto& ch = qt->children[0];
        auto e_type = ch.first;
        auto* subtree = ch.second;
        if (e_type == QueryEdgeType::relation)
        {
            if (!inverse && !this->kg->has_forward_edge(subtree->answer, subtree->parent_r, answer))
                return false;
            if (inverse && !this->kg->has_forward_edge(answer, subtree->parent_r, subtree->answer))
                return false;
            return verify_sampled_query(subtree, subtree->answer, inverse);
        } else
            return !verify_sampled_query(subtree, answer, inverse);
    }
}

template<typename Dtype>
QueryTree<Dtype>* Sampler<Dtype>::instantiate_query(QueryTree<Dtype>* query_template)
{
    QueryTree<Dtype>* query = query_template->copy_backbone();
    while (true) {
        Dtype ans = this->sample_entity(this->weighted_answer_sampling, query->is_inverse ? this->am_out : this->am_in);
        if (sample_actual_query(query, ans, query->is_inverse) && verify_sampled_query(query, ans, query->is_inverse))
            break;
    }
    return query;
}

template<typename Dtype>
bool Sampler<Dtype>::exec_junction(QueryTree<Dtype>* cur_root)
{
    assert(cur_root->children.size() > 1u); // it is a non-trivial interesect/join
    typedef std::tuple<QueryTree<Dtype>*, const Dtype*, const Dtype*> RType;
    std::vector< RType > ch_ranges;
    for (auto& ch : cur_root->children)
    {
        if (!ch.second->num_intermediate_answers())
            continue;
        ch_ranges.push_back(std::make_tuple(ch.second, ch.second->ans_ptr_begin(), ch.second->ans_ptr_end()));
    }
    if (ch_ranges.size() == 0 || (cur_root->node_type == QueryNodeType::intersect && ch_ranges.size() != cur_root->children.size())) // no feasible solution
        return false;
    if (ch_ranges.size() == 1) {
        cur_root->set_answers(std::get<1>(ch_ranges[0]), std::get<2>(ch_ranges[0]));
        cur_root->lazy_negation = std::get<0>(ch_ranges[0])->lazy_negation;
    } else {
        std::sort(ch_ranges.begin(), ch_ranges.end(), [](const RType& x, const RType& y){
            size_t nx = std::get<2>(x) - std::get<1>(x);
            size_t ny = std::get<2>(y) - std::get<1>(y);
            bool lx = std::get<0>(x)->lazy_negation;
            bool ly = std::get<0>(y)->lazy_negation;
            return (lx < ly) || (lx == ly && nx < ny);
        });
        std::vector<Dtype> buf;
        std::vector<Dtype> *ptr_current, *ptr_next;
        if (ch_ranges.size() % 2)
        {
            ptr_current = &buf; ptr_next = &(cur_root->intermediate_answers);
        } else {
            ptr_current = &(cur_root->intermediate_answers); ptr_next = &buf;
        }

        auto op_type = cur_root->node_type;
        auto* ch_first = std::get<0>(ch_ranges[0]), *ch_second = std::get<0>(ch_ranges[1]), *ch_last = std::get<0>(ch_ranges[ch_ranges.size() - 1]);
        cur_root->lazy_negation = ch_first->lazy_negation && ch_last->lazy_negation;
        if (cur_root->lazy_negation)
            op_type = cur_root->node_type == QueryNodeType::intersect ? QueryNodeType::union_set : QueryNodeType::intersect;

        if (ch_first->lazy_negation != ch_last->lazy_negation)
            assert(op_type == QueryNodeType::intersect);
        if (ch_first->lazy_negation != ch_second->lazy_negation)
            intersect_lazy_negation(std::get<1>(ch_ranges[0]), std::get<2>(ch_ranges[0]), std::get<1>(ch_ranges[1]), std::get<2>(ch_ranges[1]), *ptr_current);
        else
            merge_sorted_list(std::get<1>(ch_ranges[0]), std::get<2>(ch_ranges[0]), std::get<1>(ch_ranges[1]), std::get<2>(ch_ranges[1]), 
                              *ptr_current, op_type);
        for (size_t i = 2; i < ch_ranges.size(); ++i)
        {
            ptr_next->clear();
            if (cur_root->lazy_negation || !std::get<0>(ch_ranges[i])->lazy_negation)
                merge_sorted_list(ptr_current->begin(), ptr_current->end(), std::get<1>(ch_ranges[i]), std::get<2>(ch_ranges[i]),
                                    *ptr_next, op_type);
            else
                intersect_lazy_negation(ptr_current->begin(), ptr_current->end(), std::get<1>(ch_ranges[i]), std::get<2>(ch_ranges[i]), *ptr_next);
            auto* tmp = ptr_current; ptr_current = ptr_next; ptr_next = tmp;
        }
        cur_root->set_answers();
    }
    return cur_root->num_intermediate_answers() > this->max_to_keep;
}


template<typename Dtype>
bool Sampler<Dtype>::exec_relation(QueryTree<Dtype>* cur_root, bool inverse)
{
    QueryTree<Dtype>* ch = cur_root->children[0].second;
    Dtype parent_r = ch->parent_r;
    auto e_type = cur_root->children[0].first;
    if ((parent_r < this->kg->num_rel && ch->num_intermediate_answers() == 0) ||
        (parent_r == this->kg->num_rel && ch->num_intermediate_answers() == this->kg->num_ent))
        return false;
    const Dtype* ptr_begin = ch->ans_ptr_begin();
    const Dtype* ptr_end = ch->ans_ptr_end();
    if (parent_r == this->kg->num_rel) // negation
    {
        assert(e_type == QueryEdgeType::negation);
        cur_root->set_answers(ptr_begin, ptr_end);
        cur_root->lazy_negation = !ch->lazy_negation;
    } else { // relation
        assert(e_type == QueryEdgeType::relation && !ch->lazy_negation);
        auto* edges = inverse ? this->kg->ent_in : this->kg->ent_out;
        if (ptr_end - ptr_begin == 1) // only one answer
        {
            auto ch_range = get_ch_range(edges, *ptr_begin, parent_r, this->rel_bandwidth);
            cur_root->set_answers(ch_range.first, ch_range.second);
        } else {
            std::unordered_set<Dtype> answers;
            for (const Dtype* ptr = ptr_begin; ptr != ptr_end; ++ptr)
            {
                Dtype cur_ans = *ptr;
                auto ch_range = get_ch_range(edges, cur_ans, parent_r, this->rel_bandwidth);
                if (ch_range.first == nullptr)
                    continue;
                answers.insert(ch_range.first, ch_range.second);
                if (answers.size() > this->max_to_keep)
                    return true;
            }
            std::copy(answers.begin(), answers.end(), std::back_inserter(cur_root->intermediate_answers));
            std::sort(cur_root->intermediate_answers.begin(), cur_root->intermediate_answers.end());
            cur_root->set_answers();
        }
    }
    return cur_root->num_intermediate_answers() > this->max_to_keep;
}

template<typename Dtype>
bool Sampler<Dtype>::dfs_answer(QueryTree<Dtype>* cur_root, bool inverse)
{
    cur_root->reset_answers();
    bool is_over_size = false;
    for (auto& ch : cur_root->children)
    {
        is_over_size |= !dfs_answer(ch.second, inverse);
        if (is_over_size)
            return false;
    }
    if (cur_root->node_type == QueryNodeType::intersect || cur_root->node_type == QueryNodeType::union_set)
        is_over_size |= exec_junction(cur_root);
    else if (cur_root->node_type == QueryNodeType::entity_set)
        is_over_size |= exec_relation(cur_root, inverse);
    else // leaf entity
        cur_root->set_answers(cur_root->answer);
    return !is_over_size;
}


template class ISampler<unsigned>;
template class ISampler<uint64_t>;

template class Sampler<unsigned>;
template class Sampler<uint64_t>;


template<typename Dtype>
std::pair<Dtype*, Dtype*> get_ch_range(SortedList<Dtype>* sl, Dtype node, Dtype r, Dtype rel_bandwidth)
{
    auto* rel_dst = sl->locate_rel_dst(node, r);
    Dtype* ptr_st = nullptr, *ptr_ed = nullptr;
    if (rel_dst == nullptr)
        return std::make_pair(ptr_st, ptr_ed);

    Dtype num = std::min(rel_dst[2] - rel_dst[1], rel_bandwidth);
    Dtype st = rand_int((Dtype)0, (Dtype)(rel_dst[2] - rel_dst[1] - num + 1));
    return std::make_pair(sl->dst_list + rel_dst[1] + st, 
                          sl->dst_list + rel_dst[1] + st + num);    
}

template std::pair<unsigned*, unsigned*> get_ch_range(SortedList<unsigned>* sl, unsigned node, unsigned r, unsigned rel_bandwidth);
template std::pair<uint64_t*, uint64_t*> get_ch_range(SortedList<uint64_t>* sl, uint64_t node, uint64_t r, uint64_t rel_bandwidth);
// NOLINTEND

