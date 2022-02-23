// NOLINTBEGIN

#include <iostream>
#include <algorithm>
#include <unordered_set>

#include "utils.h"
#include "sampler.h"
#include "query.h"
#include "knowledge_graph.h"

template<typename Dtype>
class TreeDP
{
public:
    TreeDP(QueryTree<Dtype>* _qt) {
        qt = _qt;
        subtree_cost = upward_size = subtree_size = 0;
        children.clear();
    }
    ~TreeDP() {
        for (auto* ch : children)
            delete ch;
        children.clear();
    }

    QueryTree<Dtype>* qt;
    std::vector<TreeDP*> children;

    int estimate_cost(int up_size)
    {
        qt->sqrt_middle = false;
        upward_size = up_size;
        int tot_ch_cost = 0;
        for (auto& ch_pair : qt->children)
        {
            TreeDP<Dtype>* ch_dp = new TreeDP<Dtype>(ch_pair.second);
            children.push_back(ch_dp);
            int ch_cost = ch_dp->estimate_cost(up_size + (int)(ch_pair.first == QueryEdgeType::relation));
            tot_ch_cost = std::max(tot_ch_cost, ch_cost);
        }

        if (qt->node_type == QueryNodeType::entity)
        {
            subtree_size = subtree_cost = 0;
            tot_ch_cost = upward_size + 10ULL;  // hack, simply ignore tot_ch_cost
        } else if (qt->node_type == QueryNodeType::intersect || qt->node_type == QueryNodeType::union_set)
        {
            subtree_cost = 0;
            subtree_size = 0;
            int min_subtree_size = 0;
            for (auto* ch_dp : children)
            {
                subtree_cost = std::max(subtree_cost, ch_dp->subtree_cost);
                min_subtree_size = min_subtree_size == 0 ? ch_dp->subtree_size : std::min(ch_dp->subtree_size, min_subtree_size);
                subtree_size = std::max(subtree_size, ch_dp->subtree_size);
            }
            subtree_cost = std::max(subtree_cost, subtree_size);
            if (qt->node_type == QueryNodeType::intersect)
                subtree_size = min_subtree_size;
        } else {
            assert(children.size() == 1u);
            if (children[0]->qt->parent_edge == QueryEdgeType::relation)
            {
                subtree_size = children[0]->subtree_size + 1;
                subtree_cost = std::max(children[0]->subtree_size + 1, children[0]->subtree_cost);
            } else {
                subtree_size = children[0]->subtree_size;
                subtree_cost = children[0]->subtree_cost;
            }
        }
        local_best_cost = std::min(tot_ch_cost, std::max(std::max(upward_size, subtree_size), subtree_cost));
        return local_best_cost;
    }

    void setup_schedule()
    {
        int dst_side_cost = std::max(std::max(upward_size, subtree_size), subtree_cost);
        if (local_best_cost < dst_side_cost)
        {
            for (auto* ch : children) 
                ch->setup_schedule();
        } else
            qt->sqrt_middle = true;
    }

    int subtree_cost, subtree_size, upward_size, local_best_cost;
};

template<typename Dtype>
void set_best_schedule(QueryTree<Dtype>* qt)
{
    TreeDP<Dtype>* root = new TreeDP<Dtype>(qt);
    root->estimate_cost(0);
    root->setup_schedule();
    delete root;
}

template<typename Dtype>
void print_schedule(QueryTree<Dtype>* qt, int n)
{
    for (int i = 0; i < n; ++i)
        std::cerr << " ";
    std::cerr << qt->node_type << " " << qt->sqrt_middle << std::endl;
    for (auto& ch : qt->children)
        print_schedule(ch.second, n + 2);
}

template<typename Dtype>
bool RejectionSampler<Dtype>::bottomup_search(QueryTree<Dtype>* qt, bool inverse)
{
    bool succ = true;
    if (qt->sqrt_middle)
        succ &= this->dfs_answer(qt, inverse);
    else {
        for (auto& ch : qt->children)
            succ &= bottomup_search(ch.second, inverse);
    }
    return succ;
}

template<typename Dtype>
int RejectionSampler<Dtype>::check_neg_depth(QueryTree<Dtype>* qt, bool& false_pos_flag)
{
    if (qt->node_type == QueryNodeType::entity_set)
    {
        auto& ch = qt->children[0];
        int d = 0;
        if (ch.first == QueryEdgeType::relation)
            d += 1;
        d += check_neg_depth(ch.second, false_pos_flag);
        if (d >= 2 && ch.first == QueryEdgeType::negation)
            false_pos_flag = true;
        return ch.first == QueryEdgeType::negation ? 0 : d;
    } else {
        for (auto& ch : qt->children)
            check_neg_depth(ch.second, false_pos_flag);
        return 0;
    }
}

template<typename Dtype>
RejectionSampler<Dtype>::RejectionSampler(KG<Dtype>* _kg, py::list _query_trees, py::list _query_prob, bool _share_negative, bool _same_in_batch,
                                          bool _weighted_answer_sampling, bool _weighted_negative_sampling,
                                          Dtype _negative_sample_size, Dtype _rel_bandwidth, Dtype _max_to_keep, 
                                          Dtype _max_n_partial_answers, int num_threads, py::list no_search_list) : Sampler<Dtype>(_kg, _query_trees, _query_prob, _share_negative, _same_in_batch,
                                                                        _weighted_answer_sampling, _weighted_negative_sampling,
                                                                        _negative_sample_size, _rel_bandwidth, _max_to_keep, _max_n_partial_answers, num_threads)
{
    has_false_pos.clear();
    for (auto* qt : this->query_trees)
    {
        set_best_schedule(qt);
        bool false_pos_flag = false;
        check_neg_depth(qt, false_pos_flag);
        has_false_pos.push_back(false_pos_flag);
    }
    avg_sample_weight = -1;
    no_search_set.clear();
    for (size_t i = 0; i < py::len(no_search_list); ++i)
        no_search_set.insert(py::cast<int>(no_search_list[i]));
}

template<typename Dtype>
void RejectionSampler<Dtype>::print_queries()
{
    for (size_t i = 0; i < this->query_trees.size(); ++i)
    {
        auto* qt = this->query_trees[i];
        if (no_search_set.count(i))
            std::cerr << "not searching:" << std::endl;
        std::cerr << qt->str_bracket(true) << std::endl;
        print_schedule(qt, 0);
    }
}

template<typename Dtype>
Dtype RejectionSampler<Dtype>::negative_sampling(QuerySample<Dtype>* sample, QueryTree<Dtype>* qt, const Dtype* list_neg_candidates)
{
    bool no_search = no_search_set.count(sample->q_type);
    Dtype num_pos = no_search ? 0 : is_positive(qt, qt->answer);

    sample->negative_samples.clear();
    Dtype neg_ans;
    for (Dtype i = 0; i < this->negative_sample_size; ++i)
    {
        if (list_neg_candidates != nullptr)
        {
            bool is_negative = no_search || !is_positive(qt, list_neg_candidates[i]);
            if (!is_negative) {
                if (num_pos == 0)
                    qt->answer = list_neg_candidates[i];
                num_pos++;
            }
            sample->negative_samples.push_back(is_negative);
            continue;
        }
        while (true) {
            neg_ans = this->sample_entity(this->weighted_negative_sampling, 
                                          qt->is_inverse ? this->am_out : this->am_in);
            if ((!no_search) && is_positive(qt, neg_ans))
            {
                if (num_pos == 0)
                    qt->answer = neg_ans;
                num_pos++;
                continue;
            }
            break;
        }
        sample->negative_samples.push_back(neg_ans);
    }
    return num_pos;
}

template<typename Dtype>
bool RejectionSampler<Dtype>::is_positive(QueryTree<Dtype>* qt, Dtype candidate)
{
    if (qt->sqrt_middle)
        return std::binary_search(qt->ans_ptr_begin(), qt->ans_ptr_end(), candidate);
    std::vector<Dtype> proof_targets{candidate};
    qt->reset_answers();
    qt->set_answers(proof_targets.data(), proof_targets.data() + 1);
    return is_positive(qt, qt->is_inverse);
}

template<typename Dtype>
bool RejectionSampler<Dtype>::is_positive(QueryTree<Dtype>* qt, bool inverse)
{
    assert(!qt->sqrt_middle);
    if (qt->num_intermediate_answers() == 0)
        return false;
    if (qt->node_type == QueryNodeType::intersect || qt->node_type == QueryNodeType::union_set)
    {
        QueryNodeType qtype = qt->node_type; 
        std::vector<Dtype> buf, vec_tmp;
        std::vector<Dtype> *ptr_current, *ptr_next;
        if (qt->children.size() % 2)
        {
            ptr_current = &buf; ptr_next = &(qt->intermediate_answers);
        } else {
            ptr_current = &(qt->intermediate_answers); ptr_next = &buf;
        }
        for (size_t i = 0; i < qt->children.size(); ++i)
        {
            auto* ch = qt->children[i].second;
            if (!ch->sqrt_middle)
            {
                ch->reset_answers();
                ch->set_answers(qt->ans_ptr_begin(), qt->ans_ptr_end());
                is_positive(ch, inverse);
            }
            ptr_next->clear();
            if (i == 0) {
                if (!ch->lazy_negation)
                    std::copy(ch->ans_ptr_begin(), ch->ans_ptr_end(), std::back_inserter(*ptr_next));
                else
                    std::set_difference(qt->ans_ptr_begin(), qt->ans_ptr_end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), std::back_inserter(*ptr_next));
            } else {
                if (ch->lazy_negation)
                {
                    if (qtype == QueryNodeType::intersect)
                        intersect_lazy_negation(ptr_current->begin(), ptr_current->end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), *ptr_next);
                    else {
                        vec_tmp.clear();
                        std::set_difference(qt->ans_ptr_begin(), qt->ans_ptr_end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), std::back_inserter(vec_tmp));
                        merge_sorted_list(ptr_current->begin(), ptr_current->end(), vec_tmp.begin(), vec_tmp.end(), *ptr_next, qtype);
                    }
                } else
                    merge_sorted_list(ptr_current->begin(), ptr_current->end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), *ptr_next, qtype);
            }
            auto* tmp = ptr_current; ptr_current = ptr_next; ptr_next = tmp;
        }
        qt->set_answers();
        return qt->num_intermediate_answers();
    } else {
        assert(qt->node_type == QueryNodeType::entity_set && qt->children.size() == 1u);
        QueryTree<Dtype>* ch = qt->children[0].second;
        Dtype& parent_r = ch->parent_r;
        auto e_type = qt->children[0].first;
        const Dtype* ptr_begin = qt->ans_ptr_begin();
        const Dtype* ptr_end = qt->ans_ptr_end();
        auto* edges = inverse ? this->kg->ent_out : this->kg->ent_in;

        if (!ch->sqrt_middle)
        {
            ch->reset_answers();
            if (e_type == QueryEdgeType::negation)
                ch->set_answers(ptr_begin, ptr_end);
            else {
                if (ptr_end - ptr_begin == 1)
                {
                    auto ch_range = get_ch_range(edges, *ptr_begin, parent_r, this->rel_bandwidth);
                    ch->set_answers(ch_range.first, ch_range.second);
                } else {
                    std::unordered_set<Dtype> answers;
                    for (const Dtype* ptr = ptr_begin; ptr != ptr_end; ++ptr)
                    {
                        Dtype cur_ans = *ptr;
                        auto ch_range = get_ch_range(edges, cur_ans, parent_r, this->rel_bandwidth);
                        if (ch_range.first == nullptr)
                            continue;
                        answers.insert(ch_range.first, ch_range.second);
                    }
                    std::copy(answers.begin(), answers.end(), std::back_inserter(qt->intermediate_answers));
                    std::sort(qt->intermediate_answers.begin(), qt->intermediate_answers.end());
                    ch->set_answers(qt->intermediate_answers.data(), qt->intermediate_answers.data() + qt->intermediate_answers.size());
                }
            }
            is_positive(ch, inverse);
            qt->intermediate_answers.clear();
        }

        if (e_type == QueryEdgeType::negation)
        {
            if (ch->lazy_negation)
                std::set_intersection(qt->ans_ptr_begin(), qt->ans_ptr_end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), std::back_inserter(qt->intermediate_answers));
            else
                std::set_difference(qt->ans_ptr_begin(), qt->ans_ptr_end(), ch->ans_ptr_begin(), ch->ans_ptr_end(), std::back_inserter(qt->intermediate_answers));
        } else {
            assert(!ch->lazy_negation); // too complicated
            for (const Dtype* ptr = ptr_begin; ptr != ptr_end; ++ptr)
            {
                Dtype cur_ans = *ptr;
                auto ch_range = get_ch_range(edges, cur_ans, parent_r, this->rel_bandwidth);
                if (ch_range.first == nullptr)
                    continue;
                if (has_set_intersect(ch_range.first, ch_range.second, ch->ans_ptr_begin(), ch->ans_ptr_end()))
                    qt->intermediate_answers.push_back(cur_ans);
            }
        }
        qt->set_answers();
        return qt->num_intermediate_answers();
    }
}

template<typename Dtype>
QuerySample<Dtype>* RejectionSampler<Dtype>::gen_sample(int query_type, const Dtype* list_neg_candidates)
{
    while (true)
    {
        QueryTree<Dtype>* qt = this->instantiate_query(this->query_trees[query_type]);
        if ((!no_search_set.count(query_type)) && (!bottomup_search(qt, qt->is_inverse))) // oversize
        {
            delete qt;
            continue;
        }

        QuerySample<Dtype>* sample = new QuerySample<Dtype>(query_type);
        Dtype num_pos = negative_sampling(sample, qt, list_neg_candidates);

        bool too_many_answers = (qt->sqrt_middle && qt->num_intermediate_answers() > this->max_n_partial_answers);
        if (num_pos > this->max_n_partial_answers || too_many_answers)
        {
            delete qt;
            delete sample;
            continue;
        }
        double weight;
        if (qt->sqrt_middle)
        {
            weight = 1.0 / sqrt(qt->num_intermediate_answers() + 1.0);
            if (avg_sample_weight < 0)
                avg_sample_weight = weight;
            else
                avg_sample_weight = 0.9 * avg_sample_weight + 0.1 * weight;
        } else
            weight = avg_sample_weight < 0 ? 1.0 : avg_sample_weight;

        sample->query_args.clear();
        qt->get_query_args(sample->query_args);
        sample->positive_answer = qt->answer;
        sample->sample_weight = weight;
        delete qt;
        return sample;
    }
}

template class RejectionSampler<unsigned>;
template class RejectionSampler<uint64_t>;
// NOLINTEND
