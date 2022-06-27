// NOLINTBEGIN

#include "query.h"

#include <iostream>
#include <cassert>

template<typename Dtype>
QueryTree<Dtype>::QueryTree(QueryNodeType _node_type) :  node_type(_node_type), parent_r(0), answer(0)
{
    parent_edge = QueryEdgeType::no_op;
    lazy_negation = false;
    is_inverse = false;
    children.clear();
}

template<typename Dtype>
void QueryTree<Dtype>::add_child(QueryEdgeType edge_type, QueryTree<Dtype>* child)
{
    assert(child->parent_edge == QueryEdgeType::no_op);
    child->parent_edge = edge_type;
    children.push_back(std::make_pair(edge_type, child));
}

template<typename Dtype>
void QueryTree<Dtype>::get_children(py::list ch_list)
{
    for (auto& ch : children)
        ch_list.append(ch.second);
}

template<typename Dtype>
QueryTree<Dtype>::~QueryTree()
{
    for (auto& c : children)
        delete c.second;
    children.clear();
}

template<typename Dtype>
QueryTree<Dtype>* QueryTree<Dtype>::copy_backbone()
{
    QueryTree<Dtype>* cur_node = new QueryTree<Dtype>(this->node_type);
    cur_node->sqrt_middle = this->sqrt_middle;
    cur_node->is_inverse = this->is_inverse;
    for (auto& c : children)
    {
        QueryTree<Dtype>* ch_clone = c.second->copy_backbone();
        cur_node->add_child(c.first, ch_clone);
    }
    return cur_node;
}

template<typename Dtype>
QueryTree<Dtype>* QueryTree<Dtype>::copy_instantiated_query()
{
    QueryTree<Dtype>* cur_node = new QueryTree<Dtype>(this->node_type);
    cur_node->sqrt_middle = this->sqrt_middle;
    cur_node->lazy_negation = false;
    cur_node->answer = this->answer;
    cur_node->parent_r = this->parent_r;
    for (auto& c : children)
    {
        QueryTree<Dtype>* ch_clone = c.second->copy_instantiated_query();
        cur_node->add_child(c.first, ch_clone);
    }
    return cur_node;
}

template<typename Dtype>
bool QueryTree<Dtype>::has_negation()
{
    for (auto& c : children)
    {
        if (c.first == QueryEdgeType::negation)
            return true;
        if (c.second->has_negation())
            return true;
    }
    return false;
}

template<typename Dtype>
void QueryTree<Dtype>::get_query_args(std::vector<int>& query_args)
{
    if (node_type == QueryNodeType::entity)
        query_args.push_back(answer);
    else if (node_type == QueryNodeType::entity_set) {
        auto& ch = this->children[0];
        ch.second->get_query_args(query_args);
        if (ch.first == QueryEdgeType::negation)
            query_args.push_back(-2);
        else
            query_args.push_back(ch.second->parent_r);
    } else {
        for (auto& ch : this->children)
            ch.second->get_query_args(query_args);
        if (node_type == QueryNodeType::union_set)
            query_args.push_back(-1);
    }    
}

template<typename Dtype>
void QueryTree<Dtype>::set_answers(Dtype answer)
{
    intermediate_answers.clear();
    intermediate_answers.push_back(answer);
    _ans_ptr_begin = intermediate_answers.data();
    _ans_ptr_end = _ans_ptr_begin + 1;
}

template<typename Dtype>
void QueryTree<Dtype>::set_answers()
{
    _ans_ptr_begin = intermediate_answers.data();
    _ans_ptr_end = _ans_ptr_begin + intermediate_answers.size();
}

template<typename Dtype>
void QueryTree<Dtype>::set_answers(const Dtype* ptr_begin, const Dtype* ptr_end)
{
    _ans_ptr_begin = ptr_begin;
    _ans_ptr_end = ptr_end;
}

template<typename Dtype>
const Dtype* QueryTree<Dtype>::ans_ptr_begin() const
{
    return _ans_ptr_begin;
}

template<typename Dtype>
const Dtype* QueryTree<Dtype>::ans_ptr_end() const
{
    return _ans_ptr_end;
}

template<typename Dtype>
Dtype QueryTree<Dtype>::num_intermediate_answers()
{
    if (_ans_ptr_end == nullptr || _ans_ptr_begin == _ans_ptr_end)
        return 0;
    return (Dtype)(_ans_ptr_end - _ans_ptr_begin);
}

template<typename Dtype>
void QueryTree<Dtype>::reset_answers()
{
    lazy_negation = false;
    intermediate_answers.clear();
    _ans_ptr_begin = _ans_ptr_end = nullptr;
}

QueryTree<unsigned>* create_qt32(QueryNodeType node_type)
{
    return new QueryTree<unsigned>(node_type);
}

QueryTree<uint64_t>* create_qt64(QueryNodeType node_type)
{
    return new QueryTree<uint64_t>(node_type);
}

template<typename Dtype>
std::string QueryTree<Dtype>::str_bracket(bool backbone_only) const
{
    if (node_type == QueryNodeType::entity)
        return backbone_only ? "e" : "e" + std::to_string(answer);
    
    std::string st = "";
    for (size_t i = 0; i < children.size(); ++i) {
        auto& ch = children[i];
        std::string sub = ch.second->str_bracket(backbone_only);
        if (ch.first != QueryEdgeType::no_op)
        {
            if (ch.first == QueryEdgeType::negation)
                st += "n(" + sub + ")";
            else {
                st += backbone_only ? "r" : "r" + std::to_string(ch.second->parent_r);
                st += "(" + sub + ")";
            }
        } else 
            st += sub;
        if (i + 1 < children.size())
            st += ",";
    }
    if (node_type == QueryNodeType::intersect)
        st = "i(" + st + ")";
    if (node_type == QueryNodeType::union_set)
        st = "u(" + st + ")";
    return st;
}

template class QueryTree<unsigned>;
template class QueryTree<uint64_t>;

template<typename Dtype>
QuerySample<Dtype>::QuerySample(int _q_type) : q_type(_q_type)
{
    negative_samples.clear();
    query_args.clear();
}

template<typename Dtype>
QuerySample<Dtype>::~QuerySample()
{

}

template class QuerySample<unsigned>;
template class QuerySample<uint64_t>;


template<typename Dtype>
TestQuerySample<Dtype>::TestQuerySample(int _q_type) : QuerySample<Dtype>(_q_type)
{
    true_positive.clear();
    false_positive.clear();
    false_negative.clear();
}

template<typename Dtype>
TestQuerySample<Dtype>::~TestQuerySample()
{

}

template class TestQuerySample<unsigned>;
template class TestQuerySample<uint64_t>;
// NOLINTEND
