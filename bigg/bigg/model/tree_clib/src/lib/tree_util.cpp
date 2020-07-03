#include <iostream>
#include <algorithm>
#include <cassert>

#include "config.h"  // NOLINT
#include "tree_util.h"  // NOLINT
#include "struct_util.h"  // NOLINT


AdjNode::AdjNode(AdjNode* parent, int row, int col_begin, int col_end,
                 int depth)
{
    this->init(parent, row, col_begin, col_end, depth);
}

AdjNode::~AdjNode()
{
    if (this->lch != nullptr)
        delete this->lch;
    if (this->rch != nullptr)
        delete this->rch;
}

void AdjNode::init(AdjNode* parent, int row, int col_begin, int col_end,
                   int depth)
{
    this->lch = nullptr;
    this->rch = nullptr;
    this->parent = parent;
    this->row = row;
    this->col_begin = col_begin;
    this->col_end = col_end;
    this->depth = depth;
    this->mid = (col_begin + col_end) / 2;
    this->n_cols = col_end - col_begin;
    this->is_lowlevel = this->n_cols <= cfg::bits_compress;
    this->is_leaf = (this->n_cols <= 1);
    this->is_root = (this->parent == nullptr);
    if (is_lowlevel)
        this->bits_rep = BitSet(cfg::bits_compress);
    this->has_edge = false;
    this->job_idx = -1;
}

void AdjNode::update_bits()
{
    if (!is_lowlevel)
        return;
    if (is_leaf)
    {
        if (has_edge)
            bits_rep.set(0);
    } else {
        bits_rep = lch->bits_rep.left_shift(rch->n_cols);
        bits_rep = bits_rep.or_op(rch->bits_rep);
    }
}

void AdjNode::split()
{
    if (this->lch != nullptr && this->rch != nullptr)
        return;
    if (this->is_leaf)
        return;
    this->lch = node_holder.get_pt(this, row, col_begin, mid, depth + 1);
    this->rch = node_holder.get_pt(this, row, mid, col_end, depth + 1);
}

AdjRow::AdjRow(int row, int col_start, int col_end)
{
    init(row, col_start, col_end);
}

AdjRow::~AdjRow()
{
    if (this->root != nullptr)
        delete this->root;
}

void AdjRow::init(int row, int col_start, int col_end)
{
    this->row = row;
    assert(!cfg::directed);
    int max_col = row;
    if (cfg::self_loop)
        max_col += 1;
    if (col_start < 0 || col_end < 0)
    {
        col_start = 0;
        col_end = max_col;
    }
    this->root = node_holder.get_pt(nullptr, row, col_start, col_end, 0);
}


void AdjRow::insert_edges(std::vector<int>& col_indices)
{
    auto* col_sm = new ColAutomata(col_indices);
    this->add_edges(this->root, col_sm);
    delete col_sm;
}

void AdjRow::add_edges(AdjNode* node, ColAutomata* col_sm)
{
    if (node->is_root)
    {
        node->has_edge = col_sm->num_indices > 0;
        job_collect.has_ch.push_back(node->has_edge);
    } else {
        node->has_edge = true;
    }
    if (!node->has_edge)
        return;
    job_collect.append_bool(job_collect.is_internal, node->depth,
                            !(node->is_leaf));
    if (node->is_leaf) {
        col_sm->add_edge(node->col_begin);
        node->update_bits();
    } else {
        node->split();
        bool has_left = (col_sm->next_edge() < node->mid);
        if (has_left)
            this->add_edges(node->lch, col_sm);
        job_collect.append_bool(job_collect.has_left, node->depth, has_left);
        job_collect.append_bool(job_collect.num_left, node->depth,
                                node->lch->n_cols);
        bool has_right = has_left ?
            col_sm->has_edge(node->mid, node->col_end) : true;
        if (has_right)
            this->add_edges(node->rch, col_sm);
        job_collect.append_bool(job_collect.has_right, node->depth, has_right);
        job_collect.append_bool(job_collect.num_right, node->depth,
                                node->rch->n_cols);
        node->update_bits();
        node->job_idx = job_collect.add_job(node);

        int cur_idx = (int)job_collect.has_left[node->depth].size() - 1;
        auto* ch = node->lch;
        if (ch->has_edge && !ch->is_leaf && !ch->is_lowlevel)
        {
            int pos = job_collect.job_position[ch->job_idx];
            job_collect.append_bool(job_collect.next_left_froms, node->depth,
                                    pos);
            job_collect.append_bool(job_collect.next_left_tos, node->depth,
                                    cur_idx);
        } else {
            int bid = ch->has_edge ? 1 : 0;
            if (ch->has_edge && !ch->is_leaf)
                bid = 2 + job_collect.job_position[ch->job_idx];
            job_collect.append_bool(job_collect.bot_left_froms, node->depth,
                                    bid);
            job_collect.append_bool(job_collect.bot_left_tos, node->depth,
                                    cur_idx);
        }
    }
}


PtHolder<AdjNode> node_holder;
PtHolder<AdjRow> row_holder;
