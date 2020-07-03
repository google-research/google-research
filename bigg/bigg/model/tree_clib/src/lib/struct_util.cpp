#include <algorithm>
#include <iostream>
#include <random>
#include <queue>
#include <unordered_set>
#include <cassert>

#include "config.h"  // NOLINT
#include "struct_util.h"  // NOLINT
#include "tree_util.h"  // NOLINT

std::vector<GraphStruct*> graph_list = std::vector<GraphStruct*>();
std::vector<GraphStruct*> active_graphs;

BitSet::BitSet()
{
    n_bits = n_macros = 0;
    macro_bits.clear();
}

BitSet::BitSet(uint32_t _n_bits)
{
    n_bits = _n_bits;
    n_macros = n_bits / ibits;
    if (n_bits % ibits)
        n_macros++;
    macro_bits.clear();
    for (uint32_t i = 0; i < n_macros; ++i)
        macro_bits.push_back(0);
}

void BitSet::set(uint32_t pos)
{
    uint32_t m = pos / ibits;
    pos = pos % ibits;
    macro_bits[m] |= (1 << pos);
}

bool BitSet::get(uint32_t pos)
{
    uint32_t m = pos / ibits;
    pos = pos % ibits;
    return (macro_bits[m] & ((uint32_t)1 << pos)) > 0;
}

BitSet BitSet::left_shift(uint32_t n)
{
    BitSet bs(n_bits);
    uint32_t prev = 0;
    uint32_t topn;
    uint32_t block_shift = n / ibits;
    n = n % ibits;
    for (uint32_t i = 0; i < block_shift; ++i)
        bs.macro_bits[i] = 0;
    for (uint32_t i = block_shift; i < n_macros; ++i)
    {
        topn = n ? macro_bits[i - block_shift] >> (ibits - n) : 0;
        bs.macro_bits[i] = (macro_bits[i - block_shift] << n) + prev;
        prev = topn;
    }
    return bs;
}

BitSet BitSet::or_op(BitSet& another)
{
    BitSet bs(n_bits);
    for (uint32_t i = 0; i < n_macros; ++i)
    {
        bs.macro_bits[i] = macro_bits[i] | another.macro_bits[i];
    }
    return bs;
}

int num_ones(int n)
{
    int cnt = 0;
    while (n) {
        n &= (n - 1);
        cnt += 1;
    }
    return cnt;
}

GraphStruct::GraphStruct(int graph_id, int num_nodes, int num_edges,
                         void* _edge_pairs, int n_left, int n_right)
{
    this->num_nodes = num_nodes;
    this->num_edges = num_edges;
    this->graph_id = graph_id;

    edge_list.clear();
    active_rows.clear();
    idx_map.clear();

    if (_edge_pairs == nullptr)
        return;
    int* edge_pairs = static_cast<int*>(_edge_pairs);
    for (int i = 0; i < num_edges; ++i)
    {
        int x = edge_pairs[i * 2];
        int y = edge_pairs[i * 2 + 1];
        if (n_left < 0 || n_right < 0)
        {
            if (x < y)
            {
                int t = x; x = y; y = t;
            }
        } else {
            if (x > y)
            {
                int t = x; x = y; y = t;
            }
            assert(x < n_left);
            y -= n_left;
            assert(y >= 0 && y < n_right);
        }
        if (!edge_list.count(x))
            edge_list[x] = std::vector<int>();
        edge_list[x].push_back(y);
    }

    for (auto it = edge_list.begin(); it != edge_list.end(); ++it)
        std::sort(it->second.begin(), it->second.end());
}


GraphStruct* GraphStruct::permute()
{
    if (!cfg::bfs_permute)
        return this;
    if ((int)idx_map.size() != num_nodes)
    {
        idx_map.resize(num_nodes);
    }
    for (int i = 0; i < num_nodes; ++i)
        idx_map[i] = i;
    std::random_shuffle(idx_map.begin(), idx_map.end());
    std::unordered_set<int> h;
    std::queue<int> q;

    GraphStruct* g = new GraphStruct(graph_id, num_nodes, num_edges);
    g->idx_map.resize(num_nodes);

    int t = 0;
    for (int i = 0; i < num_nodes; ++i)
    {
        int src = idx_map[i];
        if (h.count(src))
            continue;
        q.push(src);
        h.insert(src);

        while (!q.empty())
        {
            int cur_node = q.front();
            q.pop();
            g->idx_map[t++] = cur_node;

            for (auto y : edge_list[cur_node])
            {
                y = idx_map[y];
                if (h.count(y))
                    continue;
                h.insert(y);
                q.push(y);
            }
        }
    }
    assert(t == num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        idx_map[g->idx_map[i]] = i;

    for (int i = 0; i < num_nodes; ++i)
    {
        for (auto y : edge_list[i])
        {
            y = idx_map[y];
            int x = idx_map[i];
            if (x < y)
            {
                int t = x; x = y; y = t;
            }
            if (!g->edge_list.count(x))
                g->edge_list[x] = std::vector<int>();
            g->edge_list[x].push_back(y);
        }
    }
    for (auto it = g->edge_list.begin(); it != g->edge_list.end(); ++it)
        std::sort(it->second.begin(), it->second.end());
    return g;
}


void GraphStruct::realize_nodes(int node_start, int node_end, int col_start,
                                int col_end)
{
    active_rows.clear();
    for (int i = node_start; i < node_end; ++i)
        active_rows.push_back(row_holder.get_pt(i, col_start, col_end));

    for (int i = node_start; i < node_end; ++i)
    {
        auto* row = active_rows[i - node_start];
        row->insert_edges(edge_list[i]);
    }
    this->node_start = node_start;
    this->node_end = node_end;
}


ColAutomata::ColAutomata(std::vector<int>& _indices)
{
    this->indices = _indices.data();
    this->pos = 0;
    this->num_indices = (int)_indices.size();
}

void ColAutomata::add_edge(int col_idx)
{
    assert(this->pos < this->num_indices);
    assert(this->indices[this->pos] == col_idx);
    this->pos += 1;
}

int ColAutomata::next_edge()
{
    if (this->pos < this->num_indices)
        return this->indices[this->pos];
    return -1;
}

int ColAutomata::last_edge()
{
    return this->indices[this->num_indices - 1];
}

bool ColAutomata::has_edge(int range_start, int range_end)
{
    for (int i = pos; i < this->num_indices; ++i)
    {
        if (this->indices[i] >= range_start && this->indices[i] < range_end)
            return true;
        if (this->indices[i] >= range_end)
            break;
    }
    return false;
}


template<typename PtType>
PtHolder<PtType>::PtHolder()
{
    pt_buff.clear();
    cur_pos = 0;
}

template<typename PtType>
void PtHolder<PtType>::reset()
{
    cur_pos = 0;
}

template<typename PtType>
void PtHolder<PtType>::clear()
{
    for (auto* pt : pt_buff)
        delete pt;
    pt_buff.clear();
    cur_pos = 0;
}

template class PtHolder<AdjNode>;
template class PtHolder<AdjRow>;


JobCollect::JobCollect()
{
    reset();
}

void JobCollect::reset()
{
    global_job_nodes.clear();
    job_position.clear();
    n_bin_job_per_level.clear();
    n_cell_job_per_level.clear();
    binary_feat_nodes.clear();
    has_ch.clear();
    has_left.clear();
    has_right.clear();
    num_left.clear();
    num_right.clear();
    is_internal.clear();
    bot_left_froms.clear();
    bot_left_tos.clear();
    next_left_froms.clear();
    next_left_tos.clear();
    for (int i = 0; i < 2; ++i)
    {
        bot_froms[i].clear();
        bot_tos[i].clear();
        prev_froms[i].clear();
        prev_tos[i].clear();
    }
}

int JobCollect::add_job(AdjNode* node)
{
    int job_id = global_job_nodes.size();
    int cur_depth = node->depth;
    std::vector<int>* _vpt = nullptr;
    _vpt = node->is_lowlevel ? &n_bin_job_per_level : &n_cell_job_per_level;
    auto& njob_per_level = *_vpt;
    if (cur_depth >= (int)njob_per_level.size())
    {
        for (int i = njob_per_level.size(); i <= cur_depth; ++i)
        {
            njob_per_level.push_back(0);
            binary_feat_nodes.push_back(std::vector<AdjNode*>());
            if (node->is_lowlevel)
                continue;
            for (int j = 0; j < 2; ++j)
            {
                bot_froms[j].push_back(std::vector<int>());
                bot_tos[j].push_back(std::vector<int>());
                prev_froms[j].push_back(std::vector<int>());
                prev_tos[j].push_back(std::vector<int>());
            }
        }
    }
    int job_pos = njob_per_level[cur_depth];
    job_position.push_back(job_pos);
    njob_per_level[cur_depth]++;
    global_job_nodes.push_back(node);
    if (node->is_lowlevel) {
        binary_feat_nodes[cur_depth].push_back(node);
    } else {
        for (int i = 0; i < 2; ++i)
        {
            auto* ch = (i == 0) ? node->lch : node->rch;
            if (ch->has_edge && !ch->is_leaf && !ch->is_lowlevel)
            {
                prev_froms[i][cur_depth].push_back(job_position[ch->job_idx]);
                prev_tos[i][cur_depth].push_back(job_pos);
            } else {
                int bid;
                if (ch->is_leaf || !ch->has_edge)
                    bid = ch->has_edge ? 1 : 0;
                else
                    bid = 2 + job_position[ch->job_idx];
                bot_froms[i][cur_depth].push_back(bid);
                bot_tos[i][cur_depth].push_back(job_pos);
            }
        }
    }
    return job_id;
}

void JobCollect::append_bool(std::vector< std::vector<int> >& list,
                             int depth, int val)
{
    while (depth >= (int)list.size())
        list.push_back(std::vector<int>());
    list[depth].push_back(val);
}

void JobCollect::build_row_indices()
{
    for (int i = 0; i < 2; ++i)
    {
        row_bot_froms[i].clear();
        row_bot_tos[i].clear();
        row_top_froms[i].clear();
        row_top_tos[i].clear();
        row_prev_froms[i].clear();
        row_prev_tos[i].clear();
        row_top_froms[i].push_back(std::vector<int>());
        row_top_tos[i].push_back(std::vector<int>());
        row_prev_froms[i].push_back(std::vector<int>());
        row_prev_tos[i].push_back(std::vector<int>());
    }

    // build lv=1
    int offset = 0;
    int prev_offset = 0;
    layer_sizes.clear();
    bool has_next = false;
    std::vector<int> used_cnts;
    used_cnts.clear();
    for (size_t i = 0; i < active_graphs.size(); ++i)
    {
        used_cnts.push_back(0);
        auto* g = active_graphs[i];
        layer_sizes.push_back(g->active_rows.size());
        int prev_correct = g->node_start & 1;
        used_cnts[i] += prev_correct;
        int ub = (g->active_rows.size() + prev_correct) / 2;
        for (int j = 0; j < ub; ++j)
            for (int k = 0; k < 2; ++k)
            {
                int row_pos = j * 2 + k - prev_correct;
                if (row_pos < 0) {  // from prev state
                    row_prev_froms[k][0].push_back(prev_offset);
                    row_prev_tos[k][0].push_back(j + offset);
                    continue;
                }
                auto* root = g->active_rows[row_pos]->root;
                if (root->has_edge && !root->is_leaf && !root->is_lowlevel)
                {
                    row_top_froms[k][0].push_back(job_position[root->job_idx]);
                    row_top_tos[k][0].push_back(j + offset);
                } else {
                    int bid = root->has_edge ? 1 : 0;
                    if (root->has_edge && !root->is_leaf)
                        bid = 2 + job_position[root->job_idx];
                    row_bot_froms[k].push_back(bid);
                    row_bot_tos[k].push_back(j + offset);
                }
            }
        offset += ub;
        layer_sizes[i] = ub;
        if (ub + (g->node_start & (1 << 1)) > 1)
            has_next = true;
        prev_offset += num_ones(g->node_start);
    }
    int lv = 1;
    while (has_next)
    {
        has_next = false;
        for (int i = 0; i < 2; ++i)
        {
            row_top_froms[i].push_back(std::vector<int>());
            row_top_tos[i].push_back(std::vector<int>());
            row_prev_froms[i].push_back(std::vector<int>());
            row_prev_tos[i].push_back(std::vector<int>());
        }
        int old_offset = 0;
        prev_offset = 0;
        offset = 0;
        for (size_t i = 0; i < active_graphs.size(); ++i)
        {
            auto* g = active_graphs[i];
            int prev_correct = (g->node_start & (1 << lv)) > 0;
            int ub = (layer_sizes[i] + prev_correct) / 2;
            for (int j = 0; j < ub; ++j)
                for (int k = 0; k < 2; ++k)
                {
                    int row_pos = j * 2 + k - prev_correct + old_offset;
                    if (row_pos < old_offset)
                    {
                        row_prev_froms[k][lv].push_back(prev_offset +
                                                        used_cnts[i]);
                        row_prev_tos[k][lv].push_back(offset + j);
                        continue;
                    }
                    row_top_froms[k][lv].push_back(row_pos);
                    row_top_tos[k][lv].push_back(offset + j);
                }
            used_cnts[i] += prev_correct;
            old_offset += layer_sizes[i];
            layer_sizes[i] = ub;
            offset += ub;
            if (ub + (g->node_start & (1 << (lv + 1))) > 1)
                has_next = true;
            prev_offset += num_ones(g->node_start);
        }
        lv++;
    }
    max_row_merge_steps = lv;
}

void JobCollect::build_row_summary()
{
    layer_sizes.clear();
    tree_idx_map.clear();
    step_inputs.clear();
    step_tos.clear();
    step_indices.clear();
    step_froms.clear();
    step_nexts.clear();
    int tot_past = 0;
    std::vector<int> used_cnts, past_cnts;
    used_cnts.clear();
    past_cnts.clear();
    for (size_t i = 0; i < active_graphs.size(); ++i)
    {
        auto* g = active_graphs[i];
        layer_sizes.push_back(g->active_rows.size());
        tree_idx_map.push_back(std::unordered_map<int, int>());
        used_cnts.push_back(0);
        past_cnts.push_back(num_ones(g->node_start));
        tot_past += past_cnts[i];
    }

    bool has_job = true;
    int layer = 0;
    int global_offset = 3;
    if (cfg::bits_compress && n_bin_job_per_level.size())
        global_offset += n_bin_job_per_level[0];
    int past_start_offset = global_offset;
    global_offset += tot_past;

    while (has_job)
    {
        has_job = false;
        int past_offset = past_start_offset;
        for (size_t i = 0; i < layer_sizes.size(); ++i)
        {
            auto* g = active_graphs[i];
            int num_rows = (int)g->active_rows.size(), cnt = 0;
            int cur_bit = (g->node_start & (1 << layer)) > 0;
            if (layer == 0)
            {
                for (int j = 0; j < layer_sizes[i]; ++j)
                {
                    auto* root = g->active_rows[j]->root;
                    if (root->has_edge && !root->is_leaf && !root->is_lowlevel)
                    {
                        tree_idx_map[i][layer * num_rows + j + cur_bit] = cnt + global_offset;  // NOLINT
                        cnt += 1;
                    } else {
                        int bid = root->has_edge ? 2 : 1;
                        if (root->has_edge && !root->is_leaf)
                            bid = 3 + job_position[root->job_idx];
                        tree_idx_map[i][layer * num_rows + j + cur_bit] = bid;
                    }
                }
            } else {
                for (int j = 0; j < layer_sizes[i]; ++j)
                    tree_idx_map[i][layer * num_rows + j + cur_bit] = (global_offset + j);  // NOLINT
                cnt = layer_sizes[i];
            }
            if (cur_bit) {
                tree_idx_map[i][layer * num_rows] = past_offset + used_cnts[i];
                used_cnts[i] += 1;
            }
            global_offset += cnt;
            past_offset += num_ones(g->node_start);
        }
        for (size_t i = 0; i < layer_sizes.size(); ++i)
        {
            auto* g = active_graphs[i];
            int bit = (g->node_start & (1 << layer)) > 0;
            layer_sizes[i] = (layer_sizes[i] + bit) / 2;
            if (layer_sizes[i] || used_cnts[i] != past_cnts[i])
                has_job = true;
        }
        layer += 1;
        step_inputs.push_back(std::vector<int>());
        step_indices.push_back(std::vector<int>());
    }

    global_offset = 0;
    max_rowsum_steps = 0;
    next_state_froms.clear();
    for (size_t i = 0; i < active_graphs.size(); ++i)
    {
        auto* g = active_graphs[i];
        int num_nodes = (int)g->active_rows.size();
        for (int j = 0; j < num_nodes + 1; ++j)
        {
            int k = j + g->node_start;
            if (k == 0)
            {
                step_inputs[0].push_back(0);
                step_indices[0].push_back(global_offset);
                if (max_rowsum_steps == 0)
                    max_rowsum_steps = 1;
                continue;
            }
            int layer = 0, cur_bit, src, step = 0;
            while (k)
            {
                cur_bit = k & 1;
                k /= 2;
                if (cur_bit)
                {
                    int num_prev = g->node_start / (1 << layer);
                    int prev_bit = (g->node_start & (1 << layer)) > 0;
                    int pos = 2 * k - num_prev + prev_bit;
                    assert(pos >= 0);
                    src = tree_idx_map[i][layer * num_nodes + pos];
                    if (j < num_nodes)
                    {
                        assert(step < (int)step_inputs.size());
                        step_inputs[step].push_back(src);
                        step_indices[step].push_back(global_offset + j);
                        step += 1;
                    } else {
                        next_state_froms.push_back(src);
                    }
                }
                layer += 1;
            }
            if (step > max_rowsum_steps)
                max_rowsum_steps = step;
        }
        global_offset += num_nodes;
    }

    for (int i = 0; i + 1 < max_rowsum_steps; ++i)
    {
        step_tos.push_back(std::vector<int>());
        step_nexts.push_back(std::vector<int>());
        step_froms.push_back(std::vector<int>());

        size_t y = 0;
        auto& prev_list = step_indices[i];
        auto& cur_list = step_indices[i + 1];
        for (size_t x = 0; x < prev_list.size(); ++x)
        {
            if (prev_list[x] < cur_list[y] || y >= cur_list.size())
            {
                step_froms[i].push_back(x);
                step_tos[i].push_back(prev_list[x]);
            } else {
                assert(prev_list[x] == cur_list[y]);
                step_nexts[i].push_back(x);
                y += 1;
            }
        }
    }
    max_rowsum_steps -= 1;
}

JobCollect job_collect;
