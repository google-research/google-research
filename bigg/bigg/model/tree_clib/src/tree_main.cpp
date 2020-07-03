#include <signal.h>
#include <random>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

#include "config.h"  // NOLINT
#include "tree_clib.h"  // NOLINT
#include "tree_util.h"  // NOLINT
#include "cuda_ops.h"  // NOLINT

int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}

int TotalTreeNodes()
{
    return 0;
}

int MaxBinFeatDepth()
{
    return (int)job_collect.n_bin_job_per_level.size();
}

int NumBinNodes(int depth)
{
    if (depth >= (int)job_collect.n_bin_job_per_level.size())
        return 0;
    return job_collect.n_bin_job_per_level[depth];
}

int SetBinaryFeat(int d, void* _feat_ptr, int dev)
{
    int num_jobs = job_collect.n_bin_job_per_level[d];
    float* feat_ptr = static_cast<float*>(_feat_ptr);
    if (dev == 0)  // cpu
    {
        #pragma omp parallel for
        for (int i = 0; i < num_jobs + 2; ++i)
        {
            float* cur_ptr = feat_ptr + i * cfg::dim_embed;
            if (i < 2) {
                cur_ptr[0] = i ? 1 : -1;
            } else {
                auto* node = job_collect.binary_feat_nodes[d][i - 2];
                for (int j = 0; j < node->n_cols; ++j)
                    cur_ptr[j] = node->bits_rep.get(j) ? 1 : -1;
            }
        }
    } else {
        int* lens = new int[num_jobs + 2];
        lens[0] = lens[1] = 1;
        uint32_t n_ints = cfg::bits_compress / ibits;
        if (cfg::bits_compress % ibits)
            n_ints++;
        uint32_t* bits = new uint32_t[(num_jobs + 2) * n_ints];
        bits[0] = 0;
        bits[n_ints] = 1;
        #pragma omp parallel for
        for (int i = 2; i < num_jobs + 2; ++i)
        {
            auto* node = job_collect.binary_feat_nodes[d][i - 2];
            lens[i] = node->n_cols;
            uint32_t* cur_bits = bits + i * n_ints;
            assert(node->bits_rep.n_macros <= n_ints);
            for (uint32_t j = 0; j < node->bits_rep.n_macros; ++j)
                cur_bits[j] = node->bits_rep.macro_bits[j];
        }
#ifdef USE_GPU
        build_binary_mat(num_jobs + 2, n_ints, cfg::dim_embed, lens, bits, feat_ptr);  // NOLINT
#endif
        delete[] bits;
        delete[] lens;
    }
    return 0;
}

int MaxTreeDepth()
{
    int depth = (int)job_collect.n_cell_job_per_level.size();
    depth -= 1;
    return depth;
}

int NumBottomDep(int depth, int lr)
{
    return (int)job_collect.bot_froms[lr][depth].size();
}

int NumPrevDep(int depth, int lr)
{
    return (int)job_collect.prev_froms[lr][depth].size();
}

int NumRowBottomDep(int lr)
{
    return (int)job_collect.row_bot_froms[lr].size();
}

int NumRowPastDep(int lv, int lr)
{
    if (lv >= (int)job_collect.row_prev_froms[lr].size())
        return 0;
    return (int)job_collect.row_prev_froms[lr][lv].size();
}

int NumRowTopDep(int lv, int lr)
{
    if (lv >= (int)job_collect.row_top_froms[lr].size())
        return 0;
    return (int)job_collect.row_top_froms[lr][lv].size();
}

int RowSumSteps()
{
    return job_collect.max_rowsum_steps;
}

int RowMergeSteps()
{
    return job_collect.max_row_merge_steps;
}

int NumRowSumOut(int lr)
{
    return (int)job_collect.step_froms[lr].size();
}

int NumRowSumNext(int lr)
{
    return (int)job_collect.step_nexts[lr].size();
}

int SetRowSumInit(void* _init_idx)
{
    int* init_idx = static_cast<int*>(_init_idx);
    std::memcpy(init_idx, job_collect.step_inputs[0].data(),
                job_collect.step_inputs[0].size() * sizeof(int));
    return 0;
}


int HasChild(void* _has_child)
{
    int* has_child = static_cast<int*>(_has_child);
    std::memcpy(has_child, job_collect.has_ch.data(),
                job_collect.has_ch.size() * sizeof(int));
    return 0;
}

int NumCurNodes(int depth)
{
    if (depth >= (int)job_collect.is_internal.size())
        return 0;
    return (int)job_collect.is_internal[depth].size();
}

int GetInternalMask(int depth, void* _internal_mask)
{
    int* internal_mask = static_cast<int*>(_internal_mask);
    std::memcpy(internal_mask, job_collect.is_internal[depth].data(),
                job_collect.is_internal[depth].size() * sizeof(int));
    return 0;
}

int NumInternalNodes(int depth)
{
    if (depth >= (int)job_collect.has_left.size())
        return 0;
    return (int)job_collect.has_left[depth].size();
}

int NumLeftBot(int depth)
{
    return (int)job_collect.bot_left_froms[depth].size();
}

int GetChMask(int lr, int depth, void* _ch_mask)
{
    int* ch_mask = static_cast<int*>(_ch_mask);
    const int* ptr = lr < 0 ? job_collect.has_left[depth].data()
        : job_collect.has_right[depth].data();
    size_t n = job_collect.has_left[depth].size();
    std::memcpy(ch_mask, ptr, n * sizeof(int));
    return 0;
}

int GetNumCh(int lr, int depth, void* _num_ch)
{
    int* num_ch = static_cast<int*>(_num_ch);
    const int* ptr = lr < 0 ? job_collect.num_left[depth].data()
        : job_collect.num_right[depth].data();
    size_t n = job_collect.num_left[depth].size();
    std::memcpy(num_ch, ptr, n * sizeof(int));
    return 0;
}

int LeftRightSelect(int depth, void* _left_from, void* _left_to,
                    void* _right_from, void* _right_to)
{
    int* left_from = static_cast<int*>(_left_from);
    int* left_to = static_cast<int*>(_left_to);
    int* right_from = static_cast<int*>(_right_from);
    int* right_to = static_cast<int*>(_right_to);

    int n_left = 0, n_right = 0, pos = 0;
    auto& has_left = job_collect.has_left[depth];
    auto& has_right = job_collect.has_right[depth];
    for (int i = 0; i < (int)has_left.size(); ++i)
    {
        if (has_left[i]) {
            left_from[n_left] = i;
            left_to[n_left] = pos;
            pos++;
            n_left++;
        }
        if (has_right[i]) {
            right_from[n_right] = i;
            right_to[n_right] = pos;
            pos++;
            n_right++;
        }
    }
    return 0;
}

int SetLeftState(int depth, void* _bot_from, void* _bot_to,
                 void* _prev_from, void* _prev_to)
{
    int* bot_from = static_cast<int*>(_bot_from);
    int* bot_to = static_cast<int*>(_bot_to);
    int* prev_from = static_cast<int*>(_prev_from);
    int* prev_to = static_cast<int*>(_prev_to);
    if (depth < (int)job_collect.bot_left_froms.size() &&
        job_collect.bot_left_froms[depth].size())
    {
        std::memcpy(bot_from, job_collect.bot_left_froms[depth].data(),
                    job_collect.bot_left_froms[depth].size() * sizeof(int));
        std::memcpy(bot_to, job_collect.bot_left_tos[depth].data(),
                    job_collect.bot_left_tos[depth].size() * sizeof(int));
    }
    if (depth < (int)job_collect.next_left_froms.size() &&
        job_collect.next_left_froms[depth].size())
    {
        std::memcpy(prev_from, job_collect.next_left_froms[depth].data(),
                    job_collect.next_left_froms[depth].size() * sizeof(int));
        std::memcpy(prev_to, job_collect.next_left_tos[depth].data(),
                    job_collect.next_left_tos[depth].size() * sizeof(int));
    }
    return 0;
}

int SetRowSumLast(void* _last_idx)
{
    int* last_idx = static_cast<int*>(_last_idx);
    int last_step = job_collect.max_rowsum_steps;
    std::memcpy(last_idx, job_collect.step_indices[last_step].data(),
                job_collect.step_indices[last_step].size() * sizeof(int));
    return 0;
}

int SetRowSumIds(int lr, void* _step_from, void* _step_to,
                 void* _next_input, void* _next_states)
{
    int* step_from = static_cast<int*>(_step_from);
    std::memcpy(step_from, job_collect.step_froms[lr].data(),
                job_collect.step_froms[lr].size() * sizeof(int));

    int* step_to = static_cast<int*>(_step_to);
    std::memcpy(step_to, job_collect.step_tos[lr].data(),
                job_collect.step_tos[lr].size() * sizeof(int));

    int* next_input = static_cast<int*>(_next_input);
    std::memcpy(next_input, job_collect.step_inputs[lr + 1].data(),
                job_collect.step_inputs[lr + 1].size() * sizeof(int));

    int* next_states = static_cast<int*>(_next_states);
    std::memcpy(next_states, job_collect.step_nexts[lr].data(),
                job_collect.step_nexts[lr].size() * sizeof(int));
    return 0;
}

int SetTreeEmbedIds(int depth, int lr, void* _bot_from, void* _bot_to,
                    void* _prev_from, void* _prev_to)
{
    int* bot_from = static_cast<int*>(_bot_from);
    int* bot_to = static_cast<int*>(_bot_to);
    int* prev_from = static_cast<int*>(_prev_from);
    int* prev_to = static_cast<int*>(_prev_to);
    std::memcpy(bot_from, job_collect.bot_froms[lr][depth].data(),
                job_collect.bot_froms[lr][depth].size() * sizeof(int));
    std::memcpy(bot_to, job_collect.bot_tos[lr][depth].data(),
                job_collect.bot_tos[lr][depth].size() * sizeof(int));
    std::memcpy(prev_from, job_collect.prev_froms[lr][depth].data(),
                job_collect.prev_froms[lr][depth].size() * sizeof(int));
    std::memcpy(prev_to, job_collect.prev_tos[lr][depth].data(),
                job_collect.prev_tos[lr][depth].size() * sizeof(int));
    return 0;
}

int SetRowEmbedIds(int lr, int level, void* _bot_from, void* _bot_to,
                   void* _prev_from, void* _prev_to,
                   void* _past_from, void* _past_to)
{
    if (level == 0)
    {
        int* bot_from = static_cast<int*>(_bot_from);
        int* bot_to = static_cast<int*>(_bot_to);
        std::memcpy(bot_from, job_collect.row_bot_froms[lr].data(),
                    job_collect.row_bot_froms[lr].size() * sizeof(int));
        std::memcpy(bot_to, job_collect.row_bot_tos[lr].data(),
                    job_collect.row_bot_tos[lr].size() * sizeof(int));
    }

    int* prev_from = static_cast<int*>(_prev_from);
    int* prev_to = static_cast<int*>(_prev_to);
    std::memcpy(prev_from, job_collect.row_top_froms[lr][level].data(),
                job_collect.row_top_froms[lr][level].size() * sizeof(int));
    std::memcpy(prev_to, job_collect.row_top_tos[lr][level].data(),
                job_collect.row_top_tos[lr][level].size() * sizeof(int));
    if (job_collect.row_prev_froms[lr][level].size())
    {
        int* past_from = static_cast<int*>(_past_from);
        int* past_to = static_cast<int*>(_past_to);
        std::memcpy(past_from, job_collect.row_prev_froms[lr][level].data(),
                    job_collect.row_prev_froms[lr][level].size() * sizeof(int));
        std::memcpy(past_to, job_collect.row_prev_tos[lr][level].data(),
                    job_collect.row_prev_tos[lr][level].size() * sizeof(int));
    }
    return 0;
}

int PrepareTrain(int num_graphs, void* _list_ids, void* _list_start_node,
                 void* _list_col_start, void* _list_col_end,
                 int num_nodes, int new_batch)
{
    int* list_ids = static_cast<int*>(_list_ids);
    int* list_start_node = static_cast<int*>(_list_start_node);
    int* list_col_start = static_cast<int*>(_list_col_start);
    int* list_col_end = static_cast<int*>(_list_col_end);
    job_collect.reset();
    node_holder.reset();
    row_holder.reset();

    if (new_batch)
    {
        if (cfg::bfs_permute)
        {
            for (auto* g : active_graphs)
                delete g;
        }
        active_graphs.clear();
    }
    int gid, node_start, node_end;
    for (int i = 0; i < num_graphs; ++i)
    {
        gid = list_ids[i];
        assert(gid >= 0 && gid < (int)graph_list.size());
        GraphStruct* g;
        if (new_batch)
        {
            g = graph_list[gid]->permute();
            active_graphs.push_back(g);
        } else {
            g = active_graphs[i];
        }
        node_start = list_start_node[i];
        assert(node_start >= 0);
        node_end = (num_nodes < 0) ? g->num_nodes : node_start + num_nodes;
        assert(node_start >= 0 && node_end <= g->num_nodes);
        g->realize_nodes(node_start, node_end,
                         list_col_start[i], list_col_end[i]);
    }
    job_collect.build_row_indices();
    job_collect.build_row_summary();
    return 0;
}

int AddGraph(int graph_id, int num_nodes, int num_edges,
             void* edge_pairs, int n_left, int n_right)
{
    auto* g = new GraphStruct(graph_id, num_nodes, num_edges,
                              edge_pairs, n_left, n_right);
    assert(graph_id == (int)graph_list.size());
    graph_list.push_back(g);
    return 0;
}

int GetNextStates(void* _state_idx)
{
    int* state_idx = static_cast<int*>(_state_idx);
    std::memcpy(state_idx, job_collect.next_state_froms.data(),
                job_collect.next_state_froms.size() * sizeof(int));
    return 0;
}

int GetNumNextStates()
{
    return (int)job_collect.next_state_froms.size();
}

int GetCurPos(void* _pos)
{
    int* pos = static_cast<int*>(_pos);
    int t = 0;
    for (auto* g : active_graphs)
    {
        for (int i = g->node_start; i < g->node_end; ++i)
        {
            pos[t++] = g->num_nodes - i;
        }
    }
    return 0;
}
