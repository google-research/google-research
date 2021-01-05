// Copyright 2021 The Google Research Authors.
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

#ifndef STRUCT_UTIL_H
#define STRUCT_UTIL_H

#include <vector>
#include <map>
#include <cassert>
#include <atomic>
#include <unordered_map>

class AdjRow;
class AdjNode;

const uint32_t ibits = 32;

int num_ones(int n);

class BitSet
{
 public:
    BitSet();
    BitSet(uint32_t _n_bits);
    BitSet left_shift(uint32_t n);
    BitSet or_op(BitSet& another);

    void set(uint32_t pos);
    bool get(uint32_t pos);

    uint32_t n_bits, n_macros;
    std::vector<uint32_t> macro_bits;
};

class GraphStruct
{
 public:
    GraphStruct(int graph_id, int num_nodes, int num_edges,
                void* _edge_pairs = nullptr, int n_left = -1, int n_right = -1);

    void realize_nodes(int node_start, int node_end,
                       int col_start, int col_end);
    GraphStruct* permute();
    std::map<int, std::vector<int> > edge_list;
    std::vector<AdjRow*> active_rows;
    std::vector<int> idx_map;
    int num_nodes, num_edges, graph_id;
    int node_start, node_end;
};

extern std::vector<GraphStruct*> graph_list;
extern std::vector<GraphStruct*> active_graphs;

class AdjNode;

class JobCollect
{
 public:
    JobCollect();
    void reset();
    void build_row_indices();
    void build_row_summary();
    int add_job(AdjNode* node);
    void append_bool(std::vector< std::vector<int> >& list, int depth, int val);
    std::vector<AdjNode*> global_job_nodes;
    std::vector<int> job_position;
    std::vector<int> has_ch;
    std::vector< std::vector<int> > has_left, has_right, num_left, num_right;
    std::vector< std::vector<int> > is_internal;
    std::vector<int> n_cell_job_per_level, n_bin_job_per_level;
    std::vector< std::vector<int> > bot_froms[2], bot_tos[2], prev_froms[2], prev_tos[2]; // NOLINT
    std::vector< std::vector<AdjNode*> > binary_feat_nodes;
    std::vector<int> row_bot_froms[2], row_bot_tos[2];
    std::vector< std::vector<int> > row_top_froms[2], row_top_tos[2], row_prev_froms[2], row_prev_tos[2];  // NOLINT
    std::vector<int> layer_sizes;
    std::vector< std::unordered_map<int, int> > tree_idx_map;

    std::vector<int> next_state_froms;
    std::vector< std::vector<int> > bot_left_froms, bot_left_tos, next_left_froms, next_left_tos;  // NOLINT
    std::vector< std::vector<int> > step_inputs, step_nexts, step_froms, step_tos, step_indices;  // NOLINT
    int max_rowsum_steps, max_tree_depth, max_row_merge_steps;
};

extern JobCollect job_collect;

class ColAutomata
{
 public:
    ColAutomata(std::vector<int>& indices);

    void add_edge(int col_idx);
    int next_edge();
    int last_edge();
    bool has_edge(int range_start, int range_end);

    int* indices;
    int pos, num_indices;
};

class AdjNode;

template<typename PtType>
class PtHolder
{
 public:
    PtHolder();
    void reset();
    void clear();

    template<typename...Args>
    PtType* get_pt(Args&&... args)
    {
        PtType* ret;
        if (cur_pos >= pt_buff.size())
        {
            ret = new PtType(std::forward<Args>(args)...);
            pt_buff.push_back(ret);
        } else {
            ret = pt_buff[cur_pos];
            ret->init(std::forward<Args>(args)...);
        }
        assert(cur_pos < pt_buff.size());
        cur_pos++;
        return ret;
    }

    std::vector<PtType*> pt_buff;
    size_t cur_pos;
};


#endif
