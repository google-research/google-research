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

#ifndef TREE_CLIB_H
#define TREE_CLIB_H

#include "config.h"  // NOLINT

extern "C" int Init(const int argc, const char **argv);

extern "C" int PrepareTrain(int num_graphs, void* list_ids,
                            void* list_start_node, void* list_col_start,
                            void* list_col_end, int num_nodes, int new_batch);

extern "C" int AddGraph(int graph_idx, int num_nodes, int num_edges,
                        void* edge_pairs, int n_left, int n_right);

extern "C" int TotalTreeNodes();

extern "C" int SetTreeEmbedIds(int depth, int lr, void* _bot_from,
                               void* _bot_to, void* _prev_from, void* _prev_to);

extern "C" int SetRowEmbedIds(int lr, int level, void* _bot_from,
                              void* _bot_to, void* _prev_from,
                              void* _prev_to, void* _past_from, void* _past_to);

extern "C" int MaxTreeDepth();

extern "C" int NumBottomDep(int depth, int lr);

extern "C" int NumPrevDep(int depth, int lr);

extern "C" int NumRowBottomDep(int lr);

extern "C" int NumRowPastDep(int lv, int lr);

extern "C" int NumRowTopDep(int lv, int lr);

extern "C" int RowSumSteps();

extern "C" int RowMergeSteps();

extern "C" int NumRowSumOut(int lr);

extern "C" int NumRowSumNext(int lr);

extern "C" int SetRowSumIds(int lr, void* _step_from, void* _step_to,
                            void* _next_input, void* _next_states);

extern "C" int SetRowSumInit(void* _init_idx);

extern "C" int SetRowSumLast(void* _last_idx);

extern "C" int HasChild(void* _has_child);

extern "C" int NumCurNodes(int depth);

extern "C" int GetInternalMask(int depth, void* _internal_mask);

extern "C" int NumInternalNodes(int depth);

extern "C" int GetChMask(int lr, int depth, void* _ch_mask);

extern "C" int GetNumCh(int lr, int depth, void* _num_ch);

extern "C" int SetLeftState(int depth, void* _bot_from, void* _bot_to,
                            void* _prev_from, void* _prev_to);

extern "C" int NumLeftBot(int depth);

extern "C" int LeftRightSelect(int depth, void* _left_from, void* _left_to,
                               void* _right_from, void* _right_to);

extern "C" int MaxBinFeatDepth();

extern "C" int NumBinNodes(int depth);

extern "C" int SetBinaryFeat(int d, void* _feat_ptr, int dev);

extern "C" int GetNextStates(void* _state_idx);

extern "C" int GetNumNextStates();

extern "C" int GetCurPos(void* _pos);

#endif
