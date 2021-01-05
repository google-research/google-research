# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import numpy as np
import random
import os
import sys
import networkx as nx
from tqdm import tqdm
# pylint: skip-file

try:
    import torch
except:
    print('no torch loaded')


class CtypeGraph(object):
    def __init__(self, g):
        self.num_nodes = len(g)
        self.num_edges = len(g.edges())

        self.edge_pairs = np.zeros((self.num_edges * 2, ), dtype=np.int32)
        for i, (x, y) in enumerate(g.edges()):
            self.edge_pairs[i * 2] = x
            self.edge_pairs[i * 2 + 1] = y


class _tree_lib(object):

    def __init__(self):
        pass

    def setup(self, config):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libtree.so' % dir_path)

        self.lib.Init.restype = ctypes.c_int
        self.lib.PrepareTrain.restype = ctypes.c_int
        self.lib.AddGraph.restype = ctypes.c_int
        self.lib.TotalTreeNodes.restype = ctypes.c_int
        self.lib.MaxTreeDepth.restype = ctypes.c_int
        self.lib.NumPrevDep.restype = ctypes.c_int
        self.lib.NumBottomDep.restype = ctypes.c_int
        self.lib.NumRowBottomDep.restype = ctypes.c_int
        self.lib.NumRowPastDep.restype = ctypes.c_int
        self.lib.NumRowTopDep.restype = ctypes.c_int
        self.lib.RowSumSteps.restype = ctypes.c_int
        self.lib.RowMergeSteps.restype = ctypes.c_int
        self.lib.NumRowSumOut.restype = ctypes.c_int
        self.lib.NumRowSumNext.restype = ctypes.c_int
        self.lib.NumCurNodes.restype = ctypes.c_int
        self.lib.NumInternalNodes.restype = ctypes.c_int
        self.lib.NumLeftBot.restype = ctypes.c_int
        self.lib.GetNumNextStates.restype = ctypes.c_int

        args = 'this -bits_compress %d -embed_dim %d -gpu %d -bfs_permute %d -seed %d' % (config.bits_compress, config.embed_dim, config.gpu, config.bfs_permute, config.seed)
        args = args.split()
        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args

        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)
        self.embed_dim = config.embed_dim
        self.device = config.device
        self.num_graphs = 0
        self.graph_stats = []

    def TotalTreeNodes(self):
        return self.lib.TotalTreeNodes()

    def InsertGraph(self, nx_g, bipart_stats=None):
        gid = self.num_graphs
        self.num_graphs += 1
        if isinstance(nx_g, CtypeGraph):
            ctype_g = nx_g
        else:
            ctype_g = CtypeGraph(nx_g)
        self.graph_stats.append((ctype_g.num_nodes, ctype_g.num_edges))
        if bipart_stats is None:
            n, m = -1, -1
        else:
            n, m = bipart_stats
        self.lib.AddGraph(gid, ctype_g.num_nodes, ctype_g.num_edges,
                          ctypes.c_void_p(ctype_g.edge_pairs.ctypes.data), n, m)
        return gid

    def PrepareMiniBatch(self, list_gids, list_node_start=None, num_nodes=-1, list_col_ranges=None, new_batch=True):
        n_graphs = len(list_gids)
        list_gids = np.array(list_gids, dtype=np.int32)
        if list_node_start is None:
            list_node_start = np.zeros((n_graphs,), dtype=np.int32)
        else:
            list_node_start = np.array(list_node_start, dtype=np.int32)
        if list_col_ranges is None:
            list_col_start = np.zeros((n_graphs,), dtype=np.int32) - 1
            list_col_end = np.zeros((n_graphs,), dtype=np.int32) - 1
        else:
            list_col_start, list_col_end = zip(*list_col_ranges)
            list_col_start = np.array(list_col_start, dtype=np.int32)
            list_col_end = np.array(list_col_end, dtype=np.int32)

        self.lib.PrepareTrain(n_graphs,
                              ctypes.c_void_p(list_gids.ctypes.data),
                              ctypes.c_void_p(list_node_start.ctypes.data),
                              ctypes.c_void_p(list_col_start.ctypes.data),
                              ctypes.c_void_p(list_col_end.ctypes.data),
                              num_nodes,
                              int(new_batch))
        list_nnodes = []
        for i, gid in enumerate(list_gids):
            tot_nodes = self.graph_stats[gid][0]
            if num_nodes <= 0:
                cur_num = tot_nodes - list_node_start[i]
            else:
                cur_num = min(num_nodes, tot_nodes - list_node_start[i])
            list_nnodes.append(cur_num)
        self.list_nnodes = list_nnodes
        return list_nnodes

    def PrepareTreeEmbed(self):
        max_d = self.lib.MaxTreeDepth()

        all_ids = []
        for d in range(max_d + 1):
            ids_d = []
            for i in range(2):
                num_prev = self.lib.NumPrevDep(d, i)
                num_bot = self.lib.NumBottomDep(d, i)

                bot_froms = np.empty((num_bot,), dtype=np.int32)
                bot_tos = np.empty((num_bot,), dtype=np.int32)

                prev_froms = np.empty((num_prev,), dtype=np.int32)
                prev_tos = np.empty((num_prev,), dtype=np.int32)
                self.lib.SetTreeEmbedIds(d,
                                         i,
                                         ctypes.c_void_p(bot_froms.ctypes.data),
                                         ctypes.c_void_p(bot_tos.ctypes.data),
                                         ctypes.c_void_p(prev_froms.ctypes.data),
                                         ctypes.c_void_p(prev_tos.ctypes.data))
                ids_d.append((bot_froms, bot_tos, prev_froms, prev_tos))
            all_ids.append(ids_d)
        return all_ids

    def PrepareBinary(self):
        max_d = self.lib.MaxBinFeatDepth()
        all_bin_feats = []
        base_feat = torch.zeros(2, self.embed_dim)
        base_feat[0, 0] = -1
        base_feat[1, 0] = 1
        base_feat = base_feat.to(self.device)
        for d in range(max_d):
            num_nodes = self.lib.NumBinNodes(d)
            if num_nodes == 0:
                all_bin_feats.append(base_feat)
            else:
                if self.device == torch.device('cpu'):
                    feat = torch.zeros(num_nodes + 2, self.embed_dim)
                    dev = 0
                else:
                    feat = torch.cuda.FloatTensor(num_nodes + 2, self.embed_dim).fill_(0)
                    dev = 1
                self.lib.SetBinaryFeat(d, ctypes.c_void_p(feat.data_ptr()), dev)
                all_bin_feats.append(feat)
        return all_bin_feats, (base_feat, base_feat)

    def PrepareRowEmbed(self):
        tot_levels = self.lib.RowMergeSteps()
        lv = 0
        all_ids = []
        for lv in range(tot_levels):
            ids_d = []
            for i in range(2):
                num_prev = self.lib.NumRowTopDep(lv, i)
                num_bot = self.lib.NumRowBottomDep(i) if lv == 0 else 0
                num_past = self.lib.NumRowPastDep(lv, i)
                bot_froms = np.empty((num_bot,), dtype=np.int32)
                bot_tos = np.empty((num_bot,), dtype=np.int32)
                prev_froms = np.empty((num_prev,), dtype=np.int32)
                prev_tos = np.empty((num_prev,), dtype=np.int32)
                past_froms = np.empty((num_past,), dtype=np.int32)
                past_tos = np.empty((num_past,), dtype=np.int32)
                self.lib.SetRowEmbedIds(i,
                                        lv,
                                        ctypes.c_void_p(bot_froms.ctypes.data),
                                        ctypes.c_void_p(bot_tos.ctypes.data),
                                        ctypes.c_void_p(prev_froms.ctypes.data),
                                        ctypes.c_void_p(prev_tos.ctypes.data),
                                        ctypes.c_void_p(past_froms.ctypes.data),
                                        ctypes.c_void_p(past_tos.ctypes.data))
                ids_d.append((bot_froms, bot_tos, prev_froms, prev_tos, past_froms, past_tos))
            all_ids.append(ids_d)

        return all_ids

    def PrepareRowSummary(self):
        total_steps = self.lib.RowSumSteps()
        all_ids = []
        total_nodes = np.sum(self.list_nnodes)
        init_ids = np.empty((total_nodes,), dtype=np.int32)
        self.lib.SetRowSumInit(ctypes.c_void_p(init_ids.ctypes.data))
        for i in range(total_steps):
            num_done = self.lib.NumRowSumOut(i)
            num_next = self.lib.NumRowSumNext(i)
            step_from = np.empty((num_done,), dtype=np.int32)
            step_to = np.empty((num_done,), dtype=np.int32)

            step_next = np.empty((num_next,), dtype=np.int32)
            step_input = np.empty((num_next,), dtype=np.int32)
            self.lib.SetRowSumIds(i,
                                  ctypes.c_void_p(step_from.ctypes.data),
                                  ctypes.c_void_p(step_to.ctypes.data),
                                  ctypes.c_void_p(step_input.ctypes.data),
                                  ctypes.c_void_p(step_next.ctypes.data))
            all_ids.append((step_from, step_to, step_next, step_input))
            total_nodes -= num_done
        last_ids = np.empty((total_nodes,), dtype=np.int32)
        self.lib.SetRowSumLast(ctypes.c_void_p(last_ids.ctypes.data))

        num_next = self.lib.GetNumNextStates()
        next_ids = np.empty((num_next,), dtype=np.int32)
        self.lib.GetNextStates(ctypes.c_void_p(next_ids.ctypes.data))

        np_pos = np.empty((np.sum(self.list_nnodes),), dtype=np.int32)
        self.lib.GetCurPos(ctypes.c_void_p(np_pos.ctypes.data))
        return init_ids, all_ids, last_ids, next_ids, torch.tensor(np_pos, dtype=torch.float32).to(self.device)

    def GetChLabel(self, lr, depth=-1, dtype=None):
        if lr == 0:
            total_nodes = np.sum(self.list_nnodes)
            has_ch = np.empty((total_nodes,), dtype=np.int32)
            self.lib.HasChild(ctypes.c_void_p(has_ch.ctypes.data))
            num_ch = None
        else:
            n = self.lib.NumInternalNodes(depth)
            has_ch = np.empty((n,), dtype=np.int32)
            self.lib.GetChMask(lr, depth,
                               ctypes.c_void_p(has_ch.ctypes.data))
            num_ch = np.empty((n,), dtype=np.int32)
            self.lib.GetNumCh(lr, depth,
                              ctypes.c_void_p(num_ch.ctypes.data))
            num_ch = torch.tensor(num_ch, dtype=torch.float32).to(self.device)
        if dtype is not None:
            has_ch = has_ch.astype(dtype)
        return has_ch, num_ch

    def QueryNonLeaf(self, depth):
        n = self.lib.NumCurNodes(depth)
        if n == 0:
            return None
        is_internal = np.empty((n,), dtype=np.int32)
        self.lib.GetInternalMask(depth, ctypes.c_void_p(is_internal.ctypes.data))
        return is_internal.astype(np.bool)

    def GetLeftRootStates(self, depth):
        n = self.lib.NumInternalNodes(depth)
        left_bot = self.lib.NumLeftBot(depth)
        left_next = n - left_bot
        bot_froms = np.empty((left_bot,), dtype=np.int32)
        bot_tos = np.empty((left_bot,), dtype=np.int32)
        next_froms = np.empty((left_next,), dtype=np.int32)
        next_tos = np.empty((left_next,), dtype=np.int32)
        self.lib.SetLeftState(depth,
                              ctypes.c_void_p(bot_froms.ctypes.data),
                              ctypes.c_void_p(bot_tos.ctypes.data),
                              ctypes.c_void_p(next_froms.ctypes.data),
                              ctypes.c_void_p(next_tos.ctypes.data))
        if left_bot == 0:
            bot_froms = bot_tos = None
        if left_next == 0:
            next_froms = next_tos = None
        return bot_froms, bot_tos, next_froms, next_tos

    def GetLeftRightSelect(self, depth, num_left, num_right):
        left_froms = np.empty((num_left,), dtype=np.int32)
        left_tos = np.empty((num_left,), dtype=np.int32)
        right_froms = np.empty((num_right,), dtype=np.int32)
        right_tos = np.empty((num_right,), dtype=np.int32)

        self.lib.LeftRightSelect(depth,
                                 ctypes.c_void_p(left_froms.ctypes.data),
                                 ctypes.c_void_p(left_tos.ctypes.data),
                                 ctypes.c_void_p(right_froms.ctypes.data),
                                 ctypes.c_void_p(right_tos.ctypes.data))
        return left_froms, left_tos, right_froms, right_tos


TreeLib = _tree_lib()

def setup_treelib(config):
    global TreeLib
    dll_path = '%s/build/dll/libtree.so' % os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(dll_path):
        TreeLib.setup(config)
