# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file

import os
import sys
import csv
import networkx as nx
import numpy as np
import pickle as cp
import argparse
from bigg.common.configs import cmd_args
from bigg.data_process.data_util import create_graphs, get_graph_data, get_node_map

cmd_opt = argparse.ArgumentParser(description='Argparser for syn_gen')
cmd_opt.add_argument('-folder_name', default=None, type=str, help='folder name')

local_args, _ = cmd_opt.parse_known_args()


def check_graph(graph, n_var, n_clause):
    for e in graph.edges():
        if e[1] < e[0]:
            e = (e[1], e[0])
        assert e[1] >= 2 * n_var and e[0] < n_var * 2
    assert(graph.number_of_nodes() == n_var * 2 + n_clause)


def reorder_graph(g, n_var, n_clause, nodelist):
    left_list = list(range(2 * n_var))
    left_list = sorted(left_list, key=lambda x: nodelist[x])
    node_map = get_node_map(left_list)

    right_list = list(range(n_clause))
    right_list = sorted(right_list, key=lambda x: nodelist[x + 2 * n_var])

    right_map = get_node_map(right_list, shift=2 * n_var)
    node_map.update(right_map)
    g = nx.relabel_nodes(g, node_map)
    check_graph(g, n_var, n_clause)
    return g


if __name__ == '__main__':
    cmd_args.__dict__.update(local_args.__dict__)

    data_folder = '../../data/G2SAT/%s_set' % cmd_args.folder_name
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

    with open(os.path.join(data_folder, '..', 'lcg_stats.csv')) as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        stats = []
        for stat in data:
            stats.append(stat)

    g_list = []
    graph_stats = []
    for fname in os.listdir(data_folder):
        if not 'lcg_edge' in fname:
            continue
        print(fname)
        with open(os.path.join(data_folder, fname), 'rb') as f:
            graph = nx.read_edgelist(f)

        filename = fname[:-14] # remove postfix
        for stat in stats:
            if not filename in stat[0]:
                continue
            n = graph.number_of_nodes()
            n_var = int(stat[1])
            n_clause = int(stat[2])
            if graph.number_of_nodes() != n_var * 2 + n_clause:
                continue
            keys = [str(i + 1) for i in range(n)]
            vals = range(n)
            mapping = dict(zip(keys, vals))
            nx.relabel_nodes(graph, mapping, copy=False)
            break
        check_graph(graph, n_var, n_clause)
        g_list.append(graph)
        graph_stats.append((n_var * 2, n_clause))

    for phase in ['train', 'val', 'test']:
        out_stats = []
        with open(os.path.join(cmd_args.save_dir, '%s-graphs.pkl' % phase), 'wb') as f:
            for g, stat in zip(g_list, graph_stats):
                n_var, n_clause = stat
                n_var = n_var // 2
                sub_list = [g]
                for g in sub_list:
                    cp.dump(g, f, cp.HIGHEST_PROTOCOL)
                    out_stats.append(stat)
        with open(os.path.join(cmd_args.save_dir, '%s-graph-stats.pkl' % phase), 'wb') as f:
            cp.dump(out_stats, f, cp.HIGHEST_PROTOCOL)

        print('num', phase, len(out_stats))
