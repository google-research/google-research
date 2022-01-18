# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
import networkx as nx
import random
import numpy as np
import pickle as cp
import argparse
from tqdm import tqdm
from bigg.common.configs import cmd_args
from bigg.data_process.data_util import create_graphs, get_graph_data

cmd_opt = argparse.ArgumentParser(description='Argparser for syn_gen')
cmd_opt.add_argument('-g_type', default=None, type=str, help='graph type')

local_args, _ = cmd_opt.parse_known_args()

if __name__ == '__main__':
    cmd_args.__dict__.update(local_args.__dict__)
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    graphs = create_graphs(cmd_args.g_type)
    num_graphs = len(graphs)
    num_train = int(float(num_graphs) * cmd_args.train_ratio)
    num_dev = int(float(num_graphs) * cmd_args.dev_ratio)
    num_test_gt = num_graphs - num_train

    # npr = np.random.RandomState(cmd_args.seed)
    # npr.shuffle(graphs)

    graph_splits = {}
    for phase, g_list in zip(['train', 'val', 'test'], [graphs[:num_train], graphs[:num_dev], graphs[num_train:]]):
        with open(os.path.join(cmd_args.save_dir, '%s-graphs.pkl' % phase), 'wb') as f:
            for g in tqdm(g_list):
                cano_g = get_graph_data(g, cmd_args.node_order)
                for gc in cano_g:
                    cp.dump(gc, f, cp.HIGHEST_PROTOCOL)
                    if phase != 'train':
                        break
        print('num', phase, len(g_list))
        graph_splits[phase] = g_list
