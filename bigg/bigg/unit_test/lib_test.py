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
from tqdm import tqdm
import torch
import torch.optim as optim

import numpy as np
import random
import networkx as nx
from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.model.tree_model import RecurTreeGen

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    train_graphs = [nx.barabasi_albert_graph(10, 2)]
    TreeLib.InsertGraph(train_graphs[0])
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes

    model = RecurTreeGen(cmd_args).to(cmd_args.device)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    for i in range(2):
        optimizer.zero_grad()
        ll, _ = model.forward_train([0])
        loss = -ll / max_num_nodes
        print('iter', i, 'loss', loss.item())
        loss.backward()
        optimizer.step()
