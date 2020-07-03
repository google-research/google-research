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
import pickle as cp
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import OrderedDict
from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_model import RecurTreeGen
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.experiments.train_utils import sqrtn_forward_backward, get_node_dist


def load_graphs(graph_pkl):
    graphs = []
    with open(graph_pkl, 'rb') as f:
        while True:
            try:
                g = cp.load(f)
            except:
                break
            graphs.append(g)
    for g in graphs:
        TreeLib.InsertGraph(g)
    return graphs


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    train_graphs = load_graphs(os.path.join(cmd_args.data_dir, 'train-graphs.pkl'))
    max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
    cmd_args.max_num_nodes = max_num_nodes

    model = RecurTreeGen(cmd_args).to(cmd_args.device)
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))

    if cmd_args.phase != 'train':
        # get num nodes dist
        num_node_dist = get_node_dist(train_graphs)
        gt_graphs = load_graphs(os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % cmd_args.phase))
        print('# gt graphs', len(gt_graphs))
        gen_graphs = []
        with torch.no_grad():
            for _ in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
                _, pred_edges, _ = model(num_nodes, display=cmd_args.display)
                for e in pred_edges:
                    assert e[0] > e[1]
                pred_g = nx.Graph()
                pred_g.add_edges_from(pred_edges)
                gen_graphs.append(pred_g)
        print('saving graphs')
        with open(cmd_args.model_dump + '.graphs-%s' % str(cmd_args.greedy_frac), 'wb') as f:
            cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        print('evaluating')
        sys.exit()

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    indices = list(range(len(train_graphs)))
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        pbar = tqdm(range(cmd_args.epoch_save))

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            batch_indices = indices[:cmd_args.batch_size]

            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
                ll, _ = model.forward_train(batch_indices)
                loss = -ll / num_nodes
                loss.backward()
                loss = loss.item()
            else:
                ll = 0.0
                for i in batch_indices:
                    n = len(train_graphs[i])
                    cur_ll, _ = sqrtn_forward_backward(model, graph_ids=[i], list_node_starts=[0],
                                                    num_nodes=n, blksize=cmd_args.blksize, loss_scale=1.0/n)
                    ll += cur_ll
                loss = -ll / num_nodes
            if (idx + 1) % cmd_args.accum_grad == 0:
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))

        torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
