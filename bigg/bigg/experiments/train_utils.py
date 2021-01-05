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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: skip-file

import os
import sys
import numpy as np
import random
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm
from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
import torch.distributed as dist
from torch.multiprocessing import Process
from bigg.model.tree_model import RecurTreeGen


def get_node_dist(graphs):
  num_node_dist = np.bincount([len(gg.nodes) for gg in graphs])
  num_node_dist = num_node_dist / np.sum(num_node_dist)
  return num_node_dist


def sqrtn_forward_backward(model,
                           graph_ids,
                           list_node_starts,
                           num_nodes,
                           blksize,
                           loss_scale,
                           init_states=[None, None],
                           top_grad=None,
                           **kwargs):
    assert len(graph_ids) == 1
    if blksize < 0 or blksize > num_nodes:
        blksize = num_nodes

    prev_states = init_states
    cache_stages = list(range(0, num_nodes, blksize))

    list_caches = []
    for st_delta in cache_stages[:-1]:
        node_st = list_node_starts[0] + st_delta
        with torch.no_grad():
            cur_num = num_nodes - node_st if node_st + blksize > num_nodes else blksize
            _, new_states = model.forward_row_summaries(graph_ids,
                                                        list_node_starts=[node_st],
                                                        num_nodes=cur_num,
                                                        prev_rowsum_states=prev_states,
                                                        **kwargs)
            prev_states = new_states
            list_caches.append(new_states)

    tot_ll = 0.0
    for i in range(len(cache_stages) - 1, -1, -1):
        st_delta = cache_stages[i]
        node_st = list_node_starts[0] + st_delta
        cur_num = num_nodes - node_st if node_st + blksize > num_nodes else blksize
        prev_states = list_caches[i - 1] if i else init_states
        if prev_states[0] is not None:
            for x in prev_states:
                x.requires_grad = True
        ll, cur_states = model.forward_train(graph_ids,
                                             list_node_starts=[node_st],
                                             num_nodes=cur_num,
                                             prev_rowsum_states=prev_states,
                                             **kwargs)
        tot_ll += ll.item()
        loss = -ll * loss_scale
        if top_grad is not None:
            torch.autograd.backward([loss, *cur_states], [None, *top_grad])
        else:
            loss.backward()
        if i:
            top_grad = [x.grad.detach() for x in prev_states]

    return tot_ll, top_grad


def setup_dist(rank):
    if cmd_args.gpu >= 0:
        set_device(rank)
    else:
        set_device(-1)
    setup_treelib(cmd_args)
    random.seed(cmd_args.seed + rank)
    torch.manual_seed(cmd_args.seed + rank)
    np.random.seed(cmd_args.seed + rank)


def build_model(rank):
    model = RecurTreeGen(cmd_args).to(cmd_args.device)
    if rank == 0 and cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))
    for param in model.parameters():
        dist.broadcast(param.data, 0)
    return model


def dist_graphgen(rank, total_num_gen, train_graphs):
    world_size = dist.get_world_size()
    model = build_model(rank)

    num_node_dist = get_node_dist(train_graphs)
    num_gen = total_num_gen // world_size
    rest = total_num_gen % world_size
    num_gen += int(rank < rest)

    gen_graphs = []
    with torch.no_grad():
        for _ in tqdm(range(num_gen)):
            num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
            _, pred_edges, _ = model(num_nodes)
            for e in pred_edges:
                assert e[0] > e[1]
            pred_g = nx.Graph()
            pred_g.add_edges_from(pred_edges)
            gen_graphs.append(pred_g)
    return gen_graphs


def data_parallel_main(rank, train_graphs, fn_kwargs):
    world_size = float(dist.get_world_size())
    model = build_model(rank)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)

    random.seed(cmd_args.seed)
    indices = list(range(len(train_graphs)))
    for epoch in range(cmd_args.num_epochs):
        pbar = range(cmd_args.epoch_save)
        if rank == 0:
            pbar = tqdm(pbar)

        optimizer.zero_grad()
        for idx in pbar:
            random.shuffle(indices)
            st = cmd_args.batch_size * rank
            batch_indices = indices[st : st + cmd_args.batch_size]
            num_nodes = sum([len(train_graphs[i]) for i in batch_indices])
            if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
                ll, _ = model.forward_train(batch_indices, **(fn_kwargs(batch_indices, None)))
                loss = -ll / num_nodes / cmd_args.accum_grad / world_size
                loss.backward()
                loss = loss.item() * cmd_args.accum_grad * world_size
            else:
                ll = 0.0
                for i in batch_indices:
                    n = len(train_graphs[i]) * len(batch_indices) * world_size * cmd_args.accum_grad
                    cur_ll, _ = sqrtn_forward_backward(model, graph_ids=[i], list_node_starts=[0], blksize=cmd_args.blksize, loss_scale=1.0/n,
                                                       **(fn_kwargs(batch_indices, i)))
                    ll += cur_ll
                loss = -ll / num_nodes

            if (idx + 1) % cmd_args.accum_grad == 0:
                for param in model.parameters():
                    if param.grad is None:
                        param.grad = param.data.new(param.shape).zero_()
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            if rank == 0:
                pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / cmd_args.epoch_save, loss))

        if rank == 0:
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % epoch))
