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
import pickle as cp
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_model import RecurTreeGen
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.experiments.train_utils import sqrtn_forward_backward

def num_expect(n):
    cnt = 0
    while n > 0:
        cnt += 1
        n = (n - 1) & n
    return cnt

def send_states(states, rank):
    for x in states:
        dist.send(tensor=x.detach().data.cpu(), dst=rank)

def recv_states(n, rank):
    states_recv = []
    for _ in range(2):
        x = torch.FloatTensor(n, cmd_args.embed_dim)
        dist.recv(tensor=x, src=rank)
        if cmd_args.gpu >= 0:
            x = x.cuda()
        states_recv.append(x)
    return states_recv

def get_stage_stats(stage, blksize, rank, g):
    stage_st = (blksize * cmd_args.num_proc) * stage

    local_st = stage_st + blksize * rank
    local_num = blksize if local_st + blksize <= len(g) else len(g) - local_st
    if stage_st + blksize * cmd_args.num_proc <= len(g):
        rank_last = cmd_args.num_proc - 1
    else:
        rest = len(g) - stage_st
        num_segs = rest // blksize + (rest % blksize > 0)
        rank_last = num_segs - 1
    return local_st, local_num, rank_last

def main(rank):
    if cmd_args.gpu >= 0:
        set_device(rank)
    else:
        set_device(-1)
    setup_treelib(cmd_args)
    model = RecurTreeGen(cmd_args).to(cmd_args.device)

    if rank == 0 and cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        model.load_state_dict(torch.load(cmd_args.model_dump))
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
    for param in model.parameters():
        dist.broadcast(param.data, 0)

    graphs = []
    with open(os.path.join(cmd_args.data_dir, 'train-graphs.pkl'), 'rb') as f:
        while True:
            try:
                g = cp.load(f)
                TreeLib.InsertGraph(g)
            except:
                break
            graphs.append(g)

    for epoch in range(cmd_args.num_epochs):
        pbar = range(cmd_args.epoch_save)
        if rank == 0:
            pbar = tqdm(pbar)

        g = graphs[0]
        graph_ids = [0]
        blksize = cmd_args.blksize
        if blksize < 0 or blksize > len(g):
            blksize = len(g)

        for e_it in pbar:
            optimizer.zero_grad()
            num_stages = len(g) // (blksize * cmd_args.num_proc) + (len(g) % (blksize * cmd_args.num_proc) > 0)
            states_prev = [None, None]
            list_caches = []
            prev_rank_last = None
            for stage in range(num_stages):
                local_st, local_num, rank_last = get_stage_stats(stage, blksize, rank, g)
                if local_num <= 0:
                    break
                with torch.no_grad():
                    fn_hc_bot, h_buf_list, c_buf_list = model.forward_row_trees(graph_ids,
                                                                                list_node_starts=[local_st],
                                                                                num_nodes=local_num)
                    if stage and rank == 0:
                        states_prev = recv_states(num_expect(local_st), prev_rank_last)
                    if rank:
                        num_recv = num_expect(local_st)
                        states_prev = recv_states(num_recv, rank - 1)
                    _, next_states = model.row_tree.forward_train(*(fn_hc_bot(0)), h_buf_list[0], c_buf_list[0], *states_prev)
                    list_caches.append(states_prev)
                    if rank != rank_last:
                        send_states(next_states, rank + 1)
                    elif stage + 1 < num_stages:
                        send_states(next_states, 0)
                prev_rank_last = rank_last

            tot_ll = torch.zeros(1).to(cmd_args.device)
            for stage in range(num_stages - 1, -1, -1):
                local_st, local_num, rank_last = get_stage_stats(stage, blksize, rank, g)
                if local_num <= 0:
                    continue
                prev_states = list_caches[stage]
                if prev_states[0] is not None:
                    for x in prev_states:
                        x.requires_grad = True
                ll, cur_states = model.forward_train(graph_ids,
                                                    list_node_starts=[local_st],
                                                    num_nodes=local_num,
                                                    prev_rowsum_states=prev_states)
                tot_ll = tot_ll + ll
                loss = -ll / len(g)
                if stage + 1 == num_stages and rank == rank_last:
                    loss.backward()
                else:
                    top_grad = recv_states(cur_states[0].shape[0], rank + 1 if rank != rank_last else 0)
                    torch.autograd.backward([loss, *cur_states], [None, *top_grad])
                if prev_states[0] is not None:
                    grads = [x.grad.detach() for x in prev_states]
                    dst = rank - 1 if rank else cmd_args.num_proc - 1
                    send_states(grads, dst)
            dist.all_reduce(tot_ll.data, op=dist.ReduceOp.SUM)

            for param in model.parameters():
                if param.grad is None:
                    param.grad = param.data.new(param.data.shape).zero_()
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
            optimizer.step()

            if rank == 0:
                loss = -tot_ll.item() / len(g)
                pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (e_it + 1) / cmd_args.epoch_save, loss))
                torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % epoch))


    print('done rank', rank)

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo')

    main(dist.get_rank())
