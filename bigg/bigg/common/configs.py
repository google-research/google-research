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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# pylint: skip-file
import argparse
import os
import pickle as cp
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for grecur', allow_abbrev=False)
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_dir', default='.', help='data dir')
cmd_opt.add_argument('-eval_folder', default=None, help='data eval_dir')
cmd_opt.add_argument('-train_method', default='full', help='full/stage')
cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-g_type', default=None, help='graph type')
cmd_opt.add_argument('-model_dump', default=None, help='load model dump')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-num_proc', type=int, default=1, help='number of processes')
cmd_opt.add_argument('-node_order', default='default', help='default/DFS/BFS/degree_descent/degree_accent/k_core/all, or any of them concat by +')

cmd_opt.add_argument('-dist_backend', default='gloo', help='dist package backend', choices=['gloo', 'nccl'])

cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-bits_compress', default=256, type=int, help='num of bits to compress')
cmd_opt.add_argument('-param_layers', default=1, type=int, help='num of param groups')
cmd_opt.add_argument('-num_test_gen', default=-1, type=int, help='num of graphs generated for test')
cmd_opt.add_argument('-max_num_nodes', default=-1, type=int, help='max num of nodes')


cmd_opt.add_argument('-rnn_layers', default=2, type=int, help='num layers in rnn')
cmd_opt.add_argument('-seed', default=34, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='gradient clip')
cmd_opt.add_argument('-train_ratio', default=0.8, type=float, help='ratio for training')
cmd_opt.add_argument('-dev_ratio', default=0.2, type=float, help='ratio for dev')
cmd_opt.add_argument('-greedy_frac', default=0, type=float, help='prob for greedy decode')

cmd_opt.add_argument('-num_epochs', default=100000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=10, type=int, help='batch size')
cmd_opt.add_argument('-pos_enc', default=True, type=eval, help='pos enc?')
cmd_opt.add_argument('-pos_base', default=10000, type=int, help='base of pos enc')

cmd_opt.add_argument('-old_model', default=False, type=eval, help='old model dumps?')

cmd_opt.add_argument('-tree_pos_enc', default=False, type=eval, help='pos enc for tree?')

cmd_opt.add_argument('-blksize', default=-1, type=int, help='num blksize steps')
cmd_opt.add_argument('-accum_grad', default=1, type=int, help='accumulate grad for batching purpose')

cmd_opt.add_argument('-epoch_save', default=100, type=int, help='num epochs between save')
cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')

cmd_opt.add_argument('-batch_exec', default=False, type=eval, help='run with dynamic batching?')

cmd_opt.add_argument('-share_param', default=True, type=eval, help='share param in each level?')
cmd_opt.add_argument('-directed', default=False, type=eval, help='is directed graph?')
cmd_opt.add_argument('-self_loop', default=False, type=eval, help='has self-loop?')
cmd_opt.add_argument('-bfs_permute', default=False, type=eval, help='random permute with bfs?')
cmd_opt.add_argument('-display', default=False, type=eval, help='display progress?')

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.epoch_load is not None and cmd_args.model_dump is None:
    cmd_args.model_dump = os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % cmd_args.epoch_load)

print(cmd_args)

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
