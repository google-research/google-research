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

# pylint: skip-file
import argparse
import os
import pickle as cp
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for debm', allow_abbrev=False)
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_dir', default='.', help='data dir')
cmd_opt.add_argument('-data', default=None, help='data name')

cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-model_dump', default=None, help='load model dump')
cmd_opt.add_argument('-energy_model_dump', default=None, help='load energy model dump')
cmd_opt.add_argument('-sampler_model_dump', default=None, help='load sampler model dump')

cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-num_proc', type=int, default=1, help='number of processes')

cmd_opt.add_argument('-embed_dim', default=256, type=int, help='embed size')
cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='gradient clip')
cmd_opt.add_argument('-weight_clip', default=-1, type=float, help='weight clip')

cmd_opt.add_argument('-ent_coeff', default=1.0, type=float, help='coeff of entropy')
cmd_opt.add_argument('-base_coeff', default=1.0, type=float, help='coeff of baseline reg loss')
cmd_opt.add_argument('-gamma', default=1.0, type=float, help='gamma in rl')

cmd_opt.add_argument('-num_epochs', default=100000, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size')
cmd_opt.add_argument('-gibbs_rounds', default=1, type=int, help='rounds of gibbs sampling')
cmd_opt.add_argument('-num_importance_samples', default=5, type=int, help='number of importance samples')

cmd_opt.add_argument('-num_q_steps', default=0, type=int, help='q steps')
cmd_opt.add_argument('-persistent_frac', default=0, type=float, help='frac of persistent samples')

cmd_opt.add_argument('-epoch_save', default=100, type=int, help='num epochs between save')
cmd_opt.add_argument('-iter_per_epoch', default=100, type=int, help='num iterations per epoch')

cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch for loading')


cmd_opt.add_argument('-proposal', default='uniform', help='uniform/inv/downhill')

cmd_opt.add_argument('-mu0', default=0.5, type=float, help='percentage of sampling from init')

# synthetic
cmd_opt.add_argument('-discrete_dim', default=32, type=int, help='embed size')
cmd_opt.add_argument('-binmode', default='normal', help='normal/remap')
cmd_opt.add_argument('-learn_mode', default='ebm', help='ebm/mle', choices=['ebm', 'mle'])
cmd_opt.add_argument('-lb_type', default='uni', help='uni/is', choices=['uni', 'is'])
cmd_opt.add_argument('-base_type', default='mlp', help='mlp/rnn/iid', choices=['mlp', 'rnn', 'iid'])

cmd_opt.add_argument('-f_lr_scale', default=0.2, type=float, help='scale of f learning rate')
cmd_opt.add_argument('-q_iter', default=1, type=int, help='# rounds for updating q')

cmd_opt.add_argument('-f_bound', default=-1, type=float, help='scale of f')

cmd_opt.add_argument('-int_scale', default=None, type=float, help='scale for float2int')
cmd_opt.add_argument('-plot_size', default=None, type=float, help='scale for plot')
cmd_opt.add_argument("-shuffle_edit", default=True, type=eval, help="shuffle edit?")
cmd_opt.add_argument("-learn_q0", default=True, type=eval, help="learn q0?")
cmd_opt.add_argument("-with_replacement", default=True, type=eval, help="sampling with replacement?")
cmd_opt.add_argument("-learn_stop", default=True, type=eval, help="learn when to stop?")

#robust fill

cmd_opt.add_argument("-maxInputTokens", default=5, type=int, help="maximum number of type of tokens in inupt strings")
cmd_opt.add_argument("-maxInputLength", default=20, type=int, help="maximum size of input strings")
cmd_opt.add_argument("-maxOutputLength", default=50, type=int, help="maximum size of output strings")
cmd_opt.add_argument("-numExamples", default=10, type=int, help="number of input output examples")
cmd_opt.add_argument("-numPublicIO", default=4, type=int, help="number of public input output examples")
cmd_opt.add_argument("-numPrivateIO", default=6, type=int, help="number of private input output examples")

cmd_opt.add_argument("-maxNumConcats", default=6, type=int, help="maximum number of concats in our final program")
cmd_opt.add_argument("-maxK", default=4, type=int, help="maximum value of K parameter, -5 to 5")

cmd_opt.add_argument("-rnn_layers", default=3, type=int, help="n_layers in lstm")
cmd_opt.add_argument("-eval_topk", default=1, type=int, help="topk evaluation")

cmd_opt.add_argument("-mh", default='local', type=str, help="local/naive")
cmd_opt.add_argument("-score_func", default=None, type=str, help="score func type")
cmd_opt.add_argument("-io_enc", default='rnn', type=str, help="rnn/mlp")
cmd_opt.add_argument("-masked", default=False, type=eval, help="masked pred?")
cmd_opt.add_argument("-rnn_state_proj", default=False, type=eval, help="learn rnn state proj?")

cmd_opt.add_argument("-rlang", default='short', type=str, help="short/long version of serial robust fill")

cmd_opt.add_argument("-inf_type", default='argmax', type=str, help="inference type")
cmd_opt.add_argument("-eval_method", default='argmax', type=str, help="evaluation method")
cmd_opt.add_argument("-io_agg_type", default='mean', type=str, help="mean/max pooling of io pairs")
cmd_opt.add_argument("-cell_type", default='lstm', type=str, help="gru/lstm")
cmd_opt.add_argument("-tok_type", default='embed', type=str, help="embed/onehot")
cmd_opt.add_argument("-io_embed_type", default='normal', type=str, help="normal/masked")

# fuzz
cmd_opt.add_argument('-window_size', default=100, type=int, help='window size')
cmd_opt.add_argument('-kernel_size', default=7, type=int, help='kernel size')
cmd_opt.add_argument('-num_change', default=100, type=int, help='number of positions to change')

cmd_opt.add_argument('-stride', default=1, type=int, help='stride')
cmd_opt.add_argument('-num_gen', default=100, type=int, help='num gen for fuzzing')

cmd_opt.add_argument('-f_scale', default=1, type=float, help='scale of f func')
cmd_opt.add_argument("-f_out", default='identity', type=str, help="out func of f", choices=["identity", "tanh", "elu", "relu"])

# gnn
cmd_opt.add_argument('-gnn_msg_dim', default=64, type=int, help='dim of message passing in gnn')
cmd_opt.add_argument('-max_lv', default=3, type=int, help='# layers of gnn')
cmd_opt.add_argument('-msg_agg_type', default='sum', help='how to aggregate the message')
cmd_opt.add_argument('-att_type', default='inner_prod', help='mlp/inner_prod')
cmd_opt.add_argument('-readout_agg_type', default='sum', help='how to aggregate all node embeddings', choices=['sum', 'max', 'mean'])
cmd_opt.add_argument('-gnn_out', default='last', help='how to aggregate readouts from different layers', choices=['last', 'sum', 'max', 'gru', 'mean'])
cmd_opt.add_argument('-gnn_type', default='s2v_single', help='type of graph neural network', choices=['s2v_code2inv', 's2v_single', 's2v_multi', 'ggnn'])
cmd_opt.add_argument('-act_func', default='tanh', help='default activation function')

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

if cmd_args.epoch_load is not None and cmd_args.model_dump is None:
    cmd_args.energy_model_dump = os.path.join(cmd_args.save_dir, 'score_func-%d.ckpt' % cmd_args.epoch_load)
    cmd_args.sampler_model_dump = os.path.join(cmd_args.save_dir, 'sampler-%d.ckpt' % cmd_args.epoch_load)

print(cmd_args)
assert cmd_args.numPublicIO + cmd_args.numPrivateIO == cmd_args.numExamples

def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
