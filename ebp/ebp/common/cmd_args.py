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

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
from argparse import Namespace
import os

cmd_opt = argparse.ArgumentParser(
    description='Argparser for fp gan', allow_abbrev=False)
cmd_opt.add_argument(
    '-saved_model', default=None, help='start from existing model')
cmd_opt.add_argument('-save_dir', default=None, help='save folder')
cmd_opt.add_argument('-cfg_file', default=None, help='cfg')
cmd_opt.add_argument('-init_model_dump', default=None, help='load model dump')

cmd_opt.add_argument('-ctx', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-phase', default='train', help='train/test')
cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
cmd_opt.add_argument(
    '-batch_size', type=int, default=100, help='minibatch size')
cmd_opt.add_argument(
    '-num_ctx', type=int, default=10, help='max num of ctx points')

cmd_opt.add_argument('-iter_eval', type=int, default=100, help='iters per eval')

cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-epoch_load', type=int, default=-1, help='epoch')
cmd_opt.add_argument(
    '-gnorm_lambda', type=float, default=0.0, help='lambda for gradient norm')
cmd_opt.add_argument(
    '-kl_lambda', type=float, default=1.0, help='lambda for kl')
cmd_opt.add_argument(
    '-ent_lam', type=float, default=1.0, help='lambda for entropy')
cmd_opt.add_argument(
    '-gnorm_type',
    type=str,
    default='lp1',
    help='type for gradient norm (lp1 || norm2)')
cmd_opt.add_argument('-f_depth', type=int, default=2, help='depth of f')
cmd_opt.add_argument(
    '-num_epochs', type=int, default=50000, help='number of epochs')
cmd_opt.add_argument(
    '-iters_per_eval', type=int, default=100, help='iterations per evaluation')
cmd_opt.add_argument(
    '-nn_hidden_size', type=int, default=128, help='dimension of mlp layers')
cmd_opt.add_argument(
    '-learning_rate', type=float, default=0.001, help='init learning_rate')
cmd_opt.add_argument('-verb', type=int, default=1, help='display info')
cmd_opt.add_argument(
    '-f_bd', type=float, default=0.01, help='kernel bandwidth of f')
cmd_opt.add_argument(
    '-kde_sigma', type=float, default=0.1, help='kernel bandwidth of kde')
cmd_opt.add_argument(
    '-mmd_bd', type=float, default=0.1, help='kernel bandwidth of mmd')
cmd_opt.add_argument(
    '-gp_lambda', type=float, default=0, help='use gradient penalty')
cmd_opt.add_argument('-hmc_clip', type=float, default=-1, help='hmc clip')
cmd_opt.add_argument('-ema_decay', type=float, default=0.99, help='ema decay')

cmd_opt.add_argument(
    '-z_dim', type=int, default=64, help='dimension of latent variable')
cmd_opt.add_argument(
    '-g_iter', type=int, default=5, help='iters of generator update')
cmd_opt.add_argument('-grad_clip', type=int, default=5, help='clip of gradient')
cmd_opt.add_argument(
    '-energy_type', type=str, default='mlp', help='type for energy func')
cmd_opt.add_argument(
    '-num_landmarks', type=int, default=1000, help='number of landmarks')

# args for gaussian experiment
cmd_opt.add_argument(
    '-gauss_dim', type=int, default=1, help='dimension of gaussian')

# args for ring experiment
cmd_opt.add_argument(
    '-ring_dim', type=int, default=2, help='dimension of ring data')
cmd_opt.add_argument('-fix_phi', type=int, default=0, help='fix phi or not')
cmd_opt.add_argument(
    '-ring_radius',
    type=str,
    default='1,3,5',
    help='list of int, radius of each ring')
cmd_opt.add_argument(
    '-data_dump', type=str, default=None, help='synthetic data dump')
cmd_opt.add_argument(
    '-data_name', type=str, default=None, help='synthetic data name')

# args for generator
cmd_opt.add_argument(
    '-score_type', type=str, default='agg', help='score_type [agg, prod]')
cmd_opt.add_argument(
    '-score_func',
    type=str,
    default='single',
    help='score func [single, mixture]')

cmd_opt.add_argument(
    '-flow_type',
    type=str,
    default='planar',
    help='type for flows (planar || ires)')
cmd_opt.add_argument(
    '-flow_form',
    type=str,
    default='param',
    help='form of flows (param || hyper)')
cmd_opt.add_argument(
    '-num_flows',
    type=int,
    default=1,
    help='number of flows in the mixture model')
cmd_opt.add_argument('-gen_depth', type=int, default=10, help='depth of flow')

cmd_opt.add_argument(
    '-iaf_hidden',
    type=int,
    default=16,
    help='hidden dimension of autoregressive nn')
cmd_opt.add_argument(
    '-sp_iters', type=int, default=0, help='spectrum norm iters')

# args for hmc
cmd_opt.add_argument(
    '-mcmc_type',
    type=str,
    default='None',
    help='type for mcmc',
    choices=['None', 'HMC', 'GeneralHmc', 'ResGeneralHmc', 'SGLD'])
cmd_opt.add_argument(
    '-use_mh',
    type=eval,
    default=True,
    help='use rejection sampling?',
    choices=[True, False])
cmd_opt.add_argument(
    '-mcmc_steps', type=int, default=1, help='number of mcmc steps')
cmd_opt.add_argument(
    '-hmc_inner_steps', type=int, default=1, help='number of hmc inner steps')
cmd_opt.add_argument(
    '-use_2nd_order_grad', type=eval, default=True, choices=[True, False])
cmd_opt.add_argument(
    '-clip_samples', type=eval, default=False, choices=[True, False])
cmd_opt.add_argument(
    '-sigma_eps',
    type=float,
    default=1e-1,
    help='std scale of reparameterization')

# args for sgld
cmd_opt.add_argument(
    '-sgld_noise_std',
    type=float,
    default=1.0,
    help='std of injected noise for LD update')
cmd_opt.add_argument(
    '-sgld_clip_value',
    type=float,
    default=1.0,
    help='clip value for gradient in LD update')
cmd_opt.add_argument(
    '-sgld_clip_mode',
    type=str,
    default='norm',
    help='type of gradient clipping in LD update',
    choices=['norm', 'value'])
cmd_opt.add_argument(
    '-moment_penalty',
    type=float,
    default=0,
    help='coefficient for norm of momentums in hmc')
cmd_opt.add_argument(
    '-hmc_adaptive_mode',
    type=str,
    default='human',
    help='adaptive tuning for hmc',
    choices=['auto', 'human', 'none'])

cmd_opt.add_argument(
    '-hmc_step_size', type=float, default=1e-2, help='hmc step size')
cmd_opt.add_argument(
    '-hmc_p_sigma', type=float, default=1.0, help='momentum p ~ N(0, sigma)')

# args for adam params
cmd_opt.add_argument('-beta1', type=float, default=0.9)
cmd_opt.add_argument('-beta2', type=float, default=0.999)

cmd_opt.add_argument('-img_size', type=int, default=28, help='size of img')
cmd_opt.add_argument('-binary', type=int, default=0, help='binary img')
cmd_opt.add_argument('-vis_num', type=int, default=0, help='vis img')

cmd_opt.add_argument(
    '-test_batch_size', type=int, default=100, help='bsize test')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='latent')

cmd_opt.add_argument('-data_mean', type=float, default=0.0, help='mean')
cmd_opt.add_argument('-data_std', type=float, default=1.0, help='std')
cmd_opt.add_argument(
    '-net_type', type=str, default='mlp', help='type for vae (cnn || mlp)')

cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
  if not os.path.isdir(cmd_args.save_dir):
    os.makedirs(cmd_args.save_dir)
print(cmd_args)
