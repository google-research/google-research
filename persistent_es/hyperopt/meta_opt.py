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

"""Script to train MLPs or ResNets on MNIST or CIFAR-10, tuning optimization hyperparameters with ES/PES.

# ES
# --
python meta_opt.py \
    --dataset=mnist \
    --batch_size=100 \
    --shuffle=True \
    --model=mlp \
    --nlayers=2 \
    --nhid=100 \
    --inner_optimizer=sgdm \
    --sgdm_type=0 \
    --objective=train_sum_loss \
    --resample_fixed_minibatch=True \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=3e-2 \
    --outer_iterations=50000 \
    --log_every=5000 \
    --print_every=100 \
    --random_hparam_init=False \
    --lr0=0.01 \
    --lr1=10.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.3 \
    --N=100 \
    --estimate=pes \
    --save_dir=saves/mnist

# PES
# ---
python meta_opt.py \
    --dataset=mnist \
    --batch_size=100 \
    --shuffle=True \
    --model=mlp \
    --nlayers=2 \
    --nhid=100 \
    --inner_optimizer=sgdm \
    --sgdm_type=0 \
    --objective=train_sum_loss \
    --resample_fixed_minibatch=True \
    --tune_params=lr:inverse-time-decay \
    --outer_optimizer=adam \
    --outer_lr=3e-2 \
    --outer_iterations=50000 \
    --log_every=5000 \
    --print_every=100 \
    --random_hparam_init=False \
    --lr0=0.01 \
    --lr1=10.0 \
    --T=5000 \
    --K=10 \
    --sigma=0.3 \
    --N=100 \
    --estimate=pes \
    --save_dir=saves/mnist
"""
import os
import sys
# import csv
# import ipdb
import time
import random
import pickle as pkl
from functools import partial
from collections import defaultdict

import numpy as onp

import jax
from jax.config import config
# config.update('jax_disable_jit', True)
import jax.numpy as jnp
from jax import flatten_util
from jax.tree_util import tree_flatten, tree_unflatten

import haiku as hk

import tensorflow.io.gfile as gfile

from absl import app
from absl import flags

# Local imports
import models
import aug_utils
import data_utils
import hparam_utils
import general_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'Dataset')
flags.DEFINE_string('model', 'mlp', 'Model')
flags.DEFINE_string('model_size', 'small', 'Choose model size, that affects #channels in the ResNet (tiny, small, medium, or large)')
flags.DEFINE_integer('nlayers', 4, 'Number of layers in the MLP')
flags.DEFINE_integer('nhid', 100, 'Number of hidden units in each layer of the MLP')
flags.DEFINE_bool('use_bn', False, 'Whether to use BatchNorm in the MLP')
flags.DEFINE_bool('with_bias', False, 'Whether to use a bias term in each layer of the MLP')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_bool('shuffle', False, 'Set this to not shuffle the training minibatches (should yield deterministic data ordering)')
flags.DEFINE_bool('resample_fixed_minibatch', False, 'Whether to randomly re-sample the fixed minibatch at the start of each inner problem')
flags.DEFINE_bool('reset_params', False, 'Randomly initialize the base model parameters differently at the start of each inner problem')

flags.DEFINE_integer('outer_iterations', 50000, 'Number of meta-optimization iterations')
flags.DEFINE_string('outer_optimizer', 'adam', 'Outer optimizer')
flags.DEFINE_float('outer_lr', 1e-2, 'Outer learning rate')
flags.DEFINE_float('outer_b1', 0.9, 'Outer optimizer Adam b1 hyperparameter')
flags.DEFINE_float('outer_b2', 0.99, 'Outer optimizer Adam b2 hyperparameter')
flags.DEFINE_float('outer_eps', 1e-8, 'Outer optimizer Adam epsilon hyperparameter')
flags.DEFINE_float('outer_gamma', 0.9, 'Outer RMSprop gamma hyperparameter')
flags.DEFINE_float('outer_momentum', 0.9, 'Outer optimizer momentum')
flags.DEFINE_float('outer_lr_decay', 0.01, 'Factor by which to decay the outer LR')
flags.DEFINE_integer('outer_lr_decay_iter', -1, 'Total iteration number at which to decay the outer lr')

flags.DEFINE_string('lr_transform', 'base_e', 'Exponential function to transform the LR, either base 2, base e, or base 10')
flags.DEFINE_float('outer_clip', -1, 'Outer gradient clipping (-1 means no clipping)')
flags.DEFINE_float('grad_clip', -1, 'Gradient clipping for each step of the inner unroll (-1 means no grad clipping)')
flags.DEFINE_string('grad_clip_type', 'hardclip', 'Choose the type of gradient clipping for inner unrolls (normclip or hardclip)')
flags.DEFINE_float('inner_step_clip', -1, 'Step clipping for the inner unroll (-1 means no step clipping)')

flags.DEFINE_string('loss', 'softmax', 'Choose the type of loss (softmax, mse)')
flags.DEFINE_string('inner_optimizer', 'sgdm', 'Inner optimizer')
flags.DEFINE_integer('sgdm_type', 1, 'Choose which implementation of SGDm to use (0 or 1)')
flags.DEFINE_string('tune_params', 'lr:fixed', 'A comma-separated string of hyperparameters to search over')
flags.DEFINE_string('objective', 'train_sum_loss', 'The objective for meta-optimization')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_float('b1', 0.99, 'Adam b1 hyperparameter')
flags.DEFINE_float('b2', 0.999, 'Adam b2 hyperparameter')
flags.DEFINE_float('eps', 1e-8, 'Adam epsilon hyperparameter')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_float('weight_decay', 1e-10, 'Weight decay')
flags.DEFINE_integer('num_pieces', 2, 'Number of pieces of a piecewise linear schedule')
flags.DEFINE_float('lr0', 1e-2, 'Initial learning rate for linear LR schedule')
flags.DEFINE_float('lr1', 1e-2, 'Final learning rate for linear LR schedule')
flags.DEFINE_bool('random_hparam_init', True, 'Whether to initialize the hyperparameters to random values')

flags.DEFINE_string('estimate', 'pes', 'Type of gradient estimate (es or pes)')
flags.DEFINE_integer('T', 2500, 'Maximum number of iterations of the inner loop')
flags.DEFINE_integer('K', 10, 'Number of steps to unroll (== truncation length)')
flags.DEFINE_integer('N', 4, 'Number of ES/PES particles')
flags.DEFINE_float('sigma', 0.1, 'Variance for ES /PESperturbations')
flags.DEFINE_bool('use_new_indexing', False, 'Whether to use the new indexing')
flags.DEFINE_bool('clip_objective', False, 'Whether or not to clip the objective values during ES/PES meta-opt when they get huge/NaN')

flags.DEFINE_string('preset', None, 'Optional preset to set a specific configuration of arguments')
flags.DEFINE_integer('print_every', 100, 'Print theta every N iterations')
flags.DEFINE_integer('log_every', 1000, 'Log the full training and val losses to the CSV log every N iterations')
flags.DEFINE_string('save_dir', 'saves', 'Save directory')
flags.DEFINE_integer('seed', 3, 'Random seed')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tune_params_str = FLAGS.tune_params or 'None'

  if FLAGS.tune_params:
      FLAGS.tune_params = [{'param': hparam_utils.abbreviation_dict[p.split(':')[0]],
                            'sched': hparam_utils.abbreviation_dict[p.split(':')[1]]} for p in FLAGS.tune_params.split(',')]

  schedule_info = {}
  for item in FLAGS.tune_params:
    schedule_info[item['param']]= item['sched']

  # Set random seed for reproducibility
  if FLAGS.seed is not None:
      random.seed(FLAGS.seed)
      onp.random.seed(FLAGS.seed)

  # The experiment directory name contains everything except for the seed, so that multiple exps with identical params
  # except for the random seed will be put into the same subdir
  exp_dir = '{}'.format(FLAGS.objective)
  exp_name = '{}-{}-{}-obj:{}-tune:{}-T:{}-K:{}-N:{}-sigma:{}-olr:{}-seed:{}'.format(
              FLAGS.estimate, FLAGS.dataset, FLAGS.model, FLAGS.objective, tune_params_str,
              FLAGS.T, FLAGS.K, FLAGS.N, FLAGS.sigma, FLAGS.outer_lr, FLAGS.seed)

  save_dir = os.path.join(FLAGS.save_dir, exp_dir, exp_name)
  gfile.makedirs(save_dir)

  key_flags = FLAGS.get_key_flags_for_module(argv[0])
  myflagdict = {myflag.name: myflag.value for myflag in key_flags}
  with gfile.GFile(os.path.join(save_dir, 'args.pkl'), 'w') as f:
    pkl.dump(myflagdict, f)

  with gfile.GFile(os.path.join(save_dir, 'command.txt'), 'w') as f:
      f.write('\n'.join(['python {} \\'.format(sys.argv[0])] + ["    {} \\".format(line) for line in sys.argv[1:-1]] + ['    {}'.format(sys.argv[-1])]))

  # Create dataloaders
  # ----------------------------------------
  (train_data, train_targets), (val_data, val_targets), (test_data, test_targets) = data_utils.load_data(FLAGS.dataset)

  key = jax.random.PRNGKey(FLAGS.seed)
  key, skey = jax.random.split(key, 2)
  all_train_xs, all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
  all_val_xs, all_val_ys = data_utils.create_minibatches(skey, val_data, val_targets, FLAGS.batch_size)
  all_test_xs, all_test_ys = data_utils.create_minibatches(skey, test_data, test_targets, FLAGS.batch_size)
  num_train_batches = len(all_train_xs)

  fixed_val_inputs = all_val_xs[0]
  fixed_val_targets = all_val_ys[0]
  fixed_train_inputs = all_train_xs[0]
  fixed_train_targets = all_train_ys[0]

  # Make these fixed minibatches 4x the size of standard minibatches
  # fixed_val_inputs = jnp.concatenate(all_val_xs[0:4])
  # fixed_val_targets = jnp.concatenate(all_val_ys[0:4])
  # fixed_train_inputs = jnp.concatenate(all_train_xs[0:4])
  # fixed_train_targets = jnp.concatenate(all_train_ys[0:4])

  print('all_train_xs.shape: {} | all_train_ys.shape: {}'.format(all_train_xs.shape, all_train_ys.shape))
  print('all_val_xs.shape: {} | all_val_ys.shape: {}'.format(all_val_xs.shape, all_val_ys.shape))
  print('all_test_xs.shape: {} | all_test_ys.shape: {}'.format(all_test_xs.shape, all_test_ys.shape))
  # ----------------------------------------

  # Initialize model
  # =======================================================================
  def net_fn(x, is_training):
    if FLAGS.model == 'resnet':
      net = models.Net(FLAGS.model_size)
      return net(x, is_training=is_training)
    elif FLAGS.model == 'mlp':
      mlp = models.MLP(nlayers=FLAGS.nlayers, nhid=FLAGS.nhid, with_bias=FLAGS.with_bias, batch_norm=FLAGS.use_bn)
      return mlp(x, is_training)
    elif FLAGS.model == 'tinyconv':
      net = models.TinyConv()
      return net(x)

  net = hk.transform_with_state(net_fn)
  apply_jit = jax.jit(net.apply, static_argnums=(4,))

  # Initialize the network parameters
  init_images = all_train_xs[0]
  print('Initializing parameters...')
  key_for_init = jax.random.PRNGKey(FLAGS.seed)
  params, state = net.init(key_for_init, init_images, is_training=True)
  print('Num parameters: {}'.format(general_utils.count_params(params)))
  sys.stdout.flush()

  num_classes = 10
  def loss_fn_softmax(params, images, labels, state, is_training):
      logits, state = apply_jit(params, state, None, images, is_training)
      labels = hk.one_hot(labels, num_classes)
      softmax_xent = jnp.mean(-jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))
      return softmax_xent, (logits, state)

  def loss_fn_mse(params, images, labels, state, is_training):
      preds, state = apply_jit(params, state, None, images, is_training)
      labels = hk.one_hot(labels, num_classes)
      mse = jnp.mean(jnp.sum((labels - preds)**2, axis=-1))
      return mse, (preds, state)

  if FLAGS.loss == 'mse':
      loss_fn = loss_fn_mse
  elif FLAGS.loss == 'softmax':
      loss_fn = loss_fn_softmax

  loss_fn = jax.jit(loss_fn, static_argnums=(4,))
  loss_grad = jax.jit(jax.grad(loss_fn, has_aux=True), static_argnums=(4,))
  loss_value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=(4,))
  # =======================================================================


  # Hyperparameter schedules
  # =======================================================================
  def get_fixed_value(theta, param_name, t, T):
      return theta

  def get_linear_sched_value(theta, param_name, t, T):
      """Here theta is assumed to be 2D"""
      init_value = theta[0]
      final_value = theta[1]
      return init_value * (T - t) / T + final_value * t / T

  def get_inverse_time_decay_sched_value(theta, param_name, t, T):
      init_value = theta[0]
      decay = theta[1]
      return init_value / ((1 + t / 5000.0)**decay)

  def get_piecewise_sched_value(theta, param_name, t, T):
      ts = jax.nn.softmax(theta[:FLAGS.num_pieces])  # Get the values in theta that represent the locations of the knots
      ts = jnp.cumsum(ts)  # Convert [0.25, 0.25, 0.25, 0.25] --> [0.25, 0.5, 0.75, 1]
      ts = jnp.concatenate([jnp.array([0.0]), ts])  # Explicitly add the 0 mark --> [0, 0.25, 0.5, 0.75, 1]
      ts = ts * T  # Make sure to convert the fractional values to iteration numbers like [0, 25, 50, 75, 100] if T=100
      theta_constrained = hparam_utils.cons_func_dict[param_name](theta[FLAGS.num_pieces:])  # The rest of theta represent the hyperparam values at the knots
      interp = jnp.array(0)
      for i in range(1,len(ts)):
          fraction = (t - ts[i-1]) / (ts[i] - ts[i-1])
          interp_value = fraction * theta_constrained[i] + (1 - fraction) * theta_constrained[i-1]
          interp += interp_value * (ts[i-1] <= t) * (t < ts[i])
      return interp

  def get_fixed_value_single(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda x: theta, optim_params[param_name])
      return optim_tree

  def get_linear_sched_value_single(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda x: get_linear_sched_value(theta, param_name, t, T), optim_params[param_name])
      return optim_tree

  def get_inverse_time_decay_sched_value_single(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda x: get_inverse_time_decay_sched_value(theta, param_name, t, T), optim_params[param_name])
      return optim_tree

  def get_piecewise_sched_value_single(optim_params, theta, param_name, t, T):
      optim_tree = jax.tree_map(lambda x: get_piecewise_sched_value(theta, param_name, t, T), optim_params[param_name])
      return optim_tree

  def get_fixed_value_pl(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      return unflatten_func_dict[param_name](theta)

  def get_linear_sched_value_pl(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      hparam_tree = unflatten_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda theta_val: get_linear_sched_value(theta_val, param_name, t, T), hparam_tree)
      return optim_tree

  def get_inverse_time_decay_sched_value_pl(optim_params, theta, param_name, t, T):
      theta = hparam_utils.cons_func_dict[param_name](theta)
      hparam_tree = unflatten_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda theta_val: get_inverse_time_decay_sched_value(theta_val, param_name, t, T), hparam_tree)
      return optim_tree

  def get_piecewise_sched_value_pl(optim_params, theta, param_name, t, T):
      hparam_tree = unflatten_func_dict[param_name](theta)
      optim_tree = jax.tree_map(lambda theta_val: get_piecewise_sched_value(theta_val, param_name, t, T), hparam_tree)
      return optim_tree

  schedule_funcs = { 'fixed': get_fixed_value_single,
                     'linear': get_linear_sched_value_single,
                     'inverse-time-decay': get_inverse_time_decay_sched_value_single,
                     'piecewise': get_piecewise_sched_value_single,
                     'fixed-pl': get_fixed_value_pl,
                     'linear-pl': get_linear_sched_value_pl,
                     'inverse-time-decay-pl': get_inverse_time_decay_sched_value_pl,
                     'piecewise-pl': get_piecewise_sched_value_pl,
                   }

  def get_inner_optim_params(inner_optim_params, theta, t, T):
    for setting in FLAGS.tune_params:
      param = setting['param']
      sched = setting['sched']
      if param not in ['lr', 'momentum', 'b1', 'b2', 'eps', 'weight_decay']:  # Only deal with optimization hparams here
        continue
      theta_subset = theta[idx_dict[param]]
      inner_optim_params[param] = schedule_funcs[sched](inner_optim_params, theta_subset, param, t, T)
    return inner_optim_params

  # Generic way to set theta
  default_value = { 'lr': FLAGS.lr,
                    'b1': FLAGS.b1,
                    'b2': FLAGS.b2,
                    'eps': FLAGS.eps,
                    'momentum': FLAGS.momentum,
                    'weight_decay': FLAGS.weight_decay,  # Just so it's not equal to 0, which causes problems with jax.log
                    'cutoutsize': 4.0,
                    'dropout_prob': 0.1,
                    'hflip_prob': 0.1,
                    'vflip_prob': 0.1,
                    'crop_border': 2,
                    'hue_jitter': 0.1,
                    'sat_jitter': 0.1,
                  }

  key = jax.random.PRNGKey(FLAGS.seed)  # Only needed here when using --random_hparam_init

  theta_vals = []
  idx_dict = {}
  setting_idx_dict = {}
  unflatten_func_dict = {}
  idx = 0
  for setting in FLAGS.tune_params:
      key, skey = jax.random.split(key)  # Only needed when using --random_hparam_init
      param = setting['param']
      sched = setting['sched']
      default = hparam_utils.uncons_func_dict[param](default_value[param])
      min_range, max_range = hparam_utils.hparam_range_dict[param]
      if sched == 'fixed':
          if FLAGS.random_hparam_init:
              theta_vals.append(jax.random.uniform(skey, (1,)) * (max_range - min_range) + min_range)
          else:
              theta_vals.append(jnp.array([default]))
          idx_dict[param] = idx
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += 1
      elif sched == 'linear':
          if FLAGS.random_hparam_init:
              theta_vals.append(jax.random.uniform(skey, (2,)) * (max_range - min_range) + min_range)
          else:
              # theta_vals.append(jnp.array([default, default]))
              lr0 = hparam_utils.uncons_func_dict['lr'](FLAGS.lr0)
              lr1 = hparam_utils.uncons_func_dict['lr'](FLAGS.lr1)
              theta_vals.append(jnp.array([lr0, lr1]))
          idx_dict[param] = jnp.array([idx, idx+1])
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += 2
      elif sched == 'inverse-time-decay':
          if FLAGS.random_hparam_init:
              key, key1, key2 = jax.random.split(key, 3)
              decay_min_range, decay_max_range = -4, 4  # A fixed range for the decay factor regardless of which hyperparameter we're dealing with
              sampled_init_values = jax.random.uniform(key1, (1,)) * (max_range - min_range) + min_range
              sampled_decay_values = jax.random.uniform(key2, (1,)) * (decay_max_range - decay_min_range) + decay_min_range
              theta_vals.append(jnp.concatenate([sampled_init_values, sampled_decay_values], axis=0))
          else:
              # theta_vals.append(jnp.array([default, default]))
              lr0 = hparam_utils.uncons_func_dict['lr'](FLAGS.lr0)
              lr1 = hparam_utils.uncons_func_dict['lr'](FLAGS.lr1)
              theta_vals.append(jnp.array([lr0, lr1]))
          idx_dict[param] = jnp.array([idx, idx+1])
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += 2
      elif sched == 'piecewise':  # TODO: Random hparam init not supported here yet!
          knot_locations = jnp.zeros(FLAGS.num_pieces)
          knot_values = jnp.ones(FLAGS.num_pieces+1) * default  # XXXXXXXXXX ORIGINAL!!
          # knot_values = jnp.array([hparam_utils.uncons_func_dict[param](1e-9), hparam_utils.uncons_func_dict[param](0.4), hparam_utils.uncons_func_dict[param](1e-9)])  # XXXXXXXX MODIFIED MANUALLY!
          theta_vals += [knot_locations, knot_values]
          idx_dict[param] = jnp.array(list(range(idx, idx+len(knot_locations)+len(knot_values))))
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += len(knot_locations) + len(knot_values)
      elif sched == 'fixed-pl':
          if FLAGS.random_hparam_init:
              leaves, treedef = tree_flatten(params)
              random_values = jax.random.uniform(skey, (len(leaves),)) * (max_range - min_range) + min_range
              random_values = [random_values[i] for i in range(len(random_values))]
              hparam_tree = tree_unflatten(treedef, random_values)
          else:
              hparam_tree = jax.tree_map(lambda x: jnp.array(default), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += len(hparam_vector)
      elif sched in ['linear-pl', 'inverse-time-decay-pl']:
          if FLAGS.random_hparam_init:
              leaves, treedef = tree_flatten(params)
              random_values = jax.random.uniform(skey, (len(leaves),2)) * (max_range - min_range) + min_range
              random_values = [random_values[i] for i in range(len(random_values))]
              hparam_tree = tree_unflatten(treedef, random_values)
          else:
              hparam_tree = jax.tree_map(lambda x: jnp.array([default, default]), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += len(hparam_vector)
      elif sched == 'piecewise-pl':
          knot_locations = jnp.zeros(FLAGS.num_pieces)
          knot_values = jnp.ones(FLAGS.num_pieces+1) * default
          knot_concat = jnp.concatenate([knot_locations, knot_values])
          hparam_tree = jax.tree_map(lambda x: jnp.array(knot_concat), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
          setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
          idx += len(hparam_vector)

  theta = jnp.concatenate(theta_vals)
  # =======================================================================

  # CSV logger setup (needs to be after model init to access model param names for per-layer)
  # =======================================================================
  param_fieldnames = []
  for setting in FLAGS.tune_params:
      if setting['sched'] == 'fixed':
          param_fieldnames += ['{}_0'.format(setting['param'])]
      elif setting['sched'] == 'linear':
          param_fieldnames += ['{}_0'.format(setting['param']), '{}_1'.format(setting['param'])]
      elif setting['sched'] == 'inverse-time-decay':
          param_fieldnames += ['{}_0'.format(setting['param']), '{}_1'.format(setting['param'])]
      elif setting['sched'] == 'piecewise':
          param_fieldnames += ['{}_knot_{}'.format(setting['param'], i) for i in range(FLAGS.num_pieces)]
          param_fieldnames += ['{}_value_{}'.format(setting['param'], i) for i in range(FLAGS.num_pieces+1)]
      elif setting['sched'] == 'fixed-pl':
          for key in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(key, setting['param'])
              param_fieldnames += ['{}_0'.format(base_str)]
      elif setting['sched'] in ['linear-pl', 'inverse-time-decay-pl']:
          for key in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(key, setting['param'])
              param_fieldnames += ['{}_0'.format(base_str), '{}_1'.format(base_str)]
      elif setting['sched'] == 'piecewise-pl':
          for key in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(key, setting['param'])
              param_fieldnames += ['{}_knot_{}'.format(base_str, i) for i in range(FLAGS.num_pieces)]
              param_fieldnames += ['{}_value_{}'.format(base_str, i) for i in range(FLAGS.num_pieces+1)]

  print('Param fieldnames: {}'.format(param_fieldnames))

  constrained_param_fieldnames = ['cons_{}'.format(name) for name in param_fieldnames]
  param_grad_fieldnames = ['grad_{}'.format(name) for name in param_fieldnames]
  iteration_logger = general_utils.CSVLogger(fieldnames=['time_elapsed', 'iteration', 'inner_problem_steps', 'F', 'F_sum',
                                                         'full_train_loss', 'full_train_sum_loss', 'full_train_acc',
                                                         'full_val_loss', 'full_val_sum_loss', 'full_val_acc',
                                                         'full_test_loss', 'full_test_sum_loss', 'full_test_acc',
                                                         'train_fixed_acc', 'train_fixed_loss',
                                                         'train_sum_loss', 'train_sum_acc',
                                                         'val_sum_loss', 'val_sum_acc',
                                                         'train_rand_loss_unroll', 'train_rand_acc_unroll',
                                                         'train_fixed_loss_unroll', 'train_fixed_acc_unroll',
                                                         'val_rand_loss_unroll', 'val_rand_acc_unroll',
                                                         'val_fixed_loss_unroll', 'val_fixed_acc_unroll'] + \
                                                         constrained_param_fieldnames + param_fieldnames + param_grad_fieldnames,
                                             filename=os.path.join(save_dir, 'iteration.csv'))

  frequent_logger = general_utils.CSVLogger(fieldnames=['time_elapsed', 'iteration', 'inner_problem_steps', 'F', 'prev_obj'] + \
                                                       constrained_param_fieldnames + param_fieldnames + param_grad_fieldnames,
                                            filename=os.path.join(save_dir, 'frequent.csv'))
  # =======================================================================


  # =======================================================================
  # Inner Optimization
  # =======================================================================
  if FLAGS.inner_optimizer == 'adam':
      def reset_inner_optim_params(params):
          return { 'lr':           jax.tree_map(lambda x: jnp.array(FLAGS.lr), params),
                   'b1':           jax.tree_map(lambda x: jnp.array(FLAGS.b1), params),
                   'b2':           jax.tree_map(lambda x: jnp.array(FLAGS.b2), params),
                   'eps':          jax.tree_map(lambda x: jnp.array(FLAGS.eps), params),
                   'weight_decay': jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params),
                   'm':            jax.tree_map(lambda x: jnp.zeros(x.shape), params),
                   'v':            jax.tree_map(lambda x: jnp.zeros(x.shape), params)
                 }

      def reset_inner_optim_except_state(inner_optim_params, params):
          inner_optim_params['lr'] = jax.tree_map(lambda x: jnp.array(FLAGS.lr), params)
          inner_optim_params['b1'] = jax.tree_map(lambda x: jnp.array(FLAGS.b1), params)
          inner_optim_params['b2'] = jax.tree_map(lambda x: jnp.array(FLAGS.b2), params)
          inner_optim_params['eps'] = jax.tree_map(lambda x: jnp.array(FLAGS.eps), params)
          inner_optim_params['weight_decay'] = jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params)
          return inner_optim_params

      in_axes_for_inner_optim = {'lr': None, 'b1': None, 'b2': None, 'eps': None, 'weight_decay': None, 'm': 0, 'v': 0}

      def inner_optimizer_step(params, grads, inner_optim_params, t):
          # AdamW weight decay (like https://pytorch.org/docs/1.5.0/_modules/torch/optim/adamw.html#AdamW)
          params_wd = jax.tree_multimap(lambda p, wd, lr: p * (1.0 - lr * wd), params, inner_optim_params['weight_decay'], inner_optim_params['lr'])

          inner_optim_params['m'] = jax.tree_multimap(lambda b1,g,m: (1 - b1) * g + b1 * m, inner_optim_params['b1'], grads, inner_optim_params['m'])
          inner_optim_params['v'] = jax.tree_multimap(lambda b2,g,v: (1 - b2) * g**2 + b2 * v, inner_optim_params['b2'], grads, inner_optim_params['v'])
          mhat = jax.tree_multimap(lambda b1,m: m / (1 - b1**(t+1)), inner_optim_params['b1'], inner_optim_params['m'])
          vhat = jax.tree_multimap(lambda b2,v: v / (1 - b2**(t+1)), inner_optim_params['b2'], inner_optim_params['v'])
          updated_params = jax.tree_multimap(lambda lr,eps,p,m,v: p - lr * m / (jnp.sqrt(v) + eps), inner_optim_params['lr'], inner_optim_params['eps'], params, mhat, vhat)
          return updated_params, inner_optim_params

  elif FLAGS.inner_optimizer == 'sgdm':
      def reset_inner_optim_params(params):
          return { 'lr': jax.tree_map(lambda x: jnp.array(FLAGS.lr), params),
                   'momentum': jax.tree_map(lambda x: jnp.array(FLAGS.momentum), params),
                   'weight_decay': jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params),
                   'buf': jax.tree_map(lambda x: jnp.zeros(x.shape), params)
                 }

      def reset_inner_optim_except_state(inner_optim_params, params):
          inner_optim_params['lr'] = jax.tree_map(lambda x: jnp.array(FLAGS.lr), params)
          inner_optim_params['momentum'] = jax.tree_map(lambda x: jnp.array(FLAGS.momentum), params)
          inner_optim_params['weight_decay'] = jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params)
          return inner_optim_params

      in_axes_for_inner_optim = {'lr': None, 'momentum': None, 'weight_decay': None, 'buf': 0}

      if FLAGS.sgdm_type == 0:
          def inner_optimizer_step(params, grads, inner_optim_params, t):
              inner_optim_params['buf'] = jax.tree_multimap(lambda mom, v, g: mom * v - (1 - mom) * g, inner_optim_params['momentum'], inner_optim_params['buf'], grads)
              steps = jax.tree_multimap(lambda lr, v: lr * v, inner_optim_params['lr'], inner_optim_params['buf'])
              if FLAGS.inner_step_clip > 0:
                  # Inner step clipping
                  steps = jax.tree_map(lambda s: jnp.clip(s, a_min=-FLAGS.inner_step_clip, a_max=FLAGS.inner_step_clip), steps)
              updated_params = jax.tree_multimap(lambda p, s: p + s, params, steps)
              # updated_params = jax.tree_multimap(lambda lr, p, v: p + lr * v, inner_optim_params['lr'], params, inner_optim_params['buf'])
              return updated_params, inner_optim_params
      elif FLAGS.sgdm_type == 1:
          def inner_optimizer_step(params, grads, inner_optim_params, t):
              """This follows the PyTorch SGD + momentum implementation.
                From https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
              """
              nesterov = True
              # Weight decay
              d_p = jax.tree_multimap(lambda wd, g, p: g + wd * p, inner_optim_params['weight_decay'], grads, params)
              # Momentum
              inner_optim_params['buf'] = jax.tree_multimap(lambda mom, b, g: b * mom + g, inner_optim_params['momentum'], inner_optim_params['buf'], d_p)
              # Nesterov
              if nesterov:
                  d_p = jax.tree_multimap(lambda mom, g, b: g + mom * b, inner_optim_params['momentum'], d_p, inner_optim_params['buf'])
              else:
                  d_p = inner_optim_params['buf']

              steps = jax.tree_multimap(lambda lr, g: lr * g, inner_optim_params['lr'], d_p)
              if FLAGS.inner_step_clip > 0:
                  steps = jax.tree_map(lambda s: jnp.clip(s, a_min=-FLAGS.inner_step_clip, a_max=FLAGS.inner_step_clip), steps)
              updated_params = jax.tree_multimap(lambda p, s: p - s, params, steps)
              # updated_params = jax.tree_multimap(lambda lr, p, g: p - lr * g, inner_optim_params['lr'], params, d_p)
              return updated_params, inner_optim_params

  elif FLAGS.inner_optimizer == 'sgd':
      def reset_inner_optim_params(params):
          return { 'lr': jax.tree_map(lambda x: jnp.array(FLAGS.lr), params),
                   'weight_decay': jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params)
                 }

      def reset_inner_optim_except_state(inner_optim_params, params):
          inner_optim_params['lr'] = jax.tree_map(lambda x: jnp.array(FLAGS.lr), params)
          inner_optim_params['weight_decay'] = jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params)
          return inner_optim_params

      in_axes_for_inner_optim = { 'lr': None, 'weight_decay': None }

      def inner_optimizer_step(params, grads, inner_optim_params, t):
          # Weight decay
          d_p = jax.tree_multimap(lambda wd, g, p: g + wd * p, inner_optim_params['weight_decay'], grads, params)
          updated_params = jax.tree_multimap(lambda lr,p,g: p - lr * g, inner_optim_params['lr'], params, d_p)
          # updated_params = jax.tree_multimap(lambda lr, p, g: p - lr * g, inner_optim_params['lr'], params, grads)
          return updated_params, inner_optim_params
  # =======================================================================

  # Outer optimization
  # =======================================================================
  if FLAGS.outer_optimizer == 'adam':
      outer_optim_params = {
          'lr': FLAGS.outer_lr,
          'b1': FLAGS.outer_b1,
          'b2': FLAGS.outer_b2,
          'eps': FLAGS.outer_eps,
          'm': jnp.zeros(len(theta)),
          'v': jnp.zeros(len(theta)),
      }

      @jax.jit
      def outer_optimizer_step(params, grads, optim_params, t):
          lr = optim_params['lr']
          b1 = optim_params['b1']
          b2 = optim_params['b2']
          eps = optim_params['eps']

          optim_params['m'] = (1 - b1) * grads + b1 * optim_params['m']
          optim_params['v'] = (1 - b2) * (grads**2) + b2 * optim_params['v']
          mhat = optim_params['m'] / (1 - b1**(t+1))
          vhat = optim_params['v'] / (1 - b2**(t+1))

          updated_params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
          return updated_params, optim_params

  elif FLAGS.outer_optimizer == 'rmsprop':
      outer_optim_params = {
          'lr': FLAGS.outer_lr,
          'gamma': FLAGS.outer_gamma,
          'eps': FLAGS.outer_eps,
          'avg_sq_grad': jnp.ones(len(theta)),
      }

      @jax.jit
      def outer_optimizer_step(params, grads, optim_params, t):
          lr = optim_params['lr']
          gamma = optim_params['gamma']
          eps = optim_params['eps']
          optim_params['avg_sq_grad'] = optim_params['avg_sq_grad'] * gamma + (grads**2) * (1 - gamma)
          updated_params = params - lr * grads / (jnp.sqrt(optim_params['avg_sq_grad']) + eps)
          return updated_params, optim_params

  elif FLAGS.outer_optimizer == 'sgdm':
      outer_optim_params = {
          'lr': FLAGS.outer_lr,
          'momentum': FLAGS.outer_momentum,
          'buf': jnp.zeros(len(theta)),
      }

      @jax.jit
      def outer_optimizer_step(params, grads, optim_params, t):
          """This implements one version of momentum,which corresponds to sgdm_type==0 for the inner params... is this ok?"""
          lr = optim_params['lr']
          mom = optim_params['momentum']
          optim_params['buf'] = mom * optim_params['buf'] - (1 - mom) * grads
          updated_params = params + lr * optim_params['buf']
          return updated_params, optim_params

  elif FLAGS.outer_optimizer == 'sgd':
      outer_optim_params = {
          'lr': FLAGS.outer_lr
      }

      @jax.jit
      def outer_optimizer_step(params, grads, optim_params, t):
          lr = optim_params['lr']
          updated_params = params - lr * grads
          return updated_params, optim_params
  # =======================================================================

  @jax.jit
  def evaluate_minibatch(params, state, data):
    inputs, targets = data
    xentropy, (logits, _) = loss_fn(params, inputs, targets, state, False)
    pred = jnp.argmax(logits, axis=1)
    num_total = targets.shape[0]
    num_correct = (pred == targets).sum()
    return xentropy, num_correct, num_total

  @jax.jit
  def evaluate(params, state, xs, ys):
    (losses, num_correct, num_total) = jax.lax.map(lambda data: evaluate_minibatch(params=params, state=state, data=data), (xs, ys))
    return jnp.mean(losses), jnp.sum(losses), jnp.sum(num_correct) / jnp.sum(num_total)


  @jax.jit
  def get_random_minibatch(key, all_xs, all_ys):
    rand_idx = jax.random.randint(key, (), 0, len(all_xs))
    rand_xs, rand_ys = all_xs[rand_idx], all_ys[rand_idx]
    return rand_xs, rand_ys

  def clip_objective(objective):
      if 'final_loss' in FLAGS.objective:
          objective = jnp.nan_to_num(objective, nan=0.0)
          return jnp.clip(objective, a_max=10.0)
      elif 'sum_loss' in FLAGS.objective:
          objective = jnp.nan_to_num(objective, nan=jnp.log2(10) * FLAGS.K)
          return jnp.clip(objective, a_max=jnp.log2(10) * FLAGS.K)
      elif 'final_acc' in FLAGS.objective:
          objective = jnp.nan_to_num(objective, nan=0.0)
          return jnp.clip(objective, a_min=0.0, a_max=100.0)
      elif 'sum_fixed_acc' in FLAGS.objective:
          objective = jnp.nan_to_num(objective, nan=0.0)
          return jnp.clip(objective, a_min=0.0, a_max=100.0 * FLAGS.K)

  @jax.jit
  def apply_hsv_jitter(key, image, hue_jitter, sat_jitter):  # Input is (3,32,32)
    # Unnormalize to get an image with values in [0,1]
    unnormalized_image = image * data_utils.stds['cifar10'].reshape(-1,1,1) + data_utils.means['cifar10'].reshape(-1,1,1)

    # Convert to HSV representation to manipulate hue, saturation, value
    hsv_image = rgb_to_hsv(unnormalized_image.transpose((1,2,0)))
    h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]

    # Manipulate the hue, saturation, and value
    h_pert, s_pert, v_pert = jax.random.normal(key, shape=(3,)) * 1.0
    h = (h + h_pert * hue_jitter) % 1.0
    s = jnp.clip(s + s_pert * sat_jitter, 0.0, 1.0)
    # v = jnp.clip(v + v_pert * value_jitter, 0, 1)
    jittered_hsv_image = jnp.stack([h,s,v], axis=2)  # Should be (32,32,3)

    # Convert back to RGB
    rgb_image = hsv_to_rgb(jittered_hsv_image).transpose((2,0,1))  # Should be back to (3,32,32)

    # Normalize the image to have 0 mean and unit standard deviation, e.g., values somewhere around the range [-2, +2]
    normalized_rgb_image = (rgb_image - data_utils.means['cifar10'].reshape(-1, 1, 1)) / data_utils.stds['cifar10'].reshape(-1,1,1)
    return normalized_rgb_image

  @partial(jax.jit, static_argnums=4)
  def apply_augmentations(key, images, theta, t, T):
    augmented_images = images

    if 'crop_border' in tune_params_str or 'cb' in tune_params_str:
      crop_border = hparam_utils.cons_func_dict['crop_border'](theta[idx_dict['crop_border']])
      if schedule_info['crop_border'] == 'linear':
        crop_border = jnp.round(get_linear_sched_value(crop_border, 'crop_border', t, T)).astype(jnp.int32)
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(aug_utils.apply_crop, in_axes=(0,0,None))(skeys, augmented_images, crop_border)

    if 'hflip_prob' in tune_params_str or 'hfp' in tune_params_str:
      hflip_prob = hparam_utils.cons_func_dict['hflip_prob'](theta[idx_dict['hflip_prob']])
      if schedule_info['hflip_prob'] == 'linear':
        hflip_prob = get_linear_sched_value(hflip_prob, 'hflip_prob', t, T)
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(aug_utils.apply_random_hflip, in_axes=(0,0,None))(skeys, augmented_images, hflip_prob)

    if 'vflip_prob' in tune_params_str or 'vfp' in tune_params_str:
      vflip_prob = hparam_utils.cons_func_dict['vflip_prob'](theta[idx_dict['vflip_prob']])
      if schedule_info['vflip_prob'] == 'linear':
        vflip_prob = get_linear_sched_value(vflip_prob, 'vflip_prob', t, T)
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(aug_utils.apply_random_vflip, in_axes=(0,0,None))(skeys, augmented_images, vflip_prob)

    if 'cutoutsize' in tune_params_str or 'cs' in tune_params_str:
      cutoutsize = hparam_utils.cons_func_dict['cutoutsize'](theta[idx_dict['cutoutsize']])
      if schedule_info['cutoutsize'] == 'linear':
        cutoutsize = jnp.round(get_linear_sched_value(cutoutsize, 'cutoutsize', t, T)).astype(jnp.int32)
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(aug_utils.apply_cutout, in_axes=(0,0,None,None))(skeys, augmented_images, cutoutsize, cutoutsize)

    if 'dropout_prob' in tune_params_str or 'dp' in tune_params_str:
      dropout_prob = hparam_utils.cons_func_dict['dropout_prob'](theta[idx_dict['dropout_prob']])
      if schedule_info['dropout_prob'] == 'linear':
        dropout_prob = get_linear_sched_value(dropout_prob, 'dropout_prob', t, T)
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(aug_utils.apply_dropout, in_axes=(0,0,None))(skeys, augmented_images, dropout_prob)

    if 'hue_jitter' in tune_params_str or 'hj' in tune_params_str or 'sat_jitter' in tune_params_str or 'sj' in tune_params_str:
      hue_jitter = hparam_utils.cons_func_dict['hue_jitter'](theta[idx_dict['hue_jitter']])
      sat_jitter = hparam_utils.cons_func_dict['sat_jitter'](theta[idx_dict['sat_jitter']])

      if schedule_info['hue_jitter'] == 'linear':
        hue_jitter = get_linear_sched_value(hue_jitter, 'hue_jitter', t, T)
      if schedule_info['sat_jitter'] == 'linear':
        sat_jitter = get_linear_sched_value(sat_jitter, 'sat_jitter', t, T)

      # value_jitter = theta[7]
      keys = jax.random.split(key, images.shape[0]+1)  # Get batch_size number of keys
      key, skeys = keys[0], keys[1:]
      augmented_images = jax.vmap(apply_hsv_jitter, in_axes=(0,0,None,None))(skeys, augmented_images, hue_jitter, sat_jitter)

    return key, augmented_images

  @jax.jit
  def update(i, state):
      t = state['t']
      key = state['key']
      theta = state['theta']
      params = state['params']
      model_state = state['model_state']
      inner_optim_params = state['inner_optim_params']

      inputs, targets = state['all_train_xs'][i], state['all_train_ys'][i]
      key, inputs = apply_augmentations(key, inputs, theta, t, FLAGS.T)
      (loss, (logits, model_state)), grads = loss_value_and_grad(params, inputs, targets, model_state, True)

      if FLAGS.clip_objective:
          loss = clip_objective(loss)

      grads = jax.tree_map(lambda g: jnp.nan_to_num(g), grads)  # Get rid of nans by setting to 0 in the gradient

      if FLAGS.grad_clip > 0:
          if FLAGS.grad_clip_type == 'hardclip':
              grads = jax.tree_map(lambda g: jnp.clip(g, a_min=-FLAGS.grad_clip, a_max=FLAGS.grad_clip), grads)  # Clip very small and very large inner gradients
          elif FLAGS.grad_clip_type == 'normclip':
              grad_norm = general_utils.flat_norm(grads)
              grads = jax.tree_map(lambda g: g * jnp.minimum(1., FLAGS.grad_clip / grad_norm), grads)

      inner_optim_params = get_inner_optim_params(inner_optim_params, theta, t, FLAGS.T)
      params, inner_optim_params = inner_optimizer_step(params, grads, inner_optim_params, t)

      state['params'] = params
      state['model_state'] = model_state
      state['inner_optim_params'] = inner_optim_params

      # Evaluate the loss on the same example _after_ taking a gradient step
      # loss, (logits, _) = loss_fn(params, inputs, targets, model_state, True)

      if FLAGS.objective in ['train_sum_fixed_loss', 'train_sum_fixed_acc', 'val_sum_fixed_loss', 'val_sum_fixed_acc']:
        train_loss, (train_logits, _) = loss_fn(params, state['fixed_train_inputs'], state['fixed_train_targets'], model_state, True)
        train_acc = (jnp.argmax(train_logits, axis=1) == state['fixed_train_targets']).sum() / state['fixed_train_targets'].shape[0]
        val_loss, (val_logits, _) = loss_fn(params, state['fixed_val_inputs'], state['fixed_val_targets'], model_state, True)
        val_acc = (jnp.argmax(val_logits, axis=1) == state['fixed_val_targets']).sum() / state['fixed_val_targets'].shape[0]
      else:
        key, key_train, key_val = jax.random.split(key, 3)
        rand_train_xs, rand_train_ys = get_random_minibatch(key_train, state['all_train_xs'], state['all_train_ys'])
        train_loss, (train_logits, _) = loss_fn(params, rand_train_xs, rand_train_ys, model_state, True)
        train_acc = (jnp.argmax(train_logits, axis=1) == rand_train_ys).sum() / rand_train_ys.shape[0]

        rand_val_xs, rand_val_ys = get_random_minibatch(key_val, state['all_val_xs'], state['all_val_ys'])
        val_loss, (val_logits, _) = loss_fn(params, rand_val_xs, rand_val_ys, model_state, True)
        val_acc = (jnp.argmax(val_logits, axis=1) == rand_val_ys).sum() / rand_val_ys.shape[0]

      state['loss_container'] = state['loss_container'].at[state['container_idx']].set(train_loss)
      state['acc_container'] = state['acc_container'].at[state['container_idx']].set(train_acc)
      state['val_loss_container'] = state['val_loss_container'].at[state['container_idx']].set(val_loss)
      state['val_acc_container'] = state['val_acc_container'].at[state['container_idx']].set(val_acc)

      state['t'] += 1
      state['container_idx'] += 1
      state['key'] = key
      return state

  @partial(jax.jit, static_argnums=(19,20,21))
  def unroll(key,
             params,
             model_state,
             inner_optim_params,
             theta,
             loss_container,
             acc_container,
             val_loss_container,
             val_acc_container,
             all_train_xs,
             all_train_ys,
             all_val_xs,
             all_val_ys,
             fixed_train_inputs,
             fixed_train_targets,
             fixed_val_inputs,
             fixed_val_targets,
             t0,
             i,
             T,  # static_argnums starts here
             K,
             clip_obj):
      loop_state = {'key': key,
                    'params': params,
                    'model_state': model_state,
                    'inner_optim_params': inner_optim_params,
                    'all_train_xs': all_train_xs,
                    'all_train_ys': all_train_ys,
                    'all_val_xs': all_val_xs,
                    'all_val_ys': all_val_ys,
                    'fixed_train_inputs': fixed_train_inputs,
                    'fixed_train_targets': fixed_train_targets,
                    'fixed_val_inputs': fixed_val_inputs,
                    'fixed_val_targets': fixed_val_targets,
                    'loss_container': loss_container,
                    'acc_container': acc_container,
                    'val_loss_container': val_loss_container,
                    'val_acc_container': val_acc_container,
                    'theta': theta,
                    't': t0,
                    'container_idx': 0,
                   }
      loop_state = jax.lax.fori_loop(i, jnp.min(jnp.array([i+K, num_train_batches])), update, loop_state)

      train_sum_loss = jnp.sum(loop_state['loss_container'])
      train_sum_acc = jnp.sum(loop_state['acc_container'])
      val_sum_loss = jnp.sum(loop_state['val_loss_container'])
      val_sum_acc = jnp.sum(loop_state['val_acc_container'])

      key = loop_state['key']
      params = loop_state['params']
      model_state = loop_state['model_state']
      inner_optim_params = loop_state['inner_optim_params']

      # Sample random train and val minibatches to compute the value for --objective=train_sum_loss_unroll
      key, key_train, key_val = jax.random.split(key, 3)
      rand_train_xs, rand_train_ys = get_random_minibatch(key_train, all_train_xs, all_train_ys)
      train_rand_loss_unroll, (train_rand_logits, _) = loss_fn(params, rand_train_xs, rand_train_ys, model_state, True)
      train_rand_acc_unroll = (jnp.argmax(train_rand_logits, axis=1) == rand_train_ys).sum() / rand_train_ys.shape[0]

      # Choose a random validation minibatch and compute the loss and accuracy on it for the val containers
      rand_val_xs, rand_val_ys = get_random_minibatch(key_val, all_val_xs, all_val_ys)
      val_rand_loss_unroll, (val_rand_logits, _) = loss_fn(params, rand_val_xs, rand_val_ys, model_state, True)
      val_rand_acc_unroll = (jnp.argmax(val_rand_logits, axis=1) == rand_val_ys).sum() / rand_val_ys.shape[0]

      # Computing quantities for --objective=train_sum_fixed_loss_unroll and --objective=val_sum_fixed_loss_unroll
      train_fixed_loss_unroll, (train_fixed_logits, _) = loss_fn(params, fixed_train_inputs, fixed_train_targets, model_state, True)
      train_fixed_acc_unroll = (jnp.argmax(train_fixed_logits, axis=1) == fixed_train_targets).sum() / fixed_train_targets.shape[0]
      val_fixed_loss_unroll, (val_fixed_logits, _) = loss_fn(params, fixed_val_inputs, fixed_val_targets, model_state, True)
      val_fixed_acc_unroll = (jnp.argmax(val_fixed_logits, axis=1) == fixed_val_targets).sum() / fixed_val_targets.shape[0]

      metric_dict = { 'train_sum_loss': train_sum_loss,
                      'train_sum_acc': train_sum_acc,
                      'val_sum_loss': val_sum_loss,
                      'val_sum_acc': val_sum_acc,
                      'train_rand_loss_unroll': train_rand_loss_unroll,
                      'train_rand_acc_unroll': train_rand_acc_unroll,
                      'train_fixed_loss_unroll': train_fixed_loss_unroll,
                      'train_fixed_acc_unroll': train_fixed_acc_unroll,
                      'val_rand_loss_unroll': val_rand_loss_unroll,
                      'val_rand_acc_unroll': val_rand_acc_unroll,
                      'val_fixed_loss_unroll': val_fixed_loss_unroll,
                      'val_fixed_acc_unroll': val_fixed_acc_unroll,
                    }

      if FLAGS.objective in ['train_sum_loss', 'train_sum_fixed_loss']:
          objective = train_sum_loss
      elif FLAGS.objective in ['train_sum_acc', 'train_sum_fixed_acc']:
          objective = train_sum_acc
      elif FLAGS.objective in ['val_sum_loss', 'val_sum_fixed_loss']:
          objective = val_sum_loss
      elif FLAGS.objective in ['val_sum_acc', 'val_sum_fixed_acc']:
          objective = val_sum_acc

      elif FLAGS.objective in 'train_sum_loss_unroll':
          objective = train_rand_loss_unroll
      elif FLAGS.objective in 'train_sum_acc_unroll':
          objective = train_rand_acc_unroll
      elif FLAGS.objective in 'val_sum_loss_unroll':
          objective = val_rand_loss_unroll
      elif FLAGS.objective in 'val_sum_acc_unroll':
          objective = val_rand_acc_unroll

      elif FLAGS.objective in ['train_final_acc', 'train_sum_fixed_acc_unroll']:
          objective = train_fixed_acc_unroll
      elif FLAGS.objective in ['train_final_loss', 'train_sum_fixed_loss_unroll']:
          objective = train_fixed_loss_unroll
      elif FLAGS.objective in ['val_final_acc', 'val_sum_fixed_acc_unroll']:
          objective = val_fixed_acc_unroll
      elif FLAGS.objective in ['val_final_loss', 'val_sum_fixed_loss_unroll']:
          objective = val_fixed_loss_unroll

      if FLAGS.clip_objective:
          objective = clip_objective(objective)

      return key, objective, metric_dict, params, model_state, inner_optim_params

  @partial(jax.jit, static_argnums=(20,21,22,23))
  def es_grad(key,
              x,
              state,
              inner_optim_params,
              prev_obj,
              theta,
              loss_container,
              acc_container,
              val_loss_container,
              val_acc_container,
              all_train_xs,
              all_train_ys,
              all_val_xs,
              all_val_ys,
              fixed_train_inputs,
              fixed_train_targets,
              fixed_val_inputs,
              fixed_val_targets,
              t0,
              i,
              T,  # static_argnums starts here
              K,
              N,
              sigma=0.1):
      pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
      neg_pert = -pos_pert
      perts = jnp.concatenate([pos_pert, neg_pert])

      key, result, metric_dict, x, state, inner_optim_params = es_unroll(key,
                                                                         x,
                                                                         state,
                                                                         inner_optim_params,
                                                                         theta + perts,
                                                                         loss_container,
                                                                         acc_container,
                                                                         val_loss_container,
                                                                         val_acc_container,
                                                                         all_train_xs,
                                                                         all_train_ys,
                                                                         all_val_xs,
                                                                         all_val_ys,
                                                                         fixed_train_inputs,
                                                                         fixed_train_targets,
                                                                         fixed_val_inputs,
                                                                         fixed_val_targets,
                                                                         t0,
                                                                         i,
                                                                         T,
                                                                         K,
                                                                         FLAGS.clip_objective)  # Clip objective

      if 'acc' in FLAGS.objective:
        objective = -result
      else:
        objective = result

      gradient_estimate = jnp.sum(objective.reshape(-1, 1) * perts, axis=0) / (N * sigma**2)
      return key, gradient_estimate, result, metric_dict

  @partial(jax.jit, static_argnums=(21,22,23,24))
  def pes_grad(key,
               xs,
               states,
               inner_optim_params,
               perturbation_accums,
               prev_obj,
               theta,
               loss_container,
               acc_container,
               val_loss_container,
               val_acc_container,
               all_train_xs,
               all_train_ys,
               all_val_xs,
               all_val_ys,
               fixed_train_inputs,
               fixed_train_targets,
               fixed_val_inputs,
               fixed_val_targets,
               t0,
               i,
               T, # static_argnums starts here
               K,
               N,
               sigma=0.1):
      pos_pert = jax.random.normal(key, (N//2, len(theta))) * sigma
      neg_pert = -pos_pert
      perts = jnp.concatenate([pos_pert, neg_pert])

      key, result, metric_dict, xs, states, inner_optim_params = pes_unroll(key,
                                                                            xs,
                                                                            states,
                                                                            inner_optim_params,
                                                                            theta + perts,
                                                                            loss_container,
                                                                            acc_container,
                                                                            val_loss_container,
                                                                            val_acc_container,
                                                                            all_train_xs,
                                                                            all_train_ys,
                                                                            all_val_xs,
                                                                            all_val_ys,
                                                                            fixed_train_inputs,
                                                                            fixed_train_targets,
                                                                            fixed_val_inputs,
                                                                            fixed_val_targets,
                                                                            t0,
                                                                            i,
                                                                            T,
                                                                            K,
                                                                            FLAGS.clip_objective)  # Clip objective

      if 'acc' in FLAGS.objective:
        objective = -result
      else:
        objective = result

      perturbation_accums = perturbation_accums + perts
      gradient_estimate = jnp.sum(objective.reshape(-1, 1) * perturbation_accums, axis=0) / (N * sigma**2)
      return key, gradient_estimate, result, metric_dict, xs, states, inner_optim_params, perturbation_accums


  def to_constrained(theta_unconstrained):
    theta_constrained = []
    for setting in FLAGS.tune_params:
      param = setting['param']
      unconstrained_values = theta_unconstrained[onp.array(idx_dict[param])]
      constrained_values = hparam_utils.cons_func_dict[param](unconstrained_values)
      if constrained_values.ndim == 0:
        constrained_values = constrained_values.reshape(1)
      theta_constrained.append(constrained_values)
    return onp.concatenate(theta_constrained)


  key = jax.random.PRNGKey(FLAGS.seed)
  start_time = time.time()  # This one will never be reset, it will track time from the very start
  total_inner_iterations = 0
  num_minibatches_seen = 0
  t = 0

  if FLAGS.estimate == 'es':
    x, state = net.init(key_for_init, init_images, is_training=True)
    inner_optim_params = reset_inner_optim_params(x)
    es_unroll = jax.vmap(unroll, in_axes=(None,    # key
                                          None,    # params
                                          None,    # model_state
                                          None,    # inner_optim_params
                                          0,       # thetas
                                          0,       # loss_container
                                          0,       # acc_container
                                          0,       # val_loss_container
                                          0,       # val_acc_container
                                          None,    # all_train_xs
                                          None,    # all_train_ys
                                          None,    # all_val_xs
                                          None,    # all_val_ys
                                          None,    # fixed_train_inputs
                                          None,    # fixed_train_targets
                                          None,    # fixed_val_inputs
                                          None,    # fixed_val_targets
                                          None,    # t0
                                          None,    # i
                                          None,    # T
                                          None,    # K
                                          None))   # clip_obj
    if FLAGS.use_new_indexing:
      prev_obj, (_, _) = loss_fn(x, fixed_train_inputs, fixed_train_targets, state, True)
    else:
      prev_obj = 0
  elif FLAGS.estimate == 'pes':
    perturbation_accums = jnp.zeros((FLAGS.N, len(theta)))

    params, state = net.init(key, init_images, is_training=True)
    xs = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), params)
    states = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), state)

    inner_optim_params = reset_inner_optim_params(xs)
    in_axes_for_xs = jax.tree_map(lambda x: 0, xs)
    in_axes_for_states = jax.tree_map(lambda x: 0, states)
    pes_unroll = jax.vmap(unroll, in_axes=(None,                     # key
                                           in_axes_for_xs,           # params
                                           in_axes_for_states,       # model_state
                                           in_axes_for_inner_optim,  # inner_optim_params
                                           0,                        # thetas
                                           0,                        # loss_container
                                           0,                        # acc_container
                                           0,                        # val_loss_container
                                           0,                        # val_acc_container
                                           None,                     # all_train_xs
                                           None,                     # all_train_ys
                                           None,                     # all_val_xs
                                           None,                     # all_val_ys
                                           None,                     # fixed_train_inputs
                                           None,                     # fixed_train_targets
                                           None,                     # fixed_val_inputs
                                           None,                     # fixed_val_targets
                                           None,                     # t0
                                           None,                     # i
                                           None,                     # T
                                           None,                     # K
                                           None))                    # clip_obj
    if FLAGS.use_new_indexing:
      prev_obj, (_, _) = loss_fn(params, fixed_train_inputs, fixed_train_targets, state, True)
    else:
      prev_obj = jnp.zeros(FLAGS.N)

  print('Starting meta-optimization')
  sys.stdout.flush()
  for outer_iteration in range(FLAGS.outer_iterations):
    key, skey = jax.random.split(key)

    if t >= FLAGS.T:
        print('\nReset inner problem\n')
        sys.stdout.flush()
        t = 0
        num_minibatches_seen = 0

        if FLAGS.shuffle:
          all_train_xs, all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
          all_val_xs, all_val_ys = data_utils.create_minibatches(skey, val_data, val_targets, FLAGS.batch_size)

        if FLAGS.resample_fixed_minibatch:
          key, key_train, key_val = jax.random.split(key, 3)
          fixed_train_inputs, fixed_train_targets = get_random_minibatch(key_train, all_train_xs, all_train_ys)
          fixed_val_inputs, fixed_val_targets = get_random_minibatch(key_val, all_val_xs, all_val_ys)

        if FLAGS.estimate == 'es':
          if FLAGS.reset_params:
              x, state = net.init(skey, init_images, is_training=True)
          else:
              x, state = net.init(key_for_init, init_images, is_training=True)

          inner_optim_params = reset_inner_optim_params(x)

          if FLAGS.use_new_indexing:
              prev_obj, (_, _) = loss_fn(x, fixed_train_inputs, fixed_train_targets, state, True)
          else:
              prev_obj = 0
        elif FLAGS.estimate == 'pes':
          perturbation_accums = jnp.zeros((FLAGS.N, len(theta)))  # Reset perturbation accumulator

          if FLAGS.reset_params:
              params, state = net.init(skey, init_images, is_training=True)
              xs = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), params)
              states = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), state)
          else:
              params, state = net.init(key_for_init, init_images, is_training=True)
              xs = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), params)
              states = jax.tree_map(lambda x: jnp.stack([x] * FLAGS.N), state)

          if FLAGS.use_new_indexing:
              prev_obj, (_, _) = loss_fn(params, fixed_train_inputs, fixed_train_targets, state, True)
          else:
              prev_obj = jnp.zeros(FLAGS.N)

          inner_optim_params = reset_inner_optim_params(xs)

        key, skey = jax.random.split(key)

    if num_minibatches_seen >= num_train_batches:
        num_minibatches_seen = 0
        print('Resetting training minibatches at t = {} out of T = {}'.format(t, FLAGS.T))
        sys.stdout.flush()
        if FLAGS.shuffle:
          all_train_xs, all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
        key, skey = jax.random.split(key)

    loss_container = jnp.zeros((FLAGS.N, FLAGS.K))
    acc_container = jnp.zeros((FLAGS.N, FLAGS.K))
    val_loss_container = jnp.zeros((FLAGS.N, FLAGS.K))
    val_acc_container = jnp.zeros((FLAGS.N, FLAGS.K))

    if FLAGS.estimate == 'es':
      key, theta_grad, result, metric_dict = es_grad(skey,
                                                     x,
                                                     state,
                                                     inner_optim_params,
                                                     prev_obj,
                                                     theta,
                                                     loss_container,
                                                     acc_container,
                                                     val_loss_container,
                                                     val_acc_container,
                                                     all_train_xs,
                                                     all_train_ys,
                                                     all_val_xs,
                                                     all_val_ys,
                                                     fixed_train_inputs,
                                                     fixed_train_targets,
                                                     fixed_val_inputs,
                                                     fixed_val_targets,
                                                     t,
                                                     num_minibatches_seen,
                                                     FLAGS.T,
                                                     FLAGS.K,
                                                     FLAGS.N,
                                                     FLAGS.sigma)

      loss_container = jnp.zeros((FLAGS.K))
      acc_container = jnp.zeros((FLAGS.K))
      val_loss_container = jnp.zeros((FLAGS.K))
      val_acc_container = jnp.zeros((FLAGS.K))
      key, result, metric_dict, x, state, inner_optim_params = unroll(skey,
                                                                      x,
                                                                      state,
                                                                      inner_optim_params,
                                                                      theta,
                                                                      loss_container,
                                                                      acc_container,
                                                                      val_loss_container,
                                                                      val_acc_container,
                                                                      all_train_xs,
                                                                      all_train_ys,
                                                                      all_val_xs,
                                                                      all_val_ys,
                                                                      fixed_train_inputs,
                                                                      fixed_train_targets,
                                                                      fixed_val_inputs,
                                                                      fixed_val_targets,
                                                                      t,
                                                                      num_minibatches_seen,
                                                                      FLAGS.T,
                                                                      FLAGS.K,
                                                                      FLAGS.clip_objective)  # clip objective
    elif FLAGS.estimate == 'pes':
      key, theta_grad, result, metric_dict, xs, states, inner_optim_params, perturbation_accums = pes_grad(skey,
                                                                                                           xs,
                                                                                                           states,
                                                                                                           inner_optim_params,
                                                                                                           perturbation_accums,
                                                                                                           prev_obj,
                                                                                                           theta,
                                                                                                           loss_container,
                                                                                                           acc_container,
                                                                                                           val_loss_container,
                                                                                                           val_acc_container,
                                                                                                           all_train_xs,
                                                                                                           all_train_ys,
                                                                                                           all_val_xs,
                                                                                                           all_val_ys,
                                                                                                           fixed_train_inputs,
                                                                                                           fixed_train_targets,
                                                                                                           fixed_val_inputs,
                                                                                                           fixed_val_targets,
                                                                                                           t,
                                                                                                           num_minibatches_seen,
                                                                                                           FLAGS.T,
                                                                                                           FLAGS.K,
                                                                                                           FLAGS.N,
                                                                                                           FLAGS.sigma)

      key = key[0]

    t += FLAGS.K
    num_minibatches_seen += FLAGS.K

    # Stop meta-optimization early if theta_grad ever becomes nan
    if bool(jnp.any(jnp.isnan(theta_grad))):
        print('='*80)
        print('theta_grad is nan, exiting!')
        print('='*80)
        sys.exit(1)

    if FLAGS.outer_clip > 0:
        theta_grad = jnp.clip(theta_grad, -FLAGS.outer_clip, FLAGS.outer_clip)
    theta, outer_optim_params = outer_optimizer_step(theta, theta_grad, outer_optim_params, outer_iteration)

    if FLAGS.estimate == 'pes':
      inner_optim_params = reset_inner_optim_except_state(inner_optim_params, params)

    if total_inner_iterations % FLAGS.print_every == 0:
        print('time: {} | outer iteration {} | t: {} | theta: {} | theta_grad: {}, F: {}'.format(
               time.time() - start_time, outer_iteration, t, theta, theta_grad, jnp.mean(result)))
        sys.stdout.flush()

        hparams_to_log = {}
        for (param_name, value) in zip(param_fieldnames, theta):
            hparams_to_log[param_name] = value

        constrained_hparams_to_log = {}
        for (param_name, value) in zip(constrained_param_fieldnames, to_constrained(theta)):
          constrained_hparams_to_log[param_name] = value

        hparam_grads_to_log = {}
        for (param_name, value) in zip(param_fieldnames, theta_grad):
            hparam_grads_to_log['grad_{}'.format(param_name)] = value

        frequent_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                   'iteration': outer_iteration,
                                   'inner_problem_steps': total_inner_iterations,
                                   'F': float(jnp.mean(result)),
                                   'prev_obj': float(jnp.mean(prev_obj)),
                                   **constrained_hparams_to_log,
                                   **hparams_to_log,
                                   **hparam_grads_to_log })

    prev_obj = result

    if total_inner_iterations % FLAGS.log_every == 0:
        key, skey = jax.random.split(key)
        if FLAGS.reset_params:
            fresh_params, fresh_state = net.init(skey, init_images, is_training=True)
        else:
            fresh_params, fresh_state = net.init(key_for_init, init_images, is_training=True)

        fresh_inner_optim_params = reset_inner_optim_params(fresh_params)

        if FLAGS.shuffle:
          fresh_all_train_xs, fresh_all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
          fresh_all_val_xs, fresh_all_val_ys = data_utils.create_minibatches(key, val_data, val_targets, FLAGS.batch_size)
        else:
          fresh_all_train_xs, fresh_all_train_ys = all_train_xs, all_train_ys
          fresh_all_val_xs, fresh_all_val_ys = all_val_xs, all_val_ys

        fresh_t = 0
        fresh_num_minibatches_seen = 0
        F_sum = 0.0
        metric_sum_dict = defaultdict(float)

        print('='*80)
        print('Starting the evaluation optimization with theta: {}'.format(theta))
        print('='*80)

        while fresh_t < FLAGS.T:
            if fresh_num_minibatches_seen >= num_train_batches:
                print('Resetting training minibatches at fresh_t = {} out of T = {}'.format(fresh_t, FLAGS.T))
                sys.stdout.flush()
                if FLAGS.shuffle:
                  fresh_all_train_xs, fresh_all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
                  fresh_all_val_xs, fresh_all_val_ys = data_utils.create_minibatches(key, val_data, val_targets, FLAGS.batch_size)
                else:
                  fresh_all_train_xs, fresh_all_train_ys = all_train_xs, all_train_ys
                  fresh_all_val_xs, fresh_all_val_ys = all_val_xs, all_val_ys
                key, skey = jax.random.split(key)
                fresh_num_minibatches_seen = 0

            key, F, metric_dict, fresh_params, fresh_state, fresh_inner_optim_params = unroll(key,
                                                                                              fresh_params,
                                                                                              fresh_state,
                                                                                              fresh_inner_optim_params,
                                                                                              theta,
                                                                                              jnp.zeros(FLAGS.K),   # loss_container
                                                                                              jnp.zeros(FLAGS.K),   # acc_container
                                                                                              jnp.zeros(FLAGS.K),   # val_loss_container
                                                                                              jnp.zeros(FLAGS.K),   # val_acc_container
                                                                                              fresh_all_train_xs,
                                                                                              fresh_all_train_ys,
                                                                                              fresh_all_val_xs,
                                                                                              fresh_all_val_ys,
                                                                                              fixed_train_inputs,
                                                                                              fixed_train_targets,
                                                                                              fixed_val_inputs,
                                                                                              fixed_val_targets,
                                                                                              fresh_t,
                                                                                              fresh_num_minibatches_seen,
                                                                                              FLAGS.T,
                                                                                              FLAGS.K,
                                                                                              False)  # Do not clip objective for eval

            F_sum += F
            fresh_t += FLAGS.K
            fresh_num_minibatches_seen += FLAGS.K

            for metric_key in metric_dict:
              metric_sum_dict[metric_key] += float(metric_dict[metric_key])

        full_train_loss, full_train_sum_loss, full_train_acc = evaluate(fresh_params, fresh_state, all_train_xs, all_train_ys)
        full_val_loss, full_val_sum_loss, full_val_acc = evaluate(fresh_params, fresh_state, all_val_xs, all_val_ys)
        full_test_loss, full_test_sum_loss, full_test_acc = evaluate(fresh_params, fresh_state, all_test_xs, all_test_ys)

        for metric_key in metric_sum_dict:
          print('{}: {:6.3f}'.format(metric_key, metric_sum_dict[metric_key]))

        print('F_sum: {}'.format(F_sum))
        print('Final F: {}'.format(F))
        print('Full train loss: {} | full train acc: {}'.format(full_train_loss, full_train_acc))
        print('Full val loss: {} | full val acc: {}'.format(full_val_loss, full_val_acc))
        print('Full test loss: {} | full test acc: {}'.format(full_test_loss, full_test_acc))
        sys.stdout.flush()

        hparams_to_log = {}
        for (param_name, value) in zip(param_fieldnames, theta):
            hparams_to_log[param_name] = value

        constrained_hparams_to_log = {}
        for (param_name, value) in zip(constrained_param_fieldnames, to_constrained(theta)):
          constrained_hparams_to_log[param_name] = value

        hparam_grads_to_log = {}
        for (param_name, value) in zip(param_fieldnames, theta_grad):
            hparam_grads_to_log['grad_{}'.format(param_name)] = value

        iteration_logger.writerow({ 'time_elapsed': time.time() - start_time,
                                    'iteration': outer_iteration,
                                    'inner_problem_steps': outer_iteration * FLAGS.K,
                                    'F': float(F),
                                    'F_sum': float(F_sum),
                                    'full_train_loss': full_train_loss,
                                    'full_train_sum_loss': full_train_sum_loss,
                                    'full_train_acc': full_train_acc,
                                    'full_val_loss': full_val_loss,
                                    'full_val_sum_loss': full_val_sum_loss,
                                    'full_val_acc': full_val_acc,
                                    'full_test_loss': full_test_loss,
                                    'full_test_sum_loss': full_test_sum_loss,
                                    'full_test_acc': full_test_acc,
                                    **metric_sum_dict,
                                    **constrained_hparams_to_log,
                                    **hparams_to_log,
                                    **hparam_grads_to_log })

        print(outer_iteration, theta, theta_grad, F)
        sys.stdout.flush()

    total_inner_iterations += FLAGS.K

if __name__ == '__main__':
  app.run(main)
