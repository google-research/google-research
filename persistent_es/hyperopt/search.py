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

"""Run grid and random searches with an MLP/ResNet on MNIST/CIFAR-10.
"""
import os
import sys
import csv
import ipdb
import time
import math
import random
import itertools
import numpy as onp
import pickle as pkl
from tqdm import tqdm
from functools import partial
from collections import defaultdict

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

flags.DEFINE_string('dataset', 'mnist', 'Choose dataset')
flags.DEFINE_string('model', 'mlp', 'Choose the model')
flags.DEFINE_string('model_size', 'small', 'Choose the model size (for ResNets)')
flags.DEFINE_integer('nlayers', 4, 'Number of layers in the MLP')
flags.DEFINE_integer('nhid', 100, 'Number of hidden units in each layer of the MLP')
flags.DEFINE_bool('use_bn', False, 'Whether to use BatchNorm in the MLP')
flags.DEFINE_bool('with_bias', False, 'Whether to use a bias term in each layer of the MLP')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_bool('shuffle', False, 'Set this to not shuffle the training minibatches (should yield deterministic data ordering)')
flags.DEFINE_bool('resample_fixed_minibatch', False, 'Whether to randomly re-sample the fixed minibatch at the start of each inner problem')
flags.DEFINE_string('loss', 'softmax', 'Choose the type of loss (softmax, mse)')
flags.DEFINE_string('inner_optimizer', 'sgdm', 'Choose the inner optimizer')
flags.DEFINE_integer('sgdm_type', 1, 'Choose which implementation of SGDm to use (0 or 1)')
flags.DEFINE_float('lr', 1e-2, 'Learning rate')
flags.DEFINE_float('b1', 0.99, 'Adam b1 hyperparameter')
flags.DEFINE_float('b2', 0.999, 'Adam b2 hyperparameter')
flags.DEFINE_float('eps', 1e-8, 'Adam epsilon hyperparameter')
flags.DEFINE_float('momentum', 0.9, 'Momentum')
flags.DEFINE_float('weight_decay', 1e-10, 'Weight decay')
flags.DEFINE_integer('num_pieces', 2, 'Number of pieces of a piecewise linear schedule')
flags.DEFINE_float('grad_clip', -1, 'Gradient clipping for each step of the inner unroll (-1 means no grad clipping)')
flags.DEFINE_string('grad_clip_type', 'hardclip', 'Choose the type of gradient clipping for inner unrolls (normclip or hardclip)')
flags.DEFINE_string('objective', 'train_sum_loss', 'The "objective" that determines what we measure (random minibatches vs fixed ones)')

flags.DEFINE_string('search_type', 'random', 'Choose either grid search or random search')
flags.DEFINE_integer('num_points', 20, 'Num points for the grid search')
flags.DEFINE_integer('chunk_size', 1, 'How many networks to train in parallel during the grid/random search')
flags.DEFINE_string('tune_params', 'lr:fixed', 'A comma-separated string of hyperparameters to search over')

flags.DEFINE_integer('T', 500, 'Maximum number of iterations of the inner loop')
flags.DEFINE_integer('K', 50, 'Number of steps to unroll')
flags.DEFINE_float('target_acc', 0.5, 'The target accuracy to reach.')

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

  print('all_train_xs.shape: {} | all_train_ys.shape: {}'.format(all_train_xs.shape, all_train_ys.shape))
  print('all_val_xs.shape: {} | all_val_ys.shape: {}'.format(all_val_xs.shape, all_val_ys.shape))
  print('all_test_xs.shape: {} | all_test_ys.shape: {}'.format(all_test_xs.shape, all_test_ys.shape))
  # ----------------------------------------

  # Initialize model
  # ----------------------------------------
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

  init_images = all_train_xs[0]
  print('Initializing parameters...')
  key = jax.random.PRNGKey(FLAGS.seed)
  params, state = net.init(key, init_images, is_training=True)
  print('Num parameters: {}'.format(general_utils.count_params(params)))
  sys.stdout.flush()
  # -----------------------------------


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

  theta_vals = []
  idx_dict = {}
  unflatten_func_dict = {}
  idx = 0
  for setting in FLAGS.tune_params:
      param = setting['param']
      sched = setting['sched']
      default = hparam_utils.uncons_func_dict[param](default_value[param])
      if sched == 'fixed':
          theta_vals.append(jnp.array([default]))
          idx_dict[param] = idx
          idx += 1
      elif sched == 'linear':
          theta_vals.append(jnp.array([default, default]))
          idx_dict[param] = jnp.array([idx, idx+1])
          idx += 2
      elif sched == 'inverse-time-decay':
          theta_vals.append(jnp.array([default, 0.0]))
          idx_dict[param] = jnp.array([idx, idx+1])
          idx += 2
      elif sched == 'piecewise':
          knot_locations = jnp.zeros(FLAGS.num_pieces)
          knot_values = jnp.ones(FLAGS.num_pieces+1) * default
          knot_concat = jnp.concatenate([knot_locations, knot_values])
          theta_vals.append(knot_concat)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(knot_locations)+len(knot_values))))
          idx += len(knot_locations) + len(knot_values)
      elif sched == 'fixed-pl':
          hparam_tree = jax.tree_map(lambda x: jnp.array(default), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
          idx += len(hparam_vector)
      elif sched == 'linear-pl':
          hparam_tree = jax.tree_map(lambda x: jnp.array([default, default]), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
          idx += len(hparam_vector)
      elif sched == 'inverse-time-decay-pl':
          hparam_tree = jax.tree_map(lambda x: jnp.array([default, 0.0]), params)
          hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(hparam_tree)
          unflatten_func_dict[param] = hparam_unravel_pytree
          theta_vals.append(hparam_vector)
          idx_dict[param] = jnp.array(list(range(idx, idx+len(hparam_vector))))
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
          idx += len(hparam_vector)

  theta = jnp.concatenate(theta_vals)

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
          for name in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(name, setting['param'])
              param_fieldnames += ['{}_0'.format(base_str)]
      elif setting['sched'] == 'fixed-pl+':
          param_fieldnames.append('{}_global'.format(setting['param']))
          for name in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(name, setting['param'])
              param_fieldnames += ['{}_0'.format(base_str)]
      elif setting['sched'] in ['linear-pl', 'inverse-time-decay-pl']:
          for name in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(name, setting['param'])
              param_fieldnames += ['{}_0'.format(base_str), '{}_1'.format(base_str)]
      elif setting['sched'] == 'piecewise-pl':
          for name in general_utils.recursive_keys(params):
              base_str = '{}/{}'.format(name, setting['param'])
              param_fieldnames += ['{}_knot_{}'.format(base_str, i) for i in range(FLAGS.num_pieces)]
              param_fieldnames += ['{}_value_{}'.format(base_str, i) for i in range(FLAGS.num_pieces+1)]
  # =======================================================================


  # Loss function and loss grad
  # =======================================================================
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
                   'v':            jax.tree_map(lambda x: jnp.zeros(x.shape), params),
                 }

      def inner_optimizer_step(params, grads, inner_optim_params, t):
          # AdamW weight decay (like https://pytorch.org/docs/1.5.0/_modules/torch/optim/adamw.html#AdamW)
          params_wd = jax.tree_multimap(lambda p, wd, lr: p * (1.0 - lr * wd), params, inner_optim_params['weight_decay'], inner_optim_params['lr'])

          inner_optim_params['m'] = jax.tree_multimap(lambda b1,g,m: (1 - b1) * g + b1 * m, inner_optim_params['b1'], grads, inner_optim_params['m'])
          inner_optim_params['v'] = jax.tree_multimap(lambda b2,g,v: (1 - b2) * g**2 + b2 * v, inner_optim_params['b2'], grads, inner_optim_params['v'])
          mhat = jax.tree_multimap(lambda b1,m: m / (1 - b1**(t+1)), inner_optim_params['b1'], inner_optim_params['m'])
          vhat = jax.tree_multimap(lambda b2,v: v / (1 - b2**(t+1)), inner_optim_params['b2'], inner_optim_params['v'])
          updated_params = jax.tree_multimap(lambda lr,eps,p,m,v: p - lr * m / (jnp.sqrt(v) + eps), inner_optim_params['lr'], inner_optim_params['eps'], params_wd, mhat, vhat)
          # updated_params = jax.tree_multimap(lambda lr,eps,p,m,v: p - lr * m / (jnp.sqrt(v) + eps), inner_optim_params['lr'], inner_optim_params['eps'], params, mhat, vhat)
          return updated_params, inner_optim_params

  elif FLAGS.inner_optimizer == 'sgdm':
      def reset_inner_optim_params(params):
          return { 'lr':           jax.tree_map(lambda x: jnp.array(FLAGS.lr), params),
                   'momentum':     jax.tree_map(lambda x: jnp.array(FLAGS.momentum), params),
                   'weight_decay': jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params),
                   'buf':          jax.tree_map(lambda x: jnp.zeros(x.shape), params)
                 }

      if FLAGS.sgdm_type == 0:
          def inner_optimizer_step(params, grads, inner_optim_params, t):
              inner_optim_params['buf'] = jax.tree_multimap(lambda mom, v, g: mom * v - (1 - mom) * g, inner_optim_params['momentum'], inner_optim_params['buf'], grads)
              updated_params = jax.tree_multimap(lambda lr, p, v: p + lr * v, inner_optim_params['lr'], params, inner_optim_params['buf'])
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
              updated_params = jax.tree_multimap(lambda lr, p, g: p - lr * g, inner_optim_params['lr'], params, d_p)
              return updated_params, inner_optim_params

  elif FLAGS.inner_optimizer == 'sgd':
      def reset_inner_optim_params(params):
          return { 'lr':           jax.tree_map(lambda x: jnp.array(FLAGS.lr), params),
                   'weight_decay': jax.tree_map(lambda x: jnp.array(FLAGS.weight_decay), params),
                 }

      def inner_optimizer_step(params, grads, inner_optim_params, t):
          # Weight decay
          d_p = jax.tree_multimap(lambda wd, g, p: g + wd * p, inner_optim_params['weight_decay'], grads, params)
          updated_params = jax.tree_multimap(lambda lr,p,g: p - lr * g, inner_optim_params['lr'], params, d_p)
          # updated_params = jax.tree_multimap(lambda p,g: p - inner_optim_params['lr'] * g, params, grads)
          return updated_params, inner_optim_params
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
    return jnp.mean(losses), jnp.sum(num_correct) / jnp.sum(num_total)


  @jax.jit
  def get_random_minibatch(key, all_xs, all_ys):
    rand_idx = jax.random.randint(key, (), 0, len(all_xs))
    rand_xs, rand_ys = all_xs[rand_idx], all_ys[rand_idx]
    return rand_xs, rand_ys


  # =======================================================================
  # Data augmentation functions
  # =======================================================================
  @jax.jit
  def apply_hsv_jitter(key, image, hue_jitter, sat_jitter):  # Input is (3,32,32)
    # Unnormalize to get an image with values in [0,1]
    unnormalized_image = image * data_util.stds[FLAGS.dataset].reshape(-1,1,1) + data_util.means[FLAGS.dataset].reshape(-1,1,1)

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
    normalized_rgb_image = (rgb_image - data_util.means[FLAGS.dataset].reshape(-1, 1, 1)) / data_util.stds[FLAGS.dataset].reshape(-1,1,1)
    return normalized_rgb_image

  # Apply augmentations using a JAX augmentation function and vmap to do a different transformation for each example in the minibatch
  # Need to pass in a vector of random seeds and vmap over the random seed vector as well as the images right?
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

      # Gradient clipping stuff
      # =======================
      grads = jax.tree_map(lambda g: jnp.nan_to_num(g), grads)  # Get rid of nans by setting to 0 in the gradient

      if FLAGS.grad_clip > 0:
          if FLAGS.grad_clip_type == 'hardclip':
              grads = jax.tree_map(lambda g: jnp.clip(g, a_min=-FLAGS.grad_clip, a_max=FLAGS.grad_clip), grads)  # Clip very small and very large inner gradients
          elif FLAGS.grad_clip_type == 'normclip':
              grad_norm = general_utils.flat_norm(grads)
              grads = jax.tree_map(lambda g: g * jnp.minimum(1., FLAGS.grad_clip / grad_norm), grads)
      # ===============================

      inner_optim_params = get_inner_optim_params(inner_optim_params, theta, t, FLAGS.T)
      params, inner_optim_params = inner_optimizer_step(params, grads, inner_optim_params, t)

      state['params'] = params
      state['model_state'] = model_state
      state['inner_optim_params'] = inner_optim_params

      # Evaluate the loss on the same example _after_ taking a gradient step
      # loss, (logits, _) = loss_fn(params, inputs, targets, model_state, True)
      # acc = (jnp.argmax(logits, axis=1) == targets).sum() / targets.shape[0]

      if FLAGS.objective in ['train_sum_fixed_loss', 'train_sum_fixed_acc', 'val_sum_fixed_loss', 'val_sum_fixed_acc']:
        train_loss, (train_logits, _) = loss_fn(params, state['fixed_train_inputs'], state['fixed_train_targets'], model_state, True)
        train_acc = (jnp.argmax(train_logits, axis=1) == state['fixed_train_targets']).sum() / state['fixed_train_targets'].shape[0]
        val_loss, (val_logits, _) = loss_fn(params, state['fixed_val_inputs'], state['fixed_val_targets'], model_state, True)
        val_acc = (jnp.argmax(val_logits, axis=1) == state['fixed_val_targets']).sum() / state['fixed_val_targets'].shape[0]
      else:
        key, key_train, key_val = jax.random.split(key, 3)
        rand_idx = jax.random.randint(key_train, (), 0, len(state['all_train_ys']))
        rand_train_xs, rand_train_ys = state['all_train_xs'][rand_idx], state['all_train_ys'][rand_idx]
        train_loss, (train_logits, _) = loss_fn(params, rand_train_xs, rand_train_ys, model_state, True)
        train_acc = (jnp.argmax(train_logits, axis=1) == rand_train_ys).sum() / rand_train_ys.shape[0]

        # Choose a random validation minibatch and compute the loss and accuracy on it for the val containers
        rand_idx = jax.random.randint(key_val, (), 0, len(state['all_val_ys']))
        rand_val_xs, rand_val_ys = state['all_val_xs'][rand_idx], state['all_val_ys'][rand_idx]
        val_loss, (val_logits, _) = loss_fn(params, rand_val_xs, rand_val_ys, model_state, True)
        val_acc = (jnp.argmax(val_logits, axis=1) == rand_val_ys).sum() / rand_val_ys.shape[0]

      state['loss_container'] = state['loss_container'].at[state['t']].set(loss)
      # state['loss_container'] = jax.ops.index_update(state['loss_container'], state['t'], train_loss)
      state['acc_container'] = state['acc_container'].at[state['t']].set(train_acc)
      state['val_loss_container'] = state['val_loss_container'].at[state['t']].set(val_loss)
      state['val_acc_container'] = state['val_acc_container'].at[state['t']].set(val_acc)

      state['t'] += 1
      state['key'] = key
      return state

  @jax.jit
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
             T,
             K):
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
                   }
      loop_state = jax.lax.fori_loop(i, jnp.min(jnp.array([i+K, len(all_train_xs)])), update, loop_state)

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

      return key, metric_dict, params, model_state, inner_optim_params

  def wrapper(key,
              params,
              state,
              inner_optim_params,
              all_train_xs,
              all_train_ys,
              all_val_xs,
              all_val_ys,
              fixed_train_inputs,
              fixed_train_targets,
              fixed_val_inputs,
              fixed_val_targets,
              theta):
      t = 0
      num_minibatches_seen = 0
      loss_container = jnp.zeros(FLAGS.T)
      acc_container = jnp.zeros(FLAGS.T)
      val_loss_container = jnp.zeros(FLAGS.T)
      val_acc_container = jnp.zeros(FLAGS.T)

      metric_sum_dict = defaultdict(float)

      while t < FLAGS.T:
          key, skey = jax.random.split(key)

          if num_minibatches_seen >= len(all_train_xs):
              print('Resetting training minibatches at t = {} out of T = {}'.format(t, FLAGS.T))
              sys.stdout.flush()
              if FLAGS.shuffle:
                  all_train_xs, all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
                  all_val_xs, all_val_ys = data_utils.create_minibatches(key, val_data, val_targets, FLAGS.batch_size)

              if FLAGS.resample_fixed_minibatch:
                  key, key_train, key_val = jax.random.split(key, 3)
                  fixed_train_inputs, fixed_train_targets = get_random_minibatch(key_train, all_train_xs, all_train_ys)
                  fixed_val_inputs, fixed_val_targets = get_random_minibatch(key_val, all_val_xs, all_val_ys)

              num_minibatches_seen = 0

          key, metric_dict, params, model_state, inner_optim_params = unroll(key,
                                                                             params,
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
                                                                             FLAGS.K)
          t += FLAGS.K
          num_minibatches_seen += FLAGS.K

          for metric_key in ['train_sum_loss', 'train_sum_acc', 'val_sum_loss', 'val_sum_acc',
                             'train_rand_loss_unroll', 'train_rand_acc_unroll', 'train_fixed_loss_unroll', 'train_fixed_acc_unroll',
                             'val_rand_loss_unroll', 'val_rand_acc_unroll', 'val_fixed_loss_unroll', 'val_fixed_acc_unroll']:
            metric_sum_dict[metric_key] += metric_dict[metric_key]

      full_train_loss, full_train_acc = evaluate(params, state, all_train_xs, all_train_ys)
      full_val_loss, full_val_acc = evaluate(params, state, all_val_xs, all_val_ys)
      full_test_loss, full_test_acc = evaluate(params, state, all_test_xs, all_test_ys)

      metric_sum_dict.update({ 'full_train_loss': full_train_loss,
                               'full_train_acc': full_train_acc,
                               'full_val_loss': full_val_loss,
                               'full_val_acc': full_val_acc,
                               'full_test_loss': full_test_loss,
                               'full_test_acc': full_test_acc,
                             })

      return metric_sum_dict

  exp_dir = '{}/{}_{}_{}_{}_{}_T_{}_N_{}'.format(
             FLAGS.save_dir, FLAGS.search_type, FLAGS.dataset, FLAGS.model,
             tune_params_str, FLAGS.inner_optimizer, FLAGS.T, FLAGS.num_points)

  exp_dir = os.path.join(exp_dir, 'seed_{}'.format(FLAGS.seed))
  gfile.makedirs(exp_dir)

  key_flags = FLAGS.get_key_flags_for_module(argv[0])
  myflagdict = {myflag.name: myflag.value for myflag in key_flags}
  with gfile.GFile(os.path.join(exp_dir, 'args.pkl'), 'w') as f:
    pkl.dump(myflagdict, f)

  num_leaves = len(jax.tree_leaves(params))
  key = jax.random.PRNGKey(FLAGS.seed)

  hparam_names = [setting['param'] for setting in FLAGS.tune_params]
  thetas = []

  if FLAGS.search_type == 'grid':
      if tune_params_str == 'lr:inverse-time-decay':  # Special case so that we can have handcrafted hparam ranges
          grid_dim = 2
          X = onp.linspace(-3, 1, FLAGS.num_points)
          Y = onp.linspace(-2.2, 2.2, FLAGS.num_points)
          X_for_base_e = onp.linspace(onp.log(10**(-3.0)), onp.log(10**1.0), FLAGS.num_points)  # Base e conversion
          Y_for_base_e = onp.linspace(onp.log(10**(-2.2)), onp.log(10**2.2), FLAGS.num_points)
          thetas = jnp.array(list(itertools.product(X_for_base_e, Y_for_base_e)))
      elif len(FLAGS.tune_params) == 1:
          setting = FLAGS.tune_params[0]
          param = setting['param']
          sched = setting['sched']
          if sched == 'fixed':
              grid_dim = 1
              thetas = jnp.linspace(*hparam_utils.hparam_range_dict[param], FLAGS.num_points).reshape(-1,1)
          elif sched == 'linear':
              grid_dim = 2
              X = jnp.linspace(*hparam_utils.hparam_range_dict[param], FLAGS.num_points)
              Y = jnp.linspace(*hparam_utils.hparam_range_dict[param], FLAGS.num_points)
              thetas = jnp.array(list(itertools.product(X, Y)))
          else:
              raise Exception('Grid search not supported for that configuration!')
      elif len(FLAGS.tune_params) == 2:
          grid_dim = 2
          param1 = FLAGS.tune_params[0]['param']
          sched1 = FLAGS.tune_params[0]['sched']
          param2 = FLAGS.tune_params[1]['param']
          sched2 = FLAGS.tune_params[1]['sched']
          if not (sched1 == sched2 == 'fixed'):
              raise Exception('Both params must have fixed schedules for the 2D grid search!')

          X = jnp.linspace(*hparam_utils.hparam_range_dict[param1], FLAGS.num_points)
          Y = jnp.linspace(*hparam_utils.hparam_range_dict[param2], FLAGS.num_points)
          thetas = jnp.array(list(itertools.product(X, Y)))
  elif FLAGS.search_type == 'random':
      for setting, theta_val in zip(FLAGS.tune_params, theta_vals):
          param = setting['param']
          sched = setting['sched']
          min_range, max_range = hparam_utils.hparam_range_dict[param]

          theta = []
          if sched == 'inverse-time-decay':
              key, key1, key2 = jax.random.split(key, 3)
              decay_min_range, decay_max_range = -4, 4  # A fixed range for the decay factor regardless of which hyperparameter we're dealing with
              sampled_init_values = jax.random.uniform(key1, (FLAGS.num_points,len(theta_val)//2)) * (max_range - min_range) + min_range
              sampled_decay_values = jax.random.uniform(key2, (FLAGS.num_points,len(theta_val)//2)) * (decay_max_range - decay_min_range) + decay_min_range
              theta = jnp.concatenate([sampled_init_values, sampled_decay_values], axis=1)  # (500, 2)
              thetas.append(theta)
          elif sched == 'inverse-time-decay-pl':
              decay_min_range, decay_max_range = -4, 4  # A fixed range for the decay factor regardless of which hyperparameter we're dealing with
              for i in range(FLAGS.num_points):
                  point_hparams = []
                  for j in range(num_leaves):
                      key, key1, key2 = jax.random.split(key, 3)
                      leaf_init = jax.random.uniform(key1) * (max_range - min_range) + min_range
                      leaf_decay = jax.random.uniform(key2) * (decay_max_range - decay_min_range) + decay_min_range
                      point_hparams.append(jnp.array([leaf_init, leaf_decay]))
                  point_hparams = jnp.concatenate(point_hparams)
                  theta.append(point_hparams)
              theta = jnp.stack(theta)  # (500, 12)
              thetas.append(theta)
          elif sched == 'piecewise':
              key, key1, key2 = jax.random.split(key, 3)
              knot_loc_min_range, knot_loc_max_range = 0, 10
              sampled_knot_locs = jax.random.uniform(key1, (FLAGS.num_points, FLAGS.num_pieces)) * (knot_loc_max_range - knot_loc_min_range) + knot_loc_min_range
              sampled_knot_values = jax.random.uniform(key2, (FLAGS.num_points, FLAGS.num_pieces+1)) * (max_range - min_range) + min_range
              theta = jnp.concatenate([sampled_knot_locs, sampled_knot_values], axis=1)  # (500, 5)
              thetas.append(theta)
          elif sched == 'piecewise-pl':
              knot_loc_min_range, knot_loc_max_range = 0, 10
              for i in range(FLAGS.num_points):
                  point_hparams = []
                  for j in range(num_leaves):
                      key, key1, key2 = jax.random.split(key, 3)
                      leaf_knot_locs = jax.random.uniform(key1, (FLAGS.num_pieces,)) * (knot_loc_max_range - knot_loc_min_range) + knot_loc_min_range
                      leaf_knot_values = jax.random.uniform(key2, (FLAGS.num_pieces+1,)) * (max_range - min_range) + min_range
                      point_hparams.append(jnp.concatenate([leaf_knot_locs, leaf_knot_values]))
                  point_hparams = jnp.concatenate(point_hparams)
                  theta.append(point_hparams)
              theta = jnp.stack(theta)  # (500, 30) if num_points==2, (500, 42) if num_points==3
              thetas.append(theta)
          else:
              key, skey = jax.random.split(key)
              min_range, max_range = hparam_utils.hparam_range_dict[param]
              theta = jax.random.uniform(skey, (FLAGS.num_points,len(theta_val))) * (max_range - min_range) + min_range
              thetas.append(theta)

      thetas = jnp.concatenate(thetas, axis=1)

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

  if FLAGS.chunk_size == -1:
      FLAGS.chunk_size = len(thetas)

  num_chunks = int(math.ceil(len(thetas) / FLAGS.chunk_size))

  results_dict = defaultdict(list)
  for chunk in tqdm(range(num_chunks)):
      key, skey = jax.random.split(key)

      # if FLAGS.shuffle:
      #     all_train_xs, all_train_ys = data_utils.create_minibatches(key, train_data, train_targets, FLAGS.batch_size)
      #     all_val_xs, all_val_ys = data_utils.create_minibatches(key, val_data, val_targets, FLAGS.batch_size)

      start = chunk * FLAGS.chunk_size
      end = min(start + FLAGS.chunk_size, len(thetas))
      chunk_thetas = thetas[start:end]
      inner_optim_params = reset_inner_optim_params(params)

      wrapper_vmap = jax.vmap(wrapper, in_axes=(None,  # key
                                                None,  # params
                                                None,  # state
                                                None,  # inner_optim_params
                                                None,  # all_train_xs
                                                None,  # all_train_ys
                                                None,  # all_val_xs
                                                None,  # all_val_ys
                                                None,  # fixed_train_inputs
                                                None,  # fixed_train_targets
                                                None,  # fixed_val_inputs
                                                None,  # fixed_val_targets
                                                0      # theta
                                               ))
      # vmap over the random key and the value of theta
      # wrapper_vmap = jax.vmap(wrapper, in_axes=(0,None,None,None,None,None,None,None,0))
      # keys = jax.random.split(key, len(chunk_thetas)+1)
      # key = keys[0]
      # keys_for_vmap = keys[1:]
      metric_dict = wrapper_vmap(# keys_for_vmap,
                                 key,
                                 params,
                                 state,
                                 inner_optim_params,
                                 all_train_xs,
                                 all_train_ys,
                                 all_val_xs,
                                 all_val_ys,
                                 fixed_train_inputs,
                                 fixed_train_targets,
                                 fixed_val_inputs,
                                 fixed_val_targets,
                                 chunk_thetas)

      for metric_key in metric_dict:
          results_dict[metric_key].append(onp.array(metric_dict[metric_key]))

      if FLAGS.search_type == 'grid':
          if grid_dim == 1:
              with gfile.GFile(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
                pkl.dump({ 'thetas': onp.array(thetas[:end]),  # The thetas SO FAR, just to make it easier to read/plot
                           'thetas_constrained': onp.array([to_constrained(theta_value.reshape(1)) for theta_value in thetas[:end]]).reshape(-1),
                           'hparam_names': hparam_names,  # Just recording a list of the hyperparameter names to make it easier to plot with labels!
                           'hparam_fieldnames': param_fieldnames,
                           **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
                         }, f)
          elif grid_dim == 2:
              xv, yv = onp.meshgrid(X, Y)
              with gfile.GFile(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
                pkl.dump({ 'thetas': onp.array(thetas[:end]),  # The thetas SO FAR, just to make it easier to read/plot
                           'thetas_constrained': onp.array([to_constrained(theta_value) for theta_value in thetas[:end]]),
                           'xv': xv,
                           'yv': yv,
                           'hparam_names': hparam_names,  # Just recording a list of the hyperparameter names to make it easier to plot with labels!
                           'hparam_fieldnames': param_fieldnames,
                           **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
                         }, f)
      elif FLAGS.search_type == 'random':
          with gfile.GFile(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
            pkl.dump({ 'thetas': onp.array(thetas[:end]),  # The thetas SO FAR, just to make it easier to read/plot
                       'thetas_constrained': onp.array([to_constrained(theta_value) for theta_value in thetas[:end]]),
                       'hparam_names': hparam_names,  # Just recording a list of the hyperparameter names to make it easier to plot with labels!
                       'hparam_fieldnames': param_fieldnames,
                       **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
                     }, f)

if __name__ == '__main__':
  app.run(main)
