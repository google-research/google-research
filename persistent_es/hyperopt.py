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

"""Hyperparameter optimization with ES/PES."""
import os
import sys
import pdb
import copy
import time
import random
import numpy as onp
from functools import partial
from typing import NamedTuple, Optional, Any

import jax
import jax.numpy as jnp
from jax import flatten_util

import optax
import haiku as hk

# Local imports
import models
import schedule
import data_utils
import inner_optim
import hparam_utils
import parser_utils
import general_utils
import gradient_estimators
from logger import CSVLogger

parser = parser_utils.create_parser()
args = parser.parse_args()

tune_params_str = args.tune_params or 'None'

if args.tune_params:
  args.tune_params = [{
      'param': hparam_utils.abbreviation_dict[p.split(':')[0]],
      'sched': hparam_utils.abbreviation_dict[p.split(':')[1]]
  } for p in args.tune_params.split(',')]

schedule_info = {}
for item in args.tune_params:
  schedule_info[item['param']] = item['sched']

# Set random seed for reproducibility
if args.seed is not None:
  random.seed(args.seed)
  onp.random.seed(args.seed)

# The experiment directory name contains everything except for the seed, so
# that multiple exps with identical params except for the random seed will be
# put into the same subdir
exp_dir = '{}'.format(args.objective)
exp_name = '{}-{}-{}-obj:{}-tune:{}-T:{}-K:{}-nc:{}-npc:{}-sigma:{}-olr:{}-ob1:{}-ob2:{}-ic:{}-oc:{}-seed:{}'.format(
    args.estimate, args.dataset, args.model, args.objective, tune_params_str,
    args.T, args.K, args.n_chunks, args.n_per_chunk, args.sigma, args.outer_lr,
    args.outer_b1, args.outer_b2, args.inner_clip, args.outer_clip, args.seed)

save_dir = os.path.join(args.save_dir, exp_dir, exp_name)
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

with open(os.path.join(save_dir, 'command.txt'), 'w') as f:
  f.write('\n'.join(['python {} \\'.format(sys.argv[0])] + \
          ['    {} \\'.format(line) for line in sys.argv[1:-1]] + \
          ['    {}'.format(sys.argv[-1])]))

# Create dataloaders
# ----------------------------------------
data_dict = data_utils.load_data(args.dataset)
train_data, train_targets = data_dict['train_data'], data_dict['train_targets']
val_data, val_targets = data_dict['val_data'], data_dict['val_targets']
test_data, test_targets = data_dict['test_data'], data_dict['test_targets']

# ----------------------------------------


# Model
# ----------------------------------------
def net_fn(inputs, theta=None, is_training=True):
  if args.model == 'mlp':
    mlp = models.MLP(
        nlayers=args.nlayers, nhid=args.nhid, with_bias=True, batch_norm=False)
    return mlp(inputs, theta, is_training)
  elif args.model == 'resnet':
    network = models.Net(args.model_size)
    return network(inputs, is_training)


net = hk.without_apply_rng(hk.transform_with_state(net_fn))
# ----------------------------------------


@partial(jax.jit, static_argnames=('is_training',))
def loss_with_logits(params, state, inputs, targets, theta, is_training):
  if 'mask' in tune_params_str:
    mask_props = hparam_utils.cons_funcs['mask'](theta[idx_dict['mask']])
  else:
    mask_props = None

  logits, updated_state = net.apply(params, state, inputs, mask_props,
                                    is_training)
  labels = jax.nn.one_hot(targets, 10)
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]
  return softmax_xent, (logits, updated_state)


@partial(jax.jit, static_argnames=('is_training',))
def loss(params, state, inputs, targets, theta, is_training):
  softmax_xent, (logits,
                 updated_state) = loss_with_logits(params, state, inputs,
                                                   targets, theta, is_training)
  return softmax_xent, updated_state


loss_and_grad = jax.jit(
    jax.value_and_grad(loss, has_aux=True), static_argnames=('is_training',))

opt_funcs = inner_optim.init_optimizer(args.inner_optimizer)
reset_opt_params = opt_funcs['reset_opt_params']
opt_step = opt_funcs['opt_step']

default_values = {
    'lr': args.lr,
    'b1': args.b1,
    'b2': args.b2,
    'eps': args.eps,
    'mom': args.mom,
    'wd': args.wd,
    'mask': 0.5,
}


class InnerState(NamedTuple):
  params: hk.Params
  inner_opt_state: dict
  t: jnp.ndarray
  model_state: Any = None
  prev_obj: Optional[jnp.ndarray] = None
  pert_accums: Optional[jnp.ndarray] = None


class LoopState(NamedTuple):
  inner_state: InnerState
  key: Any


@partial(jax.jit, static_argnames='batch_size')
def get_minibatch(key, data, targets, batch_size):
  idxs = jax.random.permutation(key, len(targets))[:batch_size]
  minibatch_inputs = data[idxs]
  minibatch_targets = targets[idxs]
  return minibatch_inputs, minibatch_targets


def get_current_opt_params(inner_optim_params, theta, t, T):
  updated_inner_opt_params = copy.deepcopy(inner_optim_params)
  for setting in args.tune_params:
    param = setting['param']
    sched = setting['sched']
    # Only deal with optimization hparams here
    if param not in ['lr', 'mom', 'b1', 'b2', 'eps', 'wd']:
      continue
    theta_subset = theta[idx_dict[param]]
    updated_inner_opt_params[param] = schedule.schedule_funcs[sched](
        inner_optim_params,
        theta_subset,
        param,
        t,
        T,
        unflatten_func_dict=unflatten_func_dict)
  return updated_inner_opt_params


def get_train_or_val_minibatch(key):
  if 'train' in args.objective:
    inputs_to_eval, targets_to_eval = get_minibatch(key, train_data,
                                                    train_targets,
                                                    args.objective_batch_size)
  elif 'val' in args.objective:
    inputs_to_eval, targets_to_eval = get_minibatch(key, val_data, val_targets,
                                                    args.objective_batch_size)
  return inputs_to_eval, targets_to_eval


@partial(jax.jit, static_argnames='is_training')
def compute_obj(params, model_state, inputs, targets, theta, is_training):
  loss_value, (logits, _) = loss_with_logits(params, model_state, inputs,
                                             targets, theta, is_training)

  if 'acc' in args.objective:
    pred = jnp.argmax(logits, axis=1)
    acc = jnp.mean(pred == targets)
    obj = -acc
  else:
    obj = loss_value

  return obj


@partial(jax.jit, static_argnames=('T', 'K'))
def unroll(key, theta, inner_state, T, K):

  def update_fn(loop_state, x):
    key = loop_state.key
    inner_state = loop_state.inner_state
    inputs, targets = get_minibatch(key, train_data, train_targets,
                                    args.batch_size)
    (loss_value,
     updated_model_state), grads = loss_and_grad(inner_state.params,
                                                 inner_state.model_state,
                                                 inputs, targets, theta, True)

    # Gradient clipping
    # =================
    grads = jax.tree_map(lambda g: jnp.nan_to_num(g), grads)

    if args.inner_clip > 0:
      grads = jax.tree_map(
          lambda g: jnp.clip(g, a_min=-args.inner_clip, a_max=args.inner_clip),
          grads)
    # =================

    inner_opt_params = get_current_opt_params(inner_state.inner_opt_state,
                                              theta, inner_state.t, T)
    updated_params, updated_inner_opt_params = opt_step(inner_state.params,
                                                        grads, inner_opt_params)

    # Sample a minibatch to compute the loss after the gradient step
    # --------------------------------------------------------------
    key, skey = jax.random.split(key)
    inputs_to_eval, targets_to_eval = get_train_or_val_minibatch(skey)
    obj = compute_obj(updated_params, updated_model_state, inputs_to_eval,
                      targets_to_eval, theta, False)
    # --------------------------------------------------------------

    # Update inner_state and loop_state
    updated_key, _ = jax.random.split(key)
    inner_state = inner_state._replace(
        t=inner_state.t + 1,
        params=updated_params,
        model_state=updated_model_state,
        inner_opt_state=updated_inner_opt_params)
    loop_state = loop_state._replace(key=updated_key, inner_state=inner_state)
    return loop_state, obj

  key_unroll, key_eval = key
  loop_state = LoopState(key=key_unroll, inner_state=inner_state)
  updated_loop_state, unroll_objs = jax.lax.scan(
      update_fn, loop_state, None, length=K)
  updated_inner_state = updated_loop_state.inner_state

  # Final evaluation on a random minibatch after the unroll
  inputs_to_eval, targets_to_eval = get_train_or_val_minibatch(key_eval)
  final_obj = compute_obj(updated_inner_state.params,
                          updated_inner_state.model_state, inputs_to_eval,
                          targets_to_eval, theta, False)

  if 'final' in args.objective:
    meta_obj = final_obj
  elif 'sum' in args.objective:
    meta_obj = jnp.sum(unroll_objs)

  if args.telescoping:
    meta_obj_to_return = meta_obj - updated_inner_state.prev_obj
    updated_inner_state = updated_inner_state._replace(prev_obj=meta_obj)
  else:
    meta_obj_to_return = meta_obj

  return meta_obj_to_return, updated_inner_state


def init_state_fn(key):
  inner_params, model_state = net.init(key, train_data[:10], is_training=True)
  inner_opt_state = reset_opt_params(inner_params, default_values)

  if args.telescoping:
    prev_obj = jnp.array(0.0)
  else:
    prev_obj = None

  return InnerState(
      params=inner_params,
      model_state=model_state,
      inner_opt_state=inner_opt_state,
      t=jnp.array(0).astype(jnp.int32),
      prev_obj=prev_obj)


@jax.jit
def evaluate_minibatch(params, state, data, theta):
  inputs, targets = data
  if targets.ndim == 0:
    inputs = inputs.reshape(1, *inputs.shape)
    targets = targets.reshape(1)
  xentropy, (logits, _) = loss_with_logits(params, state, inputs, targets,
                                           theta, False)
  pred = jnp.argmax(logits, axis=1)
  num_total = targets.shape[0]
  num_correct = (pred == targets).sum()
  return xentropy, num_correct, num_total


@jax.jit
def evaluate(params, state, xs, ys, theta):
  (losses, num_correct, num_total) = jax.lax.map(
      lambda data: evaluate_minibatch(params, state, data, theta), (xs, ys))
  return (jnp.mean(losses), jnp.sum(losses),
          jnp.sum(num_correct) / jnp.sum(num_total))


def full_eval(key, theta):
  key, model_key, unroll_key, eval_key = jax.random.split(key, 4)
  inner_state = init_state_fn(model_key)
  unroll_obj, inner_state_new = unroll((unroll_key, eval_key), theta,
                                       inner_state, args.T, args.T)
  train_mean_loss, train_sum_loss, train_acc = evaluate(
      inner_state_new.params, inner_state_new.model_state, train_data,
      train_targets, theta)
  val_mean_loss, val_sum_loss, val_acc = evaluate(inner_state_new.params,
                                                  inner_state_new.model_state,
                                                  val_data, val_targets, theta)

  return {
      'unroll_obj': unroll_obj,
      'train_mean_loss': train_mean_loss,
      'train_sum_loss': train_sum_loss,
      'train_acc': train_acc,
      'val_mean_loss': val_mean_loss,
      'val_sum_loss': val_sum_loss,
      'val_acc': val_acc
  }


@partial(jax.jit, static_argnames='num_eval_runs')
def full_evaluation_runs(key, theta, num_eval_runs=1):
  eval_run_keys = jax.random.split(key, num_eval_runs)
  stats_dict = jax.vmap(full_eval, in_axes=(0, None))(eval_run_keys, theta)
  return stats_dict


# CSV logger setup
# ================
temp_state = init_state_fn(jax.random.PRNGKey(3))
temp_params = temp_state.params

param_fieldnames = []
for setting in args.tune_params:
  if setting['param'] == 'mask':
    num_masks = args.nlayers
    param_fieldnames += ['mask_{}'.format(i) for i in range(args.nlayers)]
  elif setting['sched'] == 'fixed':
    param_fieldnames += ['{}_0'.format(setting['param'])]
  elif setting['sched'] in ['linear', 'inverse-time-decay']:
    param_fieldnames += [
        '{}_0'.format(setting['param']), '{}_1'.format(setting['param'])
    ]
  elif setting['sched'] == 'fixed-pl':
    for key in general_utils.recursive_keys(temp_params):
      base_str = '{}/{}'.format(key, setting['param'])
      param_fieldnames += ['{}_0'.format(base_str)]
  elif setting['sched'] in ['linear-pl', 'inverse-time-decay-pl']:
    for key in general_utils.recursive_keys(temp_params):
      base_str = '{}/{}'.format(key, setting['param'])
      param_fieldnames += ['{}_0'.format(base_str), '{}_1'.format(base_str)]

print('Param fieldnames: {}'.format(param_fieldnames))

cons_param_fieldnames = ['cons/{}'.format(name) for name in param_fieldnames]
param_grad_fieldnames = ['grad/{}'.format(name) for name in param_fieldnames]

iteration_logger = CSVLogger(
    fieldnames=[
        'perf/{}'.format(name) for name in [
            'time_elapsed', 'outer_iteration', 'total_inner_iterations',
            'train_sum_loss', 'train_acc', 'train_mean_loss', 'val_sum_loss',
            'val_acc', 'val_mean_loss', 'unroll_obj'
        ]
    ] + cons_param_fieldnames + param_fieldnames,
    filename=os.path.join(save_dir, 'iteration.csv'))

frequent_logger = CSVLogger(
    fieldnames=[
        'frequent/time_elapsed',
        'frequent/outer_iteration',
        'frequent/total_inner_iterations',
        'frequent/F',
    ] + cons_param_fieldnames + param_fieldnames + param_grad_fieldnames,
    filename=os.path.join(save_dir, 'frequent.csv'))
# =======================================================================


def to_constrained(theta_unconstrained):
  theta_constrained = []
  for setting in args.tune_params:
    param = setting['param']
    unconstrained_values = theta_unconstrained[onp.array(idx_dict[param])]
    constrained_values = hparam_utils.cons_funcs[param](unconstrained_values)
    if constrained_values.ndim == 0:
      constrained_values = constrained_values.reshape(1)
    theta_constrained.append(constrained_values)
  return onp.concatenate(theta_constrained)


key = jax.random.PRNGKey(args.seed)

theta_vals = []
idx_dict = {}
setting_idx_dict = {}
unflatten_func_dict = {}
idx = 0
for setting in args.tune_params:
  # Only needed when using --random_hparam_init
  key, skey = jax.random.split(key)
  param = setting['param']
  sched = setting['sched']
  default = hparam_utils.uncons_funcs[param](default_values[param])

  if param in hparam_utils.meta_opt_init_range_dict:
    min_range, max_range = hparam_utils.meta_opt_init_range_dict[param]
  else:
    min_range, max_range = hparam_utils.hparam_range_dict[param]

  # Special case for masks, since we need as many masks as layers in the model
  if param == 'mask':
    if args.random_hparam_init:
      theta_vals.append(
          jax.random.uniform(skey, (args.nlayers,)) * (max_range - min_range) +
          min_range)
    else:
      theta_vals.append(jnp.array([default] * args.nlayers))

    idx_dict[param] = jnp.array(list(range(idx, idx + args.nlayers)))
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += args.nlayers
    continue

  if sched == 'fixed':
    if args.random_hparam_init:
      theta_vals.append(
          jax.random.uniform(skey, (1,)) * (max_range - min_range) + min_range)
    else:
      theta_vals.append(jnp.array([default]))
    idx_dict[param] = idx
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += 1
  elif sched == 'linear':
    if args.random_hparam_init:
      theta_vals.append(
          jax.random.uniform(skey, (2,)) * (max_range - min_range) + min_range)
    else:
      theta_vals.append(
          jnp.array([
              hparam_utils.uncons_funcs[param](args.lr0),
              hparam_utils.uncons_funcs[param](args.lr1)
          ]))
    idx_dict[param] = jnp.array([idx, idx + 1])
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += 2
  elif sched == 'inverse-time-decay':
    if args.random_hparam_init:
      key, key1, key2 = jax.random.split(key, 3)
      # A fixed range for the decay factor regardless of which hyperparameter we're dealing with
      decay_min_range, decay_max_range = -4, 4
      sampled_init_values = jax.random.uniform(
          key1, (1,)) * (max_range - min_range) + min_range
      sampled_decay_values = jax.random.uniform(
          key2, (1,)) * (decay_max_range - decay_min_range) + decay_min_range
      theta_vals.append(
          jnp.concatenate([sampled_init_values, sampled_decay_values], axis=0))
    else:
      theta_vals.append(
          jnp.array([
              hparam_utils.uncons_funcs[param](args.lr0),
              hparam_utils.uncons_funcs[param](args.lr1)
          ]))
    idx_dict[param] = jnp.array([idx, idx + 1])
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += 2
  elif sched == 'fixed-pl':
    if args.random_hparam_init:
      leaves, treedef = jax.tree_flatten(temp_params)
      random_values = jax.random.uniform(
          skey, (len(leaves),)) * (max_range - min_range) + min_range
      random_values = [random_values[i] for i in range(len(random_values))]
      hparam_tree = jax.tree_unflatten(treedef, random_values)
    else:
      hparam_tree = jax.tree_map(lambda x: jnp.array(default), temp_params)
    hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(
        hparam_tree)
    unflatten_func_dict[param] = hparam_unravel_pytree
    theta_vals.append(hparam_vector)
    idx_dict[param] = jnp.array(list(range(idx, idx + len(hparam_vector))))
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += len(hparam_vector)
  elif sched in ['linear-pl', 'inverse-time-decay-pl']:
    if args.random_hparam_init:
      leaves, treedef = jax.tree_flatten(temp_params)
      random_values = jax.random.uniform(
          skey, (len(leaves), 2)) * (max_range - min_range) + min_range
      random_values = [random_values[i] for i in range(len(random_values))]
      hparam_tree = jax.tree_unflatten(treedef, random_values)
    else:
      hparam_tree = jax.tree_map(lambda x: jnp.array([default, default]),
                                 temp_params)
    hparam_vector, hparam_unravel_pytree = flatten_util.ravel_pytree(
        hparam_tree)
    unflatten_func_dict[param] = hparam_unravel_pytree
    theta_vals.append(hparam_vector)
    idx_dict[param] = jnp.array(list(range(idx, idx + len(hparam_vector))))
    setting_idx_dict['{}:{}'.format(param, sched)] = idx_dict[param]
    idx += len(hparam_vector)

theta = jnp.concatenate(theta_vals)

outer_opt = optax.adam(
    args.outer_lr, b1=args.outer_b1, b2=args.outer_b2, eps=args.outer_eps)
outer_opt_state = outer_opt.init(theta)

key = jax.random.PRNGKey(args.seed)
estimator = gradient_estimators.MultiParticleEstimator(
    key=key,
    theta_shape=theta.shape,
    n_chunks=args.n_chunks,
    n_particles_per_chunk=args.n_per_chunk,
    K=args.K,
    T=args.T,
    sigma=args.sigma,
    method='lockstep',
    telescoping=args.telescoping,
    estimator_type=args.estimate,
    init_state_fn=init_state_fn,
    unroll_fn=unroll,
)

start_time = time.time()
total_inner_iterations = 0
total_inner_iterations_including_N = 0

# Meta-optimization loop
for outer_iteration in range(args.outer_iterations):
  outer_grad = estimator.grad_estimate(theta)

  if args.outer_clip > 0:
    outer_grad = jax.tree_map(
        lambda g: jnp.clip(g, a_min=-args.outer_clip, a_max=args.outer_clip),
        outer_grad)

  outer_update, outer_opt_state = outer_opt.update(outer_grad, outer_opt_state)
  theta = optax.apply_updates(theta, outer_update)

  total_inner_iterations = args.K * outer_iteration
  total_inner_iterations_including_N = (
      args.K * args.n_chunks * args.n_per_chunk * outer_iteration)

  if jnp.any(jnp.isnan(theta)):
    print('=' * 80 + '\nExiting early.\n' + '=' * 80)
    sys.exit(0)

  if outer_iteration % args.print_every == 0:
    print('Outer iter: {} | Theta: {} | Theta constrained: {}'.format(
        outer_iteration, theta, to_constrained(theta)))
    sys.stdout.flush()

  if outer_iteration % args.eval_every == 0:
    key, skey = jax.random.split(key)
    stats_dict = full_evaluation_runs(
        skey, theta, num_eval_runs=args.num_eval_runs)
    mean_stats_dict = {
        metric: onp.mean(stats_dict[metric]) for metric in stats_dict
    }
    print(
        'Train acc: {:6.4f} | Val acc: {:6.4f} | Trn sum loss: {:6.4f} | Val sum loss: {:6.4f}'
        .format(mean_stats_dict['train_acc'], mean_stats_dict['val_acc'],
                mean_stats_dict['train_sum_loss'],
                mean_stats_dict['val_sum_loss']))
    sys.stdout.flush()

    hparams_to_log = {}
    for (param_name, value) in zip(param_fieldnames, theta):
      hparams_to_log[param_name] = value

    constrained_hparams_to_log = {}
    for (param_name, value) in zip(cons_param_fieldnames,
                                   to_constrained(theta)):
      constrained_hparams_to_log[param_name] = value

    iteration_log_dict = {
        'perf/time_elapsed': time.time() - start_time,
        'perf/outer_iteration': outer_iteration,
        'perf/total_inner_iterations': total_inner_iterations_including_N,
        'perf/train_sum_loss': mean_stats_dict['train_sum_loss'],
        'perf/train_acc': mean_stats_dict['train_acc'],
        'perf/train_mean_loss': mean_stats_dict['train_mean_loss'],
        'perf/val_sum_loss': mean_stats_dict['val_sum_loss'],
        'perf/val_acc': mean_stats_dict['val_acc'],
        'perf/val_mean_loss': mean_stats_dict['val_mean_loss'],
        'perf/unroll_obj': mean_stats_dict['unroll_obj'],
        **constrained_hparams_to_log,
        **hparams_to_log,
    }

    iteration_logger.writerow(iteration_log_dict)

  if outer_iteration % args.log_every == 0:
    hparams_to_log = {}
    for (param_name, value) in zip(param_fieldnames, theta):
      hparams_to_log[param_name] = value

    constrained_hparams_to_log = {}
    for (param_name, value) in zip(cons_param_fieldnames,
                                   to_constrained(theta)):
      constrained_hparams_to_log[param_name] = value

    hparam_grads_to_log = {}
    for (param_name, value) in zip(param_grad_fieldnames, outer_grad):
      hparam_grads_to_log[param_name] = value

    frequent_log_dict = {
        'frequent/time_elapsed': time.time() - start_time,
        'frequent/outer_iteration': outer_iteration,
        'frequent/total_inner_iterations': total_inner_iterations,
        **constrained_hparams_to_log,
        **hparams_to_log,
        **hparam_grads_to_log
    }

    frequent_logger.writerow(frequent_log_dict)
