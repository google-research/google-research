"""Run grid and random searches with an MLP/ResNet on MNIST/CIFAR-10.
"""
import os
import sys
import pdb
import math
import shutil
import argparse
import itertools
import numpy as onp
import pickle as pkl
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
from jax import flatten_util

import haiku as hk

# Local imports
import models
import schedule
import data_utils
import inner_optim
import hparam_utils
import general_utils


parser = argparse.ArgumentParser(description='Grid and random search')
# Dataset arguments
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Choose dataset')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Batch size')

# Model arguments
parser.add_argument('--model', type=str, default='mlp',
                    help='Choose the model')
parser.add_argument('--model_size', type=str, default='small',
                    help='Model size, that affects #channels in the ResNet'
                         '(tiny, small, med, or large)')
parser.add_argument('--nlayers', type=int, default=2,
                    help='Number of layers in the MLP')
parser.add_argument('--nhid', type=int, default=100,
                    help='Number of hidden units in each layer')

# Inner optimization arguments
parser.add_argument('--inner_optimizer', type=str, default='sgdm',
                    help='Choose the inner optimizer')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learning rate')
parser.add_argument('--b1', type=float, default=0.99,
                    help='Adam b1 hyperparameter')
parser.add_argument('--b2', type=float, default=0.999,
                    help='Adam b2 hyperparameter')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Adam epsilon hyperparameter')
parser.add_argument('--mom', type=float, default=0.9,
                    help='Momentum')
parser.add_argument('--wd', type=float, default=1e-10,
                    help='Weight decay')
parser.add_argument('--inner_clip', type=float, default= -1,
                    help='Gradient clipping for each step of the inner unroll'
                         '(-1 means no grad clipping)')

# Search arguments
parser.add_argument('--objective', type=str, default='train_sum_loss',
                    help='The "objective" that determines what we measure'
                         '(random minibatches vs fixed ones)')
parser.add_argument('--objective_batch_size', type=int, default=100,
                    help='The batch size for computing the meta-objective'
                         'after each unroll')
parser.add_argument('--search_type', type=str, default='random',
                    help='Choose either grid search or random search')
parser.add_argument('--num_points', type=int, default=20,
                    help='Num points for the grid search')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='How many networks to train in parallel during the'
                         'grid/random search')
parser.add_argument('--tune_params', type=str, default='lr:fixed',
                    help='A comma-separated string of hyperparameters to search'
                         'over')
parser.add_argument('--T', type=int, default=1000,
                    help='Maximum number of iterations of the inner loop')
parser.add_argument('--K', type=int, default=50,
                    help='Number of steps to unroll')
parser.add_argument('--num_eval_runs', type=int, default=10,
                    help='Number of runs to average over when doing a search')

# Logging/saving arguments
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
parser.add_argument('--save_dir', type=str, default='saves',
                    help='Save directory')
args = parser.parse_args()


tune_params_str = args.tune_params or 'None'

if args.tune_params:
  args.tune_params = [{'param': hparam_utils.abbreviation_dict[p.split(':')[0]],
                       'sched': hparam_utils.abbreviation_dict[p.split(':')[1]]}
                      for p in args.tune_params.split(',')]

schedule_info = {}
for item in args.tune_params:
  schedule_info[item['param']]= item['sched']

# Set random seed for reproducibility
if args.seed is not None:
  onp.random.seed(args.seed)

exp_dir = '{}/{}_{}_{}_{}_{}_T_{}_N_{}/seed_{}'.format(
           args.save_dir, args.search_type, args.dataset, args.model,
           tune_params_str, args.inner_optimizer, args.T, args.num_points,
           args.seed)

if not os.path.exists(exp_dir):
  os.makedirs(exp_dir)

with open(os.path.join(exp_dir, 'command.txt'), 'w') as f:
  f.write('\n'.join(['python {} \\'.format(sys.argv[0])] + \
          ["    {} \\".format(line) for line in sys.argv[1:-1]] + \
          ['    {}'.format(sys.argv[-1])]))

shutil.copyfile(
    'hparam_utils.py', os.path.join(exp_dir, 'hparam_utils.py')
)

# Create dataloaders
# ----------------------------------------
data_dict = data_utils.load_data(args.dataset)
train_data, train_targets = data_dict['train_data'], data_dict['train_targets']
val_data, val_targets = data_dict['val_data'], data_dict['val_targets']
test_data, test_targets = data_dict['test_data'], data_dict['test_targets']
# ----------------------------------------

# Initialize model
# ----------------------------------------
def net_fn(inputs, theta=None, is_training=True):
  if args.model == 'mlp':
    mlp = models.MLP(nlayers=args.nlayers,
                     nhid=args.nhid,
                     with_bias=True,
                     batch_norm=False)
    return mlp(inputs, theta, is_training)
  elif args.model == 'resnet':
    network = models.Net(args.model_size)
    return network(inputs, is_training)


net = hk.without_apply_rng(hk.transform_with_state(net_fn))
apply_jit = jax.jit(net.apply, static_argnums=(4,))

init_images = train_data[:10]
print('Initializing parameters...')
key = jax.random.PRNGKey(args.seed)
params, state = net.init(key, init_images, is_training=True)
print('Num parameters: {}'.format(general_utils.count_params(params)))
sys.stdout.flush()
# -----------------------------------


@partial(jax.jit, static_argnames=('is_training',))
def loss_with_logits(params, state, inputs, targets, theta, is_training):
  if 'mask' in tune_params_str:
    mask_props = hparam_utils.cons_funcs['mask'](theta[idx_dict['mask']])
  else:
    mask_props = None

  logits, updated_state = net.apply(params, state, inputs, mask_props, is_training)
  labels = jax.nn.one_hot(targets, 10)
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]
  return softmax_xent, (logits, updated_state)


@partial(jax.jit, static_argnames=('is_training',))
def loss(params, state, inputs, targets, theta, is_training):
  softmax_xent, (logits, updated_state) = loss_with_logits(
      params, state, inputs, targets, theta, is_training
  )
  return softmax_xent, updated_state

loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True),
                        static_argnames=('is_training',))


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


class LoopState(NamedTuple):
  state: InnerState
  key: jnp.ndarray


@partial(jax.jit, static_argnames='batch_size')
def get_minibatch(key, data, targets, batch_size):
  idxs = jax.random.permutation(key, len(targets))[:batch_size]
  minibatch_inputs = data[idxs]
  minibatch_targets = targets[idxs]
  return minibatch_inputs, minibatch_targets


@jax.jit
def get_current_opt_params(inner_optim_params, theta, t, T):
  for setting in args.tune_params:
    param = setting['param']
    sched = setting['sched']
    # Only deal with optimization hparams here
    if param not in ['lr', 'mom', 'b1', 'b2', 'eps', 'wd']:
      continue
    theta_subset = theta[idx_dict[param]]
    inner_optim_params[param] = schedule.schedule_funcs[sched](
        inner_optim_params, theta_subset, param, t, T,
        unflatten_func_dict=unflatten_func_dict
    )
  return inner_optim_params


def get_train_or_val_minibatch(key):
  if 'train' in args.objective:
    inputs_to_eval, targets_to_eval = get_minibatch(
        key, train_data, train_targets, args.objective_batch_size
    )
  elif 'val' in args.objective:
    inputs_to_eval, targets_to_eval = get_minibatch(
        key, val_data, val_targets, args.objective_batch_size
    )
  return inputs_to_eval, targets_to_eval


@partial(jax.jit, static_argnames='is_training')
def compute_obj(params, model_state, inputs, targets, theta, is_training):
  loss_value, (logits, _) = loss_with_logits(
      params,
      model_state,
      inputs,
      targets,
      theta,
      is_training
  )

  if 'acc' in args.objective:
    pred = jnp.argmax(logits, axis=1)
    acc = jnp.mean(pred == targets)
    obj = -acc
  else:
    obj = loss_value

  return obj


@partial(jax.jit, static_argnames=('T', 'K'))
def unroll(key, theta, state, T, K):

  @jax.jit
  def update_fn(loop_state, x):
    key = loop_state.key
    state = loop_state.state
    inputs, targets = get_minibatch(key, train_data, train_targets, args.batch_size)
    (loss_value, updated_model_state), grads = loss_and_grad(
        state.params, state.model_state, inputs, targets, theta, True
    )

    # Gradient clipping
    # =================
    grads = jax.tree_map(lambda g: jnp.nan_to_num(g), grads)

    if args.inner_clip > 0:
      grads = jax.tree_map(
          lambda g: jnp.clip(g, a_min=-args.inner_clip, a_max=args.inner_clip),
          grads
      )
    # =================

    inner_opt_params = get_current_opt_params(
        state.inner_opt_state, theta, state.t, T
    )
    updated_params, updated_inner_opt_params = opt_step(
        state.params, grads, inner_opt_params
    )

    # Sample a minibatch to compute the loss after the gradient step
    # --------------------------------------------------------------
    key, skey = jax.random.split(key)
    inputs_to_eval, targets_to_eval = get_train_or_val_minibatch(skey)
    obj = compute_obj(
        updated_params,
        updated_model_state,
        inputs_to_eval,
        targets_to_eval,
        theta,
        False
    )
    # --------------------------------------------------------------

    # Update state and loop_state
    updated_key, _ = jax.random.split(key)
    state = state._replace(
        t=state.t+1,
        params=updated_params,
        model_state=updated_model_state,
        inner_opt_state=updated_inner_opt_params
    )
    loop_state = loop_state._replace(key=updated_key, state=state)
    return loop_state, obj

  key_unroll, key_eval = key
  loop_state = LoopState(key=key_unroll, state=state)
  updated_loop_state, unroll_objs = jax.lax.scan(update_fn, loop_state, None, length=K)
  updated_inner_state = updated_loop_state.state

  # Final evaluation on a random minibatch after the unroll
  inputs_to_eval, targets_to_eval = get_train_or_val_minibatch(key_eval)
  final_obj = compute_obj(
      updated_inner_state.params,
      updated_inner_state.model_state,
      inputs_to_eval,
      targets_to_eval,
      theta,
      False
  )

  if 'final' in args.objective:
    meta_obj = final_obj
  elif 'sum' in args.objective:
    meta_obj = jnp.sum(unroll_objs)

  return meta_obj, updated_loop_state.state


def init_state_fn(key):
  inner_params, model_state = net.init(key, train_data[:10], is_training=True)
  inner_opt_state = reset_opt_params(inner_params, default_values)
  return InnerState(
      params=inner_params,
      model_state=model_state,
      inner_opt_state=inner_opt_state,
      t=jnp.array(0).astype(jnp.int32)
  )


@jax.jit
def evaluate_minibatch(params, state, data, theta):
  inputs, targets = data
  if targets.ndim == 0:
    inputs = inputs.reshape(1, *inputs.shape)
    targets = targets.reshape(1)
  xentropy, (logits, _) = loss_with_logits(params, state, inputs, targets, theta, False)
  pred = jnp.argmax(logits, axis=1)
  num_total = targets.shape[0]
  num_correct = (pred == targets).sum()
  return xentropy, num_correct, num_total


@jax.jit
def evaluate(params, state, xs, ys, theta):
  (losses, num_correct, num_total) = jax.lax.map(
      lambda data: evaluate_minibatch(params, state, data, theta),
      (xs, ys)
  )
  return jnp.mean(losses), jnp.sum(losses), jnp.sum(num_correct) / jnp.sum(num_total)


def full_eval(key, theta):
  key, model_key, unroll_key, eval_key = jax.random.split(key, 4)
  inner_state = init_state_fn(model_key)
  unroll_obj, inner_state_new = unroll(
      (unroll_key, eval_key), theta, inner_state, args.T, args.T
  )
  train_mean_loss, train_sum_loss, train_acc = evaluate(
    inner_state_new.params, inner_state_new.model_state, train_data, train_targets, theta
  )
  val_mean_loss, val_sum_loss, val_acc = evaluate(
    inner_state_new.params, inner_state_new.model_state, val_data, val_targets, theta
  )

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
  stats_dict = jax.vmap(full_eval, in_axes=(0,None))(eval_run_keys, theta)
  return stats_dict


theta_vals = []
idx_dict = {}
unflatten_func_dict = {}
idx = 0
for setting in args.tune_params:
  param = setting['param']
  sched = setting['sched']
  default = hparam_utils.uncons_funcs[param](default_values[param])

  if param == 'mask':
    theta_vals.append(jnp.array([default] * args.nlayers))
    idx_dict[param] = jnp.array(list(range(idx, idx+args.nlayers)))
    idx += args.nlayers
    continue

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

theta = jnp.concatenate(theta_vals)

param_fieldnames = []
for setting in args.tune_params:
  if setting['param'] == 'mask':
    param_fieldnames += ['mask_{}'.format(i) for i in range(args.nlayers)]
  elif setting['sched'] == 'fixed':
    param_fieldnames += ['{}_0'.format(setting['param'])]
  elif setting['sched'] == 'linear':
    param_fieldnames += ['{}_0'.format(setting['param']),
                         '{}_1'.format(setting['param'])]
  elif setting['sched'] == 'inverse-time-decay':
    param_fieldnames += ['{}_0'.format(setting['param']),
                         '{}_1'.format(setting['param'])]
  elif setting['sched'] == 'fixed-pl':
    for name in general_utils.recursive_keys(params):
      base_str = '{}/{}'.format(name, setting['param'])
      param_fieldnames += ['{}_0'.format(base_str)]
  elif setting['sched'] in ['linear-pl', 'inverse-time-decay-pl']:
    for name in general_utils.recursive_keys(params):
      base_str = '{}/{}'.format(name, setting['param'])
      param_fieldnames += ['{}_0'.format(base_str), '{}_1'.format(base_str)]
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


num_leaves = len(jax.tree_leaves(params))
key = jax.random.PRNGKey(args.seed)

hparam_names = [setting['param'] for setting in args.tune_params]

thetas = []

if args.search_type == 'grid':
  # Special case so that we can have handcrafted hparam ranges
  if tune_params_str == 'lr:inverse-time-decay':
    grid_dim = 2
    X = onp.linspace(-3, -0.5, args.num_points)
    Y = onp.linspace(-2.2, 2.2, args.num_points)
    X_for_base_e = onp.linspace(onp.log(10**(-3.0)),
                                onp.log(10**(-0.5)),
                                args.num_points)
    Y_for_base_e = onp.linspace(onp.log(10**(-2.2)),
                                onp.log(10**2.2),
                                args.num_points)
    thetas = jnp.array(list(itertools.product(X_for_base_e, Y_for_base_e)))
  elif len(args.tune_params) == 1:
    setting = args.tune_params[0]
    param = setting['param']
    sched = setting['sched']
    if sched == 'fixed':
      grid_dim = 1
      thetas = jnp.linspace(*hparam_utils.hparam_range_dict[param], args.num_points).reshape(-1,1)
    elif sched == 'linear':
      grid_dim = 2
      X = jnp.linspace(*hparam_utils.hparam_range_dict[param], args.num_points)
      Y = jnp.linspace(*hparam_utils.hparam_range_dict[param], args.num_points)
      thetas = jnp.array(list(itertools.product(X, Y)))
    else:
      raise Exception('Grid search not supported for that configuration!')
  elif len(args.tune_params) == 2:
    grid_dim = 2
    param1 = args.tune_params[0]['param']
    sched1 = args.tune_params[0]['sched']
    param2 = args.tune_params[1]['param']
    sched2 = args.tune_params[1]['sched']
    if not (sched1 == sched2 == 'fixed'):
      raise Exception('Both params must have fixed schedules for a 2D grid search!')

    X = jnp.linspace(*hparam_utils.hparam_range_dict[param1], args.num_points)
    Y = jnp.linspace(*hparam_utils.hparam_range_dict[param2], args.num_points)
    thetas = jnp.array(list(itertools.product(X, Y)))
elif args.search_type == 'random':
  thetas = []
  for setting, theta_val in zip(args.tune_params, theta_vals):
    param = setting['param']
    sched = setting['sched']
    min_range, max_range = hparam_utils.hparam_range_dict[param]

    theta = []
    if sched == 'inverse-time-decay':
      key, key1, key2 = jax.random.split(key, 3)
      # A fixed range for the decay factor regardless of which hyperparameter
      # we're dealing with
      decay_min_range, decay_max_range = -4, 4
      sampled_init = jax.random.uniform(
          key1, (args.num_points,len(theta_val)//2)
      ) * (max_range - min_range) + min_range
      sampled_decay = jax.random.uniform(
          key2, (args.num_points,len(theta_val)//2)
      ) * (decay_max_range - decay_min_range) + decay_min_range
      theta = jnp.concatenate([sampled_init, sampled_decay], axis=1)
      thetas.append(theta)
    elif sched == 'inverse-time-decay-pl':
      # A fixed range for the decay factor regardless of which hyperparameter
      # we're dealing with
      decay_min_range, decay_max_range = -4, 4
      for i in range(args.num_points):
        point_hparams = []
        for j in range(num_leaves):
          key, key1, key2 = jax.random.split(key, 3)
          leaf_init = jax.random.uniform(key1) * (max_range - min_range) + min_range
          leaf_decay = jax.random.uniform(key2) * (decay_max_range - decay_min_range) + decay_min_range
          point_hparams.append(jnp.array([leaf_init, leaf_decay]))
        point_hparams = jnp.concatenate(point_hparams)
        theta.append(point_hparams)
      theta = jnp.stack(theta)
      thetas.append(theta)
    else:
      key, skey = jax.random.split(key)
      min_range, max_range = hparam_utils.hparam_range_dict[param]
      theta = jax.random.uniform(
          skey, (args.num_points, len(theta_val))
      ) * (max_range - min_range) + min_range
      thetas.append(theta)

  thetas = jnp.concatenate(thetas, axis=1)

if args.chunk_size == -1:
  args.chunk_size = len(thetas)

num_chunks = int(math.ceil(len(thetas) / args.chunk_size))

results_dict = defaultdict(list)
for chunk in tqdm(range(num_chunks)):
  print('Chunk {}'.format(chunk))
  sys.stdout.flush()

  key, skey = jax.random.split(key)

  start = chunk * args.chunk_size
  end = start + args.chunk_size
  thetas_in_chunk = thetas[start:end]

  keys = jax.random.split(skey, len(thetas_in_chunk))
  results = jax.vmap(full_evaluation_runs, in_axes=(None, 0, None))(
      skey, thetas_in_chunk, args.num_eval_runs
  )

  for metric_key in results:
    results_dict[metric_key].append(
        onp.array(results[metric_key]).mean(axis=1)
    )

  if args.search_type == 'grid':
    if grid_dim == 1:
      with open(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
        pkl.dump({
            'thetas': onp.array(thetas[:end]),
            'thetas_cons': onp.array([to_constrained(theta_value.reshape(1))
                                      for theta_value in thetas[:end]]).reshape(-1),
            'hparam_names': hparam_names,
            'hparam_fieldnames': param_fieldnames,
            **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
        }, f)
    elif grid_dim == 2:
      xv, yv = onp.meshgrid(X, Y)
      with open(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
        pkl.dump({
            'thetas': onp.array(thetas[:end]),
            'thetas_cons': onp.array([to_constrained(theta_value)
                                      for theta_value in thetas[:end]]),
            'xv': xv,
            'yv': yv,
            'hparam_names': hparam_names,
            'hparam_fieldnames': param_fieldnames,
            **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
        }, f)

  elif args.search_type == 'random':
    with open(os.path.join(exp_dir, 'result.pkl'), 'wb') as f:
      pkl.dump({
          'thetas': onp.array(thetas[:end]),
          'thetas_cons': onp.array([to_constrained(theta_value)
                                    for theta_value in thetas[:end]]),
          'hparam_names': hparam_names,
          'hparam_fieldnames': param_fieldnames,
          **{metric_key: onp.concatenate(results_dict[metric_key]) for metric_key in results_dict},
      }, f)
