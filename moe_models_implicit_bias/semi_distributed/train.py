# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Training code for experimenting with MoE models on synthetic data settings such as mixtures of Gaussians, mixtures of subspaces and dictionary learning.
"""

import functools
import time
from typing import Any
from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from moe_models_implicit_bias.semi_distributed import input_pipeline
from moe_models_implicit_bias.semi_distributed import models

NUM_CLASSES = 2


def create_model(*, model_cls, config, mu):
  return model_cls(config, mu)


def initialized(key, config, model):
  """Initialize a model using an PRNG key.

  Args:
    key: the PRNG key to use for initializing the model.
    config: the config describing the experiment.
    model: the model object to be initialized.

  Returns:

  """
  input_shape = (1, config.dim)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones(input_shape, jnp.float32))
  return variables['params'], variables['batch_stats']


def mse_loss(logits, labels):
  return jnp.mean((logits - labels)**2)


def compute_metrics(logits, hidden_activation, labels):
  """Compute loss and other routing related metrics from a batch.

  Args:
    logits: the logits produced by the model.
    hidden_activation: the hidden_activation corresponding to the router.
    labels: the ground truth labels.

  Returns:
    metrics: a dict containing the computed metrics.

  """
  loss = mse_loss(logits, labels)
  hidden_activation = jnp.abs(hidden_activation)
  hidden_activation /= jnp.sum(hidden_activation, axis=-1, keepdims=True)
  sparsity_entropy = jnp.mean(2.**jnp.sum(
      -hidden_activation * jnp.log2(hidden_activation + 1e-8), axis=-1))
  normalized_max_prob = jnp.mean(jnp.max(hidden_activation, axis=-1))
  collision_prob = jnp.mean(jnp.sum(hidden_activation**2, axis=-1))
  metrics = {
      'loss': loss,
      'sparsity_entropy': sparsity_entropy,
      'max_prob': normalized_max_prob,
      'collision_prob': collision_prob,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def train_step(state, batch, config, learning_rate):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    (logits, expert), new_model_state = state.apply_fn(
        {
            'params': params,
            'batch_stats': state.batch_stats
        },
        batch[0],
        config.expert_scale,
        mutable=['batch_stats'])

    loss = mse_loss(logits, batch[1])
    weight_penalty_params = jax.tree.leaves(params)
    weight_decay = config.weight_decay
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * (weight_l2)  # - weight_no_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits, expert)

  dynamic_scale = state.dynamic_scale
  lr = learning_rate

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
  new_model_state, logits, expert = aux[1]
  metrics = compute_metrics(logits, expert, batch[1])
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree.map(
            functools.partial(jnp.where, is_fin), new_state.opt_state,
            state.opt_state),
        params=jax.tree.map(
            functools.partial(jnp.where, is_fin), new_state.params,
            state.params),
        dynamic_scale=dynamic_scale)
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits, expert = state.apply_fn(
      variables, batch[0], train=False, mutable=False)
  return compute_metrics(logits, expert, batch[1])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree.map(_prepare, xs)


def create_input_iter(config, local_batch_size, train, w, mu, stats=None):
  ds, stats_new = input_pipeline.create_split(config, local_batch_size, train,
                                              w, mu, stats)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it, stats_new


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree.map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def flattened_traversal(fn):
  """Returns function that is called with `(path, param)` instead of pytree."""

  def mask(tree):
    flat = flax.traverse_util.flatten_dict(tree)
    return flax.traverse_util.unflatten_dict(
        {k: fn(k, v) for k, v in flat.items()})

  return mask


def create_train_state(rng, config: ml_collections.ConfigDict, model):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, batch_stats = initialized(rng, config, model)
  if config.opt == 'adam':

    def e_fn(k, v):
      if isinstance(v, dict) and k.startswith('expert'):
        return (1, True)
      elif isinstance(v, dict):
        return (2, True)
      else:
        return (0, False)

    def label_fn(nested_dict, e=0):
      if e == 0:
        return {
            k: (label_fn(v, e=e_fn(k, v)[0]) if e_fn(k, v)[1] else (e == 1))
            for k, v in nested_dict.items()
        }
      elif e == 1:
        return {
            k: (label_fn(v, e=1) if e_fn(k, v)[1] else True)
            for k, v in nested_dict.items()
        }
      elif e == 2:
        return {
            k: (label_fn(v, e=2) if e_fn(k, v)[1] else False)
            for k, v in nested_dict.items()
        }

    print(label_fn(params))

    tx = optax.multi_transform(
        {
            True: optax.adam(learning_rate=config.expert_learning_rate),
            False: optax.adam(learning_rate=config.learning_rate)
        }, label_fn)
  elif config.opt == 'sgd':
    tx = optax.sgd(learning_rate=config.learning_rate, momentum=0.9)

  state = TrainState.create(
      apply_fn=model.apply,
      params=params.unfreeze(),
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale)
  return state


def get_mean_sparsity(v):
  entr = jnp.sum(-v * jnp.log2(v + 1e-8), axis=-1)
  spar = jnp.mean(2.**entr)
  return spar


def get_mean_maxprob(v):
  maxprob = jnp.max(v, axis=-1)
  maxprob = jnp.mean(maxprob)
  return maxprob


def get_mean_collprob(v):
  collprob = jnp.sum(v**2, axis=-1)
  collprob = jnp.mean(collprob)
  return collprob


def log_stuff(e_metrics, v, nam):
  e_metrics['maxprob+' + nam] = get_mean_maxprob(v)
  e_metrics['collprob+' + nam] = get_mean_collprob(v)
  e_metrics['sparsity+' + nam] = get_mean_sparsity(v)
  return e_metrics


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  print('Using TensorFlow version %s' % tf.__version__)
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  np.random.seed(config.seed + 2)
  if config.inp_type == 'mog':
    mu = np.random.randn(config.num_clusters, config.dim) * config.margin / (
        config.dim**.5)
  elif config.inp_type == 'mos':
    mu = np.random.randn(config.num_clusters, config.rank,
                         config.dim) * config.margin / (
                             config.dim**.5)

  if config.out_type == 'con':
    w = np.random.randn(config.num_clusters, config.out_dim)

  if config.out_type == 'hyp':
    w = np.random.randn(config.num_clusters, config.dim)
  elif config.out_type == 'nn':
    w = []
    for ii in range(config.num_clusters):
      label_model = models.LabelModel(config)
      rng = random.PRNGKey(config.seed + 1000 * ii)
      dummy_x = random.normal(rng, (1, config.dim))
      params = label_model.init(rng, dummy_x)
      w.append(params)

  train_iter, stats = create_input_iter(config, local_batch_size, True, w, mu)
  eval_iter, _ = create_input_iter(config, local_batch_size, False, w, mu,
                                   stats)

  num_steps = config.num_train_steps

  steps_per_checkpoint = 10

  model_cls = getattr(models, config.model)
  model = create_model(model_cls=model_cls, config=config, mu=mu)

  state = create_train_state(rng, config, model)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(
          train_step, learning_rate=config.learning_rate, config=config),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  step = step_offset

  best_loss = 1000000.0

  for step, batch in zip(range(step_offset, num_steps), train_iter):
    step += 1
    print(step, flush=True)
    state, metrics = p_train_step(state, batch)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree.map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        if summary['train_loss'] < .5 and config.expert_scale < 1.9 and False:
          config.expert_scale *= 1
          p_train_step = jax.pmap(
              functools.partial(
                  train_step, learning_rate=config.learning_rate,
                  config=config),
              axis_name='batch')
        summary['expert_scale'] = config.expert_scale
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    print(config.inp_type, 'inp_type', flush=True)

    if (step + 1) % config.eval_every_steps == 0 or step < 2:

      state = sync_batch_stats(state)

      print(config.inp_type, 'inp_type', flush=True)
      if config.inp_type == 'mog':

        print('is this happening', flush=True)

        e_w = state.params['expert_1']['kernel'][0]

        if config.router_only_scale:
          e_w *= (1. + state.params['scale_param'][0, 0] * (config.dim**.5))

        if config.depth == 0:
          first_weights = state.params['layers_1']
        else:
          first_weights = state.params['layers_1']['kernel'][0]

        print(e_w.shape, flush=True)

        e_metrics = {}
        e_metrics['router_norm^2'] = jnp.mean(e_w**2) * config.dim

        e_metrics['expert_norm^2'] = jnp.mean(first_weights**2) * config.dim

        dot_prod_mu = mu @ e_w
        softmax_mu = jax.nn.softmax(config.expert_scale * dot_prod_mu)

        e_metrics = log_stuff(e_metrics, softmax_mu, 'centers')

        e_w_shuffle = (np.array(e_w).copy())
        np.random.shuffle(e_w_shuffle)

        dot_prod_mu_shuffle = mu @ e_w_shuffle
        softmax_mu_shuffle = jax.nn.softmax(config.expert_scale *
                                            dot_prod_mu_shuffle)

        e_metrics = log_stuff(e_metrics, softmax_mu_shuffle, 'centers+shuffle')

        samples = mu[:, None, :] + np.random.randn(
            *[mu.shape[0], 100, mu.shape[1]])

        print(e_w.shape, flush=True)
        print(samples.shape, flush=True)
        print(
            jnp.sum((state.params['expert_1']['kernel'][0] -
                     state.params['expert_1']['kernel'][1])**2),
            flush=True)

        dot_prod_samples = samples @ e_w
        softmax_samples = jax.nn.softmax(config.expert_scale * dot_prod_samples)

        e_metrics = log_stuff(e_metrics, softmax_samples, 'samples')

        dot_prod_samples_shuffle = samples @ e_w_shuffle
        softmax_samples_shuffle = jax.nn.softmax(config.expert_scale *
                                                 dot_prod_samples_shuffle)

        e_metrics = log_stuff(e_metrics, softmax_samples_shuffle,
                              'samples+shuffle')

        avg_softmax_samples = jnp.mean(softmax_samples, axis=1)

        e_metrics = log_stuff(e_metrics, avg_softmax_samples, 'clusters')

        avg_softmax_samples_shuffle = jnp.mean(softmax_samples_shuffle, axis=1)

        e_metrics = log_stuff(e_metrics, avg_softmax_samples_shuffle,
                              'clusters+shuffle')

        print(e_metrics, 'e_metrics', flush=True)

        writer.write_scalars(step + 1, e_metrics)
        writer.flush()

      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      print(config.steps_per_eval, flush=True)
      for _ in range(config.steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree.map(lambda x: x.mean(), eval_metrics)
      writer.write_scalars(step + 1,
                           {f'eval_{key}': val for key, val in summary.items()})
      if best_loss > summary['loss']:
        best_loss = summary['loss']
      writer.write_scalars(step + 1, {'eval_best_loss': best_loss})
      writer.flush()

    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state
