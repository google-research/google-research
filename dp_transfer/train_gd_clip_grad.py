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

"""Main train loop for DP-Adam. File intended to be mostly self-contained."""

import functools

from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib

from dp_transfer import data_utils
from dp_transfer import dataset
from dp_transfer import utils



def unreplicate_and_get(x):
  return jax.device_get(jax_utils.unreplicate(x))


def noisy(step, x, s, key):
  if 0 < s < np.inf:
    new_key = jax.random.fold_in(key, step)
    noise = jax.random.normal(new_key, shape=jnp.shape(x)) * s
    return x + noise
  return x


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def log_likelihood(weights, data, labels, bias):
  """Normalized negative log likelihood."""
  logits = jnp.einsum('d,ld->l', data, weights) + bias
  log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)

  loss = -((labels * log_p) + (1. - labels) * log_not_p)
  return jnp.mean(loss)


def log_likelihood_gradient(weights, data, labels, bias):
  """Gradient of negative log likelihood."""
  return jax.grad(lambda w: log_likelihood(w, data, labels, bias))(weights)


def clip(x, clip_norm=1.0):
  divisor = jnp.maximum(jnp.linalg.norm(x) / clip_norm, 1.)
  return x / divisor


def clipped_log_likelihood_gradient(weights, data, labels, bias, clip_norm):
  """Gradient of negative log likelihood."""
  gradi = log_likelihood_gradient(weights, data, labels, bias)
  return clip(gradi, clip_norm)


def accumulate_grad(w, label_onehot, data, grad_accum, bias, clip_norm):
  update_fn = jax.vmap(lambda data, labels: clipped_log_likelihood_gradient(
      w, data, labels, bias, clip_norm))
  grad_all = update_fn(data, label_onehot)
  gradi = grad_all.sum(0)
  return grad_accum + gradi


def update_from_accum_grad(
    step, w, final_grad, batch_size, lr, apply_noise_fn=None, reg=0.0
):
  """Make an adam update from accumulated gradient."""
  final_grad = jax.lax.psum(final_grad, axis_name='batch')
  if apply_noise_fn is not None:
    final_grad = apply_noise_fn(final_grad, step)
  update = final_grad / batch_size
  b1 = 0.9
  b2 = 0.999
  update = ((1. - b1) /
            (1. - b2)) * update / ((jnp.sqrt(jax.lax.square(update))) + 1e-8)
  update += reg * w
  w -= lr * update
  return w, jnp.zeros_like(w)


def train_and_evaluate(config, workdir):
  """Top level training and eval loop."""

  tf.io.gfile.makedirs(workdir)
  start_step = 0

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0)
  if start_step == 0:
    writer.write_hparams(dict(config))

  num_epochs = config.num_epochs
  num_train_examples = 50000 if 'cifar' in config.dataset else 1281167
  local_batch_size = 1024
  num_acc_steps = num_train_examples // local_batch_size
  batch_size = local_batch_size * num_acc_steps
  num_steps_per_epoch = (num_train_examples // local_batch_size) + 1
  num_steps = num_steps_per_epoch * num_epochs
  print(f'num_steps: {num_steps}')
  print(f'num_steps_per_epoch: {num_steps_per_epoch}')
  print(f'lr: {config.lr}')
  print(f'num_acc_steps: {num_acc_steps}')
  print(f'batch_size: {batch_size}')

  data_config = data_utils.get_data_config(config)
  train_ds, test_ds = dataset.get_datasets(
      config=config,
      data_config=data_config,
      batch_size=local_batch_size,
      repeat=True
  )

  test_xs = []
  test_labels = []
  for x in test_ds:
    test_xs.append(x['repr'])
    test_labels.append(x['label'])
  test_x_np_list = utils.to_flat_np(
      test_xs, test_labels, data_config.num_labels
  )
  eval_step = jax.jit(
      functools.partial(
          utils.eval_step,
          test_x_np_list=test_x_np_list,
          hidden_dims=data_config.hidden_dims,
          num_labels=data_config.num_labels,
      ))

  # We only consider full batch setting.
  sigma = compute_noise_from_budget_lib.compute_noise(num_train_examples,
                                                      num_train_examples,
                                                      config.epsilon,
                                                      num_epochs,
                                                      data_config.delta, 1e-7)
  sigma *= data_config.clip
  key = jax.random.PRNGKey(config.seed)
  apply_noise_fn = None
  if config.is_private and config.epsilon > 0.0:
    apply_noise_fn = jax.vmap(functools.partial(noisy, s=sigma, key=key))
  update_from_accum_grad_partial = functools.partial(
      update_from_accum_grad,
      batch_size=batch_size,
      apply_noise_fn=apply_noise_fn,
      lr=config.lr,
      reg=config.reg)
  update_from_accum_grad_partial_pmapped = jax.pmap(
      update_from_accum_grad_partial, axis_name='batch')
  accumulate_grad_pmapped = jax.pmap(
      functools.partial(
          accumulate_grad, bias=-10.0, clip_norm=data_config.clip
      ),
      axis_name='batch',
  )

  grad_accum = np.zeros(
      (data_config.num_labels, data_config.hidden_dims), np.float32
  )
  grad_accum = jax.device_put_replicated(grad_accum, devices=jax.devices())
  wopt = np.zeros((data_config.num_labels, data_config.hidden_dims), np.float32)
  wopt = jax.device_put_replicated(wopt, devices=jax.devices())

  train_iter = train_ds.as_numpy_iterator()
  for i in range(1, num_steps + 1):
    x = next(train_iter)
    data = x['repr']
    data = np.reshape(data,
                      (jax.device_count(), data.shape[0] // jax.device_count(),
                       data_config.hidden_dims))
    label_onehot = np.array(one_hot(x['label'], data_config.num_labels))
    label_onehot = np.reshape(label_onehot,
                              (jax.device_count(), label_onehot.shape[0] //
                               jax.device_count(), data_config.num_labels))
    grad_accum = accumulate_grad_pmapped(wopt, label_onehot, data, grad_accum)

    if i and i % num_acc_steps == 0:
      step = np.array([i] * jax.device_count())
      wopt, grad_accum = update_from_accum_grad_partial_pmapped(
          step, wopt, grad_accum)
      wopt_for_eval = unreplicate_and_get(wopt)
      eval_acc = eval_step(wopt_for_eval)
      print(f'eval acc at step: {i}, {eval_acc}')
      summary = {}
      summary['accuracy'] = eval_acc
      with metric_writers.ensure_flushes(writer):
        writer.write_scalars(i, summary)

