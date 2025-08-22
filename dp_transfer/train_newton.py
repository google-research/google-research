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

"""Main train loop for DP-Newton. File intended to be mostly self-contained."""

import functools

from clu import metric_writers
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import tensorflow as tf

from dp_transfer import data_utils
from dp_transfer import dataset
from dp_transfer import newton_sanitizer
from dp_transfer import utils



def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def log_likelihood(weights, data, labels, alpha=1.0):
  """Normalized negative log likelihood."""
  logits = jnp.einsum('d,bd->b', weights, data)
  log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)

  loss = -((labels * log_p) + alpha * ((1. - labels) * log_not_p))
  return jnp.sum(loss)


def log_likelihood_gradient(weights, data, labels, alpha=1.0):
  """Gradient of negative log likelihood."""
  return jax.grad(lambda w: log_likelihood(w, data, labels, alpha))(weights)


def log_likelihood_hessian(weights, data, labels, alpha=1.0):
  """Hessian of negative log likelihood."""
  return jax.hessian(lambda w: log_likelihood(w, data, labels, alpha))(weights)


@jax.vmap
def cg_solve(a, b):
  x, _ = jax.scipy.sparse.linalg.cg(a, b)
  return x


def grad_hess(w, data, labels, alpha=1.0):
  """Newton iteration with least squares solver."""
  # Use least squares solver for Hessian-gradient product
  gradient = log_likelihood_gradient(w, data, labels, alpha)
  hess = log_likelihood_hessian(w, data, labels, alpha)
  return gradient, hess


def newton_update_from_grad_hess(final_grad,
                                 final_hess,
                                 num_acc_steps,
                                 reg,
                                 batch_size,
                                 hidden_dims,
                                 least_squares=cg_solve,
                                 apply_noise_fn=None):
  """Make an update from accumulated gradient and hessian."""
  if apply_noise_fn is not None:
    def add_noise(lhs, rhs):
      return apply_noise_fn([lhs, rhs])
    final_hess, final_grad = jax.vmap(add_noise)(final_hess, final_grad)
  final_hess += reg * jnp.identity(hidden_dims)[None, :, :]
  return (
      least_squares(final_hess, final_grad),
      jnp.zeros_like(final_hess),
      jnp.zeros_like(final_grad),
  )


def split_newton_grad_hess(
    w, label_onehot, data, grad_accum, hess_accum, alpha=1.0
):
  update_fn = jax.vmap(
      lambda wopt, labels: grad_hess(wopt, data, labels, alpha)
  )
  gradi, hessi = update_fn(w, label_onehot)
  return grad_accum + gradi, hess_accum + hessi


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
      )
  )

  if data_config.num_labels == 1000:
    num_devices = 8
  elif data_config.num_labels == 100:
    num_devices = 4
  elif data_config.num_labels == 10:
    num_devices = 2

  sanitizer = newton_sanitizer.Sanitizer(
      steps=num_epochs,
      max_norm=data_config.clip,
      num_classes=data_config.num_labels,
      random_seed=config.seed)
  apply_noise_fn = None
  if config.is_private and config.epsilon > 0.0:
    sanitizer.set_sigmas(
        target_epsilon=config.epsilon,
        target_delta=data_config.delta,
        sigma_ratio1=1.0)
    apply_noise_fn = sanitizer.apply_noise
  newton_update_from_grad_hess_partial = functools.partial(
      newton_update_from_grad_hess,
      batch_size=batch_size,
      apply_noise_fn=apply_noise_fn,
      hidden_dims=data_config.hidden_dims)
  newton_update_from_grad_hess_pmapped = jax.pmap(
      newton_update_from_grad_hess_partial,
      devices=jax.local_devices()[:num_devices])
  split_newton_grad_hess_partial = functools.partial(
      split_newton_grad_hess, alpha=config.alpha)
  split_newton_grad_hess_pmapped = jax.pmap(
      split_newton_grad_hess_partial, devices=jax.local_devices()[:num_devices])

  grad_accum = np.zeros(
      (num_devices, data_config.num_labels // num_devices,
       data_config.hidden_dims), np.float32)
  hess_accum = np.zeros(
      (num_devices, data_config.num_labels // num_devices,
       data_config.hidden_dims, data_config.hidden_dims), np.float32)
  wopt = np.zeros((num_devices, data_config.num_labels // num_devices,
                   data_config.hidden_dims), np.float32)
  train_iter = train_ds.as_numpy_iterator()
  for i in range(1, num_steps + 1):
    x = next(train_iter)
    data = x['repr']

    if config.is_private and config.epsilon > 0.0:
      # Clip features instead of grads or hessian.
      data = sanitizer.clip(data)
    label_onehot = np.array(one_hot(x['label'], data_config.num_labels))
    label_onehot = np.reshape(
        label_onehot.T,
        (
            num_devices,
            data_config.num_labels // num_devices,
            label_onehot.shape[0],
        ),
    )
    data = jax.device_put_replicated(
        data, devices=jax.local_devices()[:num_devices])
    grad_accum, hess_accum = split_newton_grad_hess_pmapped(
        wopt, label_onehot, data, grad_accum, hess_accum)

    if i and i % num_acc_steps == 0:
      reg_replicated = jax.device_put_replicated(
          config.reg, devices=jax.local_devices()[:num_devices])
      num_acc_steps_replicated = jax.device_put_replicated(
          num_acc_steps, devices=jax.local_devices()[:num_devices])

      # Refresh key before sampling the noise.
      sanitizer.refresh_key()

      update, hess_accum, grad_accum = newton_update_from_grad_hess_pmapped(
          grad_accum, hess_accum, num_acc_steps_replicated, reg_replicated)
      wopt = wopt - config.lr * update

      eval_acc = eval_step(wopt)
      print(f'eval acc at step: {i}, {eval_acc}')
      summary = {}
      summary['accuracy'] = eval_acc
      with metric_writers.ensure_flushes(writer):
        writer.write_scalars(i, summary)

