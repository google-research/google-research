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

"""Evaluation script for lossless compression.

This file contains the necessary methods to do lossless compression with
ARDM models.
"""
import functools
import os
import pickle
import time

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from autoregressive_diffusion.experiments.images import checkpoint
from autoregressive_diffusion.experiments.images import custom_train_state
from autoregressive_diffusion.experiments.images import datasets
from autoregressive_diffusion.experiments.images import train
from autoregressive_diffusion.experiments.images.eval_scripts import ans_template
from autoregressive_diffusion.utils import util_fns


def compress_dataset(state, model, test_ds, sigma=None, policy=None):
  """Compress a dataset.

  Args:
    state: A train state containing the params.
    model: The model class that contains all necessary methods.
    test_ds: The dataset to compress.
    sigma: An optional order of the generative process.
    policy: A policy describing which steps to take in parallel. An arange
      would do each step individually, but would be very slow.

  Returns:
    The final bits per dimension to compress the dataset, where each example
    is encoded with its own bitstream.
  """
  assert (sigma is None) == (policy is None)

  if sigma is None and policy is None:
    logging.info('Compressing with random order since none was given.')
    policy = model.get_naive_policy(25)
    sigma = model.get_random_order(jax.random.PRNGKey(0))

  total_size = 0
  total_count = 0

  d = int(np.prod(model.config.data_shape))

  test_ds = util_fns.get_iterator(test_ds)
  for idx, batch in enumerate(test_ds):
    x = batch['image']
    x = x.reshape(-1, *x.shape[2:])  # Flatten, we are not using pmap.
    batch_size = x.shape[0]

    # Scale bits determine the rounding precision, 32 seems to work well.
    # The init_bits puts a little buffer in the bitstream. But this is not
    # necessary for our model class.
    streams = [
        ans_template.Bitstream(scale_bits=32)
        for _ in range(x.shape[0])
    ]

    if idx == 0:
      logging.info('Initial total bits in bitstream: %d', len(streams[0]))

    logging.info('Encoding...')
    start = time.time()
    streams = model.encode_with_policy_and_sigma(streams, state.ema_params, x,
                                                 policy, sigma)

    logging.info('Encoding took %.2f seconds', time.time() - start)

    # Adds the number of bits in the streams to the total size.
    for stream in streams:
      total_size += len(stream)
    total_count += batch_size

    bits_per_dim = total_size / total_count / d

    logging.info('Encoded %d Current bits per dim %f',
                 total_count, bits_per_dim)

    logging.info('Decoding...')
    start = time.time()
    decoded, streams = model.decode_with_policy_and_sigma(
        streams, state.ema_params, policy, sigma, batch_size)
    del streams
    logging.info('Decoding took %.2f seconds', time.time() - start)

    coding_error = jnp.abs(x - decoded).sum()
    assert coding_error == 0, f'Coding error non-zero: {coding_error}'

  bits_per_dim = total_size / total_count / d
  logging.info('Bits per dim %f', bits_per_dim)

  return bits_per_dim


# The axes that are broadcasted are the in- and output rng key ones, and the
# model, and the policy. The rng is the first arg, and the last return value.
@functools.partial(
    jax.pmap,
    static_broadcasted_argnums=(3,),
    in_axes=(None, 0, 0, None, None, None),
    out_axes=(0, None),
    axis_name='batch')
def eval_step_policy_and_sigma(rng, batch, state, model, policy, sigma):
  """Eval a single step."""
  rng_return, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
  elbo_value = model.log_prob_with_policy_and_sigma(
      rng, state.ema_params, batch['image'], policy=policy, sigmas=sigma,
      train=False)
  metrics = {
      'nelbo': jax.lax.pmean(-elbo_value, axis_name='batch'),
  }
  return metrics, rng_return


def eval_policy_and_sigma(policy, sigma, rng, state, model, dataset):
  """Eval for a single epoch with policy and sigma."""
  batch_metrics = []

  # Function is recompiled for this specific policy.
  dataset = util_fns.get_iterator(dataset)
  for batch in dataset:
    metrics, rng = eval_step_policy_and_sigma(rng, batch, state, model, policy,
                                              sigma)

    # Better to leave metrics on device, and off-load after finishing epoch.
    batch_metrics.append(metrics)

  # Load to CPU.
  batch_metrics = jax.device_get(flax.jax_utils.unreplicate(batch_metrics))
  # Compute mean of metrics across each batch in epoch.
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}
  stdev_metrics_np = {
      k: np.std([metrics[k] for metrics in batch_metrics])
      for k in batch_metrics[0] if 'batch' not in k}

  nelbo = epoch_metrics_np['nelbo']
  stdev = stdev_metrics_np['nelbo']
  num_samples = len(batch_metrics)
  info_string = (f'eval policy with sigma scores nelbo: {nelbo:.4f} +/- '
                 f'{stdev:.4f} on {num_samples} samples. So expected stdev '
                 f'is stdev/sqrt(n) is {stdev / np.sqrt(num_samples):.4f}')
  logging.info(info_string)

  return nelbo, stdev, num_samples


def search_sigma_given_policy(policy, rng, state, model, train_ds,
                              num_tries=5):
  """Searches for an optimal ordering given a policy.

  Args:
    policy: An evaluation policy the model should use.
    rng: Random number key, used to sample permutations.
    state: State containing model parameters.
    model: Model class wrapping all functions related to the model.
    train_ds: Dataset to tune the optimal sigma to.
    num_tries: The number of random permutations that will be tested.

  Returns:
    The best performing sigma, together with its score.
  """
  best_sigma = None
  best_score = np.inf

  for i in range(num_tries):
    rng_i = jax.random.fold_in(rng, i)
    rng_i, rng_perm = jax.random.split(rng_i)

    # Retrieves a random generation order from a model.
    sigma = model.get_random_order(rng_perm)
    logging.info('Evaluating permutation %s', str(sigma))

    nelbo, _, _ = eval_policy_and_sigma(policy, sigma, rng_i, state, model,
                                        train_ds)

    if nelbo < best_score:
      logging.info('Updated best sigma with nelbo: %.4f', nelbo)
      best_sigma = sigma
      best_score = nelbo

  logging.info('Found sigma %s with nelbo %.4f', str(best_sigma), best_score)
  return best_sigma, best_score


def evaluate_compression(work_dir, budget=50):
  """Execute model training and evaluation loop.

  Args:
    work_dir: Directory where the saved files are located.
    budget: Budget for the policy.
  """
  # Loading config file.
  config_path = os.path.join(work_dir, 'config')
  with tf.io.gfile.GFile(config_path, 'rb') as fp:
    config = pickle.load(fp)
    logging.info('Loaded config')

  # Loading loss components
  with tf.io.gfile.GFile(os.path.join(work_dir, 'loss_components'), 'rb') as fp:
    loss_components = pickle.load(fp)
    logging.info('Loaded loss components')

  config.test_batch_size = 80
  config.num_eval_passes = 1

  rng = jax.random.PRNGKey(config.seed)
  data_rng, rng = jax.random.split(rng)

  train_ds, test_ds, shape, num_classes = datasets.get_dataset(config, data_rng)

  config.data_shape = shape
  config.num_classes = num_classes

  rng, init_rng = jax.random.split(rng)

  model, variables = train.model_setup(init_rng, config)

  # From now on we want different rng across hosts:
  rng = jax.random.fold_in(rng, jax.process_index())

  tx = optax.adam(
      config.learning_rate, b1=0.9, b2=config.beta2, eps=1e-08, eps_root=0.0)
  state = custom_train_state.TrainState.create(
      params=variables['params'], tx=tx)

  state, start_epoch = checkpoint.restore_from_path(work_dir, state)

  logging.info('Loaded checkpoint at epoch %d', start_epoch)

  test_rng, train_rng = jax.random.split(rng)
  del test_rng

  # Replicate state.
  state = flax.jax_utils.replicate(state)

  # Find optimal policy.
  policies, costs = model.compute_policies_and_costs(loss_components,
                                                     budgets=[budget])
  policy, expected_cost = policies[0], costs[0]
  logging.info('Using policy %s\n with expected cost %.2f',
               str(policy), expected_cost)

  # Find optimal sigma given policy using train data.
  sigma, _ = search_sigma_given_policy(policy, train_rng, state, model,
                                       train_ds)

  # Compress the test data.
  compress_dataset(state, model, test_ds, sigma=sigma, policy=policy)
