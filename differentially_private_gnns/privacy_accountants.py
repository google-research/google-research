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

"""Privacy accountants for different training schemes."""

import functools
from typing import Callable

import dp_accounting
import ml_collections
import numpy as np
import scipy.stats


def multiterm_dpsgd_privacy_accountant(num_training_steps,
                                       noise_multiplier,
                                       target_delta, num_samples,
                                       batch_size,
                                       max_terms_per_node):
  """Computes epsilon after a given number of training steps with DP-SGD/Adam.

  Accounts for the exact distribution of terms in a minibatch,
  assuming sampling of these without replacement.

  Returns np.inf if the noise multiplier is too small.

  Args:
    num_training_steps: Number of training steps.
    noise_multiplier: Noise multiplier that scales the sensitivity.
    target_delta: Privacy parameter delta to choose epsilon for.
    num_samples: Total number of samples in the dataset.
    batch_size: Size of every batch.
    max_terms_per_node: Maximum number of terms affected by the removal of a
      node.

  Returns:
    Privacy parameter epsilon.
  """
  if noise_multiplier < 1e-20:
    return np.inf

  # Compute distribution of terms.
  terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
  terms_logprobs = [
      terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
  ]

  # Compute unamplified RDPs (that is, with sampling probability = 1).
  orders = np.arange(1, 10, 0.1)[1:]

  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
  unamplified_rdps = accountant._rdp  # pylint: disable=protected-access

  # Compute amplified RDPs for each (order, unamplified RDP) pair.
  amplified_rdps = []
  for order, unamplified_rdp in zip(orders, unamplified_rdps):
    beta = unamplified_rdp * (order - 1)
    log_fs = beta * (
        np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node))
    amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
        order - 1)
    amplified_rdps.append(amplified_rdp)

  # Verify lower bound.
  amplified_rdps = np.asarray(amplified_rdps)
  if not np.all(unamplified_rdps *
                (batch_size / num_samples)**2 <= amplified_rdps + 1e-6):
    raise ValueError('The lower bound has been violated. Something is wrong.')

  # Account for multiple training steps.
  amplified_rdps_total = amplified_rdps * num_training_steps

  # Convert to epsilon-delta DP.
  return dp_accounting.rdp.compute_epsilon(orders, amplified_rdps_total,
                                           target_delta)[0]


def dpsgd_privacy_accountant(num_training_steps, noise_multiplier,
                             target_delta,
                             sampling_probability):
  """Computes epsilon after a given number of training steps with DP-SGD/Adam.

  Assumes there is only one affected term on removal of a node.
  Returns np.inf if the noise multiplier is too small.

  Args:
    num_training_steps: Number of training steps.
    noise_multiplier: Noise multiplier that scales the sensitivity.
    target_delta: Privacy parameter delta to choose epsilon for.
    sampling_probability: The probability of sampling a single sample every
      batch. For uniform sampling without replacement, this is (batch_size /
      num_samples).

  Returns:
    Privacy parameter epsilon.
  """
  if noise_multiplier < 1e-20:
    return np.inf

  orders = np.arange(1, 200, 0.1)[1:]
  event = dp_accounting.PoissonSampledDpEvent(
      sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier))
  accountant = dp_accounting.rdp.RdpAccountant(orders)
  accountant.compose(event, num_training_steps)
  return accountant.get_epsilon(target_delta)


def get_training_privacy_accountant(
    config,
    num_training_nodes,
    max_terms_per_node):
  """Returns an accountant that computes DP epsilon for a given number of training steps."""
  if not config.differentially_private_training:
    return lambda num_training_steps: 0

  if config.model == 'mlp':
    return functools.partial(
        dpsgd_privacy_accountant,
        noise_multiplier=config.training_noise_multiplier,
        target_delta=1 / (10 * num_training_nodes),
        sampling_probability=config.batch_size / num_training_nodes)
  if config.model == 'gcn':
    return functools.partial(
        multiterm_dpsgd_privacy_accountant,
        noise_multiplier=config.training_noise_multiplier,
        target_delta=1 / (10 * num_training_nodes),
        num_samples=num_training_nodes,
        batch_size=config.batch_size,
        max_terms_per_node=max_terms_per_node)

  raise ValueError(
      'Could not create privacy accountant for model: {config.model}.')
