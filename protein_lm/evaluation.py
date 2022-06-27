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

"""Baselines and evaluation metrics for Jax language models."""
import itertools
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np

from protein_lm import utils


class EmpiricalBaseline():
  """Empirical baseline as described in the ProGen paper.

  References:
    [ProGen](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v1)
  """

  def __init__(self, domain, train_ds, alpha=1.):
    """Creates an instance of this class.

    # TODO(gandreea): It's unclear how to handle the length (EOS token). The
    #   fact that the uniform baseline is reported as (perplexity=25,
    #   accuracy=0.04) suggests that the EOS prediction step is not included.

    Args:
      domain: An instance of domains.Domain.
      train_ds: A tf.data.Dataset containing the data to be used for computing
        the empirical distribution.
      alpha: A float indicating the Laplace smoothing constant.
    """
    self._vocab_size = domain.vocab_size
    self._token_indices = [
        idx for idx in range(len(domain.vocab.tokens))
        if idx != domain.vocab.bos and idx != domain.vocab.eos]
    self._mask_token = domain.vocab.bos

    self._empirical_dist = np.zeros((len(self._token_indices),))
    for batch in train_ds:
      batch = np.atleast_2d(batch)
      batch_one_hot = np.eye(self._vocab_size)[batch]
      batch_one_hot = np.take(batch_one_hot, self._token_indices, axis=-1)
      self._empirical_dist += np.sum(np.sum(batch_one_hot, axis=0), axis=0)

    self._empirical_dist += alpha  # Laplace smoothing.
    self._empirical_dist /= np.sum(self._empirical_dist)

  def evaluate_batch(self, batch):
    """Computes all metrics on the given batch."""
    labels = np.atleast_2d(batch)
    logits = np.log(self._empirical_dist)
    logits = np.tile(logits, list(labels.shape) + [1])
    weights = np.where(labels != self._mask_token, 1, 0)
    metrics = utils.compute_metrics(logits, labels, weights)
    for key, value in metrics.items():
      metrics[key] = jnp.atleast_1d(value)
    return metrics


def combine_metrics(step_metrics):
  """Given a list of metric dicts, combine to a single summary metrics dict.

  Args:
    step_metrics: A dict with (metric name, metric value) items. Contains summed
      metrics and the corresponding denominator (the number of next-token
      prediction instances). Each metric value have at least one dimension.

  Returns:
    A dict with (metric name, metric value) items containing combined metrics.
  """
  metrics_all = common_utils.get_metrics(step_metrics)
  lr = None
  if 'learning_rate' in metrics_all:
    lr = metrics_all.pop('learning_rate').mean()
  metrics_sums = jax.tree_map(jnp.sum, metrics_all)
  denominator = metrics_sums.pop('denominator')
  summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
  if lr is not None:
    summary['learning_rate'] = lr

  # Calculate (clipped) perplexity after averaging log-perplexities:
  if 'loss' in summary:
    summary['perplexity'] = jnp.clip(jnp.exp(summary['loss']), a_max=1.0e4)
  return summary


def evaluate(model, eval_ds, num_eval_steps=None):
  """Evaluates model on eval_ds for num_eval_steps.

  Args:
    model: A model to use for evaluation. Must have an evaluate_batch() method.
    eval_ds: A tensorflow dataset containing the data to be used for evaluation.
    num_eval_steps: If given, evaluate for this many steps, otherwise use the
      entire dataset.

  Returns:
    A dictionary with (metric name, metric value) items.
  """
  eval_metrics = []
  eval_iter = iter(eval_ds)
  if num_eval_steps is None:
    num_iter = itertools.repeat(1)
  else:
    num_iter = range(num_eval_steps)
  for _, eval_batch in zip(num_iter, eval_iter):
    eval_batch = np.asarray(eval_batch)
    metrics = model.evaluate_batch(eval_batch)
    eval_metrics.append(metrics)
  eval_summary = combine_metrics(eval_metrics)
  return eval_summary
