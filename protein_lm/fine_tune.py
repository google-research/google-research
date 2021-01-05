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

# Lint as: python3
"""Utilities for fine-tuning FlaxLM models."""

import functools

from flax import jax_utils as flax_jax_utils
from flax.training import common_utils
import jax
import tensorflow.compat.v1 as tf
import tqdm
from protein_lm import models
from protein_lm import utils

_SHUFFLE_BUFFER_SIZE = 5000


def _get_dataset(sequences, example_weights, batch_size, shuffle):
  data_dict = dict(sequence=sequences)
  if example_weights is not None:
    data_dict['example_weight'] = example_weights
  dataset = tf.data.Dataset.from_tensor_slices(data_dict)
  if shuffle:
    dataset = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
  dataset = dataset.repeat().batch(batch_size)
  return dataset


class _OptimizationRunner(object):
  """Helper class for running optimization steps."""

  def __init__(self, model, learning_rate, **optimizer_kwargs):
    self._bos_token = model.bos_token
    self._pad_token = model.pad_token
    unreplicated_optimizer = model.get_weights()
    self._replicated_optimizer = utils.create_adam_optimizer(
        model=unreplicated_optimizer.target,
        learning_rate=learning_rate,
        **optimizer_kwargs)
    self._dropout_rngs = model._dropout_rngs

    self._p_train_step = jax.pmap(
        functools.partial(
            models.train_step,
            preprocess_fn=model.preprocess,
            learning_rate_fn=lambda t: learning_rate),
        axis_name='batch')

  def fit_batch(self, batch, example_weights=None):
    """Runs one optimization step on batch."""
    batch = common_utils.shard(batch)

    if example_weights is not None:
      example_weights = common_utils.shard(example_weights)
    (self._replicated_optimizer, metrics,
     self._dropout_rngs) = self._p_train_step(
         self._replicated_optimizer,
         inputs=batch,
         example_weights=example_weights,
         dropout_rng=self._dropout_rngs)

    return metrics

  def get_weights(self):
    return flax_jax_utils.unreplicate(self._replicated_optimizer)


def fine_tune(model,
              initial_weights,
              sequences,
              batch_size,
              num_epochs,
              learning_rate,
              example_weights=None,
              shuffle=True,
              progress_bar=True,
              **optimizer_kwargs):
  """Fine tunes model on sequences.

  Args:
    model: A models.FlaxLM.
    initial_weights: The model is initialized with these weights.
    sequences: A list of int-encoded sequences to train on.
    batch_size: The batch size used when optimizing the model.
    num_epochs: Number of passes to take through the input sequences.
    learning_rate: Learning rate for optimization.
    example_weights: Optional per-sequence weights for performing weighted MLE
      training.
    shuffle: Whether the input sequences should be shuffled.
    progress_bar: Whether to display a progress bar.
    **optimizer_kwargs: Additional kwargs to pass to
      utils.create_adam_optimizer().

  Returns:
    A set of fine tuned weights. The model can be set to use these using
      model.set_weights(fine_tuned_weights).
  """
  model.set_weights(initial_weights)

  runner = _OptimizationRunner(
      model, learning_rate=learning_rate, **optimizer_kwargs)

  dataset = _get_dataset(sequences, example_weights, batch_size, shuffle)
  dataset_iter = iter(dataset.repeat())

  num_iter = int(num_epochs * len(sequences) / batch_size)
  iterations = list(range(num_iter))
  if progress_bar:
    iterations = tqdm.tqdm(iterations, position=0)
  for _ in iterations:
    batch = next(dataset_iter)
    batch_example_weights = batch['example_weight'].numpy(
    ) if example_weights is not None else None
    batch_sequences = batch['sequence'].numpy()
    runner.fit_batch(batch_sequences, batch_example_weights)

  return runner.get_weights()
