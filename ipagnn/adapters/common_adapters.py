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

"""Adapters bridge the gap between datasets and models.

They provide the implementation details of workflows so that workflows can be
dataset and model agnostic.
"""

import functools

from absl import logging  # pylint: disable=unused-import
import flax
from flax import jax_utils
from flax.training import common_utils
import jax
import jax.numpy as jnp

from ipagnn.lib import optimizer_utils
from ipagnn.lib import summary_utils


def compute_weighted_cross_entropy(logits, targets, weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch x length]
  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  num_classes = logits.shape[-1]
  onehot_targets = common_utils.onehot(targets, num_classes)
  loss = -jnp.sum(onehot_targets * flax.nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = jnp.prod(jnp.array(logits.shape[:-1]))
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


class BaseAdapter:
  """The base Adapter class for connecting datasets and models."""

  def __init__(self, info, config):
    self.info = info
    self.config = config

  def preprocess(self, dataset_iter, single_device=False):
    """Preprocesses the raw dataset iterator for use e.g. in a training loop."""
    dataset_iter = map(self.as_example, dataset_iter)
    if not single_device:
      dataset_iter = self.shard(dataset_iter)
    return dataset_iter

  def as_example(self, dataset_item):
    """Converts a single item from the raw dataset iterator into an example."""
    return dataset_item

  def shard(self, dataset_iter):
    """Applies sharding to an iterator of preprocessed examples."""
    return dataset_iter

  def create_optimizer(self, run_configuration, rng=None):
    """Creates the optimizer for training."""
    @functools.partial(jax.jit, static_argnums=(1, 2))
    def create_model(rng, example, model_cls):
      with flax.nn.attention.Cache().mutate() as cache_def:
        _, initial_params = model_cls.init(
            rng,
            example,
            cache=cache_def)
      model = flax.nn.Model(model_cls, initial_params)
      return model, cache_def

    config = self.config
    dataset = run_configuration.dataset_info.dataset

    rng = rng if rng is not None else jax.random.PRNGKey(0)
    learning_rate = config.opt.learning_rate
    example = self.as_example(next(iter(dataset)))
    model_cls = run_configuration.model
    model, unused_cache_def = create_model(rng, example, model_cls)
    return optimizer_utils.create_optimizer(model, learning_rate)

  def get_train_inputs(self, example):
    """Converts a sharded example to a sharded train example.

    This should strip away any data that shouldn't be passed to a jitted
    function. E.g. string metadata should be excluded.

    Args:
      example: A single element from `shard`.
    Returns:
      A train example suitable for passing to the train step.
    """
    return example

  def write_summaries(self, example, logits, summary_writer, info, step, state):
    pass

  def make_train_step(self, single_device=False):
    """Creates a train step function.

    Note that the returned train_step function is pmapped unless single_device
    is True.

    Args:
      single_device: If True, returns a train_step suitable for running on a
        single device. This is useful for debugging with ndb.
    Returns:
      A train_step function.
    """

    config = self.config
    learning_rate_fn = optimizer_utils.create_learning_rate_scheduler(
        base_learning_rate=config.opt.learning_rate,
        factors=config.opt.learning_rate_factors,
    )

    def train_step(optimizer, example, dropout_rng):
      """Perform a single training step."""
      if self.info.supervised_keys[-1] == 'error_type':
        targets = example['error_type'][:, None]
      else:
        targets = example['target_output']

      # We handle PRNG splitting inside the top pmap, rather
      # than handling it outside in the training loop - doing the
      # latter can add some stalls to the devices.
      dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

      def loss_fn(model):
        """Loss function used for training."""
        with flax.nn.stateful() as state:
          with flax.nn.stochastic(dropout_rng):
            logits = model(example, train=True)
        loss, weight_sum = compute_weighted_cross_entropy(logits, targets)
        mean_loss = loss / weight_sum
        return mean_loss, (logits, state)

      step = optimizer.state.step
      lr = learning_rate_fn(step)
      grad_fn = jax.grad(loss_fn, has_aux=True)
      grad, (logits, state) = grad_fn(optimizer.target)
      if not single_device:
        grad = jax.lax.pmean(grad, 'batch')
      grad = optimizer_utils.clip_grad(grad, config)

      new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
      metrics = self.compute_metrics(logits, targets, None, single_device)
      metrics['learning_rate'] = lr

      state = {k: v['tag'] for k, v in state.as_dict().items()}
      return new_optimizer, metrics, new_dropout_rng, logits, state

    if single_device:
      return train_step
    else:
      return jax.pmap(train_step, axis_name='batch')

  def make_eval_step(self):
    """Creates a eval step function."""
    def eval_step(model, example):
      with flax.nn.stateful() as state:
        logits = model(example, train=False)
      if self.info.supervised_keys[-1] == 'error_type':
        targets = example['error_type'][:, None]
      else:
        targets = example['target_output']
      state = {k: v['tag'] for k, v in state.as_dict().items()}
      return self.compute_metrics(logits, targets, None), logits, state

    return eval_step

  def make_predict_step(self):
    """Creates a predict step function."""
    return self.make_eval_step()

  def handle_predict(self, metrics, logits, state):
    del metrics, logits, state  # Unused.

  def compute_metrics(self, logits, labels, weights, single_device=False):
    """Compute summary metrics."""
    loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights)
    acc, _ = compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    if not single_device:
      metrics = jax.lax.psum(metrics, 'batch')
    return metrics


class SequenceAdapter(BaseAdapter):
  """Adapts datasets for use with sequence models."""

  def as_example(self, dataset_item):
    example = jax.tree_map(lambda x: x.numpy(), dataset_item)
    return example

  def shard(self, dataset_iter):
    return map(common_utils.shard, dataset_iter)

  def get_train_inputs(self, example):
    if self.config.dataset.representation == 'code':
      train_inputs = {
          'code_statements': example['code_statements'],
          'code_length': example['code_length'],
          'target_output': example['target_output'],
      }
    elif self.config.dataset.representation == 'trace':
      pre_pad_length = example['trace_statements'].shape[-1]
      pad_amount = (256 - pre_pad_length) % 256
      statements = jnp.pad(
          example['trace_statements'],
          [(0, 0), (0, 0), (0, pad_amount)])
      train_inputs = {
          'code_statements': statements,
          'code_length': example['trace_length'],
          'target_output': example['target_output'],
      }
    else:
      raise ValueError('Unexpected representation', self.info.representation)
    if 'error_type' in example:
      train_inputs['error_type'] = example['error_type']
    return train_inputs

  def write_summaries(self, example, logits, summary_writer, info, step, state):
    example = jax_utils.unreplicate(example)
    outputs = jnp.argmax(jax_utils.unreplicate(logits), axis=-1)
    text = summary_utils.human_readable_texts(example, outputs, info)
    summary_writer.text('predictions', '<pre>{}</pre>'.format(text), step)
    self.generate_plots(state, summary_writer, step)

  def generate_plots(self, state, summary_writer=None, step=None):
    """Generates plots."""
