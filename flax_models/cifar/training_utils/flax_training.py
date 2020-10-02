# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Functions to train the networks for image classification tasks."""

import functools
import math
import os
import time
from typing import Callable, Dict, Tuple

from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import lr_schedule
import jax
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.io import gfile

from flax_models.cifar.datasets import dataset_source as dataset_source_lib


FLAGS = flags.FLAGS


# Training hyper-parameters
flags.DEFINE_float('gradient_clipping', 5.0, 'Gradient clipping.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_bool('use_learning_rate_schedule', True,
                  'Whether to use a cosine schedule or keep the learning rate '
                  'constant. Training on cifar should always use the schedule '
                  ', this flag is mostly for testing purpose.')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay coefficient.')
flags.DEFINE_integer('run_seed', 0,
                     'Seed to use to generate pseudo random number during '
                     'training (for dropout for instance). Has no influence on '
                     'the dataset shuffling.')

# Additional flags that don't affect the model.
flags.DEFINE_integer('save_progress_seconds', 3600, 'Save progress every...s')
flags.DEFINE_multi_integer(
    'additional_checkpoints_at_epochs', [],
    'Additional epochs when we should save the model for later analysis. '
    'No matter the value of this flag, the most recent version of the model '
    'will be saved regularly to resume training if needed.')
flags.DEFINE_bool('also_eval_on_training_set', False,
                  'If set to true, the model will also be evaluated on the '
                  '(non-augmented) training set at the end of each epoch.')


def restore_checkpoint(
    optimizer,
    model_state,
    directory):
  """Restores a model and its state from a given checkpoint.

  If several checkpoints are saved in the checkpoint directory, the latest one
  will be loaded (based on the `epoch`).

  Args:
    optimizer: The optimizer containing the model that we are training.
    model_state: Current state associated with the model.
    directory: Directory where the checkpoints should be saved.

  Returns:
    The restored optimizer and model state, along with the number of epochs the
      model was trained for.
  """
  train_state = dict(optimizer=optimizer, model_state=model_state, epoch=0)
  restored_state = checkpoints.restore_checkpoint(directory, train_state)
  return (restored_state['optimizer'],
          restored_state['model_state'],
          restored_state['epoch'])


def save_checkpoint(optimizer,
                    model_state,
                    directory,
                    epoch):
  """Saves a model and its state.

  Removes a checkpoint if it already exists for a given epoch.

  Args:
    optimizer: The optimizer containing the model that we are training.
    model_state: Current state associated with the model.
    directory: Directory where the checkpoints should be saved.
    epoch: Number of epochs the model has been trained for.
  """
  train_state = dict(optimizer=optimizer,
                     model_state=model_state,
                     epoch=epoch)
  if gfile.exists(os.path.join(directory, 'checkpoint_' + str(epoch))):
    gfile.remove(os.path.join(directory, 'checkpoint_' + str(epoch)))
  checkpoints.save_checkpoint(directory, train_state, epoch, keep=2)


def create_optimizer(model,
                     learning_rate,
                     beta = 0.9):
  """Creates an SGD (Nesterov momentum) optimizer.

  Learning rate will be ignored when using a learning rate schedule.

  Args:
    model: The FLAX model to optimize.
    learning_rate: Learning rate for the gradient descent.
    beta: Momentum parameter.

  Returns:
    A SGD optimizer that targets the model.
  """
  optimizer_def = optim.Momentum(learning_rate=learning_rate,
                                 beta=beta,
                                 nesterov=True)
  optimizer = optimizer_def.create(model)
  return optimizer


def cross_entropy_loss(logits,
                       one_hot_labels):
  """Returns the cross entropy loss between some logits and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.

  Returns:
    The cross entropy, averaged over the first dimension (samples).
  """
  log_softmax_logits = jax.nn.log_softmax(logits)
  return -jnp.sum(one_hot_labels * log_softmax_logits) / one_hot_labels.shape[0]


def error_rate_metric(logits,
                      one_hot_labels):
  """Returns the error rate between some predictions and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.

  Returns:
    The error rate (1 - accuracy), averaged over the first dimension (samples).
  """
  return jnp.mean(jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1))


def tensorflow_to_numpy(xs):
  """Converts a tree of tensorflow tensors to numpy arrays.

  Args:
    xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
      tensorflow tensors.

  Returns:
    A pytree with the same structure as xs, where the leaves have been converted
      to jax numpy ndarrays.
  """
  # Use _numpy() for zero-copy conversion between TF and NumPy.
  return jax.tree_map(lambda x: x._numpy(), xs)  # pylint: disable=protected-access


def shard_batch(xs):
  """Shards a batch across all available replicas.

  Assumes that the number of samples (first dimension of xs) is divisible by the
  number of available replicas.

  Args:
    xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
      numpy ndarrays.

  Returns:
    A pytree with the same structure as xs, where the leaves where added a
      leading dimension representing the replica the tensor is on.
  """
  local_device_count = jax.local_device_count()
  def _prepare(x):
    return x.reshape((local_device_count, -1) + x.shape[1:])
  return jax.tree_map(_prepare, xs)


def load_and_shard_tf_batch(xs):
  """Converts to numpy arrays and distribute a tensorflow batch.

  Args:
    xs: A pytree (such as nested tuples, lists, and dicts) where the leaves are
      tensorflow tensors.

  Returns:
    A pytree of numpy ndarrays with the same structure as xs, where the leaves
      where added a leading dimension representing the replica the tensor is on.
  """
  return shard_batch(tensorflow_to_numpy(xs))


def get_cosine_schedule(num_epochs, learning_rate,
                        num_training_obs,
                        batch_size):
  """Returns a cosine learning rate schedule, without warm up.

  Args:
    num_epochs: Number of epochs the model will be trained for.
    learning_rate: Initial learning rate.
    num_training_obs: Number of training observations.
    batch_size: Total batch size (number of samples seen per gradient step).

  Returns:
    A function that takes as input the current step and returns the learning
      rate to use.
  """
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  learning_rate_fn = lr_schedule.create_cosine_learning_rate_schedule(
      learning_rate, steps_per_epoch, num_epochs)
  return learning_rate_fn


def global_norm(updates):
  """Returns the l2 norm of the input.

  Args:
    updates: A pytree of ndarrays representing the gradient.
  """
  return jnp.sqrt(
      sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)]))


def clip_by_global_norm(updates):
  """Clips the gradient by global norm.

  Will have no effect if FLAGS.gradient_clipping is set to zero (no clipping).

  Args:
    updates: A pytree of numpy ndarray representing the gradient.

  Returns:
    The gradient clipped by global norm.
  """
  if FLAGS.gradient_clipping > 0:
    g_norm = global_norm(updates)
    trigger = g_norm < FLAGS.gradient_clipping
    updates = jax.tree_multimap(
        lambda t: jnp.where(trigger, t, (t / g_norm) * FLAGS.gradient_clipping),
        updates)
  return updates


def train_step(
    optimizer,
    state,
    batch,
    prng_key,
    learning_rate_fn,
    l2_reg
):
  """Performs one gradient step.

  Args:
    optimizer: The optimizer targeting the model to train.
    state: Current state associated with the model (contains the batch norm MA).
    batch: Batch on which the gradient should be computed. Must have an `image`
      and `label` key.
    prng_key: A PRNG key to use for stochasticity for this gradient step (e.g.
      for sampling an eventual dropout mask).
    learning_rate_fn: Function that takes the current step as input and return
      the learning rate to use.
    l2_reg: Weight decay parameter. The total weight decay penaly added to the
      loss is equal to 0.5 * l2_reg * sum_i ||w_i||_2^2 where the sum is over
      all trainable parameters of the model (bias and batch norm parameters
      included).

  Returns:
    The updated optimizer (that includes the model), the updated state and
      a dictionary containing the training loss and error rate on the batch.
  """

  def forward_and_loss(model):
    """Returns the model's loss, updated state and predictions.

    Args:
      model: The model that we are training.
    """
    with flax.nn.stateful(state) as new_state:
      with flax.nn.stochastic(prng_key):
        logits = model(batch['image'], train=True)
    loss = cross_entropy_loss(logits, batch['label'])
    # We apply weight decay to all parameters, including bias and batch norm
    # parameters.
    weight_penalty_params = jax.tree_leaves(model.params)
    weight_l2 = sum([jnp.sum(x ** 2) for x in weight_penalty_params])
    weight_penalty = l2_reg * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_state, logits)

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(forward_and_loss, has_aux=True)
  (_, (new_state, logits)), grad = grad_fn(optimizer.target)

  # We synchronize the gradients across replicas by averaging them.
  grad = jax.lax.pmean(grad, 'batch')

  # Gradient is clipped after being synchronized.
  grad = clip_by_global_norm(grad)
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  # Compute some metrics to monitor the training.
  metrics = {'train_error_rate': error_rate_metric(logits, batch['label']),
             'train_loss': cross_entropy_loss(logits, batch['label'])}

  return new_optimizer, new_state, metrics, lr


# Shorthand notation for typing the function defined above.
# We omit the weight decay and learning rate arguments as they will be
# passed before we pmap the function.
_TrainStep = Callable[[
    flax.nn.Model,  # model.
    flax.nn.Collection,  # state.
    Dict[str, jnp.ndarray],  # batch.
    jnp.ndarray  # PRNG key
], Tuple[flax.optim.Optimizer, flax.nn.Collection, Dict[str, float],  # metrics.
         jnp.ndarray  # learning rate.
        ]]


def eval_step(model, state,
              batch):
  """Evaluates the model on a single batch.

  Args:
    model: The model to evaluate.
    state: Current state associated with the model (contains the batch norm MA).
    batch: Batch on which the model should be evaluated. Must have an `image`
      and `label` key.

  Returns:
    A dictionary containing the loss and error rate on the batch. These metrics
    are summed over the samples (and not averaged).
  """

  # Averages the batch norm moving averages.
  state = jax.lax.pmean(state, 'batch')
  with flax.nn.stateful(state, mutable=False):
    logits = model(batch['image'], train=False)

  # Because we don't have a guarantee that all batches contains the same number
  # of samples, we can't average the metrics per batch and then average the
  # resulting values. To compute the metrics correctly, we sum them (error rate
  # and cross entropy returns means, thus we multiply by the number of samples),
  # and finally sum across replicas. These sums will be divided by the total
  # number of samples outside of this function.
  num_samples = batch['image'].shape[0]
  metrics = {
      'error_rate':
          error_rate_metric(logits, batch['label']) * num_samples,
      'loss':
          cross_entropy_loss(logits, batch['label']) * num_samples
  }
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


# Shorthand notation for typing the function defined above.
_EvalStep = Callable[
        [flax.nn.Model, flax.nn.Collection, Dict[str, jnp.ndarray]],
        Dict[str, float]]


def eval_on_dataset(
    model, state, dataset,
    pmapped_eval_step):
  """Evaluates the model on the whole dataset.

  Args:
    model: The model to evaluate.
    state: Current state associated with the model (contains the batch norm MA).
    dataset: Dataset on which the model should be evaluated. Should already
      being batched.
    pmapped_eval_step: A pmapped version of the `eval_step` function (see its
      documentation for more details).

  Returns:
    A dictionary containing the loss and error rate on the batch. These metrics
    are averaged over the samples.
  """
  eval_metrics = []
  total_num_samples = 0
  for eval_batch in dataset:
    # Load and shard the TF batch.
    eval_batch = load_and_shard_tf_batch(eval_batch)
    # Compute metrics and sum over all observations in the batch.
    metrics = pmapped_eval_step(model, state, eval_batch)
    eval_metrics.append(metrics)
    # Number of samples seen in num_replicas * per_replica_batch_size.
    total_num_samples += (
        eval_batch['label'].shape[0] * eval_batch['label'].shape[1])
  # Metrics are all the same across all replicas (since we applied psum in the
  # eval_step). The next line will fetch the metrics on one of them.
  eval_metrics = common_utils.get_metrics(eval_metrics)
  # Finally, we divide by the number of samples to get the mean error rate and
  # cross entropy.
  eval_summary = jax.tree_map(lambda x: x.sum() / total_num_samples,
                              eval_metrics)
  return eval_summary


def train_for_one_epoch(
    dataset_source,
    optimizer, state,
    prng_key, pmapped_train_step,
    summary_writer
):
  """Trains the model for one epoch.

  Args:
    dataset_source: Container for the training dataset.
    optimizer: The optimizer targeting the model to train.
    state: Current state associated with the model (contains the batch norm MA).
    prng_key: A PRNG key to use for stochasticity (e.g. for sampling an eventual
      dropout mask). Is not used for shuffling the dataset.
    pmapped_train_step: A pmapped version of the `train_step` function (see its
      documentation for more details).
    summary_writer: A Tensorboard SummaryWriter to use to log metrics.

  Returns:
    The updated optimizer (with the associated updated model), state and PRNG
      key.
  """
  train_metrics = []
  for batch in dataset_source.get_train(use_augmentations=True):
    # Generate a PRNG key that will be rolled into the batch.
    step_key, prng_key = jax.random.split(prng_key)
    # Load and shard the TF batch.
    batch = tensorflow_to_numpy(batch)
    batch = shard_batch(batch)
    # Shard the step PRNG key.
    sharded_keys = common_utils.shard_prng_key(step_key)

    optimizer, state, metrics, lr = pmapped_train_step(
        optimizer, state, batch, sharded_keys)
    train_metrics.append(metrics)
  train_metrics = common_utils.get_metrics(train_metrics)
  # Get training epoch summary for logging.
  train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
  train_summary['learning_rate'] = lr[0]
  current_step = int(optimizer.state.step[0])
  for metric_name, metric_value in train_summary.items():
    summary_writer.scalar(metric_name, metric_value, current_step)
  summary_writer.flush()
  return optimizer, state, prng_key


def train(optimizer,
          state,
          dataset_source,
          training_dir, num_epochs):
  """Trains the model.

  Args:
    optimizer: The optimizer targeting the model to train.
    state: Current state associated with the model (contains the batch norm MA).
    dataset_source: Container for the training dataset.
    training_dir: Parent directory where the tensorboard logs and model
      checkpoints should be saved.
   num_epochs: Number of epochs for which we want to train the model.
  """
  checkpoint_dir = os.path.join(training_dir, 'checkpoints')
  summary_writer = tensorboard.SummaryWriter(training_dir)
  prng_key = jax.random.PRNGKey(FLAGS.run_seed)

  optimizer = jax_utils.replicate(optimizer)
  state = jax_utils.replicate(state)

  if FLAGS.use_learning_rate_schedule:
    learning_rate_fn = get_cosine_schedule(num_epochs, FLAGS.learning_rate,
                                           dataset_source.num_training_obs,
                                           dataset_source.batch_size)
  else:
    learning_rate_fn = lambda step: FLAGS.learning_rate

  # pmap the training and evaluation functions.
  pmapped_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          l2_reg=FLAGS.weight_decay),
      axis_name='batch')
  pmapped_eval_step = jax.pmap(eval_step, axis_name='batch')

  # Log initial results:
  if gfile.exists(checkpoint_dir):
    optimizer, state, epoch_last_checkpoint = restore_checkpoint(
        optimizer, state, checkpoint_dir)
    # If last checkpoint was saved at the end of epoch n, then the first
    # training epochs to do when we resume training is n+1.
    initial_epoch = epoch_last_checkpoint + 1
    info = 'Resuming training from epoch {}'.format(initial_epoch)
    logging.info(info)
  else:
    initial_epoch = 0
    logging.info('Starting training from scratch.')

  time_at_last_checkpoint = time.time()
  for epochs_id in range(initial_epoch, num_epochs):
    if epochs_id in FLAGS.additional_checkpoints_at_epochs:
      # To save additional checkpoints that will not be erase by later version,
      # we save them in a new directory.
      c_path = os.path.join(checkpoint_dir, 'additional_ckpt_' + str(epochs_id))
      save_checkpoint(optimizer, state, c_path, epochs_id)
    tick = time.time()
    optimizer, state, prng_key = train_for_one_epoch(dataset_source, optimizer,
                                                     state, prng_key,
                                                     pmapped_train_step,
                                                     summary_writer)
    tock = time.time()
    info = 'Epoch {} finished in {:.2f}s.'.format(epochs_id, tock - tick)
    logging.info(info)

    # Evaluate the model on the test set, and optionally the training set.
    tick = time.time()
    current_step = int(optimizer.state.step[0])
    if FLAGS.also_eval_on_training_set:
      train_ds = dataset_source.get_train(use_augmentations=False)
      train_metrics = eval_on_dataset(
          optimizer.target, state, train_ds, pmapped_eval_step)
      for metric_name, metric_value in train_metrics.items():
        summary_writer.scalar('eval_on_train_' + metric_name,
                              metric_value, current_step)
      summary_writer.flush()
    test_ds = dataset_source.get_test()
    test_metrics = eval_on_dataset(
        optimizer.target, state, test_ds, pmapped_eval_step)
    for metric_name, metric_value in test_metrics.items():
      summary_writer.scalar('test_' + metric_name,
                            metric_value, current_step)
    summary_writer.flush()
    tock = time.time()
    info = 'Evaluated model in {:.2f}.'.format(tock - tick)
    logging.info(info)

    # Save new checkpoint if the last one was saved more than
    # `save_progress_seconds` seconds ago.
    sec_from_last_ckpt = time.time() - time_at_last_checkpoint
    if sec_from_last_ckpt > FLAGS.save_progress_seconds:
      save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
      time_at_last_checkpoint = time.time()
      logging.info('Saved checkpoint.')

  # Always save final checkpoint
  save_checkpoint(optimizer, state, checkpoint_dir, epochs_id)
