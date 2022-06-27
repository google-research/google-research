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

"""Train functional distance prediction models.
"""
import collections
import functools
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Union

from absl import app
from absl import flags
from acme.jax import networks as networks_lib
from acme.jax.types import PRNGKey, TrainingMetrics  # pylint: disable=g-multiple-import
from acme.utils import paths
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from func_dist.data_utils import pickle_datasets
from func_dist.dist_learning import logging
from func_dist.dist_learning import losses
from func_dist.dist_learning import train_utils


# Input data
flags.DEFINE_string(
    'paired_data_path', None, 'Path to gzipped pickle file with paired data.')
flags.DEFINE_string(
    'demo_data_path', None,
    'Path to gzipped pickle file with demonstrator (e.g. human) data.')
flags.DEFINE_string(
    'robot_data_path', None,
    'Path to gzipped pickle file with robot interaction data.')
flags.DEFINE_string(
    'test_data_path', None,
    'Path to gzipped pickle file with robot interaction data for testing.')

# Training hyperparameters
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate for training.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train for.')

flags.DEFINE_float(
    'distance_loss_scale', 1.0, 'Weight for distance prediction loss.')
flags.DEFINE_float(
    'paired_loss_scale', 1.0, 'Weight for paired embedding loss.')
flags.DEFINE_float(
    'domain_confusion_scale', 1.0, 'Weight for domain adversarial loss.')

flags.DEFINE_float(
    'val_fraction', 0.1, 'Fraction of data to use for validation.')
flags.DEFINE_bool(
    'augment_goals', True,
    'If True, use any future frame in a demonstration as the goal in distance '
    'learning. Else use the final frame only.')
flags.DEFINE_bool(
    'augment_frames', True,
    'If True, pad and randomly crop image observations.')
flags.DEFINE_bool(
    'include_zero_distance_pairs', False,
    'If True, include d(g, g) = 0 examples in the training data.')
flags.DEFINE_bool(
    'val_full_episodes', True,
    'If True, split training and validation split according to episode '
    'boundaries.')
flags.DEFINE_float(
    'subsampling_threshold', 0.0,
    'Subsample video frames such that all consecutive frames differ in L2 norm '
    'by at least this threshold.')

flags.DEFINE_integer(
    'shuffle_buffer_size', 50_000,
    'The size of the buffer for shuffling in the input pipeline.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer(
    'split_seed', 0, 'Random seed for defining train-val splits.')

# Networks
flags.DEFINE_list(
    'encoder_conv_filters', [16, 16, 32],
    'Number and sizes of convolutional filters in the embedding network.')  # pytype: disable=wrong-arg-types
flags.DEFINE_integer(
    'encoder_conv_size', 5, 'Convolution kernel size in the embedding network.')

# Evaluation
flags.DEFINE_list(
    'test_data_cutoffs', [1, 10],
    'Counts of test episodes for which to evaluate test metrics.')  # pytype: disable=wrong-arg-types

# Logging
flags.DEFINE_string(
    'logdir', 'data/dist_learning/dev', 'Directory for output logs.')


FLAGS = flags.FLAGS

TrainingState = train_utils.TrainingState
CompoundDataset = Mapping[str, Iterator[Any]]
CutOffs = Mapping[Union[int, str], Optional[int]]


def prepare_data_for_jax(
    dataset,
    generator_fn,
    dataset_type = None,
    eval_mode = False,
    **generator_kwargs):
  """Prepare dataset from generator, batch and optionally shuffle."""
  generator_fn = functools.partial(
      generator_fn, dataset_type, eval_mode, **generator_kwargs)
  signature = dataset.get_iterator_signature(dataset_type)

  tf_dataset = tf.data.Dataset.from_generator(generator_fn, signature)

  if not eval_mode:
    tf_dataset = tf_dataset.shuffle(
        buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)

  tf_dataset = tf_dataset.batch(FLAGS.batch_size, drop_remainder=not eval_mode)
  tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
  return tf_dataset


def prepare_splits_for_jax(dataset, dataset_type=None):
  train_set = prepare_data_for_jax(
      dataset, dataset.generate_train_data, dataset_type)
  val_set = prepare_data_for_jax(
      dataset, dataset.generate_val_data, dataset_type, eval_mode=True)
  return train_set, val_set


def affine_transform(preds, labels):
  """Computes the closest affine transform W * preds + b matching labels."""
  if len(preds.shape) == 1:
    preds = np.expand_dims(preds, axis=1)
  if len(labels.shape) == 1:
    labels = np.expand_dims(labels, axis=1)
  x = np.concatenate([preds, np.ones([preds.shape[0], 1])], axis=1)
  xt = np.transpose(x)
  xt_x = np.matmul(xt, x)
  try:
    theta = np.matmul(np.linalg.inv(xt_x), np.matmul(xt, labels))
  except np.linalg.LinAlgError:
    theta = [1, 0]
  preds = np.squeeze(preds, axis=1)
  preds = theta[0] * preds + theta[1]
  return preds


def affine_transform_mse(preds, labels):
  """Computes MSE after the closest affine transform."""
  preds = affine_transform(preds, labels)
  return losses.mse(preds, labels)


def embed_images(model, params, batch):
  """Embed image inputs in a batch, leaving other inputs unchanged."""
  if isinstance(batch, tuple):
    batch = [model(params, x) if len(x.shape) == 4 else x for x in batch]
  elif len(batch.shape) == 4:
    batch = model(params, batch)
  return batch


@jax.jit
def apply_model(
    state,
    batch):
  """Compute model gradients and loss for a batch.

  Args:
    state: training state containing model inference functions and parameters.
    batch: batch of each dataset, keyed by dataset name.

  Returns:
    dist_grad: gradient of the distance model (encoder and regression model)
        parameters.
    domain_grad: gradient of the domain discriminator parameters.
    metrics: dictionary of metrics to return, keyed by metric name.
  """

  def regression_loss_fn(
      params,
      dist_batch,
      ):
    """Distance regression loss."""
    dist_batch = embed_images(state.encoder_fn, params, dist_batch)
    concat_embs = jnp.concatenate(dist_batch[:2], axis=1)
    pred_dists = state.distance_fn(params, concat_embs)
    pred_dists = jnp.squeeze(pred_dists, axis=1)
    dists = dist_batch[2]
    dist_loss = losses.mse(pred_dists, dists)
    dist_loss *= FLAGS.distance_loss_scale
    dist_error = losses.mean_error(pred_dists, dists)
    return dist_loss, dist_error, pred_dists

  def paired_loss_fn(
      params,
      paired_batch,
      ):
    """Embedding loss for paired observations."""
    paired_batch = embed_images(state.encoder_fn, params, paired_batch)
    first_embs = paired_batch[0]
    second_embs = paired_batch[1]
    paired_loss = losses.mse(first_embs, second_embs)
    paired_loss *= FLAGS.paired_loss_scale
    return paired_loss

  def domain_adversarial_loss(
      params,
      discriminator_params,
      obs_batch,
      int_batch,
      ):
    """Domain adversarial loss."""
    obs_embs = embed_images(state.encoder_fn, params, obs_batch)
    int_embs = embed_images(state.encoder_fn, params, int_batch)
    both_embs = jnp.concatenate([obs_embs, int_embs], axis=0)
    domains = jnp.concatenate(
        [jnp.ones([obs_embs.shape[0], 1]), jnp.zeros([int_embs.shape[0], 1])],
        axis=0)
    pred_domains = state.domain_discriminator_fn(
        discriminator_params, both_embs)
    domain_loss = losses.adversarial_discrimination_loss(pred_domains, domains)
    domain_loss *= FLAGS.domain_confusion_scale
    return domain_loss

  def dist_loss_fn(
      params,
      discriminator_params,
      batch,
      ):
    loss_breakdown = {}
    preds = {}
    loss = 0
    if 'dist' in batch:
      dist_loss, dist_error, pred_dists = regression_loss_fn(
          {'params': params}, batch['dist'])
      preds['dist'] = pred_dists
      loss += dist_loss
      loss_breakdown['distance_loss'] = dist_loss
      loss_breakdown['distance_error'] = dist_error
    if 'goal_augm_dist' in batch:  # For evaluation only, not used for training.
      augm_dist_loss, augm_dist_error, augm_pred_dists = regression_loss_fn(
          {'params': params}, batch['goal_augm_dist'])
      preds['goal_augm_dist'] = augm_pred_dists
      loss_breakdown['goal_augm_distance_loss'] = augm_dist_loss
      loss_breakdown['goal_augm_distance_error'] = augm_dist_error
    if 'paired' in batch:
      paired_loss = paired_loss_fn({'params': params}, batch['paired'])
      loss += paired_loss
      loss_breakdown['paired_loss'] = paired_loss
    if 'obs' in batch and 'int' in batch:
      domain_loss = domain_adversarial_loss(
          {'params': params}, {'params': discriminator_params}, batch['obs'],
          batch['int'])
      loss += domain_loss
      loss_breakdown['adversarial_domain_loss'] = domain_loss

      if 'dist' in batch and 'paired' in batch:
        loss_breakdown['total_distance_model_loss'] = loss
    return loss, (loss_breakdown, preds)

  def domain_loss_fn(
      params,
      encoder_params,
      batch,
      ):
    obs_embs = embed_images(
        state.encoder_fn, {'params': encoder_params}, batch['obs'])
    int_embs = embed_images(
        state.encoder_fn, {'params': encoder_params}, batch['int'])
    domains = jnp.concatenate(
        [jnp.ones([obs_embs.shape[0], 1]), jnp.zeros([int_embs.shape[0], 1])],
        axis=0)
    both_embs = jax.lax.stop_gradient(
        jnp.concatenate([obs_embs, int_embs], axis=0))
    pred_domains = state.domain_discriminator_fn(
        {'params': params}, both_embs)
    loss = losses.discrimination_loss(pred_domains, domains)
    return loss

  distance_params = state.distance_optimizer.target
  domain_params = state.domain_optimizer.target

  # TODO(minttu): pass in obs and int embeddings to both domain_adversarial
  # loss and domain loss to avoid creating them twice.
  dist_grad_fn = jax.value_and_grad(dist_loss_fn, has_aux=True)
  (_, aux), dist_grad = dist_grad_fn(
      distance_params, domain_params, batch)
  metrics, pred_dists = aux
  domain_grad = None
  if 'obs' in batch and 'int' in batch:
    domain_grad_fn = jax.value_and_grad(domain_loss_fn)
    discriminator_loss, domain_grad = domain_grad_fn(
        domain_params, distance_params, batch)
    metrics['domain_discriminator_loss'] = discriminator_loss

  return dist_grad, domain_grad, metrics, pred_dists


@jax.jit
def update_model(state, dist_grad, domain_grad):
  """Update model parameters according to dist_grad and domain_grad.

  Args:
    state: training state containing model parameters and optimizer states.
    dist_grad: gradient of distance model parameters.
    domain_grad: gradient of domain discriminator parameters.

  Returns:
    new_state: updated training state.
  """
  distance_optimizer = state.distance_optimizer.apply_gradient(dist_grad)
  domain_optimizer = state.domain_optimizer.apply_gradient(domain_grad)
  new_state = state.replace(
      distance_optimizer=distance_optimizer,
      domain_optimizer=domain_optimizer,
      step=state.step + 1,
  )
  return new_state


def eval_on_dataset(
    state,
    ds,
    cutoffs = None,
    ):
  """Evaluate losses and other metrics for one pass over a dataset.

  If multiple datasets of different sizes are included, dummy batches will be
  used after a dataset iterator ends.

  Args:
    state: A TrainingState capturing model parameters and apply_fns to be
        passed to apply_model.
    ds: A dictionary containing potentially different sized datasets with
        which to evaluate the model.
    cutoffs: A list of dataset sizes for which to measure MSE and affine MSE.

  Returns:
    metrics: Evaluation metrics averaged over the full dataset.
  """

  def next_or_none(
      iterator,
      ):
    try:
      batch = next(iterator)
    except StopIteration:
      batch = None
    return batch

  def create_dummy_batch(
      shape
      ):
    if shape and isinstance(shape[0], tuple):
      batch = tuple([np.zeros([1, *s]) for s in shape])
    else:
      batch = np.zeros([1, *shape])
    return batch

  metrics = collections.defaultdict(float)
  counts = collections.defaultdict(int)
  ds_iter = jax.tree_map(iter, ds)
  shapes = None
  all_pred_dists = []
  all_dists = []
  all_goal_augm_pred_dists = []
  all_goal_augm_dists = []

  while True:
    batch_or_none = {k: next_or_none(v) for k, v in ds_iter.items()}
    if np.all([v is None for v in batch_or_none.values()]):
      break
    shapes = shapes or jax.tree_map(lambda x: x.shape[1:], batch_or_none)
    batch = {
        k: create_dummy_batch(shapes[k]) if v is None else v
        for k, v in batch_or_none.items()}

    _, _, batch_metrics, batch_pred_dists = apply_model(state, batch)
    update_incremental_mean(metrics, counts, batch_metrics, batch_or_none)
    all_pred_dists.append(batch_pred_dists['dist'])
    all_dists.append(batch['dist'][2])
    if 'goal_augm_dist' in batch:
      all_goal_augm_pred_dists.append(batch_pred_dists['goal_augm_dist'])
      all_goal_augm_dists.append(batch['goal_augm_dist'][2])

  if cutoffs is not None:
    all_pred_dists = np.concatenate(all_pred_dists, axis=0)
    all_dists = np.concatenate(all_dists, axis=0)
    all_goal_augm_pred_dists = np.concatenate(all_goal_augm_pred_dists, axis=0)
    all_goal_augm_dists = np.concatenate(all_goal_augm_dists, axis=0)
    for label, cutoff in cutoffs.items():
      metrics[f'mse_{label}'] = (
          losses.mse(all_pred_dists[:cutoff], all_dists[:cutoff]))
      metrics[f'affine_mse_{label}'] = (
          affine_transform_mse(all_pred_dists[:cutoff], all_dists[:cutoff]))
      metrics[f'goal_augm_mse_{label}'] = (
          losses.mse(
              all_goal_augm_pred_dists[:cutoff], all_goal_augm_dists[:cutoff]))
      metrics[f'goal_augm_affine_mse_{label}'] = (
          affine_transform_mse(
              all_goal_augm_pred_dists[:cutoff], all_goal_augm_dists[:cutoff]))

  metrics['total_distance_model_loss'] = (
      metrics['distance_loss'] + metrics['paired_loss']
      + metrics['adversarial_domain_loss'])
  return dict(metrics)


def update_incremental_mean(
    metrics,
    counts,
    batch_metrics,
    batch_or_none):
  """Update averaged metrics in-place with batch metrics weighted by batch size.

  Args:
    metrics: incremental averages to update in-place, keyed by metric name.
    counts: the number of data points that have contributed to the incremental
        average, keyed by metric name.
    batch_metrics: values to update incremental averages towards, keyed by
        metric name.
    batch_or_none: data contributing to batch_metrics calculation keyed by
        dataset type (None if no batch was drawn from that dataset), used here
        to compute batch sizes.
  """

  def get_batch_size(
      batch):
    if batch is None:
      return None
    if isinstance(batch, tuple):
      return len(batch[0])
    else:
      return len(batch)

  metric_to_inputs = {
      'distance_loss': ['dist'],
      'distance_error': ['dist'],
      'goal_augm_distance_loss': ['goal_augm_dist'],
      'goal_augm_distance_error': ['goal_augm_dist'],
      'paired_loss': ['paired'],
      'domain_discriminator_loss': ['int', 'obs'],
      'adversarial_domain_loss': ['int', 'obs'],
  }
  batch_metrics.pop('total_distance_model_loss', None)
  batch_sizes = {k: get_batch_size(v) for k, v in batch_or_none.items()}
  for k, v in batch_metrics.items():
    # If all inputs are not dummy batches.
    # TODO(minttu): This still misses out on some batches if e.g. observation
    # dataset is larger than interaction dataset.
    if np.all([batch_or_none[b] is not None for b in metric_to_inputs[k]]):
      # Weight the newest batch_metrics according to the batch size to average
      # over the full dataset. Taking the sum allows proportionate weighting
      # for domain classification losses, which can handle different sized
      batch_size = np.sum([batch_sizes[b] for b in metric_to_inputs[k]])
      metrics[k] = (
          (counts[k] * metrics[k] + batch_size * v) / (counts[k] + batch_size))
      counts[k] += batch_size


def update_best_loss(metrics, state, ckpt_dir, best_epoch, prefix=''):
  """Record the best value per loss and the epoch it was observed.

  Saves checkpoints for the best value for each loss.

  Args:
    metrics: evaluation metrics from the most recent epoch.
    state: current TrainingState.
    ckpt_dir: directory to which to write checkpoints.
    best_epoch: dictionary keeping track of the best epoch per loss type.
    prefix: string identifier to add to checkpoints.
  """
  if prefix:
    prefix = prefix + '_'
  for k, v in metrics.items():
    k = f'{prefix}{k}'
    if k not in state.best_loss or v < state.best_loss[k]:
      state.best_loss[k] = v
      best_epoch[k] = state.epoch
      if k == 'val_distance_loss':
        print(f'best val loss: {v} (epoch {best_epoch[k]})\n')
      train_utils.save_checkpoint(ckpt_dir, state, label=f'best_{k}')


def train_and_evaluate(
    train_ds,
    eval_ds,
    test_ds,
    key,
    test_cutoffs = None):
  """Main training loop for distance learning."""
  ckpt_dir = paths.process_path(FLAGS.logdir, 'ckpts', add_uid=True)
  state = train_utils.restore_or_initialize(
      FLAGS.encoder_conv_filters, FLAGS.encoder_conv_size, key, ckpt_dir,
      FLAGS.learning_rate)

  summary_dir = paths.process_path(FLAGS.logdir, 'tb', add_uid=True)
  summary_writer = tf.summary.create_file_writer(summary_dir)

  best_epoch = {}
  if state.epoch == 0:
    # Evaluate initialization.
    eval_metrics = eval_on_dataset(state, eval_ds)
    test_metrics = eval_on_dataset(state, test_ds, cutoffs=test_cutoffs)
    logging.log_metrics(eval_metrics, summary_writer, state.step, 'val')
    logging.log_metrics_to_stdout(eval_metrics, state.epoch, 'val epoch  ')
    logging.log_metrics(test_metrics, summary_writer, state.step, 'test')
    update_best_loss(eval_metrics, state, ckpt_dir, best_epoch, 'val')
    update_best_loss(test_metrics, state, ckpt_dir, best_epoch, 'test')
  for e in range(state.epoch, FLAGS.num_epochs):
    metrics = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    train_iter = {k: iter(v) for k, v in train_ds.items()}
    for batch in zip(*train_iter.values()):
      batch = dict(zip(train_iter.keys(), batch))
      dist_grads, domain_grads, batch_metrics, _ = apply_model(state, batch)
      state = update_model(state, dist_grads, domain_grads)
      update_incremental_mean(metrics, counts, batch_metrics, batch)
    state = state.replace(epoch=e + 1)
    # Note: train 'epoch' repeats short datasets, validation epoch does not.
    metrics['total_distance_model_loss'] = (
        metrics['distance_loss'] + metrics['paired_loss']
        + metrics['adversarial_domain_loss'])
    logging.log_metrics(metrics, summary_writer, e + 1, 'train')
    logging.log_metrics_to_stdout(metrics, e + 1, 'train epoch')

    eval_metrics = eval_on_dataset(state, eval_ds)
    logging.log_metrics(eval_metrics, summary_writer, e + 1, 'val')
    logging.log_metrics_to_stdout(eval_metrics, e + 1, 'val epoch  ')
    update_best_loss(eval_metrics, state, ckpt_dir, best_epoch, 'val')
    test_metrics = eval_on_dataset(state, test_ds, cutoffs=test_cutoffs)
    logging.log_metrics(test_metrics, summary_writer, e + 1, 'test')
    update_best_loss(test_metrics, state, ckpt_dir, best_epoch, 'test')
    train_utils.save_checkpoint(ckpt_dir, state)


def create_datasets(
    val_fraction,
    ):
  """Prepare datasets for distance learning."""
  # Generates (human frame, robot frame) pairs. Domain labels (1, 0)
  paired_data = pickle_datasets.PairedDataset(
      FLAGS.paired_data_path, val_fraction, augment_frames=FLAGS.augment_frames)

  observation_data = pickle_datasets.ObservationDataset(
      path=FLAGS.demo_data_path,
      val_fraction=val_fraction,
      include_zero_distance_pairs=FLAGS.include_zero_distance_pairs,
      split_by_episodes=FLAGS.val_full_episodes,
      shuffle_seed=FLAGS.split_seed,
      augment_goals=FLAGS.augment_goals,
      augment_frames=FLAGS.augment_frames,
      subsampling_threshold=FLAGS.subsampling_threshold)

  interaction_data = pickle_datasets.InteractionDataset(
      path=FLAGS.robot_data_path,
      val_fraction=val_fraction,
      end_on_success=True,
      include_zero_distance_pairs=FLAGS.include_zero_distance_pairs,
      split_by_episodes=FLAGS.val_full_episodes,
      shuffle_seed=FLAGS.split_seed,
      augment_goals=False,
      augment_frames=FLAGS.augment_frames)
  interaction_test_data = pickle_datasets.InteractionDataset(
      path=FLAGS.test_data_path,
      val_fraction=1.0,
      end_on_success=True,
      include_zero_distance_pairs=FLAGS.include_zero_distance_pairs,
      split_by_episodes=FLAGS.val_full_episodes,
      shuffle_seed=FLAGS.split_seed,
      augment_goals=False,
      augment_frames=FLAGS.augment_frames)
  cum_episode_lengths = np.cumsum(interaction_test_data.episode_lengths())
  test_cutoffs = {i: cum_episode_lengths[i] for i in FLAGS.test_data_cutoffs}
  test_cutoffs['all'] = None

  paired_set, paired_valset = prepare_splits_for_jax(paired_data)
  observation_set, observation_valset = prepare_splits_for_jax(
      observation_data, 'obs')
  interaction_set, interaction_valset = prepare_splits_for_jax(
      interaction_data, 'obs')
  distance_set, distance_valset = prepare_splits_for_jax(
      observation_data, 'dist')
  goal_augm_distance_valset = prepare_data_for_jax(
      observation_data, observation_data.generate_val_data, 'dist',
      eval_mode=True, augment_goals=True)
  _, test_distance_set = prepare_splits_for_jax(interaction_test_data, 'dist')

  goal_augm_test_distance_set = prepare_data_for_jax(
      interaction_test_data, interaction_test_data.generate_val_data, 'dist',
      eval_mode=True, augment_goals=True)

  paired_set = paired_set.repeat()
  interaction_set = interaction_set.repeat()
  distance_set = distance_set.repeat()

  train_sets = {
      'dist': distance_set, 'obs': observation_set, 'int': interaction_set,
      'paired': paired_set}
  val_sets = {
      'dist': distance_valset, 'goal_augm_dist': goal_augm_distance_valset,
      'obs': observation_valset, 'int': interaction_valset,
      'paired': paired_valset}
  test_sets = {
      'dist': test_distance_set, 'goal_augm_dist': goal_augm_test_distance_set}

  train_sets = tfds.as_numpy(train_sets)
  val_sets = tfds.as_numpy(val_sets)
  test_sets = tfds.as_numpy(test_sets)
  return train_sets, val_sets, test_sets, test_cutoffs


def main(_):
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  key = jax.random.PRNGKey(FLAGS.seed)
  train_ds, eval_ds, test_ds, test_cutoffs = create_datasets(FLAGS.val_fraction)
  train_and_evaluate(train_ds, eval_ds, test_ds, key, test_cutoffs)


if __name__ == '__main__':
  app.run(main)
