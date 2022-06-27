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

"""Base for train or eval loops for any experiments script."""

import os
from typing import Dict, Optional, Any

from absl import logging
import tensorflow as tf


def get_metrics(task='classification'):
  """Get Keras Metrics class for task of interest."""

  if task == 'classification':
    metrics = {
        # (name, metric)
        'categorical_cross_entropy':
            tf.keras.metrics.CategoricalCrossentropy(),
        'top_1_accuracy':
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=1, name='top_1_accuracy'),
        'top_5_accuracy':
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=5, name='top_5_accuracy'),
    }

  elif task == 'ml_classification':
    metrics = {
        # (name, metric)
        'binary_cross_entropy':
            tf.keras.metrics.BinaryCrossentropy(),
        'mAP':
            tf.keras.metrics.AUC(curve='PR', multi_label=True),
        'AUC':
            tf.keras.metrics.AUC(curve='ROC', multi_label=True),
    }

  elif task == 'dummy':
    metrics = {}

  return metrics


def convert_metric_to_numpy(metrics):
  metric_result = {
      name: m.result().numpy().astype(float) for name, m in metrics.items()
  }
  return metric_result


def reset_metrics_states(metrics):
  for metric in metrics.values():
    metric.reset_states()


def get_optimizer_step(checkpoint_path):
  return tf.train.load_checkpoint(checkpoint_path).get_tensor(
      'optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE'
      )


def create_strategy(strategy_config):
  """Constructs a strategy given the strategy config."""

  distribution_strategy = strategy_config.distribution_strategy.lower()

  if distribution_strategy == 'tpu':
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=strategy_config.tpu
        )
    if strategy_config.tpu not in ('', 'local'):
      tf.config.experimental_connect_to_cluster(cluster_resolver)

    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    return tf.distribute.experimental.TPUStrategy(cluster_resolver)

  if distribution_strategy == 'multi_worker_mirrored':
    return tf.distribute.experimental.MultiWorkerMirroredStrategy()

  if distribution_strategy == 'mirrored':
    return tf.distribute.MirroredStrategy()

  raise ValueError('Invalid Distribution Strategy: %r' % distribution_strategy)


class Replicator(object):
  """A container for replica information."""

  def __init__(self, replica_context):
    self.current_replica_id = replica_context.replica_id_in_sync_group
    self.num_replicas = replica_context.num_replicas_in_sync


class SummaryWriter(object):
  """Base SummaryWriter for Tensorboard."""

  def __init__(self, path, exp_name):
    self.writer = tf.summary.create_file_writer(os.path.join(path, exp_name))

  def __call__(self, metrics, step):
    with self.writer.as_default():
      for k, v in metrics.items():
        tf.summary.scalar(k, v, step=step)
      self.writer.flush()


class Executor(object):
  """An executor containing the train/evaluation base loop."""

  def __init__(self,
               model,
               data,
               params,
               strategy,
               metrics):
    self.model = model
    self.data = data
    self.params = params
    self.strategy = strategy
    self.metrics = metrics or {}

    self._num_workers = int(self.strategy.num_replicas_in_sync + 7) // 8
    self._is_multi_host = (int(self._num_workers) >= 2)
    self._manual_restore = False

  def prepare_inputs(self, inputs):
    # this should be implemented by the user
    raise NotImplementedError

  def get_dataloaders(self, dataloaders, strategy):
    """Initiates the distributed iterators."""

    iterators = ()
    if self._is_multi_host:
      for loader in dataloaders:
        iterators = iterators + ({
            'name': loader.name,
            'mode': loader.mode,
            'iterator': iter(
                strategy.experimental_distribute_datasets_from_function(loader))
        },)
    else:
      for loader in dataloaders:
        iterators = iterators + ({
            'name': loader.name,
            'mode': loader.mode,
            'iterator': iter(
                strategy.experimental_distribute_dataset(loader()))
        },)
    return iterators

  def create_replicated_train_step(self, strategy, model):
    """Constructs an op that runs one train step on a replica."""

    optimizer = model.optimizer
    trainable_variables = model.trainable_variables
    gradient_clip_norm = self.params.train.gradient_clip_norm

    @tf.function
    def replicated_step(inputs):
      """Replicated training step."""
      replicator = Replicator(
          tf.distribute.get_replica_context()
          )
      inputs, labels = self.prepare_inputs(inputs)

      outputs = model(inputs, training=True)
      all_losses = model.loss_fn(labels=labels,
                                 outputs=outputs,
                                 replicator=replicator)
      losses = {}
      for k, v in all_losses.items():
        losses[k] = tf.reduce_mean(v)
        losses[k] = tf.where(tf.math.is_nan(v), 0., v)

      per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync

      grads = tf.gradients(per_replica_loss, trainable_variables)
      grads = [tf.where(tf.math.is_nan(g), 0., g) for g in grads]
      if gradient_clip_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, gradient_clip_norm)
      optimizer.apply_gradients(zip(grads, trainable_variables))
      return losses

    return replicated_step

  def create_train_step(self):
    """Constructs an op that runs a train step and returns the metrics."""

    model = self.model
    strategy = self.strategy

    @tf.function
    def train_step(iterator, num_iterations):
      replicated_step = self.create_replicated_train_step(strategy, model)

      for _ in tf.range(num_iterations-1):
        _ = strategy.run(
            replicated_step, args=(next(iterator),))

      per_replica_metrics = strategy.run(
          replicated_step, args=(next(iterator),))

      # Aggregate the metrics
      def aggregate_fn(loss):
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

      aggregated_metrics = tf.nest.map_structure(aggregate_fn,
                                                 per_replica_metrics)
      return aggregated_metrics

    return train_step

  def create_inference_step(self, outputs_filter = None):
    """Constructs an op that feeds inputs to the model and returns outputs."""

    model = self.model
    strategy = self.strategy

    @tf.function
    def inference_step(iterator):
      def inference_step_fn(inputs):
        inputs, labels = self.prepare_inputs(inputs)
        outputs = model(inputs, training=False)

        outputs['labels'] = labels

        if outputs_filter is not None:
          outputs = outputs_filter(outputs)

        return outputs

      outputs = strategy.run(inference_step_fn, args=(next(iterator),))
      return outputs

    return inference_step

  def create_evaluation_step(self):
    """Constructs an op for running one step of inference + metric calc."""

    model = self.model
    strategy = self.strategy
    metrics = self.metrics
    assert metrics is not None, 'Metrics should be defined in evaluation mode'

    @tf.function
    def evaluation_step(iterator):
      def evaluation_step_fn(inputs):
        inputs, labels = self.prepare_inputs(inputs)
        outputs = model(inputs, training=False)
        for m in metrics.values():
          m.update_state(labels['one_hot'], outputs['probabilities'])

      strategy.run(evaluation_step_fn, args=(next(iterator),))

    return evaluation_step

  def train(self):
    """The main train function involving the necessary loops and modules."""

    # set the variables and pointers
    model = self.model
    strategy = self.strategy
    params = self.params
    auxiliary_metrics = self.metrics
    model_dir = self.params.model_dir

    # construct the dataloaders and data iterators
    dataloaders = self.get_dataloaders(self.data, self.strategy)
    assert len(dataloaders) == 1, 'Train only accepts one dataloader!'
    data_iterator = dataloaders[0]['iterator']

    # set checkpoint saving frequency and max number of checkpoints to keep
    ckpt_save_freq = self.params.train.save_checkpoint_freq
    max_ckpt_to_keep = self.params.train.max_checkpoints

    with strategy.scope():
      # load the optimizer and construct the checkpoint manager
      optimizer = model.optimizer
      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
      manager = tf.train.CheckpointManager(checkpoint,
                                           directory=model_dir,
                                           max_to_keep=max_ckpt_to_keep)

      # check if override checkpoint is provided
      override_checkpoint_path = params.checkpoint_path

      # check if the latest checkpoint exists
      latest_checkpoint_path = tf.train.latest_checkpoint(model_dir)

      if override_checkpoint_path and not self._manual_restore:
        logging.info(
            'Override checkpoint found. Restoring the model from the '
            'checkpoint at %s.', override_checkpoint_path)
        checkpoint.restore(override_checkpoint_path)

      elif latest_checkpoint_path:
        logging.info(
            'Latest checkpoint found. Restoring the model from the '
            'latest checkpoint at %s.', latest_checkpoint_path)
        checkpoint.restore(latest_checkpoint_path)

      # construct summary writer for Tensorboard
      summary_writer = SummaryWriter(model_dir, 'train')

      # construct train step
      train_step = self.create_train_step()
      current_step = optimizer.iterations.numpy()
      total_steps = params.train.optimizer.learning_rate.total_steps

      if current_step == total_steps:
        logging.info('Training complete!')
        return {}

      while current_step < total_steps:
        if current_step + params.train.iterations_per_loop > total_steps:
          num_iterations = total_steps - current_step
        else:
          num_iterations = params.train.iterations_per_loop

        num_iterations = tf.convert_to_tensor(num_iterations, dtype=tf.int32)
        metrics = train_step(data_iterator, num_iterations)
        current_step += num_iterations

        metrics = tf.nest.map_structure(lambda x: x.numpy().astype(float),
                                        metrics)

        metrics.update(convert_metric_to_numpy(auxiliary_metrics))

        if callable(optimizer.lr):
          metrics.update({'learning_rate': optimizer.lr(current_step).numpy()})
        else:
          metrics.update({'learning_rate': optimizer.lr.numpy()})

        logging.info('Train Step: %d/%d  / training metric = %s',
                     current_step, total_steps, metrics)

        summary_writer(metrics=metrics, step=optimizer.iterations)
        reset_metrics_states(auxiliary_metrics)

        # saving the checkpoint
        if (current_step % ckpt_save_freq == 0 or
            current_step == params.train.iterations_per_loop):
          ckpt_save_path = manager.save(checkpoint_number=current_step)
          logging.info('Checkpoint saved at step %d at path: %s',
                       current_step, ckpt_save_path)

      ckpt_save_path = manager.save(checkpoint_number=current_step)
      logging.info('Final checkpoint saved at step %d at path: %s',
                   current_step, ckpt_save_path)

      return metrics

  def infer(self,
            iterator,
            num_steps = None,
            outputs_filter = None):
    """Iterates over data and returns the aggregated model outputs."""

    inference_step = self.create_inference_step(outputs_filter=outputs_filter)
    outputs = {}
    cnt = -1
    while True:
      if num_steps is not None:
        if cnt > num_steps:
          return outputs, cnt

      cnt += 1
      try:
        with tf.experimental.async_scope():
          step_outputs = inference_step(iterator)
          for k, v in step_outputs.items():
            # aggregate cross-replica results for each step
            value = tf.concat(
                tf.nest.flatten(v, expand_composites=True),
                axis=0,
                ).numpy()

            if k not in outputs:
              outputs[k] = [value]
            else:
              outputs[k] += [value]

      except (StopIteration, tf.errors.OutOfRangeError):
        tf.experimental.async_clear_error()
        return outputs, cnt

  def evaluation_loop(self):
    """Iterates over data and returns the aggregated metrics."""

    metrics = self.metrics

    # construct the dataloaders and data iterators
    dataloaders = self.get_dataloaders(self.data, self.strategy)
    assert len(dataloaders) == 1, 'Evaluation only accepts one dataloader!'
    iterator = dataloaders[0]['iterator']

    evaluation_step = self.create_evaluation_step()
    current_step = 0
    while True:
      try:
        with tf.experimental.async_scope():
          evaluation_step(iterator)
          current_step += 1
          if current_step % 100 == 0:
            logging.info('Evaluation step: [%r], metrics = %r',
                         current_step, convert_metric_to_numpy(metrics))
      except (StopIteration, tf.errors.OutOfRangeError):
        tf.experimental.async_clear_error()
        break

    metrics = convert_metric_to_numpy(metrics)

    logging.info('Total evaluation steps: [%d]', current_step)
    logging.info('Evaluation metric = %r', metrics)

    return metrics

  def evaluate(self):
    """Iterates over checkpoints OR gets a ckpt path and evaluates the model."""

    model = self.model
    model_dir = self.params.model_dir
    checkpoint = tf.train.Checkpoint(model=model)

    # construct summary writer for Tensorboard
    summary_writer = SummaryWriter(model_dir, 'eval')

    if self.params.checkpoint_path and not self._manual_restore:
      logging.info(
          'Override checkpoint found. Restoring the model from the '
          'checkpoint at %s.', self.params.checkpoint_path)
      checkpoint.restore(self.params.checkpoint_path).expect_partial(
          ).assert_existing_objects_matched()
      current_step = get_optimizer_step(self.params.checkpoint_path)
      evaluation_metrics = self.evaluation_loop()
      summary_writer(metrics=evaluation_metrics, step=current_step)
      reset_metrics_states(self.metrics)

      logging.info('Evaluation metrics at step %d: %s',
                   current_step,
                   evaluation_metrics)

      return evaluation_metrics

    else:
      for latest_checkpoint_path in tf.train.checkpoints_iterator(
          model_dir,
          min_interval_secs=180,
          timeout=2*3600,
          timeout_fn=None,
          ):

        logging.info(
            'New checkpoint found. Restoring the model from the '
            'latest checkpoint at %s.', latest_checkpoint_path)

        checkpoint.restore(latest_checkpoint_path).expect_partial(
        ).assert_existing_objects_matched()
        current_step = get_optimizer_step(latest_checkpoint_path)
        evaluation_metrics = self.evaluation_loop()
        summary_writer(metrics=evaluation_metrics, step=current_step)
        reset_metrics_states(self.metrics)

        logging.info('Evaluation metrics at step %d: %s',
                     current_step,
                     evaluation_metrics)

        if current_step == self.params.train.optimizer.learning_rate.total_steps:
          logging.info('Reached total steps: %d, exitting...', current_step)
          break

      return evaluation_metrics

    return

  def run(self, mode):
    """Route to proper execution command."""

    if mode == 'train':
      self.train()

    elif mode == 'eval':
      self.evaluate()

    else:
      raise ValueError('Mode not supported!')
