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

"""Metric util functions for contrastive learning experiments.

Defines custom metrics and metrics helper classes for experiments where we want
to track performance along multiple dataset-specific axes.
"""

import abc
import os

from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class R2Metric(tf.keras.metrics.Metric):
  """Compute and store running R^2 score."""

  def __init__(self, tss, name='R^2', **kwargs):
    super().__init__(name=name, **kwargs)
    if tf.rank(tss) > 0:
      # TODO(zeef) Find a way to store TSS and RSS values as arrays.
      # Currently, resetting the metric will throw an error if RSS is not 0-dim.
      self.tss = tf.reduce_mean(tss)
    else:
      self.tss = tss
    self.rss = self.add_weight(name='rss', initializer='zeros')

  def update_state(self, actual, preds):
    res_squared = tf.reduce_mean(
        (actual - preds)**2, axis=tf.range(1, tf.rank(actual)))
    res_squared_summed = tf.reduce_sum(res_squared)
    self.rss.assign_add(res_squared_summed)

  def result(self):
    return 1 - self.rss / self.tss


class DspritesAccuracy(tf.keras.metrics.Metric):
  """A measure of correctness for dsprites (non-shape) latents.

  We treat a predicted set of latents as 'correct' if it is closer to the
  correct values than to nearby values.
  """

  def __init__(self, tolerance, name='dsprites_accuracy', **kwargs):
    super().__init__(name=name, **kwargs)
    self.tolerance = tf.constant(tolerance)
    self.correct = self.add_weight(name='correct', initializer='zeros')
    self.seen = self.add_weight(name='seen', initializer='zeros')

  def update_state(self, actual, preds):
    is_correct = tf.math.abs(actual - preds) < self.tolerance
    is_correct = tf.cast(is_correct, tf.float32)
    # need to count no. of examples seen but can't use .shape[0] in graph mode
    is_seen = tf.reduce_sum(tf.ones_like(is_correct))
    self.correct.assign_add(tf.reduce_sum(is_correct))
    self.seen.assign_add(is_seen)

  def result(self):
    return self.correct / self.seen


class DspritesShapeAccuracy(tf.keras.metrics.Metric):
  """A measure of correctness for dsprites shape prediction.

  We treat a prediction as 'correct' if it is close to 1 for the correct shape
  AND close to 0 for the other shapes.
  """

  def __init__(self, tolerance, name='dsprites_shape_accuracy', **kwargs):
    super().__init__(name=name, **kwargs)
    self.tolerance = tf.constant(tolerance)
    self.correct = self.add_weight(name='correct', initializer='zeros')
    self.seen = self.add_weight(name='seen', initializer='zeros')

  def update_state(self, actual, preds):
    # actual and preds are shape (batch_size, 3)
    is_correct = tf.math.abs(actual - preds) < self.tolerance
    # require all three shape predictions to be accurate to count as correct
    is_correct = tf.experimental.numpy.all(is_correct, axis=1)
    is_correct = tf.cast(is_correct, tf.float32)
    # need to count no. of examples seen but can't use .shape[0] in graph mode
    is_seen = tf.reduce_sum(tf.ones_like(is_correct))
    self.correct.assign_add(tf.reduce_sum(is_correct))
    self.seen.assign_add(is_seen)

  def result(self):
    return self.correct / self.seen


class MetricsInterface(object):
  """Interface for managing metric definition, collection, and updating.
  """

  @abc.abstractmethod
  def __init__(self, data_dir):
    pass

  def setup_metrics(self):
    """Creates a consistent way of managing metrics for each individual latent.

    Returns:
      Dictionary with a key for each axis to be measured.
    """
    raise NotImplementedError()

  def update_metrics(self):
    """Updates the metric values for each axis created in setup_metrics."""
    raise NotImplementedError()

  def setup_summary_writers(self, data_dir, writer_names):
    """Creates a tf summary writer for each name in writer_names.

    Args:
      data_dir: Str, path to folder where summary writers should write to.
      writer_names: List of writer names, e.g. ['train', 'test'] or
        ['eval_overall', 'eval_shape_accuracy', 'eval_position'], etc.

    Returns:
      Dict with (key,value) pairs of the form ('writer_name': writer).
    """
    all_summary_writers = {}
    for name in writer_names:
      log_dir = os.path.join(data_dir, name)
      summary_writer = tf.summary.create_file_writer(log_dir)
      all_summary_writers[name] = summary_writer
    return all_summary_writers

  def write_metrics_to_summary(self, all_metrics, global_step):
    """Updates the summary writers at the end of each step.

    Call this at the end of each step, from within a
    `with summary_writer_name.as_default():` context.

    Args:
      all_metrics: List of tf.keras.metrics objects.
      global_step: Int.
    """
    for metric in all_metrics:
      metric_value = metric.result().numpy().astype(float)
      logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
      tf.summary.scalar(metric.name, metric_value, step=global_step)


class DspritesEvalMetrics(MetricsInterface):
  """Handles storing and updating metrics during dsprites evaluation loops.

  Simplifies the process of collecting metrics on multiple individual latents
  as well as overall performance by abstracting it away from the training loop.
  To add a new axis of metric collection: simply specify its name and behaviour
  in setup_metrics and update_metrics, and (optionally) create a separate
  summary writer for it by adding it to writer_names.
  """

  def __init__(self, data_dir, tss):
    super().__init__(data_dir)
    self.writer_names = [
        'eval_overall', 'eval_shapes', 'eval_scale', 'eval_orientation',
        'eval_x_pos', 'eval_y_pos'
    ]
    self.tss = tss
    self.summary_writers = self.setup_summary_writers(data_dir,
                                                      self.writer_names)
    self.metrics_dict = self.setup_metrics()

  def setup_metrics(self):
    """Sets up metrics for dsprites eval loop.

    Returns:
      Dictionary with a key for each axis to be measured (overall performance,
        individual latents, etc).
    """
    metrics_dict = {}
    tss = self.tss
    metrics_dict['eval_overall'] = [tf.keras.metrics.Mean('MSE loss')]
    metrics_dict['eval_shapes'] = self.create_metric_for_latent(
        0.1, tf.reduce_mean(tss[0:3]), is_shape=True)
    metrics_dict['eval_scale'] = self.create_metric_for_latent(
        1 / (2 * 10), tss[3])
    metrics_dict['eval_orientation'] = self.create_metric_for_latent(
        1 / (2 * 40), tss[4])
    metrics_dict['eval_x_pos'] = self.create_metric_for_latent(
        1 / (2 * 32), tss[5])
    metrics_dict['eval_y_pos'] = self.create_metric_for_latent(
        1 / (2 * 32), tss[6])
    return metrics_dict

  def update_metrics(self, total_loss, actual, preds):
    """Updates all metric values for dsprites eval.

    Args:
      total_loss: Float, loss score for current global step.
      actual: 2d array of shape (minibatch_size, 7) of actual latent values.
      preds: 2d array of shape (minibatch_size, 7) of predicted latent values.
    """
    metrics_dict = self.metrics_dict
    for k in metrics_dict:
      if k == 'eval_overall':
        self.update_individual_metrics(metrics_dict[k], total_loss)
      elif k == 'eval_shapes':
        self.update_individual_metrics(
            metrics_dict[k], actual=actual[:, :3], preds=preds[:, :3])
      elif k == 'eval_scale':
        self.update_individual_metrics(
            metrics_dict[k], actual=actual[:, 3], preds=preds[:, 3])
      elif k == 'eval_orientation':
        self.update_individual_metrics(
            metrics_dict[k], actual=actual[:, 4], preds=preds[:, 4])
      elif k == 'eval_x_pos':
        self.update_individual_metrics(
            metrics_dict[k], actual=actual[:, 5], preds=preds[:, 5])
      elif k == 'eval_y_pos':
        self.update_individual_metrics(
            metrics_dict[k], actual=actual[:, 6], preds=preds[:, 6])
      else:
        pass

  def create_metric_for_latent(self, tolerance, tss, is_shape=False):
    """Creates the tf.keras.metrics objects for an individual latent axis.

    Args:
      tolerance: Specifies how close to correct a measurement must be to the
        ground truth, for determining accuracy.
      tss: Total sum of squares value (over entire dataset) for individual
        latent.
      is_shape: Whether to use the DspritesShapeAccuracy metric for accuracy.

    Returns:
      List of tf.keras.metrics objects for latent.
    """
    metrics = []
    metrics.append(tf.keras.metrics.Mean('MSE loss'))
    if is_shape:
      metrics.append(DspritesShapeAccuracy(tolerance, 'accuracy'))
    else:
      metrics.append(DspritesAccuracy(tolerance, 'accuracy'))
    metrics.append(R2Metric(tss, 'R^2'))
    return metrics

  def update_individual_metrics(self,
                                metrics_list,
                                total_loss=None,
                                actual=None,
                                preds=None):
    """Logic for updating individual dsprites eval metrics within a collection.

    Args:
      metrics_list: List of tf.keras.metrics objects.
      total_loss: Optional float, total loss score from current step.
      actual: 2d array, actual latent values for minibatch.
      preds: 2d array, predicted latent values for minibatch.
    """
    for metric in metrics_list:
      if metric.name == 'MSE loss':
        if total_loss is not None:
          metric.update_state(total_loss)
        else:
          mse = (actual - preds)**2
          metric.update_state(mse)
      elif metric.name == 'accuracy':
        metric.update_state(actual, preds)
      elif metric.name == 'R^2':
        metric.update_state(actual, preds)
      else:
        logging.info(
            'Received unknown metric %s, please add desired behaviour to dsprites update_individual_metrics function',
            metric.name)


class DspritesTrainMetrics(MetricsInterface):
  """Handles storing and updating metrics during dsprites train loops.
  """

  def __init__(self, data_dir):
    super().__init__(data_dir)
    self.writer_names = ['train']
    self.summary_writers = self.setup_summary_writers(data_dir,
                                                      self.writer_names)
    self.metrics_dict = self.setup_metrics()

  def setup_metrics(self):
    metrics_dict = {}
    metrics_dict['train'] = [tf.keras.metrics.Mean('MSE loss')]
    return metrics_dict

  def update_metrics(self, total_loss, actual, preds):
    del actual, preds  # not used here
    for k in self.metrics_dict:
      if k == 'train':
        if self.metrics_dict[k][0].name == 'MSE loss':
          self.metrics_dict[k][0].update_state(total_loss)
      else:
        pass


@tf.function
def get_tss_for_r2(strategy, ds, num_classes, num_examples, batch_size=1):
  """Computes dataset-wide stats for use in R^2 computation.

  Args:
    strategy: tf.distribute.Strategy object.
    ds: tf.data.Dataset object.
    num_classes: Int.
    num_examples: Int, number of examples in dataset.
    batch_size: If ds is batched, specify batch size here.

  Returns:
    Tuple (y_bar, tss): arrays of size (7,), containing the average value and
      total sum of squares for each of the seven latents.
  """

  def y_bar_step(x):
    return tf.reduce_sum(x['values'], axis=0)

  def tss_step(x, y_bar):
    return tf.reduce_sum((y_bar - x['values'])**2, axis=0)

  y_bar = tf.zeros(num_classes)
  num_steps = num_examples // batch_size
  ds_iter = iter(ds)
  for _ in tf.range(num_steps):
    x = next(ds_iter)
    per_replica = strategy.run(y_bar_step, args=(x,))
    y_bar += strategy.reduce('SUM', per_replica, axis=None)
  y_bar = y_bar / num_examples
  tss = tf.zeros(num_classes)
  ds_iter = iter(ds)
  for _ in tf.range(num_steps):
    x = next(ds_iter)
    per_replica = strategy.run(tss_step, args=(x, y_bar))
    tss += strategy.reduce('SUM', per_replica, axis=None)
  return y_bar, tss
