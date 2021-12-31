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

"""Log data and metrics. Inspired by combini/tools/logger."""

import itertools
import os.path
from typing import Mapping, NamedTuple, Optional, Sequence, Tuple, Type, Union

from absl import logging
import gin
import tensorflow.compat.v2 as tf

from dedal import multi_task
from dedal.train import timer


# For each head at each level (embeddings/alignments) a list of metrics.
MetricCLS = Type[tf.metrics.Metric]
MetaKeys = Union[str, Sequence[str]]
MetricCLSWithOptionalMetaKeys = Union[MetricCLS, Tuple[MetricCLS, MetaKeys]]
MultiTaskMetrics = multi_task.Backbone[Sequence[MetricCLSWithOptionalMetaKeys]]


class MetricWithMetaKeys(NamedTuple):
  metric: tf.metrics.Metric
  metakeys: Optional[Sequence[str]] = None


def metric_factory(m):
  """Instantiates a tf.metrics.Metric, keeping track of optional metadata keys.

  Aims to extend tf.metrics.Metric default update_state, allowing to pass extra
  metadata when necessary. If metadata keys are provided, the metadata tensors
  indexed by those keys will be passed to the metric's update_state method as an
  extra arg `metadata`, containing a tuple of tf.Tensor of length equal to that
  of metadata keys. When no metadata keys are given, the update_state method of
  the metric is invoked as usual.

  Args:
    m: Either a tuple (metric_cls, metakeys), where metric_cls is a subclass of
      tf.metrics.Metric and metakeys a sequence of str-valued keys indexing
      metadata needed by the metric's update_state method, or just metric_cls,
      in which case metakeys will be assumed to be empty and no metadata will
      be passed to update_state.

  Returns:
    A namedtuple MetricWithMetaKeys such that:
      + metric contains an instantiated tf.metrics.Metric object of class
        metric_cls.
      + metakeys contains a (possibly None) sequence of str-valued keys indexing
        additional metadata tensors required by the metric's update_state
        method.
  """
  if isinstance(m, Sequence):  # m: Tuple[MetricCLS, MetaKeys].
    metric_cls, metakeys = m
    metakeys = (metakeys,) if isinstance(metakeys, str) else metakeys
  else:  # m: MetricCLS.
    metric_cls = m
    metakeys = None
  return MetricWithMetaKeys(metric=metric_cls(), metakeys=metakeys)


@gin.configurable
class Logger:
  """A class responsible for logging data and metrics."""

  def __init__(
      self,
      workdir,
      strategy,
      split = None,
      task = None,
      scalars = multi_task.Backbone(),
      images = multi_task.Backbone(),
      means = (),
      every = 1000,
      reset_every_step = False,
      start_clock = True):
    """Initialization.

    Args:
      workdir: the parent directory where to store data.
      strategy: distribution strategy.
      split: usually the name of the phase (train, test, valid).
      task: usually the name of the task (train, evaluate, downstream).
      scalars: the scalar metrics to be computed and dumped.
      images: the image metrics to be computed and dumped.
      means: the name of the scalar metrics that will be means. At the very
        least, "loss" and "gradient_norm" will be present.
      every: the periodicity to log the metrics.
      reset_every_step: whether to reset the metrics at every step.
      start_clock: whether or not to start the clock at instantiation.
    """
    split = '' if split is None else split
    self.workdir = os.path.join(workdir, split).rstrip('/')
    self._split = split
    self._task = task
    self._timer = timer.Timer()
    self._reset_every_step = reset_every_step
    self.training = task == 'train'

    # Take the bigger network structure.
    shape = tuple(max(scalars.shape[i], images.shape[i]) for i in range(2))
    enveloppe = multi_task.Backbone.constant_from_shape([], shape)

    means = set(means).union(['loss'])
    if self.training:
      means = means.union(['gradient_norm'])

    with strategy.scope():
      self._scalars = enveloppe.pack(
          [[metric_factory(m) for m in ms] for ms in scalars], default_value=[])
      self._images = enveloppe.pack(
          [[metric_factory(m) for m in ms] for ms in images], default_value=[])
      self._means = {name: tf.keras.metrics.Mean(name) for name in means}

    self._summary_writer = tf.summary.create_file_writer(self.workdir)
    self._every = every
    self._last_step = None if self.training else 0

    if start_clock:
      self.restart_clock()

  def update_mean(self, name, loss):
    if name not in self._means:
      self._means[name] = tf.keras.metrics.Mean(name=name)
    self._means[name].update_state(loss)

  def restart_clock(self):
    return self._timer.restart()

  def update(self,
             y_true,
             y_pred,
             weights,
             metadata):
    """Update the different metrics with the new values."""
    # TODO(oliviert): improve this flatten/unflatten danse.
    # TODO(fllinares): raise exception if key not in metadata?
    y_true = y_pred.unflatten(y_true)
    weights = y_pred.unflatten(weights)
    all_metrics_with_metakeys = self._scalars.pack(
        [a + b for a, b in zip(self._scalars, self._images)])
    for metrics_with_metakeys, label, pred, batch_w in zip(
        all_metrics_with_metakeys, y_true, y_pred, weights):
      for metric, metakeys in metrics_with_metakeys:
        kwargs = ({} if metakeys is None else
                  dict(metadata=tuple(metadata.get(k) for k in metakeys)))
        metric.update_state(label, pred, sample_weight=batch_w, **kwargs)

  def reset(self):
    for metric in self.metrics:
      metric.reset_states()

  def log(self, step):
    """Log the tf summaries."""
    delta = self.restart_clock()
    with self._summary_writer.as_default():
      n_steps = self._every if self.training else (step - self._last_step)
      tf.summary.scalar('steps_per_sec', n_steps / delta, step=step)
      for metric in self.scalars:
        curr = metric.result()
        curr = curr if isinstance(curr, Mapping) else {metric.name: curr}
        for name, value in curr.items():
          tf.summary.scalar(name, value, step=step)
      for metric in self.images:
        tf.summary.image(metric.name, metric.result(), step=step)
    self._last_step = None if self.training else step

  @property
  def metrics(self):
    return self.images + self.scalars

  @property
  def images(self):
    return list(m.metric for m in itertools.chain.from_iterable(self._images))

  @property
  def scalars(self):
    without_means = list(
        m.metric for m in itertools.chain.from_iterable(self._scalars))
    return without_means + list(self._means.values())

  def debug(self, step):
    def metric_to_str(m):
      result = m.result()
      if isinstance(result, Mapping):
        return ', '.join(f'{k}: {v:.3f}' for k, v in result.items())
      return f'{m.name}: {m.result():.3f}'

    metrics_str = ', '.join(metric_to_str(m) for m in self.scalars)
    return f'{self._split} step {step}: {metrics_str}'

  def log_and_reset(self, step, force = True):
    """Log the metrics to summaries if the step allows, and reset them.

    Args:
      step: the step where we are at now.
      force: should we force the behavior (typically for the last step).

    Returns:
      True if the metrics have been logged, False otherwise.
    """
    if step % self._every == 0 or force:
      logging.info(self.debug(step))
      self.log(step)
      self.reset()
      return True
    if self._reset_every_step:
      self.reset()
    return False


@gin.configurable
class DummyLogger:
  """A logger that logs nothing."""

  def update_mean(self, name, value):
    del name, value
    return

  def update(self, *args):
    del args
    return

  def log_and_reset(self, step, force = True):
    del step, force
    return False
