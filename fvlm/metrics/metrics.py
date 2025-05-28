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

"""Metric definitions and their mappings to summary writer functions.

A `Metric` should accept positional arguments `model_output` and `labels` and
there should be a corresponding writer function for it in the `_METRIC_WRITERS`
dictionary.
"""

from typing import Dict, Mapping, Text, Union

from absl import logging
from clu import metrics
import flax
from flax import struct
from flax.metrics import tensorboard
import gin
import jax.numpy as jnp

from losses.base_losses import softmax_cross_entropy
from metrics import coco_metrics
from utils import gin_utils


MetricsCollection = Mapping[Text, metrics.Metric]


@struct.dataclass
class Accuracy(metrics.Average):
  """Computes the accuracy from model outputs `logits` and `labels`."""

  @classmethod
  @gin_utils.allow_remapping
  def from_model_output(cls, logits, labels,
                        **kwargs):
    if logits.shape != labels.shape:
      raise ValueError('labels and logits shapes must be compatible'
                       f'{labels.shape} != {logits.shape}')
    matches = (logits.argmax(axis=-1) == labels.argmax(axis=-1))
    matches = matches.astype(jnp.float32)
    return super().from_model_output(values=matches, **kwargs)


@struct.dataclass
class AccuracyFromDictionary(metrics.Average):
  """Computes the accuracy from a dictionary of model outputs and labels."""

  @classmethod
  @gin_utils.allow_remapping
  def from_model_output(cls,
                        model_outputs,
                        labels,
                        **kwargs):
    if 'predictions' in model_outputs:
      predictions = model_outputs['predictions']
    elif 'logits' in model_outputs:
      predictions = model_outputs['logits'].argmax(axis=-1)
    else:
      raise ValueError('Predictions/logits is missing from model outputs')

    if 'targets' in model_outputs:
      targets: jnp.ndarray = model_outputs['targets']
    elif 'labels' in model_outputs:
      targets: jnp.ndarray = model_outputs['labels']
    elif isinstance(labels, dict) and 'mc_labels' in labels:
      targets: jnp.ndarray = labels['mc_labels']
    elif isinstance(labels, jnp.ndarray):
      targets: jnp.ndarray = labels
    else:
      raise ValueError('labels are missing from model outputs')

    if len(targets.shape) > 1:
      targets = targets.argmax(axis=-1)

    if predictions.shape != targets.shape:
      raise ValueError('targets and predictions shapes must be compatible'
                       f'{targets.shape} != {predictions.shape}')
    matches = (predictions == targets)
    matches = matches.astype(jnp.float32)
    return super().from_model_output(values=matches, **kwargs)


@struct.dataclass
class CrossEntropy(metrics.Average):
  """Computes the cross-entropy loss from model outputs `logits` and `labels`."""

  @classmethod
  @gin_utils.allow_remapping
  def from_model_output(cls, logits, labels,
                        **kwargs):
    if logits.shape != labels.shape:
      raise ValueError('labels and logits shapes must be compatible'
                       f'{labels.shape} != {logits.shape}')
    loss = softmax_cross_entropy(logits, labels)
    return super().from_model_output(values=loss, **kwargs)

# gin.configurable interferes with `dataclass` registration mechanism. We have
# to define our metrics as gin constants instead.
gin.constant('Accuracy', Accuracy)
gin.constant('DictAccuracy', AccuracyFromDictionary)
gin.constant('CE', CrossEntropy)
gin.constant('coco_metric', coco_metrics.COCODetectionMetric)

# This is to keep a mapping from metrics to their appropriate summary writers.
# NOTE: each new metric should have an entry below.
_DEFAULT_WRITER = tensorboard.SummaryWriter.scalar

_METRIC_WRITERS = flax.core.FrozenDict({
    Accuracy:
        tensorboard.SummaryWriter.scalar,
    AccuracyFromDictionary:
        tensorboard.SummaryWriter.scalar,
    CrossEntropy:
        tensorboard.SummaryWriter.scalar,
    coco_metrics.COCODetectionMetric:
        tensorboard.SummaryWriter.scalar,
})


def write_dict_metrics(step, output_metrics,
                       summary_writer,):
  """Helper function to write the results of evaluation metrics to tensorboard.

  Args:
    step: the training step to use for recording metrics.
    output_metrics: A dictionary of output metrics.
    summary_writer: a `SummaryWriter` to use for writing the metrics to.
  """
  for key, value in output_metrics.items():
    _DEFAULT_WRITER(summary_writer, tag=key, value=value, step=step)


@gin.configurable(allowlist=['direct_write_keys', 'strip_task_scope'])
def write_metrics(step, eval_metrics,
                  summary_writer,
                  direct_write_keys = False,
                  strip_task_scope = False):
  """Helper function to write the results of evaluation metrics to tensorboard.

  Args:
    step: the training step to use for recording metrics.
    eval_metrics: a dictionary of metric names and their corresponding
      `clu.metrics.Metric` objects.
    summary_writer: a `SummaryWriter` to use for writing the metrics to.
    direct_write_keys: Whether to write the metric keys directly without the
      metric names.
    strip_task_scope: Whether to strip the task scope of the keys. The task
      scope is assumed to be in the following format: <task_name>/<metrics>.
  """
  for name, metric in eval_metrics.items():
    metric_values = metric.compute()
    logging.info('Eval results for step %d: %s', step, metric_values)
    if isinstance(metric_values, dict):
      for key, value in metric_values.items():
        key = '/'.join(key.split('/')[1:]) if strip_task_scope else key
        tag = key if direct_write_keys else f'{name}/{key}'
        _METRIC_WRITERS.get(metric.__class__, _DEFAULT_WRITER)(
            summary_writer, tag=tag, value=value, step=step)
    else:
      _METRIC_WRITERS[metric.__class__](
          summary_writer, tag=name, value=metric_values, step=step)
