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

# Copyright 2024 Google LLC
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

"""Module for Model Evaluation (in a potentially distributed manner)."""

import functools
from typing import Dict

from absl import logging
from clu import metric_writers
from clu import metrics as metrics_lib
import data as data_lib
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


@flax.struct.dataclass
class RegressionMetrics(metrics_lib.Collection):
  mae: metrics_lib.Average.from_output("loss")
  l2_err: metrics_lib.Average.from_output("l2_err")


@flax.struct.dataclass
class AUC(metrics_lib.CollectingMetric.from_outputs(("labels", "logits"))):
  """Area Under the ROC Curve (AUC ROC)."""

  def compute(self):
    values = super().compute()
    labels = values["labels"]
    logits = values["logits"]
    if labels.ndim == 1:
      labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])

    keras_auc = tf.keras.metrics.AUC(curve="ROC", from_logits=True)
    keras_auc.update_state(y_true=labels, y_pred=logits)
    return keras_auc.result().numpy()


@flax.struct.dataclass
class ClassificationMetrics(metrics_lib.Collection):
  loss: metrics_lib.Average.from_output("loss")
  accuracy: metrics_lib.Accuracy
  auc: AUC


def regression_eval_step(
    batch,
    metrics,
):
  """Regression Metric Evaluation step for each batch."""

  additional_metrics = {}
  logits = batch[data_lib.X_KEY]
  labels = batch[data_lib.TARGET_KEY]
  metrics_collection = RegressionMetrics
  additional_metrics["l2_err"] = jnp.mean(jnp.abs(logits - labels) ** 2)

  loss = jnp.mean(jnp.abs(logits - labels))
  return metrics.merge(
      metrics_collection.gather_from_model_output(
          loss=loss, logits=logits, labels=labels, **additional_metrics
      )
  )


def cross_entropy_loss(logits, labels):
  """Computes cross-entropy loss."""
  if labels.shape[-1] == 1 or labels.ndim == 1:
    one_hot_labels = jnp.zeros_like(logits)
    one_hot_labels = one_hot_labels.at[labels].set(1)
  else:
    one_hot_labels = labels
  xe = jnp.sum(nn.log_softmax(logits) * one_hot_labels, axis=-1)
  return -jnp.mean(xe)


def classification_eval_step(
    batch,
    metrics,
):
  """Classification Metric Evaluation step for each batch."""

  logits = batch[data_lib.X_KEY]
  labels = batch[data_lib.TARGET_KEY]
  metrics_collection = ClassificationMetrics

  loss = cross_entropy_loss(logits, labels)
  return metrics.merge(
      metrics_collection.gather_from_model_output(
          loss=loss,
          logits=logits,
          labels=labels,
      )
  )


def update_auc(
    keras_auc, logits, labels
):
  """Computes AUC."""
  keras_auc.update_state(y_true=labels, y_pred=logits)


def update_acc(
    keras_acc,
    logits,
    labels,
):
  keras_acc.update_state(y_true=labels, y_pred=logits)


def evaluate_model_tf(
    model,
    test_ds,
    validation_ds,
    task_type,
    writer,
    step,
):
  """Evaluates model using TF metrics."""
  _evaluate_model_on_dataset_tf(
      model,
      test_ds,
      task_type,
      writer,
      step,
      is_validation=False,
  )
  _evaluate_model_on_dataset_tf(
      model,
      validation_ds,
      task_type,
      writer,
      step,
      is_validation=True,
  )


def _evaluate_model_on_dataset_tf(
    model,
    test_ds,
    task_type,
    writer,
    step,
    is_validation,
):
  """Evaluates model using TF metrics."""
  if task_type == "regression":
    raise ValueError("Regression is supported in the other eval method.")

  keras_auc = tf.keras.metrics.AUC(curve="ROC", from_logits=True)
  keras_acc = tf.keras.metrics.CategoricalAccuracy()
  writer_postfix = "val" if is_validation else "test"

  for step_data in test_ds:
    x = step_data[data_lib.X_KEY]
    labels = step_data[data_lib.TARGET_KEY]
    labels = tf.squeeze(labels)

    logits = model(x, training=False)
    labels = tf.one_hot(labels, logits.shape[-1], axis=-1)
    update_auc(keras_auc, logits=logits, labels=labels)
    update_acc(keras_acc, logits=logits, labels=labels)

  # logging.info("TOTAL EVAL STEP COUNT: %s", step_count)
  auc = keras_auc.result().numpy()
  accuracy = keras_acc.result().numpy()
  writer.write_scalars(
      step,
      {f"auc/{writer_postfix}": auc, f"accuracy/{writer_postfix}": accuracy},
  )
  writer.flush()


def evaluate_model(
    model,
    test_ds,
    task_type,
    writer,
    step,
):
  """Evaluates model, in a potentially distributed way."""
  if task_type == "classification":
    metrics = ClassificationMetrics.empty()
    eval_step = classification_eval_step
  else:
    metrics = RegressionMetrics.empty()
    eval_step = regression_eval_step

  # axis_name must be `batch` for `gather_from_model_output` to work.
  partial_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          metrics=metrics,
      ),
      axis_name="batch",
  )

  def tensor_to_array(x):
    return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(x))

  # This is necessary as the model is currently a TF module, not flax module.
  def get_logits(leaf):
    if leaf.ndim >= 3:
      # breakpoint()
      parallel_count, batch_sz, seq_len = leaf.shape[:3]
      outputs = model(
          tf.reshape(leaf, (parallel_count * batch_sz, seq_len)),
          training=False,
      )
      return tf.reshape(outputs, (parallel_count, batch_sz, -1))
    else:
      # Leaf is label tensor. This assumes label tensor has last dimension 1.
      return leaf

  for batch in test_ds:
    batch = jax.tree.map(get_logits, batch)
    batch = jax.tree.map(tensor_to_array, batch)
    metrics = partial_eval_step(batch)
  computed_metrics = metrics.unreplicate().compute()

  writer.write_scalars(step, computed_metrics)
  logging.info("step %d EVAL METRICS: %s", step, computed_metrics)
  writer.flush()
