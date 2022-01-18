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

# python3
"""Fixed config tasks based on MLP."""
import functools
from typing import List, Text, Callable, Dict

import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import utils
import tensorflow.compat.v1 as tf


def fc_loss_fn(
    hidden_units, losstype,
    activation):
  """Builds a simple fully connected model.

  Args:
    hidden_units: list containing hidden units
    losstype: type of loss to use for classification
    activation: activation function

  Returns:
    A fn that returns a sonnet module with a batch as input, and a scalar loss
      as output.
  """

  def _build(batch):
    """Builds the sonnet module.

    Args:
      batch: Dict with "image", "label", and "label_onehot" keys. This is the
        input batch used to compute the loss over.

    Returns:
      The loss and a metrics dict.
    """
    net = snt.BatchFlatten()(batch["image"])
    logits = snt.nets.MLP(hidden_units, activation=activation)(net)
    if losstype == "ce":
      loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=batch["label_onehot"], logits=logits)
    elif losstype == "mse":
      loss_vec = tf.reduce_mean(
          tf.square(batch["label_onehot"] - tf.nn.sigmoid(logits)), [1])
    else:
      raise ValueError("Loss type [%s] not supported." % losstype)

    aux = {"accuracy": utils.accuracy(label=batch["label"], logits=logits)}
    return base.LossAndAux(loss=tf.reduce_mean(loss_vec), aux=aux)

  return lambda: snt.Module(_build)


_datasets = ["cifar10", "mnist", "food101_32x32"]
_loss_types = ["mse", "ce"]


def _make_name(cfg):
  loss_type, dataset_name = cfg
  return "FixedMLP_%s_%s_128x128x128_relu_bs128" % (dataset_name.replace(
      "_", ""), loss_type)


def _make_task(cfg):
  loss_type, dataset_name = cfg
  dataset = datasets.get_image_datasets(
      dataset_name, batch_size=128, shuffle_buffer=5000)
  num_classes = dataset.train.output_shapes["label_onehot"].as_list()[1]
  base_model_fn = fc_loss_fn([128, 128, 128, num_classes], loss_type,
                             tf.nn.relu)
  return base.DatasetModelTask(base_model_fn, dataset)

for _dataset_name in _datasets:
  for _loss_type in _loss_types:
    _cfg = (_loss_type, _dataset_name)
    registry.task_registry.register_fixed(_make_name(_cfg))(
        functools.partial(_make_task, _cfg))


def _fc_dropout_loss_fn(hidden_units, activation, keep_probs):
  """Helper for a fully connected task with dropout.

  Args:
    hidden_units: list list of integers containing the hidden layers.
    activation: callable the activation function used by the MLP.
    keep_probs: float float between 0.0 and 1.0. Represents the probability to
      keep a given activation.

  Returns:
    A callable that returns a sonnet module representing the task loss.
  """

  def _fn(batch):
    net = snt.BatchFlatten()(batch["image"])
    logits = snt.nets.MLP(
        hidden_units, activation=activation, use_dropout=True)(
            net, dropout_keep_prob=keep_probs)
    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=logits)
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_Dropout05_128x128_relu_bs128")
def _():
  base_model_fn = _fc_dropout_loss_fn([128, 128, 10],
                                      tf.nn.relu,
                                      keep_probs=0.5)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_Dropout08_128x128_relu_bs128")
def _():
  base_model_fn = _fc_dropout_loss_fn([128, 128, 10],
                                      tf.nn.relu,
                                      keep_probs=0.8)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_Dropout02_128x128_relu_bs128")
def _():
  base_model_fn = _fc_dropout_loss_fn([128, 128, 10],
                                      tf.nn.relu,
                                      keep_probs=0.2)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


def _fc_layer_norm_loss_fn(hidden_units, activation):
  """Helper for a fully connected task with layer norm.

  Args:
    hidden_units: list list of integers containing the hidden layers.
    activation: callable the activation function used by the MLP.

  Returns:
    A callable that returns a sonnet module representing the task loss.
  """

  def _fn(batch):
    net = snt.BatchFlatten()(batch["image"])
    for i, h in enumerate(hidden_units):
      net = snt.Linear(h)(net)
      if i != (len(hidden_units) - 1):
        net = snt.LayerNorm()(net)
        net = activation(net)
    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=net)
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_LayerNorm_128x128x128_relu_bs128")
def _():
  base_model_fn = _fc_layer_norm_loss_fn([128, 128, 128, 10], tf.nn.relu)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_LayerNorm_128x128x128_tanh_bs128")
def _():
  base_model_fn = _fc_layer_norm_loss_fn([128, 128, 128, 10], tf.tanh)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


def _fc_batch_norm_loss_fn(hidden_units, activation):
  """Helper for a fully connected task with batch norm.

  Args:
    hidden_units: list list of integers containing the hidden layers.
    activation: callable the activation function used by the MLP.

  Returns:
    A callable that returns a sonnet module representing the task loss.
  """

  def _fn(batch):
    net = snt.BatchFlatten()(batch["image"])
    for i, h in enumerate(hidden_units):
      net = snt.Linear(h)(net)
      if i != (len(hidden_units) - 1):
        net = snt.BatchNormV2()(net, is_training=False)
        net = activation(net)
    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=net)
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_BatchNorm_128x128x128_relu_bs128")
def _():
  base_model_fn = _fc_batch_norm_loss_fn([128, 128, 128, 10], tf.nn.relu)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))


@registry.task_registry.register_fixed(
    "FixedMLP_cifar10_BatchNorm_64x64x64x64x64_relu_bs128")
def _():
  base_model_fn = _fc_batch_norm_loss_fn([64, 64, 64, 64, 64, 10], tf.nn.relu)
  return base.DatasetModelTask(
      base_model_fn, datasets.get_image_datasets("cifar10", batch_size=128))
