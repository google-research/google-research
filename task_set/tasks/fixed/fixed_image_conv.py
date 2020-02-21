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

# python3
"""Fixed convolutional image tasks."""

import numpy as np
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers


def ce_pool_loss(
    hidden_units,
    activation_fn,
    initializers=None,
    pool="avg",
    use_batch_norm=False,
):
  """Helper function to make a sonnet loss.

  This creates a cross entropy loss, pooling last layer conv net.
  Args:
    hidden_units: list list of hidden unit sizes
    activation_fn: callable activation function used in the convnet
    initializers: optional dict dictionary of initalizers used to initialize the
      convnet weights.
    pool: str the type of pooling. Supported values are max or avg.
    use_batch_norm: boolean to use batch norm or not in the convnet

  Returns:
    callable that returns a sonnet module representing the loss.
  """
  if not initializers:
    initializers = {}

  def _fn(batch):
    """Make the loss."""
    net = snt.nets.ConvNet2D(
        hidden_units,
        kernel_shapes=[(3, 3)],
        strides=[2] + [1] * (len(hidden_units) - 1),
        paddings=[snt.SAME],
        activation=activation_fn,
        initializers=initializers,
        use_batch_norm=use_batch_norm,
        activate_final=True)(
            batch["image"], is_training=True)
    # average pool
    if pool == "avg":
      net = tf.reduce_mean(net, [1, 2])
    elif pool == "max":
      net = tf.reduce_max(net, [1, 2])
    else:
      raise ValueError("pool type not supported")

    num_classes = batch["label_onehot"].shape.as_list()[1]
    logits = snt.Linear(num_classes)(net)
    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=logits)
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed("FixedImageConv_cifar10_32x64x64_bs128")
def _():
  base_model_fn = ce_pool_loss([32, 64, 64], tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x64_tanh_bs64")
def _():
  base_model_fn = ce_pool_loss([32, 64, 64], tf.nn.tanh)
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_he_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = tf.initializers.he_normal()
  base_model_fn = ce_pool_loss([32, 64, 128],
                               tf.nn.tanh,
                               initializers=init,
                               pool="max")
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_normal_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = tf.initializers.random_normal(0, 0.02)
  base_model_fn = ce_pool_loss([32, 64, 128],
                               tf.nn.tanh,
                               initializers=init,
                               pool="avg")
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_largenormal_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = tf.initializers.random_normal(0, 0.2)
  base_model_fn = ce_pool_loss([32, 64, 128],
                               tf.nn.tanh,
                               initializers=init,
                               pool="avg")
  dataset = datasets.get_image_datasets("cifar10", 64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_smallnormal_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = tf.initializers.random_normal(0, 0.002)
  base_model_fn = ce_pool_loss([32, 64, 128],
                               tf.nn.tanh,
                               initializers=init,
                               pool="avg")
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128x128x128_avg_he_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = tf.initializers.he_normal()
  base_model_fn = ce_pool_loss([32, 64, 128, 128, 128],
                               tf.nn.tanh,
                               initializers=init,
                               pool="avg")
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


# TODO(lmetz) add imagenet sub task problems


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_batchnorm_32x64x64_bs128")
def _():  # pylint: disable=missing-docstring
  base_model_fn = ce_pool_loss([32, 64, 64], tf.nn.relu, use_batch_norm=True)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_batchnorm_32x32x32x64x64_bs128")
def _():  # pylint: disable=missing-docstring
  base_model_fn = ce_pool_loss([32, 32, 32, 64, 64],
                               tf.nn.relu,
                               use_batch_norm=True)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_food101_batchnorm_32x32x32x64x64_bs128")
def _():
  base_model_fn = ce_pool_loss([32, 32, 32, 64, 64],
                               tf.nn.relu,
                               use_batch_norm=True)
  dataset = datasets.get_image_datasets(
      "food101_32x32", batch_size=128, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar100_bn_32x64x128x128_bs128")
def _():
  base_model_fn = ce_pool_loss([32, 64, 128, 128],
                               tf.nn.relu,
                               use_batch_norm=True)
  dataset = datasets.get_image_datasets(
      "cifar100", batch_size=128, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_coil10032x32_bn_32x64x128x128_bs128")
def _():
  base_model_fn = ce_pool_loss([32, 64, 128, 128],
                               tf.nn.relu,
                               use_batch_norm=True)
  dataset = datasets.get_image_datasets(
      "coil100_32x32", batch_size=128, shuffle_buffer=5000, num_per_valid=800)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_sun39732x32_bn_32x64x128x128_bs128")
def convnetloss1():
  base_model_fn = ce_pool_loss([32, 64, 128, 128],
                               tf.nn.relu,
                               use_batch_norm=True)
  dataset = datasets.get_image_datasets(
      "sun397_32x32", batch_size=128, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)


def ce_flatten_loss(hidden_units,
                    activation_fn,
                    hidden_layers,
                    initializers=None):
  """Helper function to make a sonnet loss.

  This creates a cross entropy loss, conv net where the last conv layer is
  flattened and run through an MLP instead of pooled.

  Args:
    hidden_units: list list of hidden unit sizes
    activation_fn: callable activation function used in the convnet
    hidden_layers: list hidden layers of the classification MLP
    initializers: optional dict dictionary of initalizers used to initialize the
      convnet weights

  Returns:
    callable that returns a sonnet module representing the loss
  """
  if not initializers:
    initializers = {}

  def _fn(batch):
    """Build the loss."""
    net = snt.nets.ConvNet2D(
        hidden_units,
        kernel_shapes=[(3, 3)],
        strides=[2] + [1] * (len(hidden_units) - 1),
        paddings=[snt.SAME],
        activation=activation_fn,
        initializers=initializers,
        activate_final=True)(
            batch["image"])

    lastdims = int(np.prod(net.shape.as_list()[1:]))
    net = tf.reshape(net, [-1, lastdims])
    for s in hidden_layers:
      net = activation_fn(snt.Linear(s, initializers=initializers)(net))

    num_classes = batch["label_onehot"].shape.as_list()[1]
    logits = snt.Linear(num_classes, initializers=initializers)(net)
    loss_vec = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=batch["label_onehot"], logits=logits)
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x64_flatten_bs128")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [])
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x64_fc_64_bs128")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [64])

  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_mnist_32x64x64_fc_64_bs128")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [64])
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "Mnist_Conv_32x16x64_flatten_FC32_tanh_bs32")
def _():
  base_model_fn = ce_flatten_loss([32, 16, 64], tf.nn.tanh, [32])
  dataset = datasets.get_image_datasets("mnist", batch_size=32)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_he_bs8")
def _():
  init = {}
  init["w"] = tf.initializers.he_normal()
  base_model_fn = ce_flatten_loss([32, 64, 128],
                                  tf.nn.tanh, [64, 32],
                                  initializers=init)
  dataset = datasets.get_image_datasets("cifar10", batch_size=8)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_variance_scaling_bs64"
)
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = contrib_layers.variance_scaling_initializer()
  base_model_fn = ce_flatten_loss([32, 64, 128],
                                  tf.nn.tanh, [64, 32],
                                  initializers=init)
  dataset = datasets.get_image_datasets("cifar10", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar100_32x64x128_FC64x32_tanh_variance_scaling_bs64")
def _():  # pylint: disable=missing-docstring
  init = {}
  init["w"] = contrib_layers.variance_scaling_initializer()
  base_model_fn = ce_flatten_loss([32, 64, 128],
                                  tf.nn.tanh, [64, 32],
                                  initializers=init)
  dataset = datasets.get_image_datasets("cifar100", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_cifar100_32x64x64_flatten_bs128")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [])
  dataset = datasets.get_image_datasets("cifar100", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_food10164x64_Conv_32x64x64_flatten_bs64")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [])
  dataset = datasets.get_image_datasets(
      "food101_64x64", batch_size=64, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConv_colorectalhistology32x32_32x64x64_flatten_bs128")
def _():
  base_model_fn = ce_flatten_loss([32, 64, 64], tf.nn.relu, [])
  dataset = datasets.get_image_datasets(
      "colorectal_histology_32x32",
      batch_size=128,
      shuffle_buffer=5000,
      num_per_valid=700)
  return base.DatasetModelTask(base_model_fn, dataset)
