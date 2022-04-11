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

"""Fixed convolutional autoencoder tasks."""
import numpy as np
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
import tensorflow.compat.v1 as tf


def conv_ae_loss_fn(enc_units, dec_units, num_latents, activation_fn):
  """Convolutional autoencoder loss module helper.

  This creates a callable that returns a sonnet module for the loss.

  Args:
    enc_units: list list of integers containing the encoder convnet number of
      units
    dec_units: list list of integers containing the decoder convnet number of
      units
    num_latents: int size of the middle layer of the autoencoder
    activation_fn: callable activation function used in the convnet

  Returns:
    callable that returns a sonnet module representing the loss
  """

  def _fn(batch):
    """Make the loss from the given batch."""
    net = batch["image"]
    net = snt.nets.ConvNet2D(
        enc_units,
        kernel_shapes=[(3, 3)],
        strides=[2, 2],
        paddings=[snt.SAME],
        activation=activation_fn,
        activate_final=True)(
            batch["image"])

    flat_dims = int(np.prod(net.shape.as_list()[1:]))
    net = tf.reshape(net, [-1, flat_dims])
    net = snt.Linear(num_latents)(net)

    if batch["image"].shape.as_list()[1] == 28:
      net = snt.Linear(7 * 7 * 32)(net)
      net = tf.reshape(net, [-1, 7, 7, 32])
      shapes = [(14, 14), (28, 28)]
    elif batch["image"].shape.as_list()[1] == 32:
      net = snt.Linear(8 * 8 * 32)(net)
      net = tf.reshape(net, [-1, 8, 8, 32])
      shapes = [(16, 16), (32, 32)]
    else:
      raise ValueError("Only 28x28, or 32x32 supported")

    net = snt.nets.ConvNet2DTranspose(
        dec_units,
        shapes,
        kernel_shapes=[(3, 3)],
        strides=[2, 2],
        paddings=[snt.SAME],
        activation=activation_fn,
        activate_final=True)(
            net)

    outchannel = batch["image"].shape.as_list()[3]
    net = snt.Conv2D(outchannel, kernel_shape=(1, 1))(net)

    loss_vec = tf.reduce_mean(
        tf.square(batch["image"] - tf.nn.sigmoid(net)), [1, 2, 3])
    return tf.reduce_mean(loss_vec)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedImageConvAE_mnist_32x32x32x32x32_bs128")
def _():
  base_model_fn = conv_ae_loss_fn([32, 32], [32, 32], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvAE_mnist_32x64x8x64x32_bs128")
def _():
  base_model_fn = conv_ae_loss_fn([32, 64], [64, 32], 8, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvAE_mnist_32x64x32x64x32_bs512")
def _():
  base_model_fn = conv_ae_loss_fn([32, 64], [64, 64], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=512)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvAE_cifar10_32x32x32x32x32_bs128")
def _():
  base_model_fn = conv_ae_loss_fn([32, 32], [32, 32], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvAE_cifar10_32x64x8x64x32_bs128")
def _():
  base_model_fn = conv_ae_loss_fn([32, 64], [64, 32], 8, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)
