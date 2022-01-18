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
"""Fixed convolutional variational autoencoders."""

import numpy as np
import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import generative_utils
import tensorflow.compat.v1 as tf


def conv_vae_loss_fn(enc_units, dec_units, n_z, activation_fn):
  """Convolutional variational autoencoder loss module helper.

  This creates a callable that returns a sonnet module for the loss.

  Args:
    enc_units: list list of interegers containing the encoder convnet number of
      units
    dec_units: list list of interegers containing the decoder convnet number of
      units
    n_z: int size of the middle layer of the autoencoder
    activation_fn: callable activation function used in the convnet

  Returns:
    callable that returns a sonnet module representing the loss
  """

  def _fn(batch):
    """Build the loss."""
    if batch["image"].shape.as_list()[1] == 28:
      shapes = [(14, 14), (28, 28)]
    elif batch["image"].shape.as_list()[1] == 32:
      shapes = [(16, 16), (32, 32)]
    else:
      raise ValueError("Only 28x28, or 32x32 supported")

    def encoder_fn(net):
      """Encoder for VAE."""
      net = snt.nets.ConvNet2D(
          enc_units,
          kernel_shapes=[(3, 3)],
          strides=[2, 2],
          paddings=[snt.SAME],
          activation=activation_fn,
          activate_final=True)(
              net)

      flat_dims = int(np.prod(net.shape.as_list()[1:]))
      net = tf.reshape(net, [-1, flat_dims])
      net = snt.Linear(2 * n_z)(net)
      return generative_utils.LogStddevNormal(net)

    encoder = snt.Module(encoder_fn, name="encoder")

    def decoder_fn(net):
      """Decoder for VAE."""
      if batch["image"].shape.as_list()[1] == 28:
        net = snt.Linear(7 * 7 * 32)(net)
        net = tf.reshape(net, [-1, 7, 7, 32])
      elif batch["image"].shape.as_list()[1] == 32:
        net = snt.Linear(8 * 8 * 32)(net)
        net = tf.reshape(net, [-1, 8, 8, 32])
      else:
        raise ValueError("Only 32x32 or 28x28 supported!")

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
      net = snt.Conv2D(2 * outchannel, kernel_shape=(1, 1))(net)
      net = tf.clip_by_value(net, -10, 10)

      return generative_utils.QuantizedNormal(mu_log_sigma=net)

    decoder = snt.Module(decoder_fn, name="decoder")

    zshape = tf.stack([tf.shape(batch["image"])[0], 2 * n_z])
    prior = generative_utils.LogStddevNormal(tf.zeros(shape=zshape))

    input_image = (batch["image"] - 0.5) * 2

    log_p_x, kl_term = generative_utils.log_prob_elbo_components(
        encoder, decoder, prior, input_image)

    elbo = log_p_x - kl_term

    metrics = {
        "kl_term": tf.reduce_mean(kl_term),
        "log_kl_term": tf.log(tf.reduce_mean(kl_term)),
        "log_p_x": tf.reduce_mean(log_p_x),
        "elbo": tf.reduce_mean(elbo),
        "log_neg_log_p_x": tf.log(-tf.reduce_mean(elbo))
    }

    return base.LossAndAux(-tf.reduce_mean(elbo), metrics)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_mnist_32x32x32x32x32_bs128")
def _():
  base_model_fn = conv_vae_loss_fn([32, 32], [32, 32], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_mnist_32x64x32x64x32_bs128")
def _():
  base_model_fn = conv_vae_loss_fn([32, 64], [64, 32], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_cifar10_32x64x128x64x32_bs128")
def _():
  base_model_fn = conv_vae_loss_fn([32, 64], [64, 32], 128, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_mnist_64x128x128x128x64_bs128")
def _():
  base_model_fn = conv_vae_loss_fn([64, 128], [128, 64], 128, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


def three_layer_conv_vae_loss_fn(enc_units, dec_units, n_z, activation_fn):
  """Convolutional variational autoencoder loss module helper.

  This creates a callable that returns a sonnet module for the loss.

  Args:
    enc_units: list list of interegers containing the encoder convnet number of
      units
    dec_units: list list of interegers containing the decoder convnet number of
      units
    n_z: int size of the middle layer of the autoencoder
    activation_fn: callable activation function used in the convnet

  Returns:
    callable that returns a sonnet module representing the loss
  """

  def _fn(batch):
    """Build the loss."""
    shapes = [(8, 8), (16, 16), (32, 32)]

    def encoder_fn(net):
      """Encoder for VAE."""
      net = snt.nets.ConvNet2D(
          enc_units,
          kernel_shapes=[(3, 3)],
          strides=[2, 2, 2],
          paddings=[snt.SAME],
          activation=activation_fn,
          activate_final=True)(
              net)

      flat_dims = int(np.prod(net.shape.as_list()[1:]))
      net = tf.reshape(net, [-1, flat_dims])
      net = snt.Linear(2 * n_z)(net)
      return generative_utils.LogStddevNormal(net)

    encoder = snt.Module(encoder_fn, name="encoder")

    def decoder_fn(net):
      """Decoder for VAE."""
      net = snt.Linear(4 * 4 * 32)(net)
      net = tf.reshape(net, [-1, 4, 4, 32])
      net = snt.nets.ConvNet2DTranspose(
          dec_units,
          shapes,
          kernel_shapes=[(3, 3)],
          strides=[2, 2, 2],
          paddings=[snt.SAME],
          activation=activation_fn,
          activate_final=True)(
              net)

      outchannel = batch["image"].shape.as_list()[3]
      net = snt.Conv2D(2 * outchannel, kernel_shape=(1, 1))(net)
      net = tf.clip_by_value(net, -10, 10)

      return generative_utils.QuantizedNormal(mu_log_sigma=net)

    decoder = snt.Module(decoder_fn, name="decoder")

    zshape = tf.stack([tf.shape(batch["image"])[0], 2 * n_z])
    prior = generative_utils.LogStddevNormal(tf.zeros(shape=zshape))

    input_image = (batch["image"] - 0.5) * 2

    log_p_x, kl_term = generative_utils.log_prob_elbo_components(
        encoder, decoder, prior, input_image)

    elbo = log_p_x - kl_term

    metrics = {
        "kl_term": tf.reduce_mean(kl_term),
        "log_kl_term": tf.log(tf.reduce_mean(kl_term)),
        "log_p_x": tf.reduce_mean(log_p_x),
        "elbo": tf.reduce_mean(elbo),
        "log_neg_log_p_x": tf.log(-tf.reduce_mean(elbo))
    }

    return base.LossAndAux(-tf.reduce_mean(elbo), metrics)

  return lambda: snt.Module(_fn)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_cifar10_64x128x256x128x256x128x64_bs128")
def _():
  base_model_fn = three_layer_conv_vae_loss_fn([64, 128, 256], [256, 128, 64],
                                               128, tf.nn.relu)

  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs128")
def _():
  base_model_fn = three_layer_conv_vae_loss_fn([32, 64, 128], [128, 64, 32], 64,
                                               tf.nn.relu)

  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs512")
def _():
  base_model_fn = three_layer_conv_vae_loss_fn([32, 64, 128], [128, 64, 32], 64,
                                               tf.nn.relu)

  dataset = datasets.get_image_datasets("cifar10", batch_size=512)
  return base.DatasetModelTask(base_model_fn, dataset)
