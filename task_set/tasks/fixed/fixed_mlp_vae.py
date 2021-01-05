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

"""Fixed tasks for MLP based variational autoencoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt

from task_set import datasets
from task_set import registry
from task_set.tasks import base
from task_set.tasks import generative_utils
import tensorflow.compat.v1 as tf


def fc_vae_loss_fn(enc_units, dec_units, n_z, activation):
  """Variational autoencoder loss with MLPs helper.

  This creates a callable that returns a sonnet module for the loss.

  Args:
    enc_units: list list of interegers containing the encoder number of units
      per layer
    dec_units: list list of interegers containing the decoder number of units
      per layer
    n_z: int size of the middle layer of the autoencoder
    activation: callable activation function used in the convnet

  Returns:
    callable that returns a sonnet module representing the loss
  """

  def _build(batch):
    """Build the sonnet module."""
    net = snt.BatchFlatten()(batch["image"])
    # shift to be zero mean
    net = (net - 0.5) * 2

    n_inp = net.shape.as_list()[1]

    def encoder_fn(x):
      mlp_encoding = snt.nets.MLP(
          name="mlp_encoder",
          output_sizes=enc_units + [2 * n_z],
          activation=activation)
      return generative_utils.LogStddevNormal(mlp_encoding(x))

    encoder = snt.Module(encoder_fn, name="encoder")

    def decoder_fn(x):
      mlp_decoding = snt.nets.MLP(
          name="mlp_decoder",
          output_sizes=dec_units + [2 * n_inp],
          activation=activation)
      net = mlp_decoding(x)
      net = tf.clip_by_value(net, -10, 10)
      return generative_utils.QuantizedNormal(mu_log_sigma=net)

    decoder = snt.Module(decoder_fn, name="decoder")

    zshape = tf.stack([tf.shape(net)[0], 2 * n_z])

    prior = generative_utils.LogStddevNormal(tf.zeros(shape=zshape))

    log_p_x, kl_term = generative_utils.log_prob_elbo_components(
        encoder, decoder, prior, net)

    elbo = log_p_x - kl_term

    metrics = {
        "kl_term": tf.reduce_mean(kl_term),
        "log_kl_term": tf.log(tf.reduce_mean(kl_term)),
        "log_p_x": tf.reduce_mean(log_p_x),
        "elbo": tf.reduce_mean(elbo),
        "log_neg_log_p_x": tf.log(-tf.reduce_mean(elbo))
    }

    return base.LossAndAux(-tf.reduce_mean(elbo), metrics)

  return lambda: snt.Module(_build)


@registry.task_registry.register_fixed("FixedMLPVAE_cifar101_128x32x128_bs128")
def _():
  base_model_fn = fc_vae_loss_fn([128], [128], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedMLPVAE_cifar101_128x128x32x128x128_bs128")
def _():
  base_model_fn = fc_vae_loss_fn([128, 128], [128, 128], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("cifar10", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMLPVAE_mnist_128x8x128x128_bs128")
def _():
  base_model_fn = fc_vae_loss_fn([128], [128, 128], 8, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMLPVAE_mnist_128x128x8x128_bs128")
def _():
  base_model_fn = fc_vae_loss_fn([128, 128], [128], 8, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=128)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed("FixedMLPVAE_mnist_128x64x32x64x128_bs64"
                                      )
def _():
  base_model_fn = fc_vae_loss_fn([128, 64], [64, 128], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets("mnist", batch_size=64)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "FixedMLPVAE_food10132x32_128x64x32x64x128_bs64")
def _():
  base_model_fn = fc_vae_loss_fn([128, 64], [64, 128], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets(
      "food101_32x32", batch_size=64, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)


@registry.task_registry.register_fixed(
    "Imagenet32x30_FC_VAE_128x64x32x64x128_relu_bs256")
def _():
  base_model_fn = fc_vae_loss_fn([128, 64], [64, 128], 32, tf.nn.relu)
  dataset = datasets.get_image_datasets(
      "food101_32x32", batch_size=256, shuffle_buffer=5000)
  return base.DatasetModelTask(base_model_fn, dataset)
