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

"""A variational autoencoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FLAGS = flags.FLAGS



class VAE(tf.keras.Model):
  """A variational autoencoder."""

  def __init__(self, conv_dims, conv_sizes):
    super(VAE, self).__init__()
    self.conv_dims = conv_dims
    self.conv_sizes = conv_sizes
    self.mean_conv_layers = []
    self.stddev_conv_layers = []
    self.deconv_layers = []
    self.pz = None

    for dim, size in zip(conv_dims, conv_sizes):
      self.mean_conv_layers.append(
          tf.keras.layers.Conv2D(dim, [size] * 2, activation=tf.nn.tanh))
      self.stddev_conv_layers.append(
          tf.keras.layers.Conv2D(dim, [size] * 2, activation=tf.nn.tanh))
      self.deconv_layers.append(
          tf.keras.layers.Conv2DTranspose(
              dim, [size] * 2, activation=tf.nn.tanh))
    # Layers are appended and then reversed because insertion is not supported
    # by tf.Checkpoint
    self.deconv_layers = self.deconv_layers[::-1]
    self.deconv_layers.append(tf.keras.layers.Conv2D(1, [1, 1]))

  def encode(self, inputs):
    """Encode inputs to a latent representation.

    Args:
      inputs: input batch we wish to encode.
    Returns:
      qz: the posterior distribution in latent space given inputs.

    """
    u = inputs
    for conv in self.mean_conv_layers:
      u = conv(u)
    v = inputs
    for conv in self.stddev_conv_layers:
      v = conv(v)
    v = tf.math.softplus(v)
    qz = tfd.Normal(u, v)
    return qz

  def decode(self, enc):
    """Decode some latent value to input space.

    Args:
      enc: some values in latent space.
    Returns:
      y: the decoded version of enc in input space.

    """
    y = enc
    for deconv in self.deconv_layers:
      y = deconv(y)
    return y

  def generate(self):
    """Generate a random value in input space.

    Returns:
      y: random samples in input space.

    """
    if self.pz is None:
      raise ValueError('Must call `call` before `generate`.')
    smp = self.pz.sample()
    y = self.decode(smp)
    return y

  def call(self, inputs):
    """Run VAE on some inputs - encode, decode, and calculate KL loss.

    Args:
      inputs: input batch we wish to run on.
    Returns:
      y: reconstructed version of inputs.
      div: KL divergence loss of the inferred posterior from these inputs.

    """
    qz = self.encode(inputs)
    enc = qz.sample()

    if self.pz is None:
      self.pz = tfd.Normal(
          loc=tf.zeros_like(enc), scale=tf.ones_like(enc))

    y = self.decode(enc)
    div = qz.kl_divergence(self.pz)
    return y, div

  def get_loss(self, batch, beta=1.):
    """Run VAE on input batch and return loss (reconstruction + KL).

    Args:
      batch (tensor): input batch we wish to run on.
      beta (float): KL weighting hyperparameter.
    Returns:
      total_loss (tensor): ELBO for each element in batch.
      div_loss (tensor): KL loss for each element in batch.
      rec_loss (tensor): reconstruction loss for each element in batch.
    """
    y, div = self.call(batch)
    div_loss = beta * tf.reduce_sum(div, axis=[1, 2, 3])
    rec_loss = tf.reduce_sum(bernoulli_logprob(batch, y),
                             axis=[1, 2, 3])
    total_loss = rec_loss + div_loss
    return total_loss, div_loss, rec_loss


def bernoulli_logprob(lbl, y):
  return -tfd.Bernoulli(logits=y).log_prob(lbl)

