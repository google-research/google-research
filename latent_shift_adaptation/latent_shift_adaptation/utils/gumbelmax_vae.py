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

"""Implementation of Gumbel-Max VAE.

Original Paper: Jang, Eric, Shixiang Gu, and Ben Poole.
  "Categorical reparameterization with gumbel-softmax." ICLR 2017.
"""

import numpy as np
import tensorflow as tf


EPS = 1e-5


class Sampling(tf.keras.layers.Layer):
  """Keras layer that samples the latent variable z.

  The Gumbel(0, 1) distribution can be sampled using inverse transform sampling
  by drawing u ~ Uniform(0, 1) and computing g = -log(-log(u)).
  """

  def call(self, logits, temp):
    sample = tf.keras.backend.random_uniform(
        tf.shape(logits), minval=0, maxval=1
    )
    noise = -tf.math.log(-tf.math.log(sample + EPS) + EPS)
    return (logits + noise) / temp


class GumbelMaxVAE(tf.keras.Model):
  """Implementation of Gumbel-Max VAE."""

  def __init__(
      self,
      encoder,
      decoder,
      temp=1.0,
      temp_anneal=0.9999,
      kl_loss_coef=1.0,
      min_temp=1e-2,
      var_x=1.0,
      dims=None,
      pos=None,
      **kwargs,
  ):
    """Constructor.

    Args:
      encoder: Keras encoder from input to latent space.
      decoder: Keras decoder from latent space to input.
      temp: Initial value of the temperature parameter.
      temp_anneal: Temperature annealing parameter. After each training step,
        the temperature is reduced by a factor of temp_anneal.
      kl_loss_coef: a tradeoff parameter between reconstruction loss and kl loss
      min_temp: minimum temperature.
      var_x: variance of x.
      dims: ConfigDict that specifies the dimensions of x, y, c, w.
      pos: ConfigDict that specifies the index of x, y, c, w, u in data tuple.
      **kwargs: any additional arguments when constructing the keras model.
    """
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder
    self.temp_anneal = temp_anneal
    self.temp = tf.Variable(
        temp, dtype=tf.float32, name="temp", trainable=False
    )
    self.min_temp = min_temp
    self.kl_loss_coef = kl_loss_coef
    self.var_x = tf.Variable(
        initial_value=var_x, name="var_x", trainable=True, dtype=tf.float32
    )

    self.latent_dim = np.prod(encoder.output_shape[1:])
    self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
        name="reconstruction_loss"
    )
    self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    self.x_loss_tracker = tf.keras.metrics.Mean(name="x_loss")
    self.w_loss_tracker = tf.keras.metrics.Mean(name="w_loss")
    self.c_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
    self.y_loss_tracker = tf.keras.metrics.Mean(name="y_loss")

    self.log_prior = -np.log(self.latent_dim)
    self.dims = dims
    self.pos = pos

  @property
  def metrics(self):
    return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
        self.x_loss_tracker,
        self.y_loss_tracker,
        self.c_loss_tracker,
        self.w_loss_tracker,
    ]

  def _parse_data(self, data):
    indices = {}
    pos_list = sorted(self.pos.items(), key=lambda item: item[1])  # sort by pos
    pos_so_far = 0
    for v, _ in pos_list:  # v is 'x', 'y', etc. p is its position
      indices[v] = (pos_so_far, pos_so_far + self.dims[v])
      pos_so_far += self.dims[v]
    x = data[:, indices["x"][0] : indices["x"][1]]
    y = data[:, indices["y"][0] : indices["y"][1]]
    c = data[:, indices["c"][0] : indices["c"][1]]
    w = data[:, indices["w"][0] : indices["w"][1]]
    return x, y, c, w

  def train_step(self, data):
    with tf.GradientTape() as tape:
      u_logits = self.encoder(data)
      batch_len = tf.shape(u_logits)[0]
      u_logits = tf.reshape(u_logits, [batch_len, -1])  # reshape u_logits
      pu = tf.math.softmax(u_logits)  # prob over categorical latent space
      # add Gumbel noise
      u = Sampling()(u_logits, self.temp)
      u = tf.math.softmax(u)

      # reconstruction loss
      reconstruction = self.decoder(u)
      x_rec, y_rec, c_rec, w_rec = self._parse_data(reconstruction)
      y_rec = tf.math.softmax(y_rec)
      w_rec = tf.math.softmax(w_rec)
      c_rec = tf.math.sigmoid(c_rec)

      x, y, c, w = self._parse_data(data)
      x_dim = tf.cast(tf.shape(x)[1], tf.float32)
      c_dim = tf.cast(tf.shape(c)[1], tf.float32)

      # remember: we have a batch
      ex_x_loss = tf.math.log(self.var_x) + 1.0 / self.var_x * tf.reduce_mean(
          tf.keras.losses.mean_squared_error(x, x_rec)
      )
      # multiply by n because x_loss is avrg over dimensions
      x_loss = ex_x_loss * 0.5 * x_dim

      w_loss = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(w, w_rec)
      )

      # multiply by c_dim because binary_crossentropy is avrg over all labels
      c_loss = (
          tf.reduce_mean(
              tf.keras.losses.binary_crossentropy(c, c_rec, from_logits=False)
          )
          * c_dim
      )

      y_loss = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(y, y_rec)
      )

      # kl loss
      log_p = tf.math.log(pu + EPS)  # add EPS to avoid log(zero)
      kl_loss = tf.math.multiply(pu, log_p - self.log_prior)
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

      reconstruction_loss = x_loss + w_loss + c_loss + y_loss

      # total loss
      total_loss = reconstruction_loss + self.kl_loss_coef * kl_loss

    # gradient descent
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    # anneal temp
    self.temp.assign(
        tf.math.maximum(self.min_temp, self.temp * self.temp_anneal)
    )

    # track metrics
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    self.x_loss_tracker.update_state(x_loss)
    self.y_loss_tracker.update_state(y_loss)
    self.c_loss_tracker.update_state(c_loss)
    self.w_loss_tracker.update_state(w_loss)
    return {
        "loss": self.total_loss_tracker.result(),
        "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        "kl_loss": self.kl_loss_tracker.result(),
        "x_loss": self.x_loss_tracker.result(),
        "c_loss": self.c_loss_tracker.result(),
        "w_loss": self.w_loss_tracker.result(),
        "y_loss": self.y_loss_tracker.result(),
    }
