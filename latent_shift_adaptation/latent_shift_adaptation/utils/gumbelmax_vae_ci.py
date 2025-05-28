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

from latent_shift_adaptation.utils import gumbelmax_vae
import tensorflow as tf


EPS = 1e-5


def mlp(input_shape, width, num_classes):
  """Multilabel Classification."""
  model_input = tf.keras.Input(shape=input_shape)
  regularizer = tf.keras.regularizers.L2(1e-6)
  # hidden layer
  if width:
    x = tf.keras.layers.Dense(
        width, use_bias=True, activation="relu", kernel_regularizer=regularizer
    )(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
  else:
    x = model_input
  # sigmoid because it is multilabel
  model_outuput = tf.keras.layers.Dense(
      num_classes,
      use_bias=True,
      kernel_regularizer=regularizer,
      activation="linear",
  )(
      x
  )  # get logits
  model = tf.keras.models.Model(model_input, model_outuput)
  model.build(input_shape)

  return model


class GumbelMaxVAECI(gumbelmax_vae.GumbelMaxVAE):
  """Implementation of Gumbel-Max VAE."""

  def __init__(
      self,
      encoder,
      width,
      latent_dim,
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

    We assume that the data is in the form (X, Y, C, W, U).

    Args:
      encoder: Keras encoder from input to latent space.
      width: Width of the hidden layer in each decoder.
      latent_dim: dimension of the latent space
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
    super(GumbelMaxVAECI, self).__init__(
        encoder,
        None,
        temp,
        temp_anneal,
        kl_loss_coef,
        min_temp,
        var_x,
        dims,
        pos,
        **kwargs,
    )

    x_dim, y_dim, c_dim, w_dim = dims["x"], dims["y"], dims["c"], dims["w"]
    self.decoder_u2x = mlp((latent_dim,), width, x_dim)
    self.decoder_u2w = mlp((latent_dim,), width, w_dim)
    self.decoder_ux2c = mlp((latent_dim + x_dim,), width, c_dim)
    self.decoder_uc2y = mlp((latent_dim + c_dim,), width, y_dim)

    self.dims = dims
    self.pos = pos

  def train_step(self, data):
    with tf.GradientTape() as tape:
      x, y, c, w = self._parse_data(data)
      x_dim = tf.cast(tf.shape(x)[1], tf.float32)
      c_dim = tf.cast(tf.shape(c)[1], tf.float32)

      u_logits = self.encoder(data)
      batch_len = tf.shape(u_logits)[0]
      u_logits = tf.reshape(u_logits, [batch_len, -1])  # reshape u_logits
      pu = tf.math.softmax(u_logits)  # prob over categorical latent space
      # add Gumbel noise
      u = gumbelmax_vae.Sampling()(u_logits, self.temp)
      u = tf.math.softmax(u)

      # reconstruction loss
      x_rec = self.decoder_u2x(u)

      w_rec = self.decoder_u2w(u)
      w_rec = tf.math.softmax(w_rec)  # one-hot encoded so use softmax

      x_and_u = tf.concat([x, u], axis=1)
      c_rec = self.decoder_ux2c(x_and_u)
      c_rec = tf.math.sigmoid(c_rec)  # multi-label

      c_and_u = tf.concat([c, u], axis=1)
      y_rec = self.decoder_uc2y(c_and_u)
      y_rec = tf.math.softmax(y_rec)  # one-hot encoded so use softmax

      ex_x_loss = tf.math.log(self.var_x) + 1.0 / self.var_x * tf.reduce_mean(
          tf.keras.losses.mean_squared_error(x, x_rec)  # x loss per example
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
      pu = tf.reduce_mean(pu, axis=0)
      log_p = tf.math.log(pu + EPS)
      kl_loss = tf.math.multiply(pu, log_p - self.log_prior)
      kl_loss = tf.reduce_sum(kl_loss)

      # total loss
      reconstruction_loss = x_loss + w_loss + c_loss + y_loss
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
