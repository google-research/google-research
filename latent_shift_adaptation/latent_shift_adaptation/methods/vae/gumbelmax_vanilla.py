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

"""Gumbel-Max VAE approach.

Similar to BBSEZ with Z=U, except that we infer U instead of observing it.

At a high level, the algorithm follows five steps:
1. Fit a VAE on source data to produce a model p(U|data).
2. For every example in the source data, append a sample of U.
  So, we now have (X, Y, C, W, U) in the source domain.
3. Fit models (e.g. logistic regressions) to obtain p(U|X) and p(Y|X, U).
4. Using label correction methods, calculuate weights q(u)/p(u).
5. Predict q(Y|X) by marginalizing over u.
    q(Y | X) ~ SUM_u p(Y | X, U) p(U | X) q(U) / p(U)
"""

from latent_shift_adaptation.methods import baseline
from latent_shift_adaptation.utils import gumbelmax_vae
from latent_shift_adaptation.utils import temp_scaling
import numpy as np
import tensorflow as tf


class Method(baseline.Method):
  """Gumbel Max VAE approach."""

  def __init__(
      self,
      vae_encoder,
      vae_decoder,
      vae_opt,
      model_x2u,
      model_xu2y,
      kl_loss_coef=1.0,
      vae_temp=1.0,
      temp_anneal=0.9999,
      min_temp=1e-2,
      num_classes=2,
      var_x=1.0,
      vae_inputs="xycw",
      evaluate=None,
      dtype=tf.float32,
      dims=None,
      pos=None,
  ):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      vae_encoder: Encoder (X, Y, C, W) -> U.
      vae_decoder: Decoder U -> (X, Y, C, W).
      vae_opt: optimizer used to train VAE.
      model_x2u: Model p(U | X).
      model_xu2y: Model p(Y | X, U).
      kl_loss_coef: a tradeoff parameter between reconstruction loss and kl loss
      vae_temp: Initial value of the temperature parameter in Gumbel-Max VAE.
      temp_anneal: Temperature annealing parameter in Gumbel-Max VAE. After each
        training step, the temperature is reduced by a factor of temp_anneal.
      min_temp: minimum temperature in Gumbel-Max VAE.
      num_classes: Number of target classes Y
      var_x: variance of x.
      vae_inputs: specify here the inputs used to predict u in the VAE
      evaluate: a tf.keras.metrics method.
      dtype: desired dtype (e.g. tf.float32).
      dims: ConfigDict that specifies the dimensions of x, y, c, w.
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(evaluate, dtype, pos)

    # models
    linear_act = tf.keras.activations.linear
    # pylint: disable=g-backslash-continuation
    if (
        vae_encoder.layers[-1].activation != linear_act
        or vae_decoder.layers[-1].activation != linear_act
        or model_x2u.layers[-1].activation != linear_act
        or model_xu2y.layers[-1].activation != linear_act
    ):
      raise ValueError(  # pylint: disable=bad-indentation
          "The four models encoder, decoder, model_x2u, model_xu2y "
          "must output logits, not probability scores."
      )

    self.model_x2u = model_x2u
    self.model_xu2y = model_xu2y
    # store initial lr to avoid issues with keras lr scheduling callbacks
    self.start_lr_x2u = self.model_x2u.optimizer.learning_rate.numpy()
    self.start_lr_xu2y = self.model_xu2y.optimizer.learning_rate.numpy()

    self.num_classes = num_classes

    # VAE
    self.vae = gumbelmax_vae.GumbelMaxVAE(
        vae_encoder,
        vae_decoder,
        temp=vae_temp,
        temp_anneal=temp_anneal,
        kl_loss_coef=kl_loss_coef,
        min_temp=min_temp,
        var_x=var_x,
        dims=dims,
        pos=pos,
    )

    self.vae.compile(optimizer=vae_opt)
    self.latent_dim = self.vae.latent_dim  # dim of latent space

    # specify here the inputs used to predict u in VAE
    self.vae_inputs = vae_inputs
    self.inputs = "x"  # model input that's available to make a prediction
    self.outputs = "y"

  def predict_u_from_all(self, *batch, dtype=tf.float32):
    """Predict u from all other variables using trained VAE."""
    eps = 1e-10
    model_input = self.get_input(*batch, inputs=self.vae_inputs)
    u_logits = self.vae.encoder(model_input)

    # use gumbel-max trick to sample from categorical distribution
    sample = tf.keras.backend.random_uniform(
        tf.shape(u_logits), minval=0, maxval=1
    )
    noise = -tf.math.log(-tf.math.log(sample + eps) + eps)
    noisy_u_logits = u_logits + noise
    u_sample = tf.math.argmax(noisy_u_logits, axis=1)
    return tf.one_hot(u_sample, self.latent_dim, dtype=dtype)

  def predict_u_from_x(self, *batch):
    """Predict u from x."""
    model_input = self.get_input(*batch, inputs="x")
    u_logits = self.model_x2u.predict(model_input)
    return tf.math.softmax(u_logits)

  def x2u(self, *batch):
    """Generate (x, u) data to train p(u|x)."""
    u = self.predict_u_from_all(*batch)
    x = self.get_input(*batch, inputs="x")
    return x, u

  def xu2y(self, *batch):
    """Inputs = (x, u). Outputs = y."""
    u = self.predict_u_from_all(*batch)
    x = self.get_input(*batch, inputs="x")
    y = self.get_output(*batch, outputs="y")
    return tf.concat([x, u], axis=1), y  # stack x with u

  def _get_freq_ratio(self, data_source_val, data_target, num_batches):
    """Apply label correction to get q(u)/p(u) using validation data."""
    # calculate confusion matrix in source p
    u_true, u_pred = self._predict_u_mult(data_source_val, num_batches)
    confusion_matrix = (
        np.sum(
            [
                np.outer(u_true[i, :], u_pred[i, :])
                for i in range(u_pred.shape[0])
            ],
            axis=0,
        )
        / u_pred.shape[0]
    )
    confusion_matrix = tf.cast(
        confusion_matrix.transpose(), tf.float64
    )  # tf.float64, otherwise error!

    # calculate mu (mean prediction) in target domain
    _, u_pred_traget = self._predict_u_mult(data_target, num_batches)
    mu = (
        np.mean(u_pred_traget, axis=0)
        .reshape((self.latent_dim, 1))
        .astype(np.float64)
    )

    # now, calculate weights
    weights = tf.squeeze(tf.linalg.pinv(confusion_matrix) @ tf.constant(mu))
    weights = np.minimum(np.maximum(1e-2, weights.numpy()), 15)  # clip weights
    class_weight = {i: weights[i] for i in range(self.latent_dim)}
    return class_weight  # weight here is frequency ratio q(u) / p(u)

  def fit(
      self,
      data_source_train,
      data_source_val,
      data_target,
      steps_per_epoch_val,
      **fit_kwargs,
  ):
    """Fit model on data. See high-level description above."""
    # verify data assumptions (e.g. shape/range)
    batch = next(iter(data_source_train))
    self.assertions(*batch)

    # fit VAE to get p(u | all)
    self.inputs = self.vae_inputs
    ds = data_source_train.map(self.get_input)
    self.vae.fit(ds, **fit_kwargs)
    self.vae.trainable = False
    self.inputs = "x"

    # fit p(u|x) in source
    tf.keras.backend.set_value(
        self.model_x2u.optimizer.learning_rate, self.start_lr_x2u
    )
    ds_x2u = data_source_train.map(self.x2u)
    self.model_x2u.fit(ds_x2u, **fit_kwargs)  # gives logits
    self._calibrate(data_source_val, "x2u", **fit_kwargs)
    self.model_x2u.trainable = False

    # estimate q(u) / p(u) using the confusion matrix
    self.freq_ratio = self._get_freq_ratio(
        data_source_val, data_target, steps_per_epoch_val
    )

    # fit p(y|x,u).
    tf.keras.backend.set_value(
        self.model_xu2y.optimizer.learning_rate, self.start_lr_xu2y
    )
    ds_xu2y = data_source_train.map(self.xu2y)
    self.model_xu2y.fit(ds_xu2y, **fit_kwargs)
    self._calibrate(data_source_val, "xu2y", **fit_kwargs)
    self.model_xu2y.trainable = False

  def predict(self, model_input, **kwargs):
    """Predict target Y given x by marginalizing over u with correction."""

    def _append_u(x, u_val):
      """Return [x, u] with u fixed to the chosen value."""
      batch_len = x.shape[0]
      x = tf.cast(x, self.dtype)
      u = tf.keras.utils.to_categorical(
          u_val * np.ones((batch_len,)), self.latent_dim
      )
      u = tf.cast(u, self.dtype)
      stack = tf.concat([x, u], axis=1)
      return stack

    # initialize predictions to zero
    batch_len = model_input.shape[0]
    result_temp = np.zeros((batch_len, self.num_classes, self.latent_dim))

    # obtain p(u | x)
    u_logits_all = self.model_x2u.predict(model_input)
    pu_all = tf.math.softmax(u_logits_all).numpy()

    # marginalize over u
    for category in range(self.latent_dim):
      input_xu = _append_u(model_input, category)
      pu = pu_all[:, category].reshape(-1, 1)
      y_xu_logits = self.model_xu2y.predict(input_xu, **kwargs)
      y_xu = tf.math.softmax(y_xu_logits).numpy()
      result_temp[:, :, category] = y_xu * pu * self.freq_ratio[category]
    # Sum over U
    result_temp = result_temp.sum(axis=-1)

    # Normalize
    y_pred = result_temp / result_temp.sum(axis=-1, keepdims=True)
    return y_pred

  def predict_mult(self, data, num_batches, **kwargs):
    """Predict target Y from the TF dataset directly. See also: predict()."""
    y_true = []
    y_pred = []
    ds_iter = iter(data)
    for _ in range(num_batches):
      batch = next(ds_iter)
      y = self.get_output(*batch, outputs="y")
      y_true.extend(y)
      model_input = self.get_input(*batch, inputs="x")
      y_pred.extend(self.predict(model_input, **kwargs))

    return np.array(y_true), np.array(y_pred)

  def _predict_u_mult(self, data, num_batches, **kwargs):
    """Used internally for label correction."""
    u_true = []
    u_pred = []
    ds_iter = iter(data)
    for _ in range(num_batches):
      batch = next(ds_iter)
      u = self.predict_u_from_all(*batch)
      u_true.extend(u)
      model_input = self.get_input(*batch, inputs="x")
      u_pred.extend(self.predict_u_from_x(model_input, **kwargs))

    return np.array(u_true), np.array(u_pred)

  def _calibrate(self, data, mode="all", **kwargs):
    """Calibrate model on validation data."""
    assert mode in ["all", "x2u", "xu2y"]

    # first, we calibrate x->u
    if mode in ["all", "x2u"]:
      calib_model = temp_scaling.TempCalibratedModel(self.model_x2u)
      opt = tf.keras.optimizers.SGD(learning_rate=1e-3)  # ok to hard-code lr
      calib_model.compile(loss=self.model_x2u.loss, optimizer=opt)
      ds = data.map(self.x2u)
      calib_model.fit(ds, **kwargs)
      self.model_x2u = calib_model

    # second, we calibrate (x, u) -> y
    if mode in ["all", "xu2y"]:
      calib_model = temp_scaling.TempCalibratedModel(self.model_xu2y)
      opt = tf.keras.optimizers.SGD(learning_rate=1e-3)  # ok to hard-code lr
      calib_model.compile(loss=self.model_xu2y.loss, optimizer=opt)
      ds = data.map(self.xu2y)
      calib_model.fit(ds, **kwargs)
      self.model_xu2y = calib_model
