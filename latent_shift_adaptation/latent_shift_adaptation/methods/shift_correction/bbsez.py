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

"""BBSE method with confounder Z = U or W.

This assumes that the confounder is observed at test time.
Computes p_t(Y | X) propto sum_z [p_s(Y | X, Z) p_s(Z | X) p_t(Z) / p_s(Z)].
"""

from latent_shift_adaptation.methods import erm
from latent_shift_adaptation.utils import temp_scaling
import numpy as np
import tensorflow as tf


class Method(erm.Method):
  """Label shift baseline method."""

  def __init__(
      self,
      x2y_model,
      x2z_model,
      evaluate=None,
      num_classes=2,
      confounder="u",
      dtype=tf.float32,
      pos=None,
  ):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      x2y_model: Compiled Keras model. It's used to predict Y from X.
      x2z_model: Keras model used to preidct the latent variable from x.
      evaluate: a tf.keras.metrics method.
      num_classes: number of classes.
      confounder: confounder variable to condition on when applying LSA
        correction. Can be either 'u', 'w', or 'c'.
      dtype: desired dtype (e.g. tf.float32).
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(x2y_model, evaluate, dtype=dtype, pos=pos)
    self.x2z_model = x2z_model

    if confounder not in ["u", "w", "c"]:
      raise ValueError("Error: confounder must be 'u', 'w', or 'c'.")

    self.num_classes = num_classes
    self.confounder = confounder
    self.num_categories = 1  # will be updated later when observing data
    self.start_lr = self.model.optimizer.learning_rate.numpy()

  def _reweight(
      self,
      data_source_train,
      data_source_val,
      data_target,
      num_batches,
      **kwargs,
  ):
    """Estimate confounder category weights q(z) / p(z)."""

    # train p(z | x), where z is either u, c or w.
    clf_x2z = erm.Method(
        model=self.x2z_model, inputs="x", outputs=self.confounder, pos=self.pos
    )
    clf_x2z.fit(
        data_source_train, data_source_val, data_target, num_batches, **kwargs
    )  # with calibration
    self.clf_x2z = clf_x2z
    self.x2z_model.trainable = False

    # Label shift correction to estimate weights proportional to q(U) / p(U)
    # p(z | source)
    z_true_source, z_pred_source = clf_x2z.predict_mult(
        data_source_val, num_batches
    )

    # obtain confusion matrix
    confusion_matrix = (
        np.sum(
            [
                np.outer(z_true_source[i, :], z_pred_source[i, :])
                for i in range(z_pred_source.shape[0])
            ],
            axis=0,
        )
        / z_pred_source.shape[0]
    )
    confusion_matrix = tf.cast(
        confusion_matrix.transpose(), tf.float64
    )  # tf.float64, otherwise error!

    # p(z | target)
    _, z_pred_target = clf_x2z.predict_mult(data_target, num_batches)
    mu_z = (
        np.mean(z_pred_target, axis=0)
        .reshape((self.num_categories, 1))
        .astype(np.float64)
    )

    # weights
    weights = tf.squeeze(tf.linalg.pinv(confusion_matrix) @ tf.constant(mu_z))
    weights = np.minimum(np.maximum(1e-2, weights.numpy()), 15)  # clip weights

    self.category_weight = {i: weights[i] for i in range(self.num_categories)}

  # pylint: disable=unused-argument
  def fit(
      self,
      data_source_train,
      data_source_val,
      data_target,
      steps_per_epoch_val,
      **fit_kwargs,
  ):
    """Fit model on data."""

    # verify data assumptions (e.g. shape/range)
    batch = next(iter(data_source_train))
    self.assertions(*batch)
    num_categories = batch[self.pos[self.confounder]].shape[1]
    self.num_categories = num_categories

    self._reweight(
        data_source_train,
        data_source_val,
        data_target,
        steps_per_epoch_val,
        **fit_kwargs,
    )
    # reset lr to avoid issue with lr scheduling callbacks
    tf.keras.backend.set_value(
        self.model.optimizer.learning_rate, self.start_lr
    )

    # build a model from 'X' + latent --> 'Y'
    self.inputs = "x" + self.confounder
    ds_source = data_source_train.map(self.split_batch)
    self.model.fit(ds_source, **fit_kwargs)

    # finally, calibrate
    self._calibrate(data_source_val, **fit_kwargs)

  def _calibrate(self, data, **kwargs):
    """Calibrate model on validation data."""
    self.inputs = "x" + self.confounder
    calib_model = temp_scaling.TempCalibratedModel(self.model)
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3)  # ok to hard-code lr
    calib_model.compile(loss=self.model.loss, optimizer=opt)
    ds = data.map(self.split_batch)
    calib_model.fit(ds, **kwargs)
    self.model = calib_model
    self.inputs = "x"  # revert inputs back to 'x' only

  def fix_confounder(self, x, confounder_val):
    """Return [x, z] with confounder z fixed to chosen value."""
    batch_len = x.shape[0]
    x = tf.cast(x, self.dtype)
    z = tf.keras.utils.to_categorical(
        confounder_val * np.ones((batch_len,)), self.num_categories
    )
    z = tf.cast(z, self.dtype)
    stack = tf.concat([x, z], axis=1)
    return stack

  def predict(self, model_input, **kwargs):
    """Predict target Y given the model input."""
    # model_input is x only. To make a prediction, we marginalize over u.
    batch_len = model_input.shape[0]
    result_temp = np.zeros((batch_len, self.num_classes, self.num_categories))
    pz_all = self.clf_x2z.predict(model_input).numpy()  # predict z from x.

    for category in range(self.num_categories):  # marginalize over z
      input_xz = self.fix_confounder(model_input, category)
      pz = pz_all[:, category].reshape(-1, 1)
      y_uz = tf.math.softmax(self.model.predict(input_xz, **kwargs))
      result_temp[:, :, category] = y_uz * pz * self.category_weight[category]

    # sum over Z
    result_temp = result_temp.sum(axis=-1)
    # normalize
    y_pred = result_temp / result_temp.sum(axis=-1, keepdims=True)
    return y_pred

  def predict_mult(self, data, num_batches, **kwargs):
    """Predict target Y from the TF dataset directly. See also: predict()."""
    y_true = []
    y_pred = []
    ds_iter = iter(data)
    for _ in range(num_batches):
      batch = next(ds_iter)
      model_input, y = self.split_batch(*batch)
      y_true.extend(y)
      y_pred.extend(self.predict(model_input, **kwargs))
    return np.array(y_true), np.array(y_pred)
