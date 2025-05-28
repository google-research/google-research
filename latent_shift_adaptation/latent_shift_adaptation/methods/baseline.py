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

"""Superclass for LSA baseline methods.
"""

from latent_shift_adaptation.utils import temp_scaling
import ml_collections as mlc
import numpy as np
import tensorflow as tf


class Method:
  """Superclass for LSA baseline methods. Shouldn't be instantiated directly."""

  def __init__(self, evaluate=None, dtype=tf.float32, pos=None):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      evaluate: a tf.keras.metrics method.
      dtype: desired dtype (e.g. tf.float32).
      pos: ConfigDict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    self.model = None
    self.evaluate = evaluate
    self.inputs = "x"  # default value
    self.outputs = "y"  # default value. It can be 'cy' or 'u' or others.
    self.dtype = dtype
    if pos is None:
      pos = mlc.ConfigDict()
      pos.x, pos.y, pos.c, pos.w, pos.u = 0, 1, 2, 3, 4
    self.pos = pos

  def get_input(self, *batch, inputs=None):
    """Fetch model input from the batch."""
    if inputs is None:
      inputs = self.inputs
    # first input
    stack = tf.cast(batch[self.pos[inputs[0]]], self.dtype)
    # fetch remaining ones
    for c in inputs[1:]:
      stack = tf.concat(
          [stack, tf.cast(batch[self.pos[c]], self.dtype)], axis=1
      )
    return stack

  def get_output(self, *batch, outputs=None):
    """Fetch outputs from the batch."""
    if outputs is None:
      outputs = self.outputs
    if len(outputs) == 1:  # e.g. return Y or U
      return tf.cast(batch[self.pos[outputs[0]]], self.dtype)
    else:  # e.g. in CBM, return C, Y
      return tf.cast(batch[self.pos["c"]], self.dtype), tf.cast(
          batch[self.pos["y"]], self.dtype
      )

  def split_batch(self, *batch):
    """Split batch into input and output."""
    return self.get_input(*batch), self.get_output(*batch)

  def assertions(self, *batch):
    """Verify that model assumptions are satisfied in the given batch."""
    for c in self.pos:
      data = batch[self.pos[c]]
      # all variables are 2D arrays; e.g. x in a single example is 1-dimensional
      if len(data.shape) != 2:
        raise ValueError(f"{c} must be a 2-dimensional array.")

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
    # verify assumptions first
    batch = next(iter(data_source_train))
    self.assertions(*batch)
    # then, train
    ds = data_source_train.map(self.split_batch)
    self.model.fit(ds, **fit_kwargs)
    # finally, calibrate
    self._calibrate(data_source_val, **fit_kwargs)

  def _calibrate(self, data, **kwargs):
    """Calibrate model on validation data."""
    # set up calibration layer
    calib_model = temp_scaling.TempCalibratedModel(self.model)
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3)  # ok to hard-code lr
    calib_model.compile(loss=self.model.loss, optimizer=opt)

    # train the temperature scaling layer on validation data
    ds = data.map(self.split_batch)
    calib_model.fit(ds, **kwargs)

    # replace model with its calibrated version
    self.model = calib_model

  def predict(self, model_input, **kwargs):
    """Predict Y (probabilities) given the input. See also: predict_mult()."""
    y_pred = self.model.predict(model_input, **kwargs)
    return tf.math.softmax(y_pred)

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

  def score(
      self, data, num_batches, evaluate=None, **kwargs
  ):
    """Evaluate model on data.

    Args:
      data: TF dataset.
      num_batches: number of batches fetched from the dataset.
      evaluate: tf.keras.metrics method
      **kwargs: arguments passed to predict() method.

    Returns:
      score: evaluation score.
    """
    if evaluate is None:
      evaluate = self.evaluate
    y_true, y_pred = self.predict_mult(data, num_batches, **kwargs)
    evaluate.reset_state()
    evaluate.update_state(y_true, y_pred)
    return evaluate.result().numpy()
