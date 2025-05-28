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

"""BBSE method: see https://arxiv.org/pdf/1802.03916.pdf.
"""

from latent_shift_adaptation.methods import erm
from latent_shift_adaptation.utils import temp_scaling
import numpy as np
import tensorflow as tf


class Method(erm.Method):
  """Label shift baseline method."""

  def __init__(
      self,
      model,
      x2y_model,
      evaluate=None,
      num_classes=2,
      inputs="x",
      outputs="y",
      dtype=tf.float32,
      pos=None,
  ):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      model: Keras model to predict Y from X.
      x2y_model: A second Keras model to predict y from x, which used for label
        correction using the confusion matrix method.
      evaluate: a tf.keras.metrics method.
      num_classes: number of classes.
      inputs: the input of a model, e.g. 'x' if x -> y, 'cw' if from C,W to Y.
      outputs: the ouptut of a model, e.g. 'y'
      dtype: desired dtype (e.g. tf.float32).
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(model, evaluate, dtype=dtype, pos=pos)
    self.x2y_model = x2y_model
    self.num_classes = num_classes
    self.start_lr = self.model.optimizer.learning_rate.numpy()
    self.class_weight = None

  def _reweight(
      self,
      data_source_train,
      data_source_val,
      data_target,
      num_batches,
      **kwargs,
  ):
    """Estimate class weights using the BBSE procedure."""

    # set up model and fit on source data
    clf = erm.Method(
        model=self.x2y_model,
        dtype=self.dtype,
        inputs="x",
        outputs="y",
        pos=self.pos,
    )
    clf.fit(
        data_source_train, data_source_val, data_target, num_batches, **kwargs
    )  # with calibration
    self.x2y_model.trainable = False

    # calculate confusion matrix in source domain
    y_true, y_pred = clf.predict_mult(data_source_val, num_batches)
    confusion_matrix = (
        np.sum(
            [
                np.outer(y_true[i, :], y_pred[i, :])
                for i in range(y_pred.shape[0])
            ],
            axis=0,
        )
        / y_pred.shape[0]
    )
    confusion_matrix = tf.cast(
        confusion_matrix.transpose(), tf.float64
    )  # tf.float64, otherwise error!
    # calculate mu (mean prediction) in target domain
    _, y_pred_traget = clf.predict_mult(data_target, num_batches)
    mu_y = (
        np.mean(y_pred_traget, axis=0)
        .reshape((self.num_classes, 1))
        .astype(np.float64)
    )
    # now, calculate weights
    weights = tf.squeeze(tf.linalg.pinv(confusion_matrix) @ tf.constant(mu_y))
    # clipping made a difference
    weights = np.minimum(np.maximum(1e-2, weights.numpy()), 15)
    self.class_weight = {i: weights[i] for i in range(self.num_classes)}

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

    # learn label shift correction
    self._reweight(
        data_source_train,
        data_source_val,
        data_target,
        steps_per_epoch_val,
        **fit_kwargs,
    )

    # reset lr to avoid potential issue with lr scheduling callbacks
    tf.keras.backend.set_value(
        self.model.optimizer.learning_rate, self.start_lr
    )

    ds_source = data_source_train.map(self.split_batch)
    self.model.fit(ds_source, class_weight=self.class_weight, **fit_kwargs)

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
    calib_model.fit(ds, class_weight=self.class_weight, **kwargs)
    self.model = calib_model
