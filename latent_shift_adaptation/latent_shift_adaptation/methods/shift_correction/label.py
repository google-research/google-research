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

"""Label shift baseline method where p(y) [source] and q(y) [target] are known.
"""

from latent_shift_adaptation.methods import erm
from latent_shift_adaptation.utils import temp_scaling
import tensorflow as tf


class Method(erm.Method):
  """Label shift baseline method."""

  def __init__(
      self, model, evaluate=None, num_classes=2, dtype=tf.float32, pos=None
  ):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      model: Compiled Keras model. Used to predict Y from X.
      evaluate: a tf.keras.metrics method.
      num_classes: number of classes.
      dtype: desired dtype (e.g. tf.float32).
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(model, evaluate, dtype=dtype, pos=pos)
    self.num_classes = num_classes
    self.start_lr = self.model.optimizer.learning_rate.numpy()

  def _reweight(
      self,
      data_source,
      data_target,
      num_batches,
  ):
    """Estimate the frequency of each class in source and target."""
    eps = 1e-3
    source_iter = iter(data_source)
    target_iter = iter(data_target)

    # estimate frequencies of targets y
    freq_source = tf.fill((1, self.num_classes), 0.0)
    freq_target = tf.fill((1, self.num_classes), 0.0)
    for _ in range(num_batches):
      # in source data, we know the true labels so we use them directly
      batch = next(source_iter)
      freq_source += tf.math.reduce_mean(
          tf.cast(batch[self.pos.y], tf.float32), axis=0
      )
      # oracle access to p(y) in target domain
      batch = next(target_iter)
      freq_target += tf.math.reduce_mean(
          tf.cast(batch[self.pos.y], tf.float32), axis=0
      )
    self.freq_source = freq_source / num_batches + eps
    self.freq_target = freq_target / num_batches + eps

    # calculate class weights
    self.freq_ratio = self.freq_target / self.freq_source
    self.freq_ratio = self.freq_ratio.numpy()[0]
    self.class_weight = {i: self.freq_ratio[i] for i in range(self.num_classes)}

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

    # class weights
    self._reweight(data_source_val, data_target, steps_per_epoch_val)

    # reset lr to avoid issue with lr scheduling callbacks
    tf.keras.backend.set_value(
        self.model.optimizer.learning_rate, self.start_lr
    )

    ds_source = data_source_train.map(self.split_batch)
    self.model.fit(ds_source, class_weight=self.class_weight, **fit_kwargs)

    # finally, calibrate
    self._calibrate(data_source_val, **fit_kwargs)

  def calibrate(self, data, **kwargs):
    """Calibrate model on validation data."""
    # set up calibration layer
    calib_model = temp_scaling.TempCalibratedModel(self.model)
    opt = tf.keras.optimizers.SGD(learning_rate=1e-3)  # ok to hard-code lr
    calib_model.compile(loss=self.model.loss, optimizer=opt)

    # train the temperature scaling layer on validation data
    ds = data.map(self.split_batch)
    calib_model.fit(ds, class_weight=self.class_weight, **kwargs)
    self.model = calib_model
