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

"""Covariate shift baseline method.
"""

from latent_shift_adaptation.methods import erm
import tensorflow as tf


class Method(erm.Method):
  """Covariate shift baseline method."""

  def __init__(
      self,
      model,
      domain_discriminator,
      evaluate=None,
      dtype=tf.float32,
      pos=None,
  ):  # pylint: disable=g-doc-args
    """Constructor.

    Args:
      model: Compiled Keras model. Used to predict Y from X.
      domain_discriminator: Keras model used to discriminate source from target.
        Must have 'sparse_categorical_crossentropy' as a loss.
      evaluate: a tf.keras.metrics method.
      dtype: desired dtype (e.g. tf.float32).
      pos: config_dict that specifies the index of x, y, c, w, u in data tuple.
        Default: data is of the form (x, y, c, w, u).
    """
    super(Method, self).__init__(model, evaluate, dtype=dtype, pos=pos)

    # verify domain_discriminator is set up properly
    self.discriminator = domain_discriminator
    if domain_discriminator.loss != "sparse_categorical_crossentropy":
      raise ValueError(
          "domain_discriminator loss must be sparse_categorical_crossentropy"
      )
    if domain_discriminator.output_shape[1] != 2:
      raise ValueError("domain_discriminator's output must be binary.")

    self.start_lr = self.model.optimizer.learning_rate.numpy()

  def _reweight(
      self, data_source, data_target, **kwargs
  ):
    """Train a domain_discriminator between source and target."""

    @tf.function
    def _preprocess_source(*batch):
      """Split batch into input and 0's."""
      instances = batch[self.pos.x]
      batch_len = tf.shape(instances)[0]
      labels = tf.fill((batch_len, 1), 0)
      return instances, labels

    @tf.function
    def _preprocess_target(*batch):
      """Split batch into input and 1's."""
      instances = batch[self.pos.x]
      batch_len = tf.shape(instances)[0]
      labels = tf.fill((batch_len, 1), 1)
      return instances, labels

    ds_source = data_source.map(_preprocess_source)
    ds_target = data_target.map(_preprocess_target)
    # mix the two datasets
    ds = tf.data.experimental.sample_from_datasets([ds_source, ds_target])
    self.discriminator.fit(ds, **kwargs)
    self.discriminator.trainable = False

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

    def _append_weights(*batch):
      """Append example weights."""
      eps = 1e-3
      # obtain weights on batch examples
      p = self.discriminator(batch[self.pos.x])[:, 1]  # prob of target domain
      weights = p / (eps + 1 - p)  # add eps to avoid div by zeros
      weights = weights / tf.math.reduce_sum(weights)  # normalize
      # use keras's built-in support for sample weights
      return batch[0], batch[1], weights

    # verify data assumptions (e.g. shape/range)
    batch = next(iter(data_source_train))
    self.assertions(*batch)

    # train a discriminator between source and target data
    self._reweight(data_source_train, data_target, **fit_kwargs)

    # reset lr to avoid potential issues with lr scheduling callbacks
    tf.keras.backend.set_value(
        self.model.optimizer.learning_rate, self.start_lr
    )

    # train model on weighted data
    ds_source = data_source_train.map(self.split_batch).map(_append_weights)
    self.model.fit(ds_source, **fit_kwargs)

    # finally, calibrate
    self._calibrate(data_source_val, **fit_kwargs)
