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

r"""Convert TRILL + Sklearn to SavedModel.

"""

import pickle

from absl import app
from absl import flags
import numpy as np
from sklearn import linear_model  # pylint:disable=unused-import

import tensorflow as tf
import tensorflow_hub as hub

flags.DEFINE_string('trill_location', None, 'Location of the SavedModel.')
flags.DEFINE_string('sklearn_location', None, 'Location of the sklearn model.')
flags.DEFINE_string('output_filepath', None, 'Output filepath.')

FLAGS = flags.FLAGS


def sklearn_logistic_regression_to_keras(sklearn_model, main_model=None):
  """Convert logistic regression sklearn model to keras."""
  model = main_model or tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(
      1,
      kernel_initializer=tf.keras.initializers.Constant(sklearn_model.coef_),
      bias_initializer=tf.keras.initializers.Constant(
          sklearn_model.intercept_),
      activation=tf.keras.activations.sigmoid))
  def _logodds_to_prob(x):
    return tf.concat([1 - x, x], axis=1)
  model.add(tf.keras.layers.Lambda(_logodds_to_prob))

  return model


def combine_models(trill_layer, sklearn_model):
  """Create a keras model that combines TRILL + average pooling + sklearn."""
  model = tf.keras.models.Sequential()
  model.add(tf.keras.Input(shape=(None,), batch_size=1))
  model.add(trill_layer)
  # Perform average pooling, as is done in the `data_prep` pipeline for sklearn
  # models.
  model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)))

  model = sklearn_logistic_regression_to_keras(sklearn_model, model)

  return model


def test_models_equal(trill_layer, sklearn_model, combined_model, seed, runs,
                      n_features=16000):
  """Tests that `TRILL + Sklearn = New SavedModel` for many inputs."""
  rng = np.random.RandomState(seed)
  for _ in range(runs):
    data = rng.lognormal(size=n_features).reshape([1, n_features])
    # Calculate TRILL + Sklearn.
    trill_output = trill_layer(data)
    assert trill_output.ndim == 3, trill_output.ndim
    average_trill_output = np.average(trill_output, axis=1)
    probs_old = sklearn_model.predict_proba(average_trill_output)

    # Calculate combined model.
    probs_new = combined_model(data)
    probs_new.shape.assert_has_rank(2)

    # Assert equality.
    np.testing.assert_almost_equal(probs_old, probs_new.numpy(), 5)


def main(unused_argv):
  trill_layer = hub.KerasLayer(
      handle=FLAGS.trill_location,
      trainable=False,
      arguments={'sample_rate': 16000},
      output_key='embedding',
      output_shape=[None, 2048]
  )
  with tf.io.gfile.GFile(FLAGS.sklearn_location, 'rb') as f:
    sklearn_model = pickle.load(f)
  combined_model = combine_models(trill_layer, sklearn_model)

  for seed in range(20):
    test_models_equal(trill_layer, sklearn_model, combined_model, seed, runs=10)

  tf.keras.models.save_model(combined_model, FLAGS.output_filepath)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['trill_location', 'sklearn_location', 'output_filepath'])

  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  app.run(main)
