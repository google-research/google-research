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

"""Unit tests for `tf_util.py`."""

from typing import List, Tuple
import unittest

from active_selective_prediction.utils import model_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


class TestModelFunctions(unittest.TestCase):
  """Tests model functions."""

  def generate_model_and_batch(self):
    """Generates a model and a batch."""
    batch_x = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
    model = tf.keras.layers.Dense(2, activation='softmax')
    model(batch_x)
    weights = np.array([[1, 3], [0, 0]], dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    model.set_weights([weights, bias])
    return model, batch_x

  def generate_simple_mlp_and_batch(self):
    """Generates a simple mlp and a batch."""
    batch_x = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
    model = model_util.get_simple_mlp(input_shape=(2,), num_classes=2)
    model(batch_x)
    return model, batch_x

  def test_get_model_feature(self):
    """Tests get_model_feature function."""
    model, batch_x = self.generate_simple_mlp_and_batch()
    batch_features = tf_util.get_model_feature(model, batch_x)
    self.assertEqual(batch_features.shape, (2, 128))

  def test_get_model_output_and_feature(self):
    """Tests get_model_output_and_feature function."""
    model, batch_x = self.generate_simple_mlp_and_batch()
    batch_outputs, batch_features = tf_util.get_model_output_and_feature(
        model,
        batch_x
    )
    self.assertEqual(batch_outputs.shape, (2, 2))
    self.assertEqual(batch_features.shape, (2, 128))

  def test_get_model_output(self):
    """Tests get_model_output function."""
    model, batch_x = self.generate_model_and_batch()
    batch_outputs = tf_util.get_model_output(model, batch_x)
    batch_output_sums = np.sum(batch_outputs.cpu().numpy(), axis=1)
    self.assertEqual(batch_outputs.shape, (2, 2))
    for val in batch_output_sums:
      self.assertAlmostEqual(val, 1.0, places=5)

  def test_get_model_prediction(self):
    """Tests get_model_prediction function."""
    model, batch_x = self.generate_model_and_batch()
    batch_preds = tf_util.get_model_prediction(model, batch_x)
    self.assertEqual(tuple(batch_preds.cpu().numpy()), (1, 0))

  def test_get_model_confidence(self):
    """Tests get_model_confidence function."""
    model, batch_x = self.generate_model_and_batch()
    batch_confs = tf_util.get_model_confidence(model, batch_x)
    batch_confs = batch_confs.cpu().numpy()
    self.assertAlmostEqual(batch_confs[0], 0.880797, places=5)
    self.assertAlmostEqual(batch_confs[1], 0.5, places=5)

  def test_get_model_margin(self):
    """Tests get_model_margin function."""
    model, batch_x = self.generate_model_and_batch()
    batch_margins = tf_util.get_model_margin(model, batch_x)
    batch_margins = batch_margins.cpu().numpy()
    self.assertAlmostEqual(batch_margins[0], 0.7615941, places=5)
    self.assertAlmostEqual(batch_margins[1], 0, places=5)


class TestEnsembleModelFunctions(unittest.TestCase):
  """Tests ensemble model functions."""

  def generate_ensemble_model_and_batch(
      self,
  ):
    """Generates an ensemble model and a batch."""
    batch_x = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
    models = []
    for k in range(2):
      model = tf.keras.layers.Dense(2, activation='softmax')
      model(batch_x)
      if k == 0:
        weights = np.array([[1, 3], [0, 0]], dtype=np.float32)
      else:
        weights = np.array([[2, 1], [0, 1]], dtype=np.float32)
      bias = np.zeros(2, dtype=np.float32)
      model.set_weights([weights, bias])
      models.append(model)
    return models, batch_x

  def generate_ensemble_simple_mlp_and_batch(
      self,
  ):
    """Generates an ensemble simple mlp and a batch."""
    batch_x = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
    models = []
    for _ in range(2):
      model = model_util.get_simple_mlp(input_shape=(2,), num_classes=2)
      model(batch_x)
      models.append(model)
    return models, batch_x

  def test_get_ensemble_model_feature(self):
    """Tests get_ensemble_model_feature function."""
    model, batch_x = self.generate_ensemble_simple_mlp_and_batch()
    batch_features = tf_util.get_ensemble_model_feature(model, batch_x)
    self.assertEqual(batch_features.shape, (2, 256))

  def test_get_ensemble_model_output_and_feature(self):
    """Tests get_ensemble_model_output_and_feature function."""
    model, batch_x = self.generate_ensemble_simple_mlp_and_batch()
    batch_outputs, batch_features = (
        tf_util.get_ensemble_model_output_and_feature(
            model,
            batch_x,
            'soft',
        )
    )
    self.assertEqual(batch_outputs.shape, (2, 2))
    self.assertEqual(batch_features.shape, (2, 256))

  def test_get_ensemble_model_output(self):
    """Tests get_ensemble_model_output function."""
    models, batch_x = self.generate_ensemble_model_and_batch()
    batch_outputs = tf_util.get_ensemble_model_output(models, batch_x, 'soft')
    batch_output_sums = np.sum(batch_outputs.cpu().numpy(), axis=1)
    self.assertEqual(batch_outputs.shape, (2, 2))
    for val in batch_output_sums:
      self.assertAlmostEqual(val, 1.0, places=5)

  def test_get_ensemble_model_prediction(self):
    """Tests get_ensemble_model_prediction function."""
    models, batch_x = self.generate_ensemble_model_and_batch()
    batch_preds = tf_util.get_ensemble_model_prediction(models, batch_x, 'soft')
    self.assertEqual(tuple(batch_preds.cpu().numpy()), (1, 0))

  def test_get_ensemble_model_confidence(self):
    """Tests get_ensemble_model_confidence function."""
    models, batch_x = self.generate_ensemble_model_and_batch()
    batch_confs = tf_util.get_ensemble_model_confidence(models, batch_x, 'soft')
    batch_confs = batch_confs.cpu().numpy()
    self.assertAlmostEqual(batch_confs[0], 0.6903985, places=5)
    self.assertAlmostEqual(batch_confs[1], 0.5, places=5)

  def test_get_ensemble_model_margin(self):
    """Tests get_ensemble_model_margin function."""
    models, batch_x = self.generate_ensemble_model_and_batch()
    batch_margins = tf_util.get_ensemble_model_margin(models, batch_x, 'soft')
    batch_margins = batch_margins.cpu().numpy()
    self.assertAlmostEqual(batch_margins[0], 0.38079706, places=5)
    self.assertAlmostEqual(batch_margins[1], 0, places=5)


if __name__ == '__main__':
  unittest.main()
