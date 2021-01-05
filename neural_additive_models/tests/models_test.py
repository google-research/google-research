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

# Lint as: python3
"""Tests functionality of loading different models."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf
from neural_additive_models import models


class LoadModelsTest(parameterized.TestCase):
  """Tests whether neural net models can be run without error."""

  @parameterized.named_parameters(('exu_nam', 'exu_nam'),
                                  ('relu_nam', 'relu_nam'), ('dnn', 'dnn'))
  def test_model(self, architecture):
    """Test whether a model with specified architecture can be run."""
    x = np.random.rand(5, 10).astype('float32')
    sess = tf.InteractiveSession()
    if architecture == 'exu_nam':
      model = models.NAM(
          num_inputs=x.shape[1], num_units=1024, shallow=True, activation='exu')
    elif architecture == 'relu_nam':
      model = models.NAM(
          num_inputs=x.shape[1], num_units=64, shallow=False, activation='relu')
    elif architecture == 'dnn':
      model = models.DNN()
    else:
      raise ValueError('Architecture {} not found'.format(architecture))
    out_op = model(x)
    sess.run(tf.global_variables_initializer())
    self.assertIsInstance(sess.run(out_op), np.ndarray)
    sess.close()


if __name__ == '__main__':
  absltest.main()
