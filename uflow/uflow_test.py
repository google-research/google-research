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

"""Tests for main train loop."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from uflow.uflow_net import UFlow


class UflowTest(absltest.TestCase):
  """Run some checks to see if loading pretrained weights works correctly."""

  def test_inference(self):
    """Test that inference runs and produces output of the right size."""
    image1 = np.random.randn(256, 256, 3).astype('float32')
    image2 = np.random.randn(256, 256, 3).astype('float32')
    uflow = UFlow()
    flow = uflow.infer(image1, image2)
    correct_shape = np.equal(flow.shape, [256, 256, 2]).all()
    self.assertTrue(correct_shape)

  def test_train_step(self):
    """Test a single training step."""
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.zeros([1, 2, 256, 256, 3], dtype=tf.float32), {})).repeat().batch(1)
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    uflow = UFlow()
    log_update = uflow.train(it, num_steps=1)
    self.assertNotEmpty(log_update)


if __name__ == '__main__':
  absltest.main()
