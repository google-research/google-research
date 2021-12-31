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

"""Tests for smurf_net."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from smurf import smurf_net


def train_step(smurf_model, inputs, weights):
  losses, gradients, variables = smurf_model.loss_and_grad(
      inputs=inputs,
      weights=weights)
  smurf_model.optimizer.apply_gradients(zip(gradients, variables))
  return losses


class SMURFNetTest(absltest.TestCase):
  """Run some checks to see if loading pretrained weights works correctly."""

  def test_inference(self):
    """Test that inference runs and produces output of the right size."""
    image1 = np.random.randn(256, 256, 3).astype('float32')
    image2 = np.random.randn(256, 256, 3).astype('float32')
    smurf_model = smurf_net.SMURFNet()
    flow = smurf_model.infer(image1, image2)
    correct_shape = np.equal(flow.shape, [256, 256, 2]).all()
    self.assertTrue(correct_shape)

  def test_train_step(self):
    """Test a single training step."""
    ds = tf.data.Dataset.from_tensor_slices({
        'images': tf.zeros([1, 2, 256, 256, 3], dtype=tf.float32)
    }).repeat().batch(1)
    it = iter(ds)
    smurf_model = smurf_net.SMURFNet()
    weights = {'smooth2': 2.0,
               'edge_constant': 100.0,
               'census': 1.}
    losses = train_step(smurf_model, it.next(), weights=weights)
    self.assertNotEmpty(losses)

  def test_supervised_train_step(self):
    """Test a single supervised training step."""
    ds = tf.data.Dataset.from_tensor_slices(
        {'images': tf.zeros([1, 2, 256, 256, 3], dtype=tf.float32),
         'flow': tf.zeros([1, 256, 256, 2], dtype=tf.float32),
         'flow_valid': tf.ones([1, 256, 256, 1], dtype=tf.float32)
        }).repeat().batch(1)
    it = iter(ds)
    smurf_model = smurf_net.SMURFNet(train_mode='supervised')
    weights = {'supervision': 1.0}
    losses = train_step(smurf_model, it.next(), weights=weights)
    self.assertNotEmpty(losses)
    self.assertGreater(losses['supervision-loss'], 0)

  def test_sequence_supervised_train_step(self):
    """Test a single supervised training step."""
    ds = tf.data.Dataset.from_tensor_slices(
        {'images': tf.zeros([1, 2, 256, 256, 3], dtype=tf.float32),
         'flow': tf.zeros([1, 256, 256, 2], dtype=tf.float32),
         'flow_valid': tf.ones([1, 256, 256, 1], dtype=tf.float32)
        }).repeat().batch(1)
    it = iter(ds)
    smurf_model = smurf_net.SMURFNet(
        train_mode='sequence-supervised',
        flow_architecture='raft',
        feature_architecture='raft')
    weights = {'supervision': 1.0}
    losses = train_step(smurf_model, it.next(), weights=weights)
    self.assertNotEmpty(losses)
    self.assertGreater(losses['supervised_sequence_loss-loss'], 0)

  def test_sequence_unsupervised_train_step(self):
    """Test a single supervised training step."""
    ds = tf.data.Dataset.from_tensor_slices(
        {'images': tf.zeros([1, 2, 256, 256, 3], dtype=tf.float32),
        }).repeat().batch(1)
    it = iter(ds)
    smurf_model = smurf_net.SMURFNet(
        train_mode='sequence-unsupervised',
        flow_architecture='raft',
        feature_architecture='raft')
    weights = {'smooth2': 2.0,
               'edge_constant': 100.0,
               'census': 1.}
    losses = train_step(smurf_model, it.next(), weights=weights)
    self.assertNotEmpty(losses)
    self.assertGreater(losses['census-loss'], 0)


if __name__ == '__main__':
  absltest.main()
