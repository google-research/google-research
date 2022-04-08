# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for non_semantic_speech_benchmark.eval_embedding.keras.train_keras."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import mock
import tensorflow as tf

from non_semantic_speech_benchmark.distillation import train_keras


def _get_data(*args, **kwargs):
  del args
  assert 'samples_key' in kwargs
  assert 'min_length' in kwargs
  assert 'batch_size' in kwargs
  bs = kwargs['batch_size']
  samples = tf.zeros((bs, 16000), tf.float32)
  targets = tf.ones([bs, 10], tf.float32)
  return tf.data.Dataset.from_tensors((samples, targets)).repeat()


class TrainKerasTest(parameterized.TestCase):

  @parameterized.parameters(
      {'alpha': 1.0},
      {'alpha': 0.5},
  )
  def test_get_model(self, alpha):
    batched_samples = tf.zeros([3, 16000])
    output_dimension = 10
    targets = tf.ones([3, output_dimension])

    model = train_keras.models.get_keras_model(
        f'mobilenet_debug_{alpha}_False',
        output_dimension=output_dimension)

    loss_obj = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.MeanSquaredError()
    train_mae = tf.keras.metrics.MeanAbsoluteError()
    summary_writer = tf.summary.create_file_writer(
        absltest.get_default_test_tmpdir())
    train_step = train_keras.get_train_step(
        model, loss_obj, opt, train_loss, train_mae, summary_writer)
    gstep = opt.iterations
    train_step(batched_samples, targets, gstep)
    self.assertEqual(1, gstep)
    train_step(batched_samples, targets, gstep)
    self.assertEqual(2, gstep)

  @mock.patch.object(train_keras.get_data, 'get_data', new=_get_data)
  @mock.patch.object(train_keras.hub, 'load')
  @flagsaver.flagsaver
  def test_full_flow(self, mock_load):
    del mock_load
    flags.FLAGS.model_type = 'mobilenet_debug_1.0_False'
    flags.FLAGS.file_patterns = 'dummy'
    flags.FLAGS.teacher_model_hub = 'dummy'
    flags.FLAGS.output_key = 'dummmy'
    flags.FLAGS.output_dimension = 10
    flags.FLAGS.shuffle_buffer_size = 4
    flags.FLAGS.samples_key = 'audio'
    flags.FLAGS.logdir = absltest.get_default_test_tmpdir()

    train_keras.train_and_report(debug=True)


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
