# coding=utf-8
# Copyright 2024 The Google Research Authors.
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
import tensorflow.compat.v2 as tf

from non_semantic_speech_benchmark.eval_embedding.keras import models
from non_semantic_speech_benchmark.eval_embedding.keras import train_keras


def _get_data(*args, **kwargs):
  del args
  assert 'embedding_dim' in kwargs
  assert 'bucket_boundaries' in kwargs
  assert 'bucket_batch_sizes' in kwargs
  assert 'label_list' in kwargs
  bs = kwargs['bucket_batch_sizes'][0]
  emb = tf.zeros((bs, 10, kwargs['embedding_dim']), tf.float32)
  labels = tf.zeros([bs], tf.int32)
  labels_onehot = tf.one_hot(labels, len(kwargs['label_list']))
  return tf.data.Dataset.from_tensors((emb, labels_onehot)).repeat()


class TrainKerasTest(parameterized.TestCase):

  @parameterized.parameters(
      {'num_clusters': 0, 'alpha_init': None},
      {'num_clusters': 4, 'alpha_init': None},
      {'num_clusters': None, 'alpha_init': 1.0},
  )
  def test_get_model(self, num_clusters, alpha_init):
    num_classes = 4
    emb = tf.zeros([3, 5, 8])
    y_onehot = tf.one_hot([0, 1, 2], num_classes)

    model = models.get_keras_model(
        num_classes, use_batchnorm=True, num_clusters=num_clusters,
        alpha_init=alpha_init)

    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    summary_writer = tf.summary.create_file_writer(
        absltest.get_default_test_tmpdir())
    train_step = train_keras.get_train_step(
        model, loss_obj, opt, train_loss, train_accuracy, summary_writer)
    gstep = opt.iterations
    train_step(emb, y_onehot, gstep)
    self.assertEqual(1, gstep)
    train_step(emb, y_onehot, gstep)
    self.assertEqual(2, gstep)

  @mock.patch.object(train_keras.get_data, 'get_data', new=_get_data)
  @flagsaver.flagsaver
  def test_full_flow(self):
    flags.FLAGS.file_pattern = 'dummy'
    flags.FLAGS.shuffle_buffer_size = 4
    flags.FLAGS.en = 'trill-distilled'
    flags.FLAGS.ed = 2048
    flags.FLAGS.nc = 2
    flags.FLAGS.label_name = 'emotion'
    flags.FLAGS.label_list = ['no', 'yes']
    flags.FLAGS.bucket_boundaries = [10, 20]
    flags.FLAGS.logdir = absltest.get_default_test_tmpdir()

    train_keras.train_and_report(debug=True)


if __name__ == '__main__':
  tf.compat.v2.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
