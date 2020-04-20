# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for non_semantic_speech_benchmark.eval_embedding.keras.train_keras."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import mock
import tensorflow.compat.v1 as tf
from non_semantic_speech_benchmark.eval_embedding.keras import train_keras


def _get_data(*args, **kwargs):
  del args
  assert 'embedding_dim' in kwargs
  assert 'batch_size' in kwargs
  assert 'label_list' in kwargs
  bs = kwargs['batch_size']
  emb = tf.zeros((bs, 10, kwargs['embedding_dim']), tf.float32)
  labels = tf.zeros([bs], tf.int32)
  labels_onehot = tf.one_hot(labels, len(kwargs['label_list']))
  return tf.data.Dataset.from_tensors((emb, labels_onehot)).repeat()


class TrainKerasTest(parameterized.TestCase):

  @parameterized.parameters(
      {'num_clusters': 0},
      {'num_clusters': 4},
  )
  def test_make_graph(self, num_clusters):
    with tf.Graph().as_default():
      emb = tf.zeros([3, 5, 8])
      y_onehot = tf.one_hot([0, 1, 2], 4)
      loss, train_op = train_keras.make_graph(
          emb, y_onehot, ubn=True, nc=num_clusters)
      with tf.train.MonitoredSession() as sess:
        sess.run([loss, train_op])

  @mock.patch.object(
      train_keras.get_data, 'get_data',
      new=_get_data)
  @flagsaver.flagsaver
  def test_full_flow(self):
    flags.FLAGS.file_pattern = 'dummy'
    flags.FLAGS.shuffle_buffer_size = 4
    flags.FLAGS.en = 'trill-distilled'
    flags.FLAGS.ed = 2048
    flags.FLAGS.nc = 2
    flags.FLAGS.label_name = 'emotion'
    flags.FLAGS.label_list = ['no', 'yes']
    flags.FLAGS.logdir = absltest.get_default_test_tmpdir()

    train_keras.train_and_report(debug=True)


if __name__ == '__main__':
  absltest.main()
