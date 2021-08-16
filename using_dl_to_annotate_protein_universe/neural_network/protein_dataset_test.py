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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import parameterized
import numpy as np
import protein_dataset
import tensorflow.compat.v1 as tf
import utils

FLAGS = flags.FLAGS


def _numpy_one_hot(x, depth):
  """Convert numpy array of indexes into a full one-hot.

  Args:
    x: np.array.
    depth: int. maximum index in array (depth of one-hot output).

  Returns:
    np.array.
  """
  return np.eye(depth)[x]


def _dataset_iterator_to_list(itr, session):
  """Convert tf.data.Dataset iterator to a python list.

  Args:
    itr: tf.data.Dataset iterator.
    session: tf.Session.

  Returns:
    list.
  """
  actual_examples = []
  while True:
    try:
      actual_examples.append(session.run(itr.get_next()))
    except tf.errors.OutOfRangeError:
      break
  return actual_examples


class ProteinDatasetTest(parameterized.TestCase):

  def test_padded_dataset(self):
    # Set up test data.
    test_data_directory = FLAGS.test_srcdir

    label_vocab_array = ['PF00001']

    batch_size = 3

    with tf.Graph().as_default():
      sess = tf.Session()
      non_padded_dataset = protein_dataset.non_batched_dataset(
          # Dev fold instead of train fold because the train fold is repeated.
          train_dev_or_test=protein_dataset.DEV_FOLD,
          label_vocab=label_vocab_array,
          data_root_dir=test_data_directory)
      batched_dataset = protein_dataset.batched_dataset(
          non_padded_dataset, batch_size=batch_size)
      batch_itr = batched_dataset.make_initializable_iterator()

      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      sess.run(batch_itr.initializer)

    # Compute actual output
    actual_examples = _dataset_iterator_to_list(batch_itr, sess)

    # Examine correctness of first element.
    actual_sequence_batch_shape = actual_examples[0][
        protein_dataset.SEQUENCE_KEY].shape
    expected_longest_sequence_len_in_first_batch = 98
    expected_first_batch_sequence_shape = (
        batch_size, expected_longest_sequence_len_in_first_batch,
        len(utils.AMINO_ACID_VOCABULARY))
    self.assertEqual(actual_sequence_batch_shape,
                     expected_first_batch_sequence_shape)

    actual_label_batch_shape = actual_examples[0][
        protein_dataset.LABEL_KEY].shape
    # Because the label vocab contains the labels in the first example, we
    # get len(label_vocab_array) as the number of labels.
    expected_batch_label_shape = (batch_size, len(label_vocab_array))
    self.assertEqual(actual_label_batch_shape, expected_batch_label_shape)


if __name__ == '__main__':
  tf.test.main()
