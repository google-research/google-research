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

"""Tests for input_utils."""
import collections

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.models import input_utils


class InputUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def tearDown(self):
    tf.keras.backend.clear_session()
    super(InputUtilsTest, self).tearDown()

  def _build_tf_example(self, features_dict):
    features = collections.OrderedDict()
    for k, v in features_dict.items():
      features[k] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(v)))
    return tf.train.Example(features=tf.train.Features(feature=features))

  def _get_test_tf_example(self):
    """Returns a sample test TFExample."""
    return self._build_tf_example({
        'token_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'long_breakpoints': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'block_ids': [1, 2],
        'is_continuation': [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'prefix_length': [0, 1],
        'annotation_begins': [0, 2, 2, 0],
        'annotation_ends': [0, 4, 6, 0],
        'annotation_labels': [1, 2, 3, 0],
        'summary_token_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    })

  def _get_test_span_prediction_tf_example(self):
    """Returns a sample test TFExample."""
    return self._build_tf_example({
        'token_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'long_breakpoints': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'block_ids': [1, 2],
        'is_continuation': [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'prefix_length': [0, 1],
        'answer_annotation_begins': [0, 2, 2, 0],
        'answer_annotation_ends': [0, 4, 6, 0],
        'answer_annotation_labels': [1, 2, 3, 0],
        'summary_token_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    })

  def test_dynamic_padding_1d(self):

    def pad(values, length):
      tensor = tf.convert_to_tensor(values, dtype=tf.int32)
      return input_utils.dynamic_padding_1d(tensor, length, padding_token_id=0)

    self.assertAllEqual([1, 2], pad([1, 2, 3], 2))
    self.assertAllEqual([1, 2, 3], pad([1, 2, 3], 3))
    self.assertAllEqual([1, 2, 3, 0, 0], pad([1, 2, 3], 5))

  def test_make_block_pos_features(self):

    def block_pos(values):
      tensor = tf.convert_to_tensor(values, dtype=tf.int32)
      return input_utils.make_block_pos_features(tensor)

    self.assertAllEqual([1, 2, 3, 1, 1, 2, 0, 0],
                        block_pos([1, 1, 1, 2, 4, 4, 0, 0]))
    self.assertAllEqual([1, 2, 3, 1, 1, 2, 1, 0],
                        block_pos([1, 1, 1, 2, 4, 4, 5, 0]))
    self.assertAllEqual([1, 2, 0, 0, 0], block_pos([9, 9, 0, 0, 0]))

  def test_decode_span_prediction_example_fn_without_annotations(self):
    """Tests the span prediction TFExample parsing function."""
    tf_example = self._get_test_tf_example()

    num_blocks_per_example = 2
    block_length = 7
    max_num_answer_annotations = None

    decode_fn = input_utils.get_span_prediction_example_decode_fn(
        num_blocks_per_example=num_blocks_per_example,
        block_length=block_length,
        max_num_answer_annotations=max_num_answer_annotations)
    features = decode_fn(tf_example.SerializeToString())

    self.assertAllEqual([num_blocks_per_example, block_length],
                        features['token_ids'].shape)
    self.assertAllEqual([1, 2], features['block_ids'])
    self.assertAllEqual([1, 1], features['block_pos'])
    self.assertAllEqual([0, 1], features['prefix_length'])

  def test_decode_span_prediction_example_fn_with_annotations(self):
    """Tests the span prediction TFExample parsing function."""
    tf_example = self._get_test_span_prediction_tf_example()

    num_blocks_per_example = 2
    block_length = 7
    max_num_answer_annotations = 2

    decode_fn = input_utils.get_span_prediction_example_decode_fn(
        num_blocks_per_example=num_blocks_per_example,
        block_length=block_length,
        max_num_answer_annotations=max_num_answer_annotations)
    features = decode_fn(tf_example.SerializeToString())

    self.assertAllEqual([num_blocks_per_example, block_length],
                        features['token_ids'].shape)
    self.assertAllEqual([1, 2], features['block_ids'])
    self.assertAllEqual([1, 1], features['block_pos'])
    self.assertAllEqual([0, 1], features['prefix_length'])
    self.assertAllEqual([[0, 2], [2, 0]], features['answer_annotation_begins'])
    self.assertAllEqual([[0, 4], [6, 0]], features['answer_annotation_ends'])
    self.assertAllEqual([[1, 2], [3, 0]], features['answer_annotation_labels'])

  def test_decode_span_prediction_example_fn_with_summaries(self):
    """Tests the span prediction TFExample parsing function."""
    tf_example = self._get_test_span_prediction_tf_example()

    num_blocks_per_example = 2
    block_length = 7

    decode_fn = input_utils.get_span_prediction_example_decode_fn(
        num_blocks_per_example=num_blocks_per_example,
        block_length=block_length,
        max_num_answer_annotations=None,
        extra_int_features_shapes=dict(
            summary_token_ids=[num_blocks_per_example, block_length]))
    features = decode_fn(tf_example.SerializeToString())

    self.assertAllEqual([num_blocks_per_example, block_length],
                        features['token_ids'].shape)
    self.assertAllEqual([1, 2], features['block_ids'])
    self.assertAllEqual([1, 1], features['block_pos'])
    self.assertAllEqual([0, 1], features['prefix_length'])

  @parameterized.named_parameters(
      dict(
          testcase_name='single',
          seq_len=7,
          max_annotation_length=1,
          annotation_labels=[[1, 2, 0]],
      ),
      dict(
          testcase_name='single_2',
          seq_len=7,
          max_annotation_length=7,
          annotation_labels=[[1, 2, 0]],
      ),
      dict(
          testcase_name='batched',
          seq_len=7,
          max_annotation_length=2,
          annotation_labels=[[1, 2, 0], [0, 3, 0]],
      ),
      dict(
          testcase_name='batched_2',
          seq_len=10,
          max_annotation_length=1,
          annotation_labels=[[1, 2, 0, 17], [0, 3, 0, 21]],
      ),
      dict(
          testcase_name='batched_3',
          seq_len=10,
          max_annotation_length=3,
          annotation_labels=[[1, 2, 0, 17], [0, 3, 0, 21]],
      ),
      dict(
          testcase_name='batched_4',
          seq_len=40,
          max_annotation_length=3,
          annotation_labels=[[1, 2, 0, 17], [0, 3, 0, 21]],
      ),
      dict(
          testcase_name='batched_5',
          seq_len=40,
          max_annotation_length=3,
          annotation_labels=[[1, 2], [0, 3], [7, 8], [9, 10], [11, 0]],
      ),
  )
  def test_make_is_span_maskable_features(self, seq_len, max_annotation_length,
                                          annotation_labels):
    np.random.seed(31415)
    annotation_labels = np.array(annotation_labels).astype(np.int32)
    batch_size, num_annotations = annotation_labels.shape
    annotation_begins = np.random.randint(
        seq_len, size=[batch_size, num_annotations], dtype=np.int32)
    annotation_length = np.random.randint(
        max_annotation_length,
        size=[batch_size, num_annotations],
        dtype=np.int32)
    annotation_ends = np.minimum(annotation_begins + annotation_length,
                                 seq_len - 1)

    is_annotation_mask_np = np.zeros((batch_size, seq_len), dtype=np.int32)
    is_annotation_cont_mask_np = np.zeros((batch_size, seq_len), dtype=np.int32)

    for i in range(batch_size):
      for j in range(seq_len):
        for k in range(num_annotations):
          if (annotation_labels[i, k] != 0 and annotation_begins[i, k] <= j and
              j <= annotation_ends[i, k]):
            is_annotation_mask_np[i, j] = 1

    for i in range(batch_size):
      for j in range(seq_len):
        for k in range(num_annotations):
          if (annotation_labels[i, k] != 0 and
              annotation_begins[i, k] + 1 <= j and j <= annotation_ends[i, k]):
            is_annotation_cont_mask_np[i, j] = 1

    is_annotation_mask_np = is_annotation_mask_np.reshape(-1)
    is_annotation_cont_mask_np = is_annotation_cont_mask_np.reshape(-1)

    def to_tensor(np_array):
      return tf.convert_to_tensor(np_array.reshape(-1), dtype=tf.int32)

    is_annotation_mask_tf_obj, is_annotation_cont_mask_tf_obj = (
        input_utils.make_is_span_maskable_features(
            batch_size,
            seq_len,
            num_annotations,
            to_tensor(annotation_begins),
            to_tensor(annotation_ends),
            to_tensor(annotation_labels),
        ))
    is_annotation_mask_tf, is_annotation_cont_mask_tf = self.evaluate(
        tf.tuple((is_annotation_mask_tf_obj, is_annotation_cont_mask_tf_obj)))
    self.assertAllEqual(is_annotation_mask_np, is_annotation_mask_tf)
    self.assertAllEqual(is_annotation_cont_mask_np, is_annotation_cont_mask_tf)

  @parameterized.named_parameters(
      dict(
          testcase_name='case_1',
          token_ids=[13, 1, 2, 13, 4, 5, 6, 7],
          annotation_begins=[0, 3, 5],
          annotation_ends=[0, 4, 7],
          annotation_labels=[100, 0, 100],
          expected_token_ids=[13, 1, 2, 13, 4, 13, 13, 13],
      ),
      dict(
          testcase_name='case_2',
          token_ids=[0, 13, 2, 3, 4, 5, 13, 7],
          annotation_begins=[0, 3, 5],
          annotation_ends=[0, 4, 7],
          annotation_labels=[100, 0, 100],
          expected_token_ids=[13, 13, 2, 3, 4, 13, 13, 13],
      ),
      dict(
          testcase_name='case_3',
          token_ids=[0, 13, 2, 3, 4, 5, 6, 7],
          annotation_begins=[0, 3, 5],
          annotation_ends=[0, 4, 7],
          annotation_labels=[100, 0, 100],
          expected_token_ids=[0, 13, 2, 3, 4, 5, 6, 7],
      ),
      dict(
          testcase_name='case_4',
          token_ids=[0, 1, 2, 3, 4, 5, 6, 7],
          annotation_begins=[0, 3, 5],
          annotation_ends=[0, 4, 7],
          annotation_labels=[100, 0, 100],
          expected_token_ids=[0, 1, 2, 3, 4, 5, 6, 7],
      ),
  )
  def test_mask_same_entity_mentions(self, token_ids, annotation_begins,
                                     annotation_ends, annotation_labels,
                                     expected_token_ids):
    mask_token_id = 13
    token_ids = np.array(token_ids, dtype=np.int32)

    def to_tensor(l):
      return tf.convert_to_tensor(np.array(l).reshape(1, -1), dtype=tf.int32)

    num_masked = (token_ids == mask_token_id).sum() * 2 + 5
    masked_lm_positions = np.zeros((num_masked), dtype=np.int32)
    masked_lm_weights = np.zeros((num_masked), dtype=np.int32)
    counter = 0
    for i in range(len(token_ids)):
      if token_ids[i] == mask_token_id:
        masked_lm_positions[counter] = i
        masked_lm_weights[counter] = 1
        counter += 1

    actual_token_ids_tf = input_utils.mask_same_entity_mentions(
        to_tensor(token_ids),
        to_tensor(annotation_begins),
        to_tensor(annotation_ends),
        to_tensor(annotation_labels),
        to_tensor(masked_lm_positions),
        to_tensor(masked_lm_weights),
        mask_token_id,
        apply_across_whole_batch=False)
    actual_token_ids = self.evaluate(actual_token_ids_tf)
    self.assertAllEqual([expected_token_ids], actual_token_ids)

  @parameterized.named_parameters(
      dict(
          testcase_name='single',
          seq_len=10,
          batch_size=1,
          max_annotation_label=2,
          max_annotation_length=1,
          num_annotations=3,
          seed=1,
      ),
      dict(
          testcase_name='multiple',
          seq_len=100,
          batch_size=1,
          max_annotation_length=1,
          max_annotation_label=10,
          num_annotations=20,
          seed=2,
      ),
      dict(
          testcase_name='multiple_possibly_overlapping_1',
          seq_len=100,
          batch_size=1,
          max_annotation_length=10,
          max_annotation_label=5,
          num_annotations=20,
          seed=7,
      ),
      dict(
          testcase_name='multiple_possibly_overlapping_2',
          seq_len=100,
          batch_size=1,
          max_annotation_length=10000,
          max_annotation_label=10,
          num_annotations=40,
          seed=8,
      ),
      dict(
          testcase_name='batched_multiple_1',
          seq_len=100,
          batch_size=10,
          max_annotation_length=1,
          max_annotation_label=10,
          num_annotations=20,
          seed=3,
      ),
      dict(
          testcase_name='batched_multiple_possibly_overlapping_1',
          seq_len=100,
          batch_size=10,
          max_annotation_length=20,
          max_annotation_label=10,
          num_annotations=20,
          seed=8,
      ),
      dict(
          testcase_name='batched_multiple_2',
          seq_len=100,
          batch_size=10,
          max_annotation_length=1,
          max_annotation_label=10,
          num_annotations=30,
          seed=4,
      ),
      dict(
          testcase_name='batched_multiple_possibly_overlapping_2',
          seq_len=100,
          batch_size=10,
          max_annotation_length=40,
          max_annotation_label=10,
          num_annotations=30,
          seed=4,
      ),
      dict(
          testcase_name='batched_multiple_3',
          seq_len=100,
          batch_size=10,
          max_annotation_length=1,
          max_annotation_label=4,
          num_annotations=30,
          seed=5,
      ),
      dict(
          testcase_name='batched_multiple_possibly_overlapping_3',
          seq_len=100,
          batch_size=10,
          max_annotation_length=20,
          max_annotation_label=4,
          num_annotations=30,
          seed=10,
      ),
      dict(
          testcase_name='batched_multiple_4',
          seq_len=100,
          batch_size=10,
          max_annotation_length=1,
          max_annotation_label=2,
          num_annotations=30,
          seed=6,
      ),
      dict(
          testcase_name='batched_multiple_possibly_overlapping_4',
          seq_len=100,
          batch_size=10,
          max_annotation_length=20,
          max_annotation_label=2,
          num_annotations=30,
          seed=11,
      ),
  )
  def test_mask_same_entity_mentions_auto(self,
                                          seq_len,
                                          batch_size,
                                          num_annotations,
                                          max_annotation_label,
                                          max_annotation_length=5,
                                          mask_rate=0.2,
                                          seed=31415):
    np.random.seed(seed)
    vocab_size = 100
    mask_token_id = 7

    annotation_labels_np = np.random.randint(
        max_annotation_label + 1,
        size=[batch_size, num_annotations],
        dtype=np.int32)
    annotation_begins_np = np.stack([
        np.random.choice(seq_len, size=[num_annotations], replace=False)
        for _ in range(batch_size)
    ])
    annotation_begins_np = annotation_begins_np.astype(np.int32)
    annotation_length = np.random.randint(
        max_annotation_length,
        size=[batch_size, num_annotations],
        dtype=np.int32)
    annotation_ends_np = np.minimum(annotation_begins_np + annotation_length,
                                    seq_len - 1)
    token_ids_np = np.random.randint(
        vocab_size, size=[batch_size, seq_len], dtype=np.int32)
    token_mask = np.random.binomial(1, mask_rate, size=[batch_size, seq_len])
    token_mask = token_mask.astype(np.bool)
    token_ids_np[token_mask] = mask_token_id

    num_masked = (token_ids_np == mask_token_id).sum(1).max() * 2 + 5
    masked_lm_positions = np.zeros((batch_size, num_masked), dtype=np.int32)
    masked_lm_weights = np.zeros((batch_size, num_masked), dtype=np.float32)

    for i in range(batch_size):
      counter = 0
      for j in range(seq_len):
        if token_ids_np[i, j] == mask_token_id:
          masked_lm_positions[i, counter] = j
          masked_lm_weights[i, counter] = 1
          counter += 1

    entity_label_per_token = {}
    for i in range(batch_size):
      for j in range(seq_len):
        entity_label_per_token[(i, j)] = set()
        for k in range(num_annotations):
          if annotation_begins_np[i, k] <= j and j <= annotation_ends_np[i, k]:
            entity_label = annotation_labels_np[i, k]
            if entity_label not in [0, 1]:
              entity_label_per_token[(i, j)].add(entity_label)

    expected_token_ids_np = np.copy(token_ids_np)
    for i in range(batch_size):
      for j in range(seq_len):
        if not entity_label_per_token[(i, j)]:
          continue
        if token_ids_np[i, j] != mask_token_id:
          continue
        for j2 in range(seq_len):
          if (entity_label_per_token[(i, j)].intersection(
              entity_label_per_token[(i, j2)])):
            expected_token_ids_np[i, j2] = mask_token_id

    expected_token_ids_np_2 = np.copy(token_ids_np)
    for i in range(batch_size):
      for j in range(seq_len):
        if not entity_label_per_token[(i, j)]:
          continue
        if token_ids_np[i, j] != mask_token_id:
          continue
        for i2 in range(batch_size):
          for j2 in range(seq_len):
            if (entity_label_per_token[(i, j)].intersection(
                entity_label_per_token[(i2, j2)])):
              expected_token_ids_np_2[i2, j2] = mask_token_id

    annotation_begins_tf = tf.compat.v1.placeholder_with_default(
        annotation_begins_np, shape=[None, None])
    annotation_ends_tf = tf.compat.v1.placeholder_with_default(
        annotation_ends_np, shape=[None, None])
    annotation_labels_tf = tf.compat.v1.placeholder_with_default(
        annotation_labels_np, shape=[None, None])
    token_ids_tf = tf.compat.v1.placeholder_with_default(
        token_ids_np, shape=[None, None])

    actual_token_ids_tf = input_utils.mask_same_entity_mentions(
        token_ids_tf,
        annotation_begins_tf,
        annotation_ends_tf,
        annotation_labels_tf,
        masked_lm_positions,
        masked_lm_weights,
        mask_token_id,
        apply_across_whole_batch=False)
    actual_token_ids_2_tf = input_utils.mask_same_entity_mentions(
        token_ids_tf,
        annotation_begins_tf,
        annotation_ends_tf,
        annotation_labels_tf,
        masked_lm_positions,
        masked_lm_weights,
        mask_token_id,
        apply_across_whole_batch=True)

    actual_token_ids = self.evaluate(actual_token_ids_tf)
    self.assertAllEqual(expected_token_ids_np, actual_token_ids)

    actual_token_ids_2 = self.evaluate(actual_token_ids_2_tf)
    self.assertAllEqual(expected_token_ids_np_2, actual_token_ids_2)


if __name__ == '__main__':
  tf.test.main()
