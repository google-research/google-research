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

"""Tests for losses library."""

from absl.testing import parameterized
import numpy as np
import scipy.special
import tensorflow.compat.v1 as tf

from readtwice.models import losses


class BatchSpanCrossEntropyLossTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="single_block_1",
          seq_length=13,
          logits_magnitude=1,
          inner_dimensions=1,
          block_ids=[1]),
      dict(
          testcase_name="single_block_2",
          seq_length=13,
          logits_magnitude=1,
          inner_dimensions=2,
          block_ids=[1]),
      dict(
          testcase_name="single_block_3",
          seq_length=13,
          logits_magnitude=100,
          inner_dimensions=2,
          block_ids=[1]),
      dict(
          testcase_name="single_block_4",
          seq_length=13,
          logits_magnitude=1000,
          inner_dimensions=2,
          block_ids=[1]),
      dict(
          testcase_name="multi_same_block_1",
          seq_length=13,
          logits_magnitude=100,
          inner_dimensions=2,
          block_ids=[1, 1, 1]),
      dict(
          testcase_name="multi_same_block_2",
          seq_length=7,
          logits_magnitude=1000,
          inner_dimensions=5,
          block_ids=[3, 3, 3]),
      dict(
          testcase_name="multi_block_1",
          seq_length=13,
          logits_magnitude=100,
          inner_dimensions=2,
          block_ids=[1, 2, 3]),
      dict(
          testcase_name="multi_block_2",
          seq_length=13,
          logits_magnitude=100,
          inner_dimensions=2,
          block_ids=[1, 1, 2]),
      dict(
          testcase_name="multi_block_3",
          seq_length=7,
          logits_magnitude=100,
          inner_dimensions=3,
          block_ids=[100, 2, 100, 3, 100]),
      dict(
          testcase_name="multi_block_4",
          seq_length=7,
          logits_magnitude=1000,
          inner_dimensions=5,
          block_ids=[1, 2, 3, 4, 5, 4, 3, 2, 1]),
      dict(
          testcase_name="multi_block_5",
          seq_length=512,
          logits_magnitude=10000000,
          inner_dimensions=2,
          block_ids=[1, 2, 3, 4, 5, 4, 3, 2, 1]),
      dict(
          testcase_name="multi_block_6",
          seq_length=10000,
          logits_magnitude=100000000,
          inner_dimensions=2,
          block_ids=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
  )
  def test_cross_batch_softmax(self, seq_length, logits_magnitude,
                               inner_dimensions, block_ids):
    np.random.seed(31415)

    block_ids = np.array(block_ids)
    batch_size = block_ids.shape[0]

    logits = np.random.random((batch_size, seq_length, inner_dimensions))
    logits = (logits - 0.5) * logits_magnitude
    logits = logits.astype(np.float32)

    probs = np.zeros_like(logits)

    for inner_dimension in range(inner_dimensions):
      for sample_index in range(batch_size):
        current_logits = [logits[sample_index, :, inner_dimension]]
        for sample_index_other in range(batch_size):
          if (sample_index != sample_index_other and
              block_ids[sample_index] == block_ids[sample_index_other]):
            current_logits.append(logits[sample_index_other, :,
                                         inner_dimension])

        current_probs = scipy.special.softmax(np.concatenate(current_logits))
        probs[sample_index, :, inner_dimension] = current_probs[:seq_length]

    cross_blocks_eq_mask = np.zeros((batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
      for j in range(batch_size):
        cross_blocks_eq_mask[i, j] = (block_ids[i] == block_ids[j])

    logits_tf = tf.compat.v1.placeholder_with_default(
        logits, shape=[None, None, inner_dimensions])
    cross_blocks_eq_mask_tf = tf.compat.v1.placeholder_with_default(
        cross_blocks_eq_mask, shape=[None, None])

    probs_tf = losses.cross_batch_softmax(logits_tf, cross_blocks_eq_mask_tf)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)

    probs_tf_result = self.evaluate(probs_tf)

    self.assertArrayNear(probs.flatten(), probs_tf_result.flatten(), err=1e-4)

  # When all samples are from different documents (have unique block_ids)
  # then the function should be equivalent to the SpanOrCrossEntropyLoss
  @parameterized.named_parameters(
      dict(
          testcase_name="single_seq",
          batch_size=1,
          seq_length=5,
          num_annotations=1),
      dict(
          testcase_name="batch", batch_size=3, seq_length=5, num_annotations=1),
      dict(
          testcase_name="multi_annotations",
          batch_size=1,
          seq_length=13,
          num_annotations=7),
      dict(
          testcase_name="batch_multi_annotations_1",
          batch_size=4,
          seq_length=13,
          num_annotations=7),
      dict(
          testcase_name="batch_multi_annotations_2",
          batch_size=8,
          seq_length=11,
          num_annotations=3),
  )
  def test_cross_entropy_loss_unique_block_ids(self, batch_size, seq_length,
                                               num_annotations):
    np.random.seed(31415)

    logits = np.random.random((batch_size, seq_length, 2))
    logits = (logits - 0.5) * 100
    logits = logits.astype(np.float32)

    annotation_begins = np.stack([
        np.random.choice(seq_length, size=num_annotations, replace=False)
        for _ in range(batch_size)
    ])
    annotation_ends = np.stack([
        np.random.choice(seq_length, size=num_annotations, replace=False)
        for _ in range(batch_size)
    ])
    one_hot_labels = np.zeros((batch_size, seq_length, 2), dtype=np.float32)
    for i in range(batch_size):
      one_hot_labels[i, annotation_begins[i], 0] = 1
      one_hot_labels[i, annotation_ends[i], 1] = 1

    logits_tf = tf.compat.v1.placeholder_with_default(
        logits, shape=[None, None, 2])
    block_ids = tf.range(batch_size)
    annotation_begins_tf = tf.compat.v1.placeholder_with_default(
        annotation_begins, shape=[None, None])
    annotation_ends_tf = tf.compat.v1.placeholder_with_default(
        annotation_ends, shape=[None, None])
    annotation_labels = tf.ones((batch_size, num_annotations), dtype=tf.float32)
    one_hot_labels_tf = tf.compat.v1.placeholder_with_default(
        one_hot_labels, shape=[None, None, 2])

    loss_layer = losses.BatchSpanCrossEntropyLoss()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)

    actual_loss = loss_layer(logits_tf, annotation_begins_tf,
                             annotation_ends_tf, annotation_labels, block_ids)

    logits_masked = logits - tf.cast(one_hot_labels_tf < 0.5, tf.float32) * 1e6
    or_cross_entropy = (
        tf.math.reduce_logsumexp(logits_tf, axis=-2) -
        tf.math.reduce_logsumexp(logits_masked, axis=-2))
    expected_loss = tf.math.reduce_sum(or_cross_entropy)

    actual_loss_value, expected_loss_value = self.evaluate(
        [actual_loss, expected_loss])

    self.assertNear(actual_loss_value, expected_loss_value, err=1e-4)

  @parameterized.named_parameters(
      dict(
          testcase_name="single_sample_single_annotation",
          seq_length=10,
          block_ids=[111],
          annotation_begins=[[0]],
          annotation_ends=[[1]],
          annotation_labels=[[1]]),
      dict(
          testcase_name="single_sample_1",
          seq_length=10,
          block_ids=[111],
          annotation_begins=[[0, 3, 9]],
          annotation_ends=[[1, 5, 9]],
          annotation_labels=[[1, 1, 0]]),
      dict(
          testcase_name="single_sample_2",
          seq_length=10,
          block_ids=[111],
          annotation_begins=[[0, 3, 9, 0, 3]],
          annotation_ends=[[1, 5, 9, 1, 5]],
          annotation_labels=[[1, 1, 0, 0, 0]]),
      dict(
          testcase_name="single_sample_no_annotations_1",
          seq_length=10,
          block_ids=[111],
          annotation_begins=[[0]],
          annotation_ends=[[1]],
          annotation_labels=[[0]]),
      dict(
          testcase_name="single_sample_no_annotations_2",
          seq_length=10,
          block_ids=[111],
          annotation_begins=[[0, 3, 9]],
          annotation_ends=[[1, 5, 9]],
          annotation_labels=[[0, 0, 0]]),
      dict(
          testcase_name="same_doc_1",
          seq_length=7,
          block_ids=[7, 7, 7],
          annotation_begins=[[0], [1], [2]],
          annotation_ends=[[1], [2], [3]],
          annotation_labels=[[1], [1], [1]]),
      dict(
          testcase_name="same_doc_2",
          seq_length=7,
          block_ids=[7, 7, 7],
          annotation_begins=[[0, 0], [1, 6], [0, 3]],
          annotation_ends=[[1, 1], [3, 6], [3, 6]],
          annotation_labels=[[1, 0], [1, 1], [0, 0]]),
      dict(
          testcase_name="same_doc_no_annotations_1",
          seq_length=7,
          block_ids=[7, 7, 7],
          annotation_begins=[[0], [1], [2]],
          annotation_ends=[[1], [2], [3]],
          annotation_labels=[[0], [0], [0]]),
      dict(
          testcase_name="same_doc_no_annotations_2",
          seq_length=7,
          block_ids=[7, 7, 7],
          annotation_begins=[[0, 0], [1, 6], [0, 3]],
          annotation_ends=[[1, 1], [3, 6], [3, 6]],
          annotation_labels=[[0, 0], [0, 0], [0, 0]]),
      dict(
          testcase_name="multi_doc_1",
          seq_length=7,
          block_ids=[2, 2, 1],
          annotation_begins=[[0], [1], [2]],
          annotation_ends=[[1], [2], [3]],
          annotation_labels=[[1], [1], [1]]),
      dict(
          testcase_name="multi_doc_2",
          seq_length=7,
          block_ids=[101, 5, 101],
          annotation_begins=[[0, 0], [1, 6], [0, 3]],
          annotation_ends=[[1, 1], [3, 6], [3, 6]],
          annotation_labels=[[1, 0], [1, 1], [0, 0]]),
      dict(
          testcase_name="multi_doc_3",
          seq_length=7,
          block_ids=[1, 2, 3, 2, 1],
          annotation_begins=[[0], [1], [2], [3], [4]],
          annotation_ends=[[0], [1], [2], [3], [4]],
          annotation_labels=[[1], [1], [1], [0], [0]]),
      dict(
          testcase_name="multi_doc_4",
          seq_length=9,
          block_ids=[1, 2, 3, 2, 1],
          annotation_begins=[[0], [1], [2], [3], [4]],
          annotation_ends=[[1], [2], [3], [4], [5]],
          annotation_labels=[[1], [0], [1], [0], [0]]),
      dict(
          testcase_name="multi_doc_no_annotations_1",
          seq_length=7,
          block_ids=[2, 2, 1],
          annotation_begins=[[0], [1], [2]],
          annotation_ends=[[1], [2], [3]],
          annotation_labels=[[0], [0], [1]]),
      dict(
          testcase_name="multi_doc_no_annotations_2",
          seq_length=7,
          block_ids=[101, 5, 101],
          annotation_begins=[[0, 0], [1, 6], [0, 3]],
          annotation_ends=[[1, 1], [3, 6], [3, 6]],
          annotation_labels=[[0, 0], [1, 1], [0, 0]]),
  )
  def test_cross_entropy_loss(self, seq_length, block_ids, annotation_begins,
                              annotation_ends, annotation_labels):
    np.random.seed(31415)

    unique_block_ids = set(block_ids)
    batch_size = len(block_ids)
    num_annotations = len(annotation_begins[0])
    for i in range(batch_size):
      self.assertLen(annotation_begins[i], num_annotations)
      self.assertLen(annotation_ends[i], num_annotations)
      self.assertLen(annotation_labels[i], num_annotations)

    logits = np.random.random((batch_size, seq_length, 2))
    logits = (logits - 0.5) * 100
    logits = logits.astype(np.float32)

    expected_loss_np = 0
    for block_id in unique_block_ids:
      current_indices = [
          i for i in range(batch_size) if block_ids[i] == block_id
      ]
      current_begin_logits = np.concatenate(
          [logits[i, :, 0] for i in current_indices])
      current_end_logits = np.concatenate(
          [logits[i, :, 1] for i in current_indices])
      current_begin_probs = scipy.special.softmax(current_begin_logits)
      current_end_probs = scipy.special.softmax(current_end_logits)

      current_begins, current_ends = set(), set()
      for i, sample_index in enumerate(current_indices):
        for j in range(num_annotations):
          if annotation_labels[sample_index][j] > 0:
            current_begins.add(annotation_begins[sample_index][j] +
                               i * seq_length)
            current_ends.add(annotation_ends[sample_index][j] + i * seq_length)

      if not current_begins:
        self.assertEmpty(current_ends)
        continue
      else:
        self.assertNotEmpty(current_ends)

      expected_loss_np -= (
          np.log(sum([current_begin_probs[i] for i in current_begins])) +
          np.log(sum([current_end_probs[i] for i in current_ends])))

    logits_tf = tf.compat.v1.placeholder_with_default(
        logits, shape=[None, None, 2])
    block_ids_tf = tf.compat.v1.placeholder_with_default(
        block_ids, shape=[None])
    annotation_begins_tf = tf.compat.v1.placeholder_with_default(
        annotation_begins, shape=[None, None])
    annotation_ends_tf = tf.compat.v1.placeholder_with_default(
        annotation_ends, shape=[None, None])
    annotation_labels_tf = tf.compat.v1.placeholder_with_default(
        annotation_labels, shape=[None, None])

    loss_layer = losses.BatchSpanCrossEntropyLoss()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)

    actual_loss = loss_layer(logits_tf, annotation_begins_tf,
                             annotation_ends_tf, annotation_labels_tf,
                             block_ids_tf)
    actual_loss_value = self.evaluate(actual_loss)

    self.assertNear(actual_loss_value, expected_loss_np, err=1e-4)


class BatchCoreferenceResolutionLossTest(tf.test.TestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="simple",
          apply_linear_layer=False,
          hidden_size=1,
          block_ids=[1, 2, 3],
          global_block_ids=[1, 2, 3],
          use_label_weights=False),
      dict(
          testcase_name="simple_2",
          apply_linear_layer=False,
          hidden_size=1,
          block_ids=[1, 1, 1],
          global_block_ids=[1, 1, 1],
          use_label_weights=False),
      dict(
          testcase_name="simple_3",
          apply_linear_layer=False,
          hidden_size=7,
          block_ids=[1, 1, 1, 2, 3],
          global_block_ids=[1, 1, 1, 5, 0, 0, 2, 0],
          use_label_weights=False),
      dict(
          testcase_name="empty",
          apply_linear_layer=False,
          hidden_size=7,
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0, 0],
          use_label_weights=False),
      dict(
          testcase_name="apply_linear.simple",
          apply_linear_layer=True,
          hidden_size=1,
          block_ids=[1, 2, 3],
          global_block_ids=[1, 2, 3],
          use_label_weights=False),
      dict(
          testcase_name="apply_linear.simple_2",
          apply_linear_layer=True,
          hidden_size=1,
          block_ids=[1, 1, 1],
          global_block_ids=[1, 1, 1],
          use_label_weights=False),
      dict(
          testcase_name="apply_linear.simple_3",
          apply_linear_layer=True,
          hidden_size=7,
          block_ids=[1, 1, 1, 2, 3],
          global_block_ids=[1, 1, 1, 5, 0, 0, 2, 0],
          use_label_weights=False),
      dict(
          testcase_name="apply_linear.empty",
          apply_linear_layer=True,
          hidden_size=7,
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0, 0],
          use_label_weights=False),
      dict(
          testcase_name="simple.with_weights",
          apply_linear_layer=False,
          hidden_size=2,
          block_ids=[1, 2, 3, 4],
          global_block_ids=[1, 2, 3, 4],
          use_label_weights=True),
      dict(
          testcase_name="simple_2.with_weights",
          apply_linear_layer=False,
          hidden_size=2,
          block_ids=[1, 1, 1],
          global_block_ids=[1, 1, 1],
          use_label_weights=True),
      dict(
          testcase_name="simple_3.with_weights",
          apply_linear_layer=False,
          hidden_size=6,
          block_ids=[1, 1, 1, 2, 3],
          global_block_ids=[1, 1, 1, 5, 0, 0, 2, 0],
          use_label_weights=True),
      dict(
          testcase_name="empty.with_weights",
          apply_linear_layer=False,
          hidden_size=5,
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0, 0],
          use_label_weights=True),
      dict(
          testcase_name="apply_linear.simple.with_weights",
          apply_linear_layer=True,
          hidden_size=1,
          block_ids=[1, 2, 3],
          global_block_ids=[1, 2, 3],
          use_label_weights=True),
      dict(
          testcase_name="apply_linear.simple_2.with_weights",
          apply_linear_layer=True,
          hidden_size=3,
          block_ids=[1, 1, 1],
          global_block_ids=[1, 1, 1],
          use_label_weights=True),
      dict(
          testcase_name="apply_linear.simple_3.with_weights",
          apply_linear_layer=True,
          hidden_size=8,
          block_ids=[1, 1, 1, 2, 3],
          global_block_ids=[1, 1, 1, 5, 0, 0, 2, 0],
          use_label_weights=True),
      dict(
          testcase_name="apply_linear.empty.with_weights",
          apply_linear_layer=True,
          hidden_size=9,
          block_ids=[0, 0],
          global_block_ids=[0, 0, 0, 0],
          use_label_weights=True),
  )
  def test_batch_coreferense_resolution_loss(self, apply_linear_layer,
                                             hidden_size, block_ids,
                                             global_block_ids,
                                             use_label_weights):
    np.random.seed(31415)

    block_ids_np = np.array(block_ids)
    global_block_ids = np.array(global_block_ids)
    item_states_np = np.random.random((len(block_ids), hidden_size))
    item_states_np = item_states_np.astype(np.float32)
    global_item_states_np = np.random.random(
        (len(global_block_ids), hidden_size))
    global_item_states_np = global_item_states_np.astype(np.float32)
    global_block_ids = np.concatenate([block_ids_np, global_block_ids], axis=0)
    global_item_states_np = np.concatenate(
        [item_states_np, global_item_states_np], axis=0)
    if use_label_weights:
      labels_weight_np = np.random.random(
          (len(block_ids), len(global_block_ids)))
      labels_weight_np = labels_weight_np.astype(np.float32)

    if not apply_linear_layer:
      loss_np = 0
      for i in range(len(block_ids)):
        if block_ids[i] == 0:
          continue
        num_predictions_per_sample, loss_np_per_sample = 0, 0
        for j in range(len(global_block_ids)):
          if global_block_ids[j] == 0:
            continue
          if j == i:
            # don't compute loss when comparing summary to itself
            continue
          x = np.dot(item_states_np[i], global_item_states_np[j])
          z = int(block_ids[i] == global_block_ids[j])
          # pylint: disable=line-too-long
          # See https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
          # pylint: enable=line-too-long
          current_loss = np.max(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))
          if use_label_weights:
            loss_np_per_sample += current_loss * labels_weight_np[i, j]
            num_predictions_per_sample += labels_weight_np[i, j]
          else:
            loss_np_per_sample += current_loss
            num_predictions_per_sample += 1
        loss_np += loss_np_per_sample / (num_predictions_per_sample + 1e-5)

    item_states = tf.compat.v1.placeholder_with_default(
        item_states_np, shape=[None, hidden_size])
    global_item_states = tf.compat.v1.placeholder_with_default(
        global_item_states_np, shape=[None, hidden_size])
    block_ids_tf = tf.compat.v1.placeholder_with_default(
        block_ids_np, shape=[None])
    global_block_ids_tf = tf.compat.v1.placeholder_with_default(
        global_block_ids, shape=[None])

    if use_label_weights:
      labels_weight_tf = tf.compat.v1.placeholder_with_default(
          labels_weight_np, shape=[None, None])
    else:
      labels_weight_tf = None

    loss_fn = losses.BatchCoreferenceResolutionLoss(apply_linear_layer)

    loss = loss_fn(
        item_states,
        block_ids_tf,
        global_item_states,
        global_block_ids_tf,
        labels_weight=labels_weight_tf)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)

    loss_result = self.evaluate(loss)
    self.assertAllEqual(loss_result.shape, [])

    if not apply_linear_layer:
      self.assertNear(loss_result, loss_np, err=1e-4)


class LanguageModelLossTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="autoregressive_lm",
          num_positions=None,
          padding_token_id=None,
          use_label_weights=False,
          use_entity_mask=False,
          seed=1),
      dict(
          testcase_name="autoregressive_lm_pad0",
          num_positions=None,
          padding_token_id=0,
          use_label_weights=False,
          use_entity_mask=False,
          seed=2),
      dict(
          testcase_name="autoregressive_lm_weighted",
          num_positions=None,
          padding_token_id=None,
          use_label_weights=True,
          use_entity_mask=False,
          seed=3),
      dict(
          testcase_name="autoregressive_lm_weighted_pad0",
          num_positions=None,
          padding_token_id=0,
          use_label_weights=True,
          use_entity_mask=False,
          seed=4),
      dict(
          testcase_name="masked_lm_1",
          num_positions=1,
          padding_token_id=None,
          use_label_weights=False,
          use_entity_mask=False,
          seed=5),
      dict(
          testcase_name="masked_lm_1_with_entity_mask",
          num_positions=1,
          padding_token_id=None,
          use_label_weights=False,
          use_entity_mask=True,
          seed=6),
      dict(
          testcase_name="masked_lm_1_pad0_with_entity_mask",
          num_positions=1,
          padding_token_id=0,
          use_label_weights=False,
          use_entity_mask=True,
          seed=7),
      dict(
          testcase_name="masked_lm_1_pad0",
          num_positions=1,
          padding_token_id=0,
          use_label_weights=False,
          use_entity_mask=False,
          seed=8),
      dict(
          testcase_name="masked_lm_1_weighted_with_entity_mask",
          num_positions=1,
          padding_token_id=None,
          use_label_weights=True,
          use_entity_mask=True,
          seed=9),
      dict(
          testcase_name="masked_lm_1_weighted",
          num_positions=1,
          padding_token_id=None,
          use_label_weights=True,
          use_entity_mask=False,
          seed=10),
      dict(
          testcase_name="masked_lm_1_weighted_pad0_with_entity_mask",
          num_positions=1,
          padding_token_id=0,
          use_label_weights=True,
          use_entity_mask=True,
          seed=11),
      dict(
          testcase_name="masked_lm_1_weighted_pad0",
          num_positions=1,
          padding_token_id=0,
          use_label_weights=True,
          use_entity_mask=False,
          seed=12),
      dict(
          testcase_name="masked_lm_2_with_entity_mask",
          num_positions=2,
          padding_token_id=None,
          use_label_weights=False,
          use_entity_mask=True,
          seed=13),
      dict(
          testcase_name="masked_lm_2",
          num_positions=2,
          padding_token_id=None,
          use_label_weights=False,
          use_entity_mask=False,
          seed=14),
      dict(
          testcase_name="masked_lm_2_pad0_with_entity_mask",
          num_positions=2,
          padding_token_id=0,
          use_label_weights=False,
          use_entity_mask=True,
          seed=15),
      dict(
          testcase_name="masked_lm_2_pad0",
          num_positions=2,
          padding_token_id=0,
          use_label_weights=False,
          use_entity_mask=False,
          seed=16),
      dict(
          testcase_name="masked_lm_2_weighted_with_entity_mask",
          num_positions=2,
          padding_token_id=None,
          use_label_weights=True,
          use_entity_mask=True,
          seed=17),
      dict(
          testcase_name="masked_lm_2_weighted",
          num_positions=2,
          padding_token_id=None,
          use_label_weights=True,
          use_entity_mask=False,
          seed=18),
      dict(
          testcase_name="masked_lm_2_weighted_pad0_with_entity_mask",
          num_positions=2,
          padding_token_id=0,
          use_label_weights=True,
          use_entity_mask=True,
          seed=19),
      dict(
          testcase_name="masked_lm_2_weighted_pad0",
          num_positions=2,
          padding_token_id=0,
          use_label_weights=True,
          use_entity_mask=False,
          seed=20),
  )
  def test_language_model_test(self, num_positions, padding_token_id,
                               use_label_weights, use_entity_mask, seed):
    np.random.seed(seed)

    seq_length = 13
    batch_size = 7
    vocab_size = 11
    hidden_size = 3
    embedding_size = 5

    embedding_table_np = np.random.random(
        (vocab_size, embedding_size)).astype(np.float32)
    embedding_table = tf.compat.v1.placeholder_with_default(
        embedding_table_np, shape=[vocab_size, embedding_size])

    input_tensor_np = np.random.random(
        (batch_size, seq_length, hidden_size)).astype(np.float32)
    input_tensor = tf.compat.v1.placeholder_with_default(
        input_tensor_np, shape=[None, None, hidden_size])

    num_labels_ids = num_positions or seq_length
    label_ids_np = np.random.randint(
        vocab_size, size=[batch_size, num_labels_ids], dtype=np.int32)
    label_ids = tf.compat.v1.placeholder_with_default(
        label_ids_np, shape=[None, num_labels_ids])

    if num_positions:
      positions_np = np.random.randint(
          seq_length, size=[batch_size, num_positions], dtype=np.int32)
      positions = tf.compat.v1.placeholder_with_default(
          positions_np, shape=[None, num_positions])
    else:
      positions = None

    if padding_token_id is not None:
      pad_mask = (label_ids_np != padding_token_id).astype(np.float32)
    else:
      pad_mask = np.ones((batch_size, num_labels_ids))

    if use_label_weights:
      label_weights_np = np.random.random(
          (batch_size, num_labels_ids)).astype(np.float32)
      label_weights = tf.compat.v1.placeholder_with_default(
          label_weights_np, shape=[None, num_labels_ids])
    else:
      label_weights_np = np.ones((batch_size, num_labels_ids))
      label_weights = None
    label_weights_np *= pad_mask

    if use_entity_mask:
      entity_mask_np = np.random.binomial(
          1, 0.5, size=(batch_size, num_labels_ids))
      entity_mask = tf.compat.v1.placeholder_with_default(
          entity_mask_np.astype(np.float32), shape=[None, num_labels_ids])
      non_entity_mask = 1 - entity_mask
    else:
      entity_mask = None
      non_entity_mask = None

    loss_fn = losses.LanguageModelLoss(
        embedding_table, activation="relu", hidden_size=hidden_size)

    loss_obj = loss_fn(input_tensor, label_ids, positions, label_weights,
                       padding_token_id, entity_mask, non_entity_mask)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.evaluate(init_op)

    self.assertEqual(
        loss_fn.linear_fn.bias.name,
        "language_model_loss/cls/predictions/transform/dense/bias:0")
    self.assertEqual(
        loss_fn.linear_fn.kernel.name,
        "language_model_loss/cls/predictions/transform/dense/kernel:0")

    weight_np = self.evaluate(loss_fn.linear_fn.kernel)

    if num_positions:
      input_tensor_np_new = np.zeros((batch_size, num_positions, hidden_size))
      for i in range(batch_size):
        for j in range(num_positions):
          input_tensor_np_new[i, j] = input_tensor_np[i, positions_np[i, j]]
      input_tensor_np = input_tensor_np_new
    x = np.dot(
        input_tensor_np.reshape(batch_size * num_labels_ids, hidden_size),
        weight_np)
    x = np.maximum(x, 0)
    x -= x.mean(axis=1, keepdims=True)
    var_x = (x**2).mean(axis=1, keepdims=True)
    x /= np.sqrt(var_x + 0.001)
    logits = np.dot(x, np.transpose(embedding_table_np))
    log_probs = np.log(scipy.special.softmax(logits, axis=1)).reshape(
        batch_size, num_labels_ids, vocab_size)

    loss_np = 0
    mlm_loss_per_sample_np = np.zeros(batch_size)
    mlm_accuracy_per_sample_np = np.zeros(batch_size)
    mlm_loss_per_entity_sample_np = np.zeros(batch_size)
    mlm_accuracy_per_entity_sample_np = np.zeros(batch_size)
    mlm_loss_per_non_entity_sample_np = np.zeros(batch_size)
    mlm_accuracy_per_non_entity_sample_np = np.zeros(batch_size)
    for i in range(batch_size):
      for j in range(num_labels_ids):
        current_loss = -log_probs[i, j, label_ids_np[i, j]]
        current_loss *= label_weights_np[i, j]
        current_accuracy = int(np.argmax(log_probs[i, j]) == label_ids_np[i, j])
        current_accuracy *= label_weights_np[i, j]
        loss_np += current_loss
        mlm_loss_per_sample_np[i] += current_loss
        mlm_accuracy_per_sample_np[i] += current_accuracy
        if use_entity_mask:
          if entity_mask_np[i, j] == 1:
            mlm_loss_per_entity_sample_np[i] += current_loss
            mlm_accuracy_per_entity_sample_np[i] += current_accuracy
          else:
            mlm_loss_per_non_entity_sample_np[i] += current_loss
            mlm_accuracy_per_non_entity_sample_np[i] += current_accuracy
    loss_np /= (label_weights_np.sum() + 1e-5)
    mlm_weight_per_sample_np = label_weights_np.sum(axis=1)
    mlm_loss_per_sample_np /= (mlm_weight_per_sample_np + 1e-5)
    mlm_accuracy_per_sample_np /= (mlm_weight_per_sample_np + 1e-5)
    if use_entity_mask:
      mlm_loss_per_entity_sample_np /= (
          (label_weights_np * entity_mask_np).sum(axis=1) + 1e-5)
      mlm_accuracy_per_entity_sample_np /= (
          (label_weights_np * entity_mask_np).sum(axis=1) + 1e-5)
      mlm_loss_per_non_entity_sample_np /= ((label_weights_np *
                                             (1 - entity_mask_np)).sum(axis=1) +
                                            1e-5)
      mlm_accuracy_per_non_entity_sample_np /= (
          (label_weights_np * (1 - entity_mask_np)).sum(axis=1) + 1e-5)

    if use_entity_mask:
      (loss, mlm_loss_per_sample, mlm_accuracy_per_sample,
       mlm_weight_per_sample, mlm_loss_per_entity_sample,
       mlm_accuracy_per_entity_sample, mlm_weight_per_entity_sample,
       mlm_loss_per_non_entity_sample, mlm_accuracy_per_non_entity_sample,
       mlm_weight_per_non_entity_sample) = self.evaluate(
           tf.tuple((loss_obj.loss, loss_obj.mlm_loss_per_sample,
                     loss_obj.mlm_accuracy_per_sample,
                     loss_obj.mlm_weight_per_sample,
                     loss_obj.mlm_loss_per_entity_sample,
                     loss_obj.mlm_accuracy_per_entity_sample,
                     loss_obj.mlm_weight_per_entity_sample,
                     loss_obj.mlm_loss_per_non_entity_sample,
                     loss_obj.mlm_accuracy_per_non_entity_sample,
                     loss_obj.mlm_weight_per_non_entity_sample)))
    else:
      (loss, mlm_loss_per_sample, mlm_accuracy_per_sample,
       mlm_weight_per_sample) = self.evaluate(
           tf.tuple((loss_obj.loss, loss_obj.mlm_loss_per_sample,
                     loss_obj.mlm_accuracy_per_sample,
                     loss_obj.mlm_weight_per_sample)))

    self.assertAllEqual(loss.shape, [])
    self.assertNear(loss, loss_np, err=1e-4)

    self.assertAllEqual(mlm_loss_per_sample.shape, [batch_size])
    self.assertArrayNear(mlm_loss_per_sample, mlm_loss_per_sample_np, err=1e-4)

    self.assertAllEqual(mlm_accuracy_per_sample.shape, [batch_size])
    self.assertArrayNear(
        mlm_accuracy_per_sample, mlm_accuracy_per_sample_np, err=1e-4)
    self.assertAllEqual(mlm_weight_per_sample.shape, [batch_size])
    self.assertArrayNear(
        mlm_accuracy_per_sample, mlm_accuracy_per_sample_np, err=1e-4)

    if use_entity_mask:
      self.assertArrayNear(
          mlm_weight_per_entity_sample,
          (label_weights_np * entity_mask_np).sum(axis=1),
          err=1e-4)
      self.assertArrayNear(
          mlm_loss_per_entity_sample, mlm_loss_per_entity_sample_np, err=1e-4)
      self.assertArrayNear(
          mlm_accuracy_per_entity_sample,
          mlm_accuracy_per_entity_sample_np,
          err=1e-4)
      self.assertArrayNear(
          mlm_weight_per_non_entity_sample,
          (label_weights_np * (1 - entity_mask_np)).sum(axis=1),
          err=1e-4)
      self.assertArrayNear(
          mlm_loss_per_non_entity_sample,
          mlm_loss_per_non_entity_sample_np,
          err=1e-4)
      self.assertArrayNear(
          mlm_accuracy_per_non_entity_sample,
          mlm_accuracy_per_non_entity_sample_np,
          err=1e-4)


if __name__ == "__main__":
  tf.test.main()
