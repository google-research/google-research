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

"""Tests for metric_utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from readtwice.models import metric_utils


class MetricUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="all_different",
          block_ids=[1, 2, 3],
          expected_mask=[0, 0, 0],
      ),
      dict(
          testcase_name="all_same",
          block_ids=[1, 1, 1],
          expected_mask=[1, 1, 1],
      ),
      dict(
          testcase_name="with_zeros",
          block_ids=[1, 0, 1, 2],
          expected_mask=[1, 0, 1, 0],
      ),
      dict(
          testcase_name="complex_1",
          block_ids=[0, 1, 1, 1, 2, 3, 2, 4, 1, 0],
          expected_mask=[0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
      ),
      dict(
          testcase_name="complex_2",
          block_ids=[-100, -100, -7, 0, 13, -100],
          expected_mask=[1, 1, 0, 0, 0, 1],
      ),
  )
  def test_get_multi_doc_mask(self, block_ids, expected_mask):
    block_ids_tf = tf.compat.v1.placeholder_with_default(
        np.array(block_ids).astype(np.int32), shape=[None])
    actual_mask_tf = metric_utils.get_multi_doc_mask(block_ids_tf)
    actual_mask = self.evaluate(actual_mask_tf)
    self.assertAllEqual(actual_mask, expected_mask)

  @parameterized.named_parameters(
      dict(testcase_name="all_different", block_ids=[1, 2, 3]),
      dict(testcase_name="all_same", block_ids=[1, 1, 1]),
      dict(testcase_name="with_zeros", block_ids=[1, 0, 1, 2]),
      dict(testcase_name="complex_1", block_ids=[0, 1, 1, 1, 2, 3, 2, 4, 1, 0]),
      dict(testcase_name="complex_2", block_ids=[-100, -100, -7, 0, 13, -100]),
      dict(testcase_name="random_1", block_ids=1),
      dict(testcase_name="random_2", block_ids=10),
      dict(testcase_name="random_3", block_ids=15),
      dict(testcase_name="random_4", block_ids=7),
      dict(testcase_name="random_5", block_ids=3),
  )
  def test_masked_lm_metrics(self, block_ids):
    np.random.seed(31415)
    if isinstance(block_ids, list):
      batch_size = len(block_ids)
      block_ids_np = np.array(block_ids).astype(np.int32)
    else:
      batch_size = block_ids
      block_ids_np = np.random.randint(10, size=[batch_size], dtype=np.int32)

    multi_block_mask_np = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
      if block_ids_np[i] == 0:
        continue
      for j in range(batch_size):
        if i != j and block_ids_np[i] == block_ids_np[j]:
          multi_block_mask_np[i] = 1
    single_block_mask_np = 1 - multi_block_mask_np
    mlm_loss_per_sample_np = np.random.random(batch_size).astype(np.float32)
    mlm_accuracy_per_sample_np = np.random.random(batch_size).astype(np.float32)
    mlm_weight_per_sample_np = np.random.random(batch_size).astype(np.float32)

    block_ids_tf = tf.compat.v1.placeholder_with_default(
        block_ids_np, shape=[None])
    mlm_loss_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_loss_per_sample_np, shape=[None])
    mlm_accuracy_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_accuracy_per_sample_np, shape=[None])
    mlm_weight_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_weight_per_sample_np, shape=[None])

    metric_dict = metric_utils.masked_lm_metrics(
        mlm_loss_per_sample_tf,
        mlm_accuracy_per_sample_tf,
        mlm_weight_per_sample_tf,
        block_ids_tf,
        mlm_loss_per_entity_sample=None,
        mlm_accuracy_per_entity_sample=None,
        mlm_weight_per_entity_sample=None,
        mlm_loss_per_non_entity_sample=None,
        mlm_accuracy_per_non_entity_sample=None,
        mlm_weight_per_non_entity_sample=None,
        is_train=True,
        metrics_name="abracadabra")

    (masked_lm_loss, masked_lm_accuracy, masked_lm_loss_multi_blocks,
     masked_lm_loss_single_blocks, masked_lm_accuracy_multi_blocks,
     masked_lm_accuracy_single_blocks, pct_multi_blocks,
     pct_single_blocks) = self.evaluate(
         tf.tuple((metric_dict["abracadabra/mlm_loss"],
                   metric_dict["abracadabra/mlm_accuracy"],
                   metric_dict["abracadabra/mlm_loss_multi_blocks"],
                   metric_dict["abracadabra/mlm_loss_single_blocks"],
                   metric_dict["abracadabra/mlm_accuracy_multi_blocks"],
                   metric_dict["abracadabra/mlm_accuracy_single_blocks"],
                   metric_dict["abracadabra/pct_multi_blocks"],
                   metric_dict["abracadabra/pct_single_blocks"])))

    def weighted_avg(values, weights):
      return values.dot(weights) / (weights.sum() + 1e-5)

    self.assertNear(
        masked_lm_loss,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_sample_np), 1e-5)
    self.assertNear(
        masked_lm_accuracy,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_sample_np),
        1e-5)

    mlm_weight_per_multi_block = mlm_weight_per_sample_np * multi_block_mask_np
    mlm_weight_per_single_block = mlm_weight_per_sample_np * single_block_mask_np
    self.assertNear(
        masked_lm_loss_multi_blocks,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_multi_block), 1e-5)
    self.assertNear(
        masked_lm_loss_single_blocks,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_single_block), 1e-5)
    self.assertNear(
        masked_lm_accuracy_multi_blocks,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_multi_block),
        1e-5)
    self.assertNear(
        masked_lm_accuracy_single_blocks,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_single_block),
        1e-5)
    self.assertNear(pct_multi_blocks, multi_block_mask_np.mean(), 1e-5)
    self.assertNear(pct_single_blocks, single_block_mask_np.mean(), 1e-5)

  @parameterized.named_parameters(
      dict(testcase_name="all_different", block_ids=[1, 2, 3]),
      dict(testcase_name="all_same", block_ids=[1, 1, 1]),
      dict(testcase_name="with_zeros", block_ids=[1, 0, 1, 2]),
      dict(testcase_name="complex_1", block_ids=[0, 1, 1, 1, 2, 3, 2, 4, 1, 0]),
      dict(testcase_name="complex_2", block_ids=[-100, -100, -7, 0, 13, -100]),
      dict(testcase_name="random_1", block_ids=1),
      dict(testcase_name="random_2", block_ids=10),
      dict(testcase_name="random_3", block_ids=15),
      dict(testcase_name="random_4", block_ids=7),
      dict(testcase_name="random_5", block_ids=3),
  )
  def test_masked_lm_metrics_with_entity(self, block_ids):
    np.random.seed(31415)
    if isinstance(block_ids, list):
      batch_size = len(block_ids)
      block_ids_np = np.array(block_ids).astype(np.int32)
    else:
      batch_size = block_ids
      block_ids_np = np.random.randint(10, size=[batch_size], dtype=np.int32)

    multi_block_mask_np = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
      if block_ids_np[i] == 0:
        continue
      for j in range(batch_size):
        if i != j and block_ids_np[i] == block_ids_np[j]:
          multi_block_mask_np[i] = 1
    single_block_mask_np = 1 - multi_block_mask_np
    mlm_loss_per_sample_np = np.random.random(batch_size).astype(np.float32)
    mlm_accuracy_per_sample_np = np.random.random(batch_size).astype(np.float32)
    mlm_weight_per_sample_np = np.random.random(batch_size).astype(np.float32)
    mlm_loss_per_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)
    mlm_accuracy_per_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)
    mlm_weight_per_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)
    mlm_loss_per_non_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)
    mlm_accuracy_per_non_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)
    mlm_weight_per_non_entity_sample_np = np.random.random(batch_size).astype(
        np.float32)

    block_ids_tf = tf.compat.v1.placeholder_with_default(
        block_ids_np, shape=[None])
    mlm_loss_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_loss_per_sample_np, shape=[None])
    mlm_accuracy_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_accuracy_per_sample_np, shape=[None])
    mlm_weight_per_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_weight_per_sample_np, shape=[None])
    mlm_loss_per_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_loss_per_entity_sample_np, shape=[None])
    mlm_accuracy_per_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_accuracy_per_entity_sample_np, shape=[None])
    mlm_weight_per_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_weight_per_entity_sample_np, shape=[None])
    mlm_loss_per_non_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_loss_per_non_entity_sample_np, shape=[None])
    mlm_accuracy_per_non_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_accuracy_per_non_entity_sample_np, shape=[None])
    mlm_weight_per_non_entity_sample_tf = tf.compat.v1.placeholder_with_default(
        mlm_weight_per_non_entity_sample_np, shape=[None])

    metric_dict = metric_utils.masked_lm_metrics(
        mlm_loss_per_sample_tf,
        mlm_accuracy_per_sample_tf,
        mlm_weight_per_sample_tf,
        block_ids_tf,
        mlm_loss_per_entity_sample=mlm_loss_per_entity_sample_tf,
        mlm_accuracy_per_entity_sample=mlm_accuracy_per_entity_sample_tf,
        mlm_weight_per_entity_sample=mlm_weight_per_entity_sample_tf,
        mlm_loss_per_non_entity_sample=mlm_loss_per_non_entity_sample_tf,
        mlm_accuracy_per_non_entity_sample=mlm_accuracy_per_non_entity_sample_tf,
        mlm_weight_per_non_entity_sample=mlm_weight_per_non_entity_sample_tf,
        is_train=True,
        metrics_name="this_is_sparta")

    (masked_lm_loss, masked_lm_accuracy, masked_lm_loss_multi_blocks,
     masked_lm_loss_single_blocks, masked_lm_accuracy_multi_blocks,
     masked_lm_accuracy_single_blocks, masked_lm_loss_entity,
     masked_lm_accuracy_entity, masked_lm_loss_entity_multi_blocks,
     masked_lm_loss_entity_single_blocks,
     masked_lm_accuracy_entity_multi_blocks,
     masked_lm_accuracy_entity_single_blocks, masked_lm_loss_non_entity,
     masked_lm_accuracy_non_entity, masked_lm_loss_non_entity_multi_blocks,
     masked_lm_loss_non_entity_single_blocks,
     masked_lm_accuracy_non_entity_multi_blocks,
     masked_lm_accuracy_non_entity_single_blocks, pct_multi_blocks,
     pct_single_blocks
    ) = self.evaluate(
        tf.tuple((
            metric_dict["this_is_sparta/mlm_loss"],
            metric_dict["this_is_sparta/mlm_accuracy"],
            metric_dict["this_is_sparta/mlm_loss_multi_blocks"],
            metric_dict["this_is_sparta/mlm_loss_single_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_multi_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_single_blocks"],
            metric_dict["this_is_sparta/mlm_loss_entity"],
            metric_dict["this_is_sparta/mlm_accuracy_entity"],
            metric_dict["this_is_sparta/mlm_loss_entity_multi_blocks"],
            metric_dict["this_is_sparta/mlm_loss_entity_single_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_entity_multi_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_entity_single_blocks"],
            metric_dict["this_is_sparta/mlm_loss_non_entity"],
            metric_dict["this_is_sparta/mlm_accuracy_non_entity"],
            metric_dict["this_is_sparta/mlm_loss_non_entity_multi_blocks"],
            metric_dict["this_is_sparta/mlm_loss_non_entity_single_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_non_entity_multi_blocks"],
            metric_dict["this_is_sparta/mlm_accuracy_non_entity_single_blocks"],
            metric_dict["this_is_sparta/pct_multi_blocks"],
            metric_dict["this_is_sparta/pct_single_blocks"])))

    def weighted_avg(values, weights):
      return values.dot(weights) / (weights.sum() + 1e-5)

    self.assertNear(
        masked_lm_loss,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_sample_np), 1e-5)
    self.assertNear(
        masked_lm_accuracy,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_sample_np),
        1e-5)
    self.assertNear(
        masked_lm_loss_entity,
        weighted_avg(mlm_loss_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np), 1e-5)
    self.assertNear(
        masked_lm_accuracy_entity,
        weighted_avg(mlm_accuracy_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np), 1e-5)
    self.assertNear(
        masked_lm_loss_non_entity,
        weighted_avg(mlm_loss_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np), 1e-5)
    self.assertNear(
        masked_lm_accuracy_non_entity,
        weighted_avg(mlm_accuracy_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np), 1e-5)

    mlm_weight_per_multi_block = mlm_weight_per_sample_np * multi_block_mask_np
    mlm_weight_per_single_block = mlm_weight_per_sample_np * single_block_mask_np
    self.assertNear(
        masked_lm_loss_multi_blocks,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_multi_block), 1e-5)
    self.assertNear(
        masked_lm_loss_single_blocks,
        weighted_avg(mlm_loss_per_sample_np, mlm_weight_per_single_block), 1e-5)
    self.assertNear(
        masked_lm_loss_entity_multi_blocks,
        weighted_avg(mlm_loss_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np * multi_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_loss_entity_single_blocks,
        weighted_avg(mlm_loss_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np * single_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_loss_non_entity_multi_blocks,
        weighted_avg(mlm_loss_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np * multi_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_loss_non_entity_single_blocks,
        weighted_avg(mlm_loss_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np *
                     single_block_mask_np), 1e-5)

    self.assertNear(
        masked_lm_accuracy_multi_blocks,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_multi_block),
        1e-5)
    self.assertNear(
        masked_lm_accuracy_single_blocks,
        weighted_avg(mlm_accuracy_per_sample_np, mlm_weight_per_single_block),
        1e-5)

    self.assertNear(
        masked_lm_accuracy_entity_multi_blocks,
        weighted_avg(mlm_accuracy_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np * multi_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_accuracy_entity_single_blocks,
        weighted_avg(mlm_accuracy_per_entity_sample_np,
                     mlm_weight_per_entity_sample_np * single_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_accuracy_non_entity_multi_blocks,
        weighted_avg(mlm_accuracy_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np * multi_block_mask_np),
        1e-5)
    self.assertNear(
        masked_lm_accuracy_non_entity_single_blocks,
        weighted_avg(mlm_accuracy_per_non_entity_sample_np,
                     mlm_weight_per_non_entity_sample_np *
                     single_block_mask_np), 1e-5)

    self.assertNear(pct_multi_blocks, multi_block_mask_np.mean(), 1e-5)
    self.assertNear(pct_single_blocks, single_block_mask_np.mean(), 1e-5)


if __name__ == "__main__":
  tf.test.main()
