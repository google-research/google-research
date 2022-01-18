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

"""Various loss functions to be used by ReadItTwice model."""

import tensorflow.compat.v1 as tf

from readtwice.layers import tpu_utils
from readtwice.models import losses


class BatchSpanCrossEntropyLoss(tf.keras.layers.Layer):
  """Loss function for yes/no/other and supporting facts classification."""

  def call(self,
           yesno_logits,
           yesno_labels,
           supporting_fact_logits,
           supporting_fact_labels,
           block_ids,
           num_replicas=None,
           eps=0):
    """Calls the layer.

    Args:
      yesno_logits: <float32>[batch_size, 3] Logits per position.
      supporting_fact_logits: <float32>[batch_size] Logits per position fro
        supporting facts classification.
      block_ids: <int32>[batch_size] Block IDs of every sample in the batch.
      num_replicas: Number of replicas to gather summaries from. If None
        (default) then cross-replicas summaries are not used.
      eps: <float> Small constant for numerical stability.

    Returns:
        total_loss: <float>
    """
    batch_size = tf.shape(supporting_fact_logits)[0]
    supporting_fact_logits = tf.expand_dims(supporting_fact_logits, 1)
    supporting_fact_labels = tf.expand_dims(supporting_fact_labels, 1)
    example_mask = tf.cast(
        tf.expand_dims(tf.not_equal(block_ids, 0), 1), tf.float32)

    # (1) Aggregate block_ids across global batch. Compute cross block mask.
    all_block_ids = block_ids
    if num_replicas:
      all_block_ids = tpu_utils.cross_replica_concat(
          tensor=all_block_ids,
          num_replicas=num_replicas,
          name='block_ids_concat')

    # [batch_size, global_batch_size]
    cross_blocks_eq_mask = tf.cast(
        tf.equal(
            tf.expand_dims(block_ids, 1), tf.expand_dims(all_block_ids, 0)),
        tf.float32)

    # (2) Apply softmax over all positions in the (global) batch
    # across the blocks with the same `block_id`.

    # [batch_size, 3, 1]
    yes_no_span_probs = losses.cross_batch_softmax(
        tf.expand_dims(yesno_logits, 2), cross_blocks_eq_mask, num_replicas)
    yes_no_span_probs = tf.squeeze(yes_no_span_probs, 2)

    # [batch_size, 1]
    supporting_facts_probs = losses.cross_batch_softmax(
        tf.expand_dims(supporting_fact_logits, 2), cross_blocks_eq_mask,
        num_replicas)
    supporting_facts_probs = tf.squeeze(supporting_facts_probs, 2)

    # (3) Prepare one-hot labels based on annotation begins and ends

    supporting_fact_labels = tf.cast(supporting_fact_labels, tf.float32)

    # [batch_size, 3]
    yes_no_span_one_hot = tf.one_hot(yesno_labels, depth=3, dtype=tf.float32)
    yes_no_span_one_hot = yes_no_span_one_hot * supporting_fact_labels

    # (4) Compute the probability of the current begin / end positions across
    # the blocks with the same `block_id`.

    def mean_loss(all_losses):
      return tf.reduce_sum(all_losses * example_mask) / (
          tf.reduce_sum(example_mask) + eps)

    supporting_facts_loss = -mean_loss(
        tf.log(supporting_facts_probs * supporting_fact_labels + eps))

    yes_no_span_loss = -mean_loss(
        tf.log(yes_no_span_probs * yes_no_span_one_hot + eps))

    return yes_no_span_loss, supporting_facts_loss
