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

"""Tests for task."""

import numpy as np
import tensorflow as tf

from multiple_user_representations.models import task


class TaskTest(tf.test.TestCase):

  def test_retrieval_task(self):

    user_embeddings = tf.convert_to_tensor(
        np.arange(12).reshape(2, 3, 2), dtype=tf.float32)
    item_embeddings = tf.convert_to_tensor(
        np.array([[0.0, 0.1], [0.2, 0.0]]), dtype=tf.float32)
    loss = task.MultiShotRetrievalTask()(user_embeddings, item_embeddings)

    self.assertAlmostEqual(loss.numpy(), 1.1955092)

  def test_multi_query_factorized_top_k_with_multi_streaming(self):

    num_candidates, num_queries, num_heads, embedding_dim = (100, 10, 3, 4)

    rng = np.random.RandomState(42)

    # Create testdata
    candidates = rng.normal(size=(num_candidates,
                                  embedding_dim)).astype(np.float32)
    query = rng.normal(size=(num_queries, num_heads,
                             embedding_dim)).astype(np.float32)
    true_candidates = rng.normal(size=(num_queries,
                                       embedding_dim)).astype(np.float32)

    # Compute positive and candidate scores
    positive_scores = (query * np.expand_dims(true_candidates, 1)).sum(
        axis=-1, keepdims=True).max(axis=1)
    candidate_scores = (query @ candidates.T).max(axis=1)

    # Concatenate all scores (Positive first)
    all_scores = np.concatenate([positive_scores, candidate_scores], axis=1)

    ks = [1, 5, 10, 50]

    # Prepare item candidate dataset
    candidates = tf.data.Dataset.from_tensor_slices(candidates).batch(32)
    candidates = task.MultiQueryStreaming().index_from_dataset(candidates)

    # Initialize and update the metric state.
    metric = task.MultiQueryFactorizedTopK(
        candidates=candidates,
        metrics=[
            tf.keras.metrics.TopKCategoricalAccuracy(k=x, name=f'HR@{x}')
            for x in ks
        ],
        k=max(ks),
    )
    metric.update_state(
        query_embeddings=query, true_candidate_embeddings=true_candidates)

    for k, metric_value in zip(ks, metric.result()):
      in_top_k = tf.math.in_top_k(
          targets=np.zeros(num_queries).astype(np.int32),
          predictions=all_scores,
          k=k)

      self.assertAllClose(metric_value, in_top_k.numpy().mean())


if __name__ == '__main__':
  tf.test.main()
