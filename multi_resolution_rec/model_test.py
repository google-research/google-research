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

"""Tests for multi_resolution_rec.model."""

import tensorflow.compat.v1 as tf
from multi_resolution_rec import model


class ModelTest(tf.test.TestCase):

  # History only model.
  def test_history_only_model(self):
    with self.test_session() as sess:
      model_history_only = self._generate_model(
          use_last_query='no_query',
          query_item_combine=None,
          num_query_attn_heads=None,
          num_query_attn_layers=None,
          query_item_attention=None)

      self._run_model(sess, model_history_only)

  # Query only model.
  def test_query_only_model(self):
    with self.test_session() as sess:
      model_query_only = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='only_query',
          num_query_attn_heads=None,
          num_query_attn_layers=None,
          query_item_attention=None)

      self._run_model(sess, model_query_only)

  # History and query combined model (no attention).
  def test_item_query_combined_model(self):
    with self.test_session() as sess:
      model_item_query = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=None,
          num_query_attn_layers=None,
          query_item_attention='none')

      self._run_model(sess, model_item_query)

  # Multihead attention model (num of heads=2).
  def test_multihead_model(self):
    with self.test_session() as sess:
      model_multihead = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=2,
          num_query_attn_layers=None,
          query_item_attention='multihead')

      self._run_model(sess, model_multihead)

  # Memory attention model (num of layers=2).
  def test_memory_model(self):
    with self.test_session() as sess:
      model_memory = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=None,
          num_query_attn_layers=2,
          query_item_attention='memory')

      self._run_model(sess, model_memory)

  # Multihead attention model with position prior (num of heads=2).
  def test_multihead_position_model(self):
    with self.test_session() as sess:
      model_multihead_position = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=2,
          num_query_attn_layers=None,
          query_item_attention='multihead_position')

      self._run_model(sess, model_multihead_position)

  # Multihead attention model with time prior (num of heads=2).
  def test_multihead_time_model(self):
    # time_exp_base=3, overlapping_chunks=False.
    with self.test_session() as sess:
      model_multihead_time = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=2,
          num_query_attn_layers=None,
          query_item_attention='multihead_time',
          time_exp_base=3,
          overlapping_chunks=False)

      self._run_model(sess, model_multihead_time)
    # time_exp_base=3, overlapping_chunks=True.
    with self.test_session(graph=tf.Graph()) as sess:  # Create a new graph.
      model_multihead_time = self._generate_model(
          use_last_query='query_bow',
          query_item_combine='concat',
          num_query_attn_heads=2,
          num_query_attn_layers=None,
          query_item_attention='multihead_time',
          time_exp_base=3,
          overlapping_chunks=True)

      self._run_model(sess, model_multihead_time)

  # Helper function to generate model.
  def _generate_model(self,
                      use_last_query,
                      query_item_combine,
                      num_query_attn_heads,
                      num_query_attn_layers,
                      query_item_attention,
                      time_exp_base=None,
                      overlapping_chunks=None):

    # Common test parameters.
    usernum = 3
    itemnum = 5
    vocab_size = 4
    maxseqlen = 4
    maxquerylen = 2
    hidden_units = 4
    l2_emb = 0
    dropout_rate = 0
    lr = 0.01
    query_layer_norm = False
    query_residual = False
    num_self_attn_heads = 1
    num_self_attn_layers = 1
    num_final_layers = 1

    return model.Model(
        usernum,
        itemnum,
        vocab_size,
        use_last_query=use_last_query,
        maxseqlen=maxseqlen,
        maxquerylen=maxquerylen,
        hidden_units=hidden_units,
        l2_emb=l2_emb,
        dropout_rate=dropout_rate,
        lr=lr,
        num_self_attn_heads=num_self_attn_heads,
        num_query_attn_heads=num_query_attn_heads,
        num_self_attn_layers=num_self_attn_layers,
        num_query_attn_layers=num_query_attn_layers,
        num_final_layers=num_final_layers,
        query_item_attention=query_item_attention,
        query_item_combine=query_item_combine,
        query_layer_norm=query_layer_norm,
        query_residual=query_residual,
        time_exp_base=time_exp_base,
        overlapping_chunks=overlapping_chunks)

  # Helper function to run model with toy/test examples.
  def _run_model(self, sess, test_model):

    # Test inputs.
    item_seq = [[1, 2, 3, 4], [2, 3, 4, 5]]  # (2, 4)
    query_seq = [[1, 2, 3, 4], [2, 3, 4, 5]]  # (2, 4)
    query_words_seq = [[[1, 1], [2, 2], [3, 3], [4, 4]]]*2  # (2, 2, 4)
    time_seq = [[1, 2, 3, 4], [2, 3, 4, 5]]  # (2, 4)
    pos = [[2, 3, 4, 5], [3, 4, 5, 5]]  # (2, 4)
    neg = [[1, 1, 1, 1], [2, 2, 2, 2]]  # (2, 4)

    sess.run(tf.global_variables_initializer())
    loss = sess.run(
        test_model.loss, {
            test_model.u: [],  # Not used.
            test_model.item_seq: item_seq,
            test_model.query_seq: query_seq,
            test_model.query_words_seq: query_words_seq,
            test_model.time_seq: time_seq,
            test_model.pos: pos,
            test_model.neg: neg,
            test_model.is_training: False
        })
    self.assertAllGreater(loss, 0)  # Expecting a non-negative loss.

if __name__ == '__main__':
  tf.test.main()
