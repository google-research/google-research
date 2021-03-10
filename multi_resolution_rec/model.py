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

"""Model."""

import functools

import tensorflow.compat.v1 as tf

from multi_resolution_rec import modules

class Model():
  """Model."""

  def __init__(self,
               usernum,
               itemnum,
               vocabsize,
               use_last_query,
               maxseqlen,
               maxquerylen,
               hidden_units,
               l2_emb,
               dropout_rate,
               lr,
               num_self_attn_heads,
               num_query_attn_heads,
               num_self_attn_layers,
               num_query_attn_layers,
               num_final_layers,
               query_item_attention,
               query_item_combine,
               query_layer_norm,
               query_residual,
               time_exp_base,
               overlapping_chunks,
               reuse=None):
    del usernum

    # Variables.
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.u = tf.placeholder(tf.int32, shape=(None))
    self.item_seq = tf.placeholder(tf.int32, shape=(None, maxseqlen))
    self.query_seq = tf.placeholder(tf.int32, shape=(None, maxseqlen))
    self.query_words_seq = tf.placeholder(
        tf.int32, shape=(None, maxseqlen, maxquerylen))
    self.time_seq = tf.placeholder(tf.int32, shape=(None, maxseqlen))
    self.pos = tf.placeholder(tf.int32, shape=(None, maxseqlen))
    self.neg = tf.placeholder(tf.int32, shape=(None, maxseqlen))
    self.query_residual = query_residual
    item_seq = self.item_seq
    pos = self.pos
    neg = self.neg
    mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, 0)), -1)
    q_words_mask = tf.expand_dims(
        tf.to_float(tf.not_equal(self.query_words_seq, 0)), -1)
    batch_size = tf.shape(self.item_seq)[0]

    with tf.variable_scope("MultiResolutionRec", reuse=reuse):
      # Sequence embedding, item embedding table.
      self.item_seq_emb, item_emb_table = modules.embedding(
          self.item_seq,
          vocab_size=itemnum + 1,
          num_units=hidden_units,
          zero_pad=True,
          scale=True,
          l2_reg=l2_emb,
          scope="item_embeddings",
          with_t=True,
          reuse=reuse)

      # Query embedding.
      if use_last_query == "query_bow":
        self.query_words_seq_emb = modules.embedding(
            self.query_words_seq,
            vocab_size=vocabsize + 1,
            num_units=hidden_units,
            zero_pad=True,
            scale=True,
            l2_reg=l2_emb,
            scope="query_bow_embeddings",
            with_t=False,
            reuse=reuse)
        # Masking padded query tokens.
        self.query_words_seq_emb *= q_words_mask
        # Computing the mean of query token embeddings (BOW). Shape changes from
        # [None, maxseqlen, maxquerylen] to [None, maxseqlen].
        self.query_seq_emb = tf.reduce_mean(self.query_words_seq_emb, 2)
        # Masking padded items.
        self.query_seq_emb *= mask

      # Positional Encoding.
      t, _ = modules.embedding(
          tf.tile(
              tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0),
              [tf.shape(self.item_seq)[0], 1]),
          vocab_size=maxseqlen,
          num_units=hidden_units,
          zero_pad=False,
          scale=False,
          l2_reg=l2_emb,
          scope="dec_pos",
          reuse=reuse,
          with_t=True)
      self.item_seq_emb += t

      # Dropout.
      self.item_seq_emb = tf.layers.dropout(
          self.item_seq_emb,
          rate=dropout_rate,
          training=tf.convert_to_tensor(self.is_training))
      self.item_seq_emb *= mask

      # Build self-attention layers.
      for i in range(num_self_attn_layers):
        with tf.variable_scope("num_self_attention_layers_%d" % i):

          # Self-attention.
          self.item_seq_emb = modules.multihead_attention(
              queries=modules.normalize(self.item_seq_emb),
              keys=self.item_seq_emb,
              num_units=hidden_units,
              num_heads=num_self_attn_heads,
              dropout_rate=dropout_rate,
              is_training=self.is_training,
              causality=True,
              scope="self_attention",
              residual=True)

          # Feed forward.
          self.item_seq_emb = modules.feedforward(
              modules.normalize(self.item_seq_emb),
              num_units=[hidden_units, hidden_units],
              dropout_rate=dropout_rate,
              is_training=self.is_training)
          self.item_seq_emb *= mask

      self.item_seq_emb = modules.normalize(self.item_seq_emb)

    # Query layer.
    if use_last_query == "query_bow":

      # Check whether to compute attention.
      if query_item_combine != "only_query" and query_item_attention != "none":
        # Compute query-attended history embeddings.
        query_attended_seq_emb = self.compute_query_to_item_attention(
            query_item_attention, self.query_seq_emb, self.item_seq_emb,
            self.time_seq, num_query_attn_heads, num_query_attn_layers,
            hidden_units, dropout_rate, self.is_training, time_exp_base,
            overlapping_chunks)
        self.item_seq_emb = tf.concat(
            [self.item_seq_emb,  # Keep non-attented history embedding.
             query_attended_seq_emb],  # Attended history embedding.
            axis=-1)
        self.item_seq_emb *= mask
        # Combine strategy can't be 'sum' due to last dimension mismatch.
        query_item_combine = "concat"

      # Combine history and query embeddings.
      self.item_query_seq_emb = self.combine_item_query_embeddings(
          query_item_combine, self.item_seq_emb, self.query_seq_emb)

      # Feed-forward layer.
      self.item_query_seq_emb = modules.query_feedforward(
          self.item_query_seq_emb,
          num_units=[hidden_units] * num_final_layers,
          dropout_rate=dropout_rate,
          is_training=self.is_training,
          residual=self.query_residual)
      self.item_query_seq_emb *= mask
      if query_layer_norm:
        self.item_query_seq_emb = modules.normalize(self.item_query_seq_emb)

      seq_emb = tf.reshape(self.item_query_seq_emb,
                           [batch_size * maxseqlen, hidden_units])
    else:
      seq_emb = tf.reshape(self.item_seq_emb,
                           [batch_size * maxseqlen, hidden_units])

    # Position-wise positives/negatives.
    item_seq = tf.reshape(item_seq, [batch_size * maxseqlen])
    pos = tf.reshape(pos, [batch_size * maxseqlen])
    neg = tf.reshape(neg, [batch_size * maxseqlen])
    pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
    neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

    self.test_item = tf.placeholder(tf.int32, shape=(101))
    test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
    self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
    self.test_logits = tf.reshape(self.test_logits,
                                  [batch_size, maxseqlen, 101])
    self.test_logits = self.test_logits[:, -1, :]

    # Prediction layer.
    self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
    self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

    # Ignore the first label item regardless of whether the query is used or not
    # for consistency.
    istarget = tf.reshape(
        tf.to_float(tf.not_equal(item_seq, 0)), [batch_size * maxseqlen])
    # Note: positives and negatives are present for exactly the same positions.
    self.loss = tf.reduce_sum(-tf.log(tf.sigmoid(self.pos_logits) + 1e-24) *
                              istarget -
                              tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) *
                              istarget) / tf.reduce_sum(istarget)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss += sum(reg_losses)

    tf.summary.scalar("loss", self.loss)

    if reuse is None:
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta2=0.98)
      self.train_op = self.optimizer.minimize(
          self.loss, global_step=self.global_step)

  def compute_query_to_item_attention(self, attn_type, query_seq_emb,
                                      item_seq_emb, time_seq,
                                      num_query_attn_heads,
                                      num_query_attn_layers, hidden_units,
                                      dropout_rate, is_training, time_exp_base,
                                      overlapping_chunks):
    """Computes query to history attention for various strategies.

    Strategies include: multihead, memory, multihead_position.
    Args:
      attn_type: A string defining attention strategy.
      query_seq_emb: A tensor of query embeddings.
      item_seq_emb: A tensor of item embeddins.
      time_seq: A tensor of timestamps.
      num_query_attn_heads: An int. Number of attention heads.
      num_query_attn_layers: An int. Number of attention layers.
      hidden_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      time_exp_base: A scalar. Base for exponential time intervals.
      overlapping_chunks: Boolean. Whether to use overlapping chunks.

    Returns:
      A tensor of attention outputs.
    """
    attention_output = []
    tf.logging.info(
        "Computing {} attention with num of heads: {} & num of layers: {}"
        .format(attn_type, num_query_attn_heads, num_query_attn_layers))

    attention_function = functools.partial(
        modules.multihead_attention,  # Attention function
        queries=query_seq_emb,  # Common arguments start.
        keys=item_seq_emb,
        num_units=hidden_units,
        num_heads=num_query_attn_heads,
        dropout_rate=dropout_rate,
        is_training=is_training)  # Common arguments end.

    # Variant 1: Query-to-history multihead attention.
    if attn_type == "multihead":
      attention_output.append(attention_function(
          scope="query_to_item_%s_attention" % attn_type))

    # Variant 2: Query-to-history memory attention. Note that this variant uses
    # single head attention.
    elif attn_type == "memory":
      memory_output = None
      memory_query = query_seq_emb
      for i in range(num_query_attn_layers):
        memory_output = attention_function(
            queries=memory_query,
            num_heads=1,
            scope="query_to_item_%s_attention_layer_%d" % (attn_type, i))
        memory_query += memory_output  # Memory connection.
      attention_output.append(memory_output)

    # Variant 3: Query-to-history multihead attention with position prior.
    elif attn_type == "multihead_position":
      attention_output.append(attention_function(
          use_prior="position",  # Using position prior.
          scope="query_to_item_%s_attention" % attn_type))

    # Variant 4: Query-to-history multihead attention with time prior.
    elif attn_type == "multihead_time":
      attention_output.append(
          attention_function(
              times=time_seq,
              use_prior="time",  # Using time prior.
              time_exp_base=time_exp_base,
              overlapping_chunks=overlapping_chunks,
              scope="query_to_item_%s_attention" % attn_type))

    # Invalid variant.
    else:
      raise ValueError("Invalid attention type.")

    attention_output = tf.concat(attention_output, axis=-1)
    tf.logging.info("Shape after {} attention layer.".format(attn_type))
    tf.logging.info(attention_output.shape)
    return attention_output

  def combine_item_query_embeddings(self, query_item_combine, item_seq_emb,
                                    query_seq_emb):
    """Helper function to combine history and query embeddings.

    Note that this function sets the query_residual variable to False in the
    case of concatenation.
    Args:
      query_item_combine: Strategy to combine query and item embeddings.
      item_seq_emb: A tensor of item embeddings.
      query_seq_emb: A tensor of query embeddings.
    Returns:
      A tensor of history and query combined embeddings or query embeddings only
      for 'only_query' strategy.
    """

    if query_item_combine == "sum":
      return item_seq_emb + query_seq_emb
    elif query_item_combine == "concat":
      # We need to project it back to num_units. Hence residual can't be used.
      self.query_residual = False
      return tf.concat([item_seq_emb, query_seq_emb], -1)
    elif query_item_combine == "only_query":
      tf.logging.info("Using only query info, disregarding the sequence emb.")
      return query_seq_emb
    else:
      raise ValueError(
          "Enter a valid combining strategy: sum, concat or only_query")

  def predict(self, sess, u, seq, q_seq, q_words_seq, t_seq, item_ids):
    return sess.run(
        self.test_logits, {
            self.u: u,
            self.item_seq: seq,
            self.query_seq: q_seq,
            self.query_words_seq: q_words_seq,
            self.time_seq: t_seq,
            self.test_item: item_ids,
            self.is_training: False
        })
