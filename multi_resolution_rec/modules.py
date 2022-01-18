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

"""Modules."""

import numpy as np
import tensorflow.compat.v1 as tf

SECS_TO_DAYS = 60 * 60 * 24


def positional_encoding(dim, sentence_length, dtype=tf.float32):
  """Positional encoding."""

  encoded_vec = np.array([
      pos / np.power(10000, 2 * i / dim)  # pylint: disable=g-complex-comprehension
      for pos in range(sentence_length)
      for i in range(dim)
  ])
  encoded_vec[::2] = np.sin(encoded_vec[::2])
  encoded_vec[1::2] = np.cos(encoded_vec[1::2])

  return tf.convert_to_tensor(
      encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
  """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing
      ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
    reuse: Boolean, whether to reuse the weights of a previous layer by the
      same name.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope(scope, reuse=reuse):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ((variance + epsilon)**(.5))
    outputs = gamma * normalized + beta

  return outputs


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              l2_reg=0.0,
              scope="embedding",
              with_t=False,
              reuse=None):
  """Embeds a given tensor.

  Args:
    inputs: A `Tensor` with type `int32` or `int64` containing the ids to be
      looked up in `lookup table`.
    vocab_size: An int. Vocabulary size.
    num_units: An int. Number of embedding hidden units.
    zero_pad: A boolean. If True, all the values of the fist row (id 0) should
      be constant zeros.
    scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
    l2_reg: L2 regularization weight.
    scope: Optional scope for `variable_scope`.
    with_t: If True, return the embedding table.
    reuse: Boolean, whether to reuse the weights of a previous layer by the
      same name.

  Returns:
    A `Tensor` with one more rank than inputs's. The last dimensionality
      should be `num_units`.

  For example,

  ```
  import tensorflow as tf

  inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
  outputs = embedding(inputs, 6, 2, zero_pad=True)
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print sess.run(outputs)
  >>
  [[[ 0.          0.        ]
    [ 0.09754146  0.67385566]
    [ 0.37864095 -0.35689294]]

   [[-1.01329422 -1.09939694]
    [ 0.7521342   0.38203377]
    [-0.04973143 -0.06210355]]]
  ```

  ```
  import tensorflow as tf

  inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
  outputs = embedding(inputs, 6, 2, zero_pad=False)
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print sess.run(outputs)
  >>
  [[[-0.19172323 -0.39159766]
    [-0.43212751 -0.66207761]
    [ 1.03452027 -0.26704335]]

   [[-0.11634696 -0.35983452]
    [ 0.50208133  0.53509563]
    [ 1.22204471 -0.96587461]]]
  ```
  """
  with tf.variable_scope(scope, reuse=reuse):
    lookup_table = tf.get_variable(
        "lookup_table",
        dtype=tf.float32,
        shape=[vocab_size, num_units],
        # initializer=tf.contrib.layers.xavier_initializer(),
        regularizer=tf.keras.regularizers.l2(l2_reg))
    if zero_pad:
      lookup_table = tf.concat(
          (tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    if scale:
      outputs = outputs * (num_units**0.5)
  if with_t:
    return outputs, lookup_table
  else:
    return outputs


def multihead_attention(queries,
                        keys,
                        times=None,
                        num_units=None,
                        num_heads=1,
                        dropout_rate=0,
                        is_training=True,
                        use_prior="none",
                        causality=True,
                        scope="multihead_attention",
                        residual=False,
                        time_exp_base=None,
                        overlapping_chunks=None,
                        reuse=None,
                        with_qk=False):
  """Applies multihead attention.

  Args:
    queries: A 3d tensor with shape of [N, T_q, C_q].
    keys: A 3d tensor with shape of [N, T_k, C_k].
    times: A 3d tensor with shape of [N, T_q, T_k].
    num_units: A scalar. Attention size.
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    is_training: Boolean. Controller of mechanism for dropout.
    use_prior: String. Whether to use prior for attention heads. Supported
      values include: none, position.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
    residual: Boolean. Whether to use residual connection.
    time_exp_base: A scalar. Base for exponential time intervals. Only used for
      the case where use_prior='time'.
    overlapping_chunks: Boolean. Whether to use (non)/overlapping chunks for the
      case where use_prior='time'.
    reuse: Boolean, whether to reuse the weights of a previous layer by the
      same name.  Returns A 3d tensor with shape of (N, T_q, C)
    with_qk: Whether to use qk.
  Returns:
    Output of multihead attention.
  """
  tf.logging.info(
      "Computing attention with prior: {} and num of heads: {}".format(
          use_prior, num_heads))
  with tf.variable_scope(scope, reuse=reuse):
    # Set the fall back option for num_units
    if num_units is None:
      num_units = queries.get_shape().as_list[-1]

    # pylint: disable=invalid-name
    # Linear projections
    # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
    # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
    # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
    Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
    K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
    V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    # pylint: enable=invalid-name

    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

    # Scale
    outputs = outputs / (K_.get_shape().as_list()[-1]**0.5)

    # Key Masking
    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
    key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
    key_masks = tf.tile(
        tf.expand_dims(key_masks, 1),
        [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

    paddings = tf.ones_like(outputs) * (-2**32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), paddings,
                       outputs)  # (h*N, T_q, T_k)

    # Causality = Future blinding
    if causality:
      diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
      tril = tf.linalg.LinearOperatorLowerTriangular(
          diag_vals).to_dense()  # (T_q, T_k)
      masks = tf.tile(tf.expand_dims(tril, 0),
                      [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

      paddings = tf.ones_like(masks) * (-2**32 + 1)
      outputs = tf.where(tf.equal(masks, 0), paddings,
                         outputs)  # (h*N, T_q, T_k)

    # Position/Time prior is only used in multi-head case.
    if num_heads > 1:
      # Scaling head weights with position prior.
      if use_prior == "position":
        # Each head focuses on a window of items whose size is computed below.
        attn_size = int(outputs.get_shape().as_list()[-1] / num_heads)
        outputs = tf.concat(
            _compute_head_weights_with_position_prior(outputs, masks, paddings,
                                                      num_heads, attn_size),
            axis=0)  # (H*N, T_q, T_k)
        tf.logging.info("After position-wise sliding window attention.")
        tf.logging.info(outputs.shape)

      # Scaling head weights with time prior.
      elif use_prior == "time":
        # Convert time deltas from seconds to days.
        if times is None:
          raise ValueError("Times tensor is needed.")
        time_deltas = _compute_time_deltas(times) / SECS_TO_DAYS
        outputs = tf.concat(_compute_head_weights_with_time_prior(
            outputs, paddings, time_deltas, num_heads, time_exp_base,
            overlapping_chunks), axis=0)  # (H*N, T_q, T_k)

    # Activation
    outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

    # Query Masking
    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
    query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
    query_masks = tf.tile(
        tf.expand_dims(query_masks, -1),
        [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
    outputs *= query_masks  # broadcasting. (h*N, T_q, C)

    # Dropouts
    outputs = tf.layers.dropout(
        outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # (h*N, T_q, C/h)

    # Restore shape
    outputs = tf.concat(
        tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    # Residual connection
    if residual:
      outputs += queries

  if with_qk:
    return Q, K
  else:
    return outputs


def _compute_head_weights_with_position_prior(weights, masks, paddings,
                                              num_heads, attn_size):
  """Computes head-specific attention weights with position prior.

  This function simply masks out the weights for items if they don't belong to a
  certain chunk, using a sliding window technique. I.e., head i only focuses on
  ith recent "chunk_size" items with respect to the query. Note that chunks are
  non-overlapping, meaning, sliding window stride is also set to attn_size.

  Args:
    weights: A 3d tensor with shape of [h*N, T_q, T_k].
    masks: A 3d tensor with shape of [h*N, T_q, T_k].
    paddings: A 3d tensor with shape of [h*N, T_q, T_k].
    num_heads: An integer denoting number of chunks.
    attn_size: An integer denoting the size of the sliding window.

  Returns:
    A list of h tensors (each shaped [N, T_q, T_k]) where tensors correspond to
    chunk specific weights.
  """
  # Masks is a lower triangular tensor with ones in the bottom and zeros in the
  # upper section. Since chunks are allocated with respect to query position, we
  # first need to count the available items prior to each query. argmin function
  # would work for this, except the last query because it returns the smallest
  # index in the case of ties. To make sure we have the accurate count for the
  # last query, we first append a zero tensor and call the argmin function.
  max_idxs = tf.argmin(tf.concat([masks, tf.zeros_like(masks)], axis=-1),
                       2)  # (h*N, T_q)

  # Split for heads.
  max_idxs_split = tf.split(max_idxs, num_heads, axis=0)  # (h x (N, T_q))
  weights_split = tf.split(weights, num_heads, axis=0)  # (h x (N, T_q, T_k))
  paddings_split = tf.split(paddings, num_heads, axis=0)  # (h x (N, T_q, T_k))

  # Collects output weights per chunk.
  chunk_outputs_list = []
  for i in range(num_heads):
    mask_left = tf.sequence_mask(
        tf.maximum(max_idxs_split[i] - (attn_size * (i + 1)), 0),
        tf.shape(weights_split[i])[2])  # (N, T_q, T_k)
    mask_right = tf.sequence_mask(
        tf.maximum(max_idxs_split[i] - (attn_size * i), 0),
        tf.shape(weights_split[i])[2])  # (N, T_q, T_k)
    mask = tf.logical_and(tf.logical_not(mask_left),
                          mask_right)  # (N, T_q, T_k)
    # Adjust weights for chunk i.
    output = tf.where(mask, weights_split[i],
                      paddings_split[i])  # (N, T_q, T_k)
    chunk_outputs_list.append(output)
  return chunk_outputs_list  # (h x (N, T_q, T_k))


def _compute_head_weights_with_time_prior(weights, paddings, time_deltas,
                                          num_heads, time_exp_base,
                                          overlapping_chunks):
  """Computes head-specific attention weights with time prior.

  This function simply masks out the weights for items if they don't belong to a
  certain chunk. Here, chunks are allocated based on time information. We use
  exponential function--pow(time_exp_base,i)--to allocate segment boundaries.
  Note that time delta values represent number of days.

  Example 1: Let overlapping_chunks=False, time_exp_base=3 and num_heads=3.
  1st head focuses on the items within time interval [0, pow(3,0)],
  2nd head focuses on the items within time interval (pow(3,0), pow(3,1)],
  3rd (last) head focuses on the items within time interval (pow(3,1), inf]

  Example 2: Let overlapping_chunks=True, time_exp_base=3 and num_heads=3.
  1st head focuses on the items within time interval [0, pow(3,0)],
  2nd head focuses on the items within time interval [0, pow(3,1)],
  3rd (last) head focuses on the items within time interval [0, inf]

  Args:
    weights: A 3d tensor with shape of [h*N, T_q, T_k].
    paddings: A 3d tensor with shape of [h*N, T_q, T_k].
    time_deltas: A 3d tensor with shape of [N, T_q, T_k].
    num_heads: An integer denoting number of chunks.
    time_exp_base: A scalar. Base for exponential time intervals.
    overlapping_chunks: Boolean. Whether to use overlapping chunks.

  Returns:
    A list of h tensors (each shaped [N, T_q, T_k]) where tensors correspond to
    chunk specific weights.
  """
  tf.logging.info(
      "Computing with time_exp_base:{} and overlapping_chunks:{}".format(
          time_exp_base, overlapping_chunks))
  chunk_outputs_list = []
  weights_split = tf.split(weights, num_heads, axis=0)
  paddings_split = tf.split(paddings, num_heads, axis=0)
  ones_tensor = tf.ones_like(time_deltas)  # (N, T_q, T_k)

  # False in previous items and True in future items.
  mask_previous_head = time_deltas < 0  # (N, T_q, T_k)
  for i in range(num_heads):
    if i == (num_heads - 1):  # Last chunk considers all the remaining items.
      # All True.
      mask_next_head = tf.ones_like(time_deltas, dtype=bool)  # (N, T_q, T_k)
    else:
      mask_next_head = tf.math.less_equal(
          time_deltas, (time_exp_base**i) * ones_tensor)  # (N, T_q, T_k)
    mask = tf.logical_and(tf.logical_not(mask_previous_head),
                          mask_next_head)  # (N, T_q, T_k)
    output = tf.where(mask, weights_split[i],
                      paddings_split[i])  # (N, T_q, T_k)
    chunk_outputs_list.append(output)

    # Update previous mask for non-overlapping chunks.
    if not overlapping_chunks:
      mask_previous_head = mask_next_head

  return chunk_outputs_list


def _compute_time_deltas(times):
  """This function computes time deltas between items.

  It is important to note that given timestamps are for queries. Hence, we need
  to consider that while calculating the time deltas between queries and items.
  Example: For items: [<PAD>, 1, 2, 3] and queries: [q1, q2, q3, q4], the times
  vector is [t1, t2, t3, t4]. Then, the time deltas will be:
    [
      [t1, 0, t1-t2, t1-t3],  # time deltas for query 1
      [t2, t2-t1, 0, t2-t3],   # time deltas for query 2
      [t3, t3-t1, t3-t2, 0],   # time deltas for query 3
      [t4, t4-t1, t4-t2, t4-t3]   # time deltas for query 4
    ]
  Args:
    times: A 2d tensor with shape of [N, T_q].

  Returns:
    A 3d tensor with shape of [N, T_q, T_q].
  """
  t1 = tf.tile(tf.expand_dims(times, 2), [1, 1, tf.shape(times)[1]])
  t2 = tf.tile(tf.expand_dims(times, 1), [1, tf.shape(times)[1], 1])
  time_deltas = t1 - t2  # (N, T_q, T_q)
  time_deltas = tf.concat([tf.expand_dims(times, 2), time_deltas],
                          2)  # (N, T_q, 1+T_q)
  time_deltas = time_deltas[:, :, :-1]  # (N, T_q, T_q)
  return time_deltas


# pylint: disable=dangerous-default-value
def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                dropout_rate=0.2,
                is_training=True,
                reuse=None):
  """Point-wise feed forward net.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    dropout_rate: Dropout rate.
    is_training: Whether to run in training mode.
    reuse: Boolean, whether to reuse the weights of a previous layer by the
      same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  """
  with tf.variable_scope(scope, reuse=reuse):
    # Inner layer
    params = {
        "inputs": inputs,
        "filters": num_units[0],
        "kernel_size": 1,
        "activation": tf.nn.relu,
        "use_bias": True
    }
    outputs = tf.layers.conv1d(**params)
    outputs = tf.layers.dropout(
        outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    # Readout layer
    params = {
        "inputs": outputs,
        "filters": num_units[1],
        "kernel_size": 1,
        "activation": None,
        "use_bias": True
    }
    outputs = tf.layers.conv1d(**params)
    outputs = tf.layers.dropout(
        outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

    # Residual connection
    outputs += inputs

    # Normalize
    # outputs = normalize(outputs)

  return outputs


# pylint: disable=dangerous-default-value
def query_feedforward(inputs,
                      num_units,
                      scope="item_and_query_combined_embedding",
                      dropout_rate=0,
                      is_training=True,
                      residual=False,
                      reuse=None):
  """Point-wise feed forward net for query-item encoder.

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    dropout_rate: Dropout rate.
    is_training: Whether to run in training mode.
    residual: Whether to use residual connections.
    reuse: Boolean, whether to reuse the weights of a previous layer by the
      same name.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  """
  with tf.variable_scope(scope, reuse=reuse):
    outputs = tf.nn.relu(inputs)
    for units in num_units:
      params = {
          "inputs": outputs,
          "filters": units,
          "kernel_size": 1,
          "activation": None,
          "use_bias": True
      }
      outputs = tf.layers.conv1d(**params)
      outputs = tf.layers.dropout(
          outputs,
          rate=dropout_rate,
          training=tf.convert_to_tensor(is_training))

    # Residual connection
    if residual:
      outputs += inputs

  return outputs
