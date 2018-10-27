# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Helpers for building SQuAD models."""
from __future__ import absolute_import

from __future__ import division
from __future__ import print_function

import collections
import os
from absl import flags

from tensor2tensor.layers import common_attention

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
import tensorflow as tf
from qanet import squad_data





def get_emb_by_name(emb_path,
                    emb_file=None,
                    draft=False,
                    words_only=False,
                    subset=None):
  """Get an `OrderedDict` that maps word to vector.

  Args:
    emb_path: `str` value, path to the emb file (e.g. `glove.6B.50d.txt`) or
      directory.
    size: `int` value, size of the vector, if `emb_path` is a directory.
      Overrides emb_path
    draft: `bool` value, whether to only load first 99 for draft mode.
    words_only: `bool` value, whether to only return the set of words in the
      vocab
    subset: If specified, only return vectors for this subset of emb.

  Returns:
    `OrderedDict` object, mapping word to vector.
  """
  if emb_file is not None:
    emb_path = os.path.join(emb_path, emb_file)
  tf.logging.info('emb file: %s' % emb_path)
  emb = collections.OrderedDict()
  with tf.gfile.GFile(emb_path, 'r') as fp:
    for idx, line in enumerate(fp):
      if emb_path.endswith('vec') and idx == 0:
        continue
      line = line.decode('utf-8')
      tokens = line.strip().split(u' ')
      word = tokens[0]
      if subset is not None:
        if word not in subset:
          continue
      vec = list(map(float, tokens[1:]))
      size = len(vec)
      if words_only:
        emb[word] = 1
      else:
        emb[word] = vec
      if draft and idx > 99:
        break

  tf.logging.info('emb vocab loaded. Is size: %s', len(emb))
  return emb, size


VERY_NEGATIVE_NUMBER = -1e29
VERY_SMALL_NUMBER = 1e-29

__all__ = ['preprocess_inputs', 'build_a_layer',
           'embed_elmo_chars', 'embed_elmo_sentences', 'embed_translation',
           'glove_layer', 'get_attention_bias',
           'compute_survival_rate', 'stochastic_depth',
           'dropout_wrapper', 'highway_layer', 'exp_mask', 'separable_conv',
           'swish', 'parametric_relu', 'bi_attention_memory_efficient_dcn',
           'get_loss', 'get_pred_ops', 'safe_log']


def preprocess_inputs(inputs, hidden_size):
  """Transform input size and add positional encodings."""
  if inputs.shape.as_list()[-1] != hidden_size:
    # Project to proper size
    inputs = common_layers.conv1d(
        inputs=inputs,
        filters=hidden_size,
        kernel_size=1,
        activation=None,
        padding='SAME')
  net = inputs
  net = common_attention.add_timing_signal_nd(net)

  return net


def build_a_layer(layer_input, layer_type, is_training, config):
  """Build a single layer."""
  batch_size = tf.shape(layer_input)[0]
  net = common_layers.layer_norm(layer_input)
  if layer_type == 'att':
    net = common_attention.multihead_attention(
        query_antecedent=net,
        memory_antecedent=None,
        bias=None,
        total_key_depth=config['hidden_size'],
        total_value_depth=config['hidden_size'],
        output_depth=config['hidden_size'],
        num_heads=config['attention_heads'],
        dropout_rate=config['attention_dropout'],
        attention_type=config['attention_type'])
  elif layer_type == 'conv':
    if config['preactivate']:
      if config['activation'] == 'relu':
        net = tf.nn.relu(net)
      elif config['activation'] == 'swish':
        net = swish(net)
      elif config['activation'] == 'prelu':
        net = parametric_relu(net)
      net = separable_conv(net, config['kernel_size'], activation=None)
    else:
      if config['activation'] == 'relu':
        net = separable_conv(
            net, config['kernel_size'], activation=tf.nn.relu)
      elif config['activation'] == 'swish':
        net = separable_conv(net, config['kernel_size'], activation=swish)
      elif config['activation'] == 'prelu':
        net = separable_conv(
            net, config['kernel_size'], activation=parametric_relu)
  elif layer_type == 'ffn':
    net = tf.reshape(net, [-1, config['hidden_size']])
    net = expert_utils.ffn_expert_fn(
        input_size=config['hidden_size'],
        hidden_sizes=[config['ffn_hs_factor'] * config['hidden_size']],
        output_size=config['hidden_size'],
    )(
        net)
    net = tf.reshape(net, [batch_size, -1, config['hidden_size']])
  else:
    raise ValueError('Unknown layer type %s' % layer_type)

  if config['layer_output_dropout'] > 0.0 and is_training:
    net = tf.nn.dropout(net, 1.0 - config['layer_output_dropout'])
  return net


def embed_elmo_chars(char_tensor, max_batch_size, elmo_pretrained_dir,
                     training, num_gpus=4, base_gpu=2):
  """Embed ELMO."""
  with tf.variable_scope('pretrained_encoder_elmo', reuse=tf.AUTO_REUSE):
    encoder_states = transfer.elmo_tensor_from_chars(
        elmo_pretrained_dir,
        char_tensor,
        max_batch_size=max_batch_size,
        is_training=training,
        is_hdf5=False,
        num_gpus=num_gpus,
        base_gpu=base_gpu)
  return encoder_states


def embed_elmo_sentences(sentence_tensor, max_batch_size, elmo_pretrained_dir,
                         training, elmo_option):
  """Embed ELMO from sentences."""
  with tf.variable_scope('pretrained_encoder_elmo', reuse=tf.AUTO_REUSE):
    encoder_states = transfer.elmo_tensor_from_sentences(
        elmo_pretrained_dir,
        sentence_tensor,
        max_batch_size=max_batch_size,
        method=elmo_option,
        is_training=training)
  return encoder_states




def _replace_zeros(sequence, vec):
  """Replace all zero vectors in the given sequence with a learned vector."""
  mask = tf.equal(sequence, 0.0)
  mask = tf.reduce_all(mask, axis=-1, keep_dims=True)
  added = tf.to_float(mask) * vec[None, None, :]
  return sequence + added


def tfdata_emb_layer(features):
  """Add learnable unk/pad vectors for inputs embedding lookups are in input.

  By convention, the UNK vector will be all 0.0 (see tf_data_pipeline).
  We replace all UNK tokens with the learned UNK vector

  Args:
    features: Input features for qanet.

  Returns:
    context and question with UNK/PAD tokens replaced.
  """
  xw = features['context_vecs']
  qw = features['question_vecs']
  vec_len = xw.get_shape()[-1]
  with tf.variable_scope('glove_layer'):
    # PAD = 0
    # UNK = 1
    unk_pad = tf.get_variable('glove_emb_mat_var', [2, vec_len])
    pad = unk_pad[0, :]
    unk = unk_pad[1, :]
  q_mask = tf.tile(
      tf.sequence_mask(features['question_num_words'],
                       dtype=tf.float32)[:, :, None], [1, 1, vec_len])
  x_mask = tf.tile(
      tf.sequence_mask(features['context_num_words'],
                       dtype=tf.float32)[:, :, None], [1, 1, vec_len])

  xw = _replace_zeros(xw, unk)
  qw = _replace_zeros(qw, unk)

  # Add learned padding token
  xw = x_mask * xw + (1.0 - x_mask) * pad[None, None, :]
  qw = q_mask * qw + (1.0 - q_mask) * pad[None, None, :]
  return xw, qw


def glove_layer(features, mode, params, scope=None):
  """GloVe embedding layer.

  The first two words of `features['emb_mat']` are <PAD> and <UNK>.
  The other words are actual words. So we learn the representations of the
  first two words but the representation of other words are fixed (GloVe).

  Args:
    features: `dict` of feature tensors.
    mode: train/eval/infer mode.
    params: `HParams` object.
    scope: `str` for scope name.

  Returns:
    A tuple of tensors, `(emb_mat, context_emb, question_emb)`.
  """
  if 'context_vecs' in features:
    tf.logging.info('Words are pre-embedded')
    # TODO(ddohan): Consider an explicit "use_tf_data" flag
    xw, qw = tfdata_emb_layer(features)
    return None, xw, qw
  else:
    tf.logging.info('Doing embeddings in graph.')
    tf.logging.info('# Glove layer')
    with tf.variable_scope(scope or 'glove_layer'):
      training = mode == tf.estimator.ModeKeys.TRAIN

      # The first two values are for UNK & PAD
      emb_mat_const = features['emb_mat'][2:, :]
      emb_mat_var = tf.get_variable('glove_emb_mat_var',
                                    [2, emb_mat_const.get_shape()[1]])
      emb_mat = tf.concat([emb_mat_var, emb_mat_const], 0)

      if training:
        emb_mat = tf.nn.dropout(
            emb_mat,
            keep_prob=1.0 - params['word_embedding_dropout'],
            noise_shape=[emb_mat.shape.as_list()[0], 1])

      xv = tf.nn.embedding_lookup(emb_mat,
                                  features['glove_indexed_context_words'])
      qv = tf.nn.embedding_lookup(emb_mat,
                                  features['glove_indexed_question_words'])
      return emb_mat, xv, qv


def get_attention_bias(sequence_length):
  """Create attention bias so attention is not applied at padding position."""
  # attention_bias: [batch, 1, 1, memory_length]
  invert_sequence_mask = tf.to_float(tf.logical_not(tf.sequence_mask(
      sequence_length)))
  attention_bias = common_attention.attention_bias_ignore_padding(
      invert_sequence_mask)
  return attention_bias


def compute_survival_rate(layer_id, n_layer,
                          survival_rate_last_layer, survival_schedule):
  """Computer survival rate for stochastic depth."""
  if survival_schedule == 'exp':
    # alpha^(n_layer - 1) = survival_rate_last_layer
    # alpha = survival_rate_last_layer^(1/(n_layer-1))
    # survival_rate = alpha^layer_id
    survival_rate = survival_rate_last_layer ** (
        float(layer_id) / (n_layer - 1))
  elif survival_schedule == 'linear':
    survival_rate = 1.0 - float(layer_id) / n_layer * (
        1 - survival_rate_last_layer)
  else:
    raise ValueError('Unknown survival_schedule %s' % survival_schedule)
  return survival_rate


def stochastic_depth(layer, layer_input, survival_rate, is_training):
  """Randomly drop a layer according to survival rate."""
  if is_training:  # Train
    # Get a value between [0, 1)
    survival_roll = tf.random_uniform(shape=[], name='survival')
    # Decide to drop
    layer_output = tf.cond(survival_roll < survival_rate,
                           lambda: layer + layer_input,  # keep
                           lambda: layer_input)  # drop
  else:  # Test
    # Set scope for compatibility
    with tf.variable_scope('layer', reuse=tf.AUTO_REUSE):
      # Multiply net with survival_rate
      layer_output = survival_rate * layer
    layer_output += layer_input

  return layer_output


def exp_mask(logits, mask, mask_is_length=True):
  """Exponential mask for logits.

  Logits cannot be masked with 0 (i.e. multiplying boolean mask)
  because exponentiating 0 becomes 1. `exp_mask` adds very large negative value
  to `False` portion of `mask` so that the portion is effectively ignored
  when exponentiated, e.g. softmaxed.

  Args:
    logits: Arbitrary-rank logits tensor to be masked.
    mask: `boolean` type mask tensor.
      Could be same shape as logits (`mask_is_length=False`)
      or could be length tensor of the logits (`mask_is_length=True`).
    mask_is_length: `bool` value. whether `mask` is boolean mask.
  Returns:
    Masked logits with the same shape of `logits`.
  """
  if mask_is_length:
    mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
  return logits + (1.0 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER


def highway_layer(inputs,
                  num_layers=2,
                  activation='relu',
                  dropout_rate=0.0,
                  batch_norm=False,
                  training=False,
                  scope=None,
                  reuse=False):
  """Multi-layer highway networks (https://arxiv.org/abs/1505.00387).

  Args:
    inputs: `float` input tensor to the highway networks.
    num_layers: `int` value, indicating the number of highway layers to build.
    dropout_rate: `float` value for the input dropout rate.
    batch_norm: `bool` value, indicating whether to use batch normalization
      or not.
    training: `bool` value, indicating whether the current run is training
     or not (e.g. eval or inference).
    scope: `str` value, variable scope. Default is `highway_net`.
    reuse: `bool` value, indicating whether the variables in this function
      are reused.
  Returns:
    The output of the highway networks, which is the same shape as `inputs`.
  """
  with tf.variable_scope(scope or 'highway_net', reuse=reuse):
    outputs = inputs
    for i in range(num_layers):
      outputs = highway(
          outputs,
          activation=activation,
          dropout_rate=dropout_rate,
          batch_norm=batch_norm,
          training=training,
          scope='layer_{}'.format(i))
    return outputs


def highway(inputs,
            outputs=None,
            activation='relu',
            dropout_rate=0.0,
            batch_norm=False,
            training=False,
            scope=None,
            reuse=False):
  """Single-layer highway networks (https://arxiv.org/abs/1505.00387).

  Args:
    inputs: Arbitrary-rank `float` tensor, where the first dim is batch size
      and the last dim is where the highway network is applied.
    outputs: If provided, will replace the perceptron layer (i.e. gating only.)
    dropout_rate: `float` value, input dropout rate.
    batch_norm: `bool` value, whether to use batch normalization.
    training: `bool` value, whether the current run is training.
    scope: `str` value variable scope, default to `highway_net`.
    reuse: `bool` value, whether to reuse variables.
  Returns:
    The output of the highway network, same shape as `inputs`.
  """
  with tf.variable_scope(scope or 'highway', reuse=reuse):
    if dropout_rate > 0.0:
      inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
    dim = inputs.get_shape()[-1]
    if outputs is None:
      outputs = tf.layers.dense(inputs, dim, name='outputs')
      if batch_norm:
        outputs = tf.layers.batch_normalization(outputs, training=training)
      if activation == 'relu':
        outputs = tf.nn.relu(outputs)
      elif activation == 'tanh':
        outputs = tf.nn.tanh(outputs)
      elif activation == 'sigmoid':
        outputs = tf.nn.sigmoid(outputs)
    gate = tf.layers.dense(inputs, dim, activation=tf.nn.sigmoid, name='gate')
    return gate * inputs + (1 - gate) * outputs


def dropout_wrapper(x, keep_prob, mode, noise_shape=None):
  if keep_prob < 1.0 and mode == tf.estimator.ModeKeys.TRAIN:
    if noise_shape is None:
      x = tf.nn.dropout(x, keep_prob)
    else:
      x = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)
  return x


def _depth(x):
  return x.shape.as_list()[-1]


def separable_conv(x, kernel_size, activation):
  """Apply a depthwise separable 1d convolution."""
  tf.assert_rank(x, 3)
  net = tf.expand_dims(x, 2)
  net = tf.layers.separable_conv2d(
      net,
      filters=_depth(x),
      kernel_size=(kernel_size, 1),
      padding='same',
      activation=activation)
  net = tf.squeeze(net, axis=2)
  return net


def swish(x):
  alpha = tf.get_variable(
      'alpha',
      x.get_shape()[-1],
      initializer=tf.constant_initializer(0.1),
      dtype=tf.float32)
  return x * tf.nn.sigmoid(alpha * x)


def parametric_relu(x):
  alphas = tf.get_variable(
      'alpha',
      x.get_shape()[-1],
      initializer=tf.constant_initializer(0.1),
      dtype=tf.float32)
  pos = tf.nn.relu(x)
  neg = alphas * (x - abs(x)) * 0.5

  return pos + neg


def bi_attention_memory_efficient_dcn(
    a,
    b,
    mask_a=None,
    mask_b=None,  # pylint: disable=unused-argument
    activation='none',
    sim_func='trilinear'):
  """Applies biattention as in the DCN paper."""
  d = a.shape.as_list()[-1]
  if activation != 'none':
    with tf.variable_scope(activation):
      if activation == 'relu':
        a = tf.nn.relu(a)
        b = tf.nn.relu(b)
      elif activation == 'swish':
        a = swish(a)
        tf.get_variable_scope().reuse_variables()
        b = swish(b)
      elif activation == 'prelu':
        a = parametric_relu(a)
        tf.get_variable_scope().reuse_variables()
        b = parametric_relu(b)

  if sim_func == 'trilinear':
    logits = tf.transpose(
        trilinear_memory_efficient(a, b, d), perm=[0, 2, 1])  # [bs,len_b,len_a]
  elif sim_func == 'dot':
    logits = tf.matmul(b, a, transpose_b=True)  # [bs,len_b,len_a]
  b2a = b2a_attention(logits, a, mask_a)
  a2b = a2b_attention_dcn(logits, b)
  return b2a, a2b


def trilinear_memory_efficient(a, b, d, use_activation=False):
  """W1a + W2b + aW3b."""
  n = tf.shape(a)[0]

  len_a = tf.shape(a)[1]
  len_b = tf.shape(b)[1]

  w1 = tf.get_variable('w1', shape=[d, 1], dtype=tf.float32)
  w2 = tf.get_variable('w2', shape=[d, 1], dtype=tf.float32)
  w3 = tf.get_variable('w3', shape=[1, 1, d], dtype=tf.float32)

  a_reshape = tf.reshape(a, [-1, d])  # [bs*len_a, d]
  b_reshape = tf.reshape(b, [-1, d])  # [bs*len_b, d]

  part_1 = tf.reshape(tf.matmul(a_reshape, w1), [n, len_a])  # [bs, len_a]
  part_1 = tf.tile(tf.expand_dims(part_1, 2),
                   [1, 1, len_b])  # [bs, len_a, len_b]

  part_2 = tf.reshape(tf.matmul(b_reshape, w2), [n, len_b])  # [bs, len_b]
  part_2 = tf.tile(tf.expand_dims(part_2, 1),
                   [1, len_a, 1])  # [bs, len_a, len_b]

  a_w3 = a * w3  # [bs, len_a, d]
  part_3 = tf.matmul(a_w3, tf.transpose(b, perm=[0, 2, 1]))  # [bs,len_a,len_b]

  ## return the unnormalized logits matrix : [bs,len_a,len_b]
  if use_activation:
    return tf.nn.relu(part_1 + part_2 + part_3)
  return part_1 + part_2 + part_3


def b2a_attention(logits, a, mask_a=None):
  """Context-to-query attention."""
  if len(mask_a.get_shape()) == 1:
    mask_a = tf.sequence_mask(mask_a, tf.shape(a)[1])
  if len(mask_a.get_shape()) == 2:
    mask_a = tf.expand_dims(mask_a, 1)
  logits = exp_mask(logits, mask_a, mask_is_length=False)
  probabilities = tf.nn.softmax(logits)  # [bs,len_b,len_a]
  b2a = tf.matmul(probabilities, a)  # [bs, len_b, d]
  return b2a


def a2b_attention_dcn(logits, b):
  """Query-to-context attention."""
  prob1 = tf.nn.softmax(logits)  # [bs,len_b,len_a]
  prob2 = tf.nn.softmax(tf.transpose(logits, perm=[0, 2,
                                                   1]))  # [bs,len_a,len_b]
  a2b = tf.matmul(tf.matmul(prob1, prob2), b)  # [bs,len_b,d]
  return a2b


def _get_specific_shape(tensor, dim=None):
  """Gets the most specific dimension size(s) of the given tensor.

  This is a wrapper around the two ways to get the shape of a tensor: (1)
  t.get_shape() to get the static shape, and (2) tf.shape(t) to get the dynamic
  shape. This function returns the most specific available size for each
  dimension. If the static size is available, that is returned. Otherwise, the
  tensor representing the dynamic size is returned.

  Args:
    tensor: Input tensor.
    dim: Desired dimension. Use None to retrieve the list of all sizes.

  Returns:
    output = Most specific static dimension size(s).
  """
  static_shape = tensor.get_shape()
  dynamic_shape = tf.shape(tensor)
  if dim is not None:
    return static_shape[dim].value or dynamic_shape[dim]
  else:
    return [d.value or dynamic_shape[i] for i, d in enumerate(static_shape)]


def compute_loss_multi_answers(sparse_labels_with_padding, n_answer, logits):
  """Compute loss for multiple positive labels.

  loss = log_sum_exp(all_logits) - log_sum_exp(positive_logits)

  Args:
    sparse_labels_with_padding: [batch_size, max_n_answers],
    n_answer: [batch_size] the number of answers in each example
    logits: [batch_size, n_context] the logit values
  Returns:
    losses: [batch_size, 1]
  """

  # [batch_size, max_n_answers]
  dense_mask = form_dense_mask_from_sparse_labels(
      sparse_labels_with_padding, n_answer, logits)

  # TODO(thangluong,mingweichang): work out how to handle no-answers
  # marginalize over positive answer logits
  # [batch_size, 1]
  positive_logsumexp_score = tf.reduce_logsumexp(
      logits + tf.log(dense_mask + VERY_SMALL_NUMBER), axis=1)

  # marginalize over all logits
  # [batch_size, 1]
  all_logsumexp_score = tf.reduce_logsumexp(logits, axis=1)

  # [batch_size, 1]
  losses = all_logsumexp_score - positive_logsumexp_score

  return losses


def form_dense_mask_from_sparse_labels(sparse_labels_with_padding, n_answer,
                                       logits):
  """Forms a dense mask for sparse labels.

  Return a [batch_size x max_n_answer] mask, where all of the values
  are zeros except that for the positive answer locations.

  If the labels is [[2,0][1,2]] and the n_answer is [1,2]
  The the return densemask would be
  [[0,0,1],
   [0,1,1]]

  Args:
    sparse_labels_with_padding: [batch_size, max_n_answers],
    n_answer: [batch_size] the number of answers in the each example
    logits: [batch_size, n_context] the logits value

  Returns:
    A tf.float32 [batch_size, max_n_answers] binary matrix

  """
  # [batch_size, max_n_answers]
  [batch_size, max_n_answers] = _get_specific_shape(sparse_labels_with_padding)
  [_, n_context] = _get_specific_shape(logits)
  # [batch_size, max_n_answers]
  smasks = tf.sequence_mask(n_answer, max_n_answers)

  # create a batch idx to form a 2-d matrix; batch idx will be used
  # to restruct the dense mask after boolean mask.
  # [batch_size, max_n_answers]
  batch_idx = tf.cast(
      tf.tile(tf.expand_dims(tf.range(batch_size), [1]), [1, max_n_answers]),
      dtype=tf.int64)

  # concat the 2-d matrix with the answers
  # [batch_size, max_n_answers, 2] # the last dimension is (batch_idx, answer)
  batch_answer_pair = tf.stack(
      [batch_idx, sparse_labels_with_padding], axis=-1)

  # remove all of the padding answers in sparse_labels_with_padding
  # [num_of_non_padding_answers, 2]
  batch_answer_pair_flatten = tf.boolean_mask(batch_answer_pair, smasks)

  n_non_pad_answer = tf.shape(batch_answer_pair_flatten)[0]

  # reconstruct back from the dense matrix using scatter_nd
  # fill all ones according to the batch_answer_pair_flatten
  # [batch_size, n_context] all zeros except ones for the answer field
  dense_mask = tf.scatter_nd(
      batch_answer_pair_flatten,
      tf.ones(n_non_pad_answer, dtype=tf.int64),
      tf.cast(tf.stack([batch_size, n_context]), dtype=tf.int64))

  # The current dense mask could have values that are larger than one.
  # While all the answer spans are different, they could share
  # the same answer start indexes or end indexes.
  # The next line makes the dense mask value binary
  dense_mask = tf.minimum(tf.constant(1, dtype=tf.int64), dense_mask)

  dense_mask = tf.cast(dense_mask, dtype=tf.float32)

  return dense_mask


def _compute_position_loss(logits, gold_positions, n_answer, no_answer_bias,
                           label_smoothing, multi_answer_loss):
  """Compute position loss, e.g., used for start and end positions."""

  logits = tf.concat([no_answer_bias, logits], 1)
  if multi_answer_loss:
    # TODO(mingweichang): add support for label_smoothing
    losses = compute_loss_multi_answers(gold_positions + 1, n_answer,
                                        logits=logits)
  else:
    # Get the first gold position
    gold_positions = gold_positions[:, 0]
    losses = _compute_entropy_loss(logits, gold_positions + 1, label_smoothing)

  return losses


def get_loss(answer_start,
             answer_end,
             n_answer,
             logits_start,
             logits_end,
             no_answer_bias,
             label_smoothing=0.0,
             multi_answer_loss=False
            ):
  """Get loss given answer and logits.

  Args:
    answer_start: [batch_size, num_answers] storing the starting answer index
    answer_end: Similar to `answer_start` but for end.
    n_answer: [batch_size] number of answers
    logits_start: [batch_size, context_size]-shaped tensor for answer start
      logits.
    logits_end: Similar to `logits_start`, but for end. This tensor can be also
      [batch_size, context_size, context_size], in which case the true answer
      start is used to index on dim 1 (context_size).
    no_answer_bias: [batch_size, 1] shaped tensor, bias for no answer decision.
    label_smoothing: whether to use label smoothing or not.
    multi_answer_loss: if True, used marginalized multiple answer spans
  Returns:
    Loss tensor of shape [batch_size].
  """
  # Loss for start.
  start_losses = _compute_position_loss(
      logits_start, answer_start, n_answer, no_answer_bias, label_smoothing,
      multi_answer_loss)

  # Loss for end.
  end_losses = _compute_position_loss(
      logits_end, answer_end, n_answer, no_answer_bias, label_smoothing,
      multi_answer_loss)

  return start_losses + end_losses


def _compute_entropy_loss(logits, answer, label_smoothing):
  """Compute entropy loss with and without label smoothing."""
  if label_smoothing > 1e-10:
    # TODO(thangluong): verify this code
    answer_one_hot = tf.one_hot(answer, tf.shape(logits)[1])
    losses = tf.losses.softmax_cross_entropy(
        onehot_labels=answer_one_hot, logits=logits,
        label_smoothing=label_smoothing)
  else:
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer, logits=logits)
  return losses


def _constrain_prob_mat(prob_mat, max_answer_size):
  """Constraint prob mat such that start <= end < start + max_answer_size."""
  # prob_mat has shape [batch, doc_len, doc_len]
  max_x_len = tf.shape(prob_mat)[1]
  max_answer_length = tf.to_int64(tf.minimum(max_x_len, max_answer_size))
  return tf.matrix_band_part(prob_mat, 0, max_answer_length - 1)


def get_answer_pred(prob_mat, max_answer_size):
  """From the probability matrix, infer start/end/prob of the answer."""
  prob_mat = _constrain_prob_mat(prob_mat, max_answer_size)

  answer_pred_start = tf.argmax(tf.reduce_max(prob_mat, 2), 1)
  answer_pred_end = tf.argmax(tf.reduce_max(prob_mat, 1), 1)  # [batch_size]
  answer_prob = tf.reduce_max(prob_mat, [1, 2])

  return answer_pred_start, answer_pred_end, answer_prob


def _compute_prob_mat(logits_start, logits_end, no_answer_bias):
  """Compute answer probability matrix and related info."""
  tf.logging.info('# logits_start %s' % str(logits_start))
  tf.logging.info('# logits_end %s' % str(logits_end))

  # Predictions and metrics.
  concat_logits_start = tf.concat([no_answer_bias, logits_start], 1)
  concat_logits_end = tf.concat([no_answer_bias, logits_end], 1)

  concat_prob_start = tf.nn.softmax(concat_logits_start)
  concat_prob_end = tf.nn.softmax(concat_logits_end)

  # No-answer
  no_answer_prob = concat_prob_start[:, 0] * concat_prob_end[:, 0]

  # Answer
  prob_start = concat_prob_start[:, 1:]
  prob_end = concat_prob_end[:, 1:]

  prob_mat = tf.expand_dims(prob_start, -1) * tf.expand_dims(prob_end, 1)

  return prob_mat, no_answer_prob, prob_start, prob_end


def get_pred_ops(features,
                 max_answer_size,
                 logits_start,
                 logits_end,
                 no_answer_bias,
                 prob_mat=None,
                 no_answer_prob=None,
                 prob_start=None,
                 prob_end=None,
                 get_answer_op=None):
  """Get prediction op dictionary given prob-related Tensors.

  Args:
    features: Features.
    max_answer_size: maximum length of answer
    logits_start: [batch_size, context_size]-shaped tensor of logits for start.
    logits_end: Similar to `logits_start`, but for end. This tensor can be also
      [batch_size, context_size, context_size], in which case the true answer
      start is used to index on dim 1 (context_size).
    no_answer_bias: [batch_size, 1]-shaped tensor, bias for no answer decision.

    # Optional
    prob_mat: Tensor of size [batch_size, context_length, context_length]
    no_answer_prob: Tensor of size [batch_size]
    prob_start: Tensor of size [batch_size, context_length]
    prob_end: Tensor of size [batch_size, context_length]

  Returns:
    A dictionary of prediction tensors.
    This dictionary will contain predictions as well as everything needed
    to produce the nominal answer and identifier (ids).
  """
  if prob_mat is None:
    prob_mat, no_answer_prob, prob_start, prob_end = _compute_prob_mat(
        logits_start, logits_end, no_answer_bias)

  answer_pred_start, answer_pred_end, answer_prob = get_answer_pred(
      prob_mat, max_answer_size)

  # Has answer
  has_answer = no_answer_prob < answer_prob
  if get_answer_op:
    answer = get_answer_op(features, answer_pred_start, answer_pred_end,
                           has_answer)
  else:
    answer = squad_data.get_answer_op(
        features['context'], features['context_words'], answer_pred_start,
        answer_pred_end, has_answer)
  predictions = {
      'logits_start': logits_start,
      'logits_end': logits_end,
      'no_answer_bias': no_answer_bias,
      'yp1': answer_pred_start,
      'yp2': answer_pred_end,
      'p1': prob_start,
      'p2': prob_end,
      'prob_mat': prob_mat,
      'a': answer,
      'id': features['id'],
      'context': features['context'],
      'context_words': features['context_words'],
      'no_answer_prob': no_answer_prob,
      'answer_prob': answer_prob,
      'has_answer': has_answer,
  }
  predictions.update(features)

  # A few special cases
  if 'disable_answer_op' in flags.FLAGS:
    # When exporting to a tf graph, we don't want to include the py_func.
    del predictions['a']
  if 'emb_mat' in predictions:
    # Unused in output & breaks the tf.estimator.predict api, which
    # expects all outputs to have a batch dimension.
    del predictions['emb_mat']

  return predictions


def safe_log(prob):
  """Clip probabiity values before taking log."""
  clipped_prob = tf.clip_by_value(prob, VERY_SMALL_NUMBER, 1)
  return tf.log(clipped_prob)
