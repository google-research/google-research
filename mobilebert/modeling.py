# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re

import numpy as np
import six
import tensorflow.compat.v1 as tf

from tensorflow.contrib import layers as contrib_layers


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               embedding_size=None,
               trigram_input=False,
               use_bottleneck=False,
               intra_bottleneck_size=None,
               use_bottleneck_attention=False,
               key_query_shared_bottleneck=False,
               num_feedforward_networks=1,
               normalization_type="layer_norm",
               classifier_activation=True):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      embedding_size: The size of the token embedding.
      trigram_input: Use a convolution of trigram as input.
      use_bottleneck: Use the bottleneck/inverted-bottleneck structure in BERT.
      intra_bottleneck_size: The hidden size in the bottleneck.
      use_bottleneck_attention: Use attention inputs from the bottleneck
        transformation.
      key_query_shared_bottleneck: Use the same linear transformation for
        query&key in the bottleneck.
      num_feedforward_networks: Number of FFNs in a block.
      normalization_type: The normalization type in BERT.
      classifier_activation: Using the tanh activation for the final
        representation of the [CLS] token in fine-tuning.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.embedding_size = embedding_size
    self.trigram_input = trigram_input
    self.use_bottleneck = use_bottleneck
    self.intra_bottleneck_size = intra_bottleneck_size
    self.use_bottleneck_attention = use_bottleneck_attention
    self.key_query_shared_bottleneck = key_query_shared_bottleneck
    self.num_feedforward_networks = num_feedforward_networks
    self.normalization_type = normalization_type
    self.classifier_activation = classifier_activation

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    if config.embedding_size is None:
      config.embedding_size = config.hidden_size
    if config.intra_bottleneck_size is None:
      config.intra_bottleneck_size = config.hidden_size
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               use_einsum=True,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) float32 or int32 Tensor
        of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      use_einsum: (optional) use einsum for faster tpu computation.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.float32)
    else:
      if input_mask.dtype == tf.int32:
        input_mask = tf.cast(input_mask, tf.float32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.embedding_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        if config.trigram_input:
          inputs = self.word_embedding_output
          self.embedding_output = tf.concat(
              [tf.pad(inputs[:, 1:], ((0, 0), (0, 1), (0, 0))),
               inputs,
               tf.pad(inputs[:, :-1], ((0, 0), (1, 0), (0, 0)))],
              axis=2)
        else:
          self.embedding_output = self.word_embedding_output

        if (config.trigram_input or
            config.embedding_size != config.hidden_size):
          self.embedding_output = dense_layer_2d(
              self.embedding_output,
              config.hidden_size,
              create_initializer(config.initializer_range),
              None,
              name="embedding_transformation",
              use_einsum=use_einsum)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            normalization_type=config.normalization_type)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers, self.all_attention_maps = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            use_bottleneck=config.use_bottleneck,
            intra_bottleneck_size=config.intra_bottleneck_size,
            use_bottleneck_attention=config.use_bottleneck_attention,
            key_query_shared_bottleneck=config.key_query_shared_bottleneck,
            num_feedforward_networks=config.num_feedforward_networks,
            normalization_type=config.normalization_type,
            use_einsum=use_einsum,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        if config.classifier_activation:
          self.pooled_output = dense_layer_2d(
              first_token_tensor,
              config.hidden_size,
              create_initializer(config.initializer_range),
              tf.tanh,
              name="dense",
              use_einsum=use_einsum)
        else:
          self.pooled_output = first_token_tensor

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_all_attention_maps(self):
    return self.all_attention_maps

  def get_word_embedding_output(self):
    """Get output of the word(piece) embedding lookup.

    This is BEFORE positional embeddings and token type embeddings have been
    added.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the word(piece) embedding layer.
    """
    return self.word_embedding_output

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars,
                                       init_checkpoint,
                                       init_from_teacher=False,
                                       init_embedding=False):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}
  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)
  assignment_map = collections.OrderedDict()
  if init_embedding:
    for x in init_vars:
      (name, var) = (x[0], x[1])
      if name not in name_to_variable:
        continue
      if not name.startswith("bert/encoder"):
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
  elif not init_from_teacher:
    for x in init_vars:
      (name, var) = (x[0], x[1])
      if name not in name_to_variable:
        continue
      assignment_map[name] = name
      initialized_variable_names[name] = 1
      initialized_variable_names[name + ":0"] = 1
  else:
    for x in init_vars:
      (name, var) = (x[0], x[1])
      if "teacher/" + name not in name_to_variable:
        continue
      assignment_map[name] = "teacher/" + name
      initialized_variable_names["teacher/" + name] = 1
      initialized_variable_names["teacher/" + name + ":0"] = 1
  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, rate=dropout_prob)
  return output


def layer_norm(input_tensor, normalization="layer_norm", name=None):
  """Run layer normalization on the last dimension of the tensor."""
  if normalization == "layer_norm":
    return contrib_layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1,
        begin_params_axis=-1, scope=name)
  elif normalization == "no_norm":
    filters = get_shape_list(input_tensor)[-1]
    with tf.variable_scope(name or "FakeLayerNorm"):
      bias = tf.get_variable(
          "beta", [filters], initializer=tf.zeros_initializer())
      scale = tf.get_variable(
          "gamma", [filters], initializer=tf.ones_initializer())
      return input_tensor * scale + bias
  elif normalization == "manual_layer_norm":
    filters = get_shape_list(input_tensor)[-1]
    with tf.variable_scope(name or "LayerNorm"):
      bias = tf.get_variable(
          "beta", [filters], initializer=tf.zeros_initializer())
      scale = tf.get_variable(
          "gamma", [filters], initializer=tf.ones_initializer())
      mean = tf.reduce_mean(input_tensor, axis=[-1], keepdims=True)
      difference = input_tensor - mean
      variance = tf.reduce_mean(
          difference * difference, axis=[-1], keepdims=True)
      norm_x = (input_tensor - mean) * tf.rsqrt(variance + 1e-6)
      return norm_x * scale + bias
  else:
    raise ValueError("Unsupported normalization: %s" % normalization)


def layer_norm_and_dropout(input_tensor, dropout_prob,
                           normalization="layer_norm", name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, normalization, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=False,
                            normalization_type="layer_norm"):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.
    normalization_type: string. The normalization used.
  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))

    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    if use_one_hot_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
      token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
    else:
      token_type_embeddings = tf.nn.embedding_lookup(
          token_type_table, token_type_ids)

    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob, normalization_type)
  return output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.reshape(to_mask, [batch_size, 1, to_seq_length])

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   size_per_head,
                   initializer,
                   activation,
                   name=None,
                   use_einsum=True):
  """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    num_attention_heads: Number of attention heads.
    size_per_head: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.
    use_einsum: Use tf.einsum.

  Returns:
    float logits Tensor.
  """

  batch_size, seq_len, last_dim = get_shape_list(input_tensor)

  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[last_dim, num_attention_heads * size_per_head],
        initializer=initializer)
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * size_per_head],
        initializer=tf.zeros_initializer)

    if use_einsum:
      w = tf.reshape(w, [last_dim, num_attention_heads, size_per_head])
      b = tf.reshape(b, [num_attention_heads, size_per_head])
      ret = tf.einsum("abc,cde->abde", input_tensor, w)
      ret += b
    else:
      input_tensor = tf.reshape(input_tensor, [batch_size * seq_len, last_dim])
      ret = tf.matmul(input_tensor, w)
      ret += b
      ret = tf.reshape(
          ret, [batch_size, seq_len, num_attention_heads, size_per_head])

    if activation is not None:
      return activation(ret)
    else:
      return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        num_attention_heads,
                        head_size,
                        initializer,
                        activation,
                        name=None,
                        use_einsum=True):
  """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    num_attention_heads: The size of output dimension.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.
    use_einsum: Use tf.einsum.

  Returns:
    float logits Tensor.
  """
  batch_size = get_shape_list(input_tensor)[0]
  seq_len = get_shape_list(input_tensor)[1]

  head_size = hidden_size // num_attention_heads
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[num_attention_heads * head_size, hidden_size],
        initializer=initializer)
    b = tf.get_variable(
        name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)

    if use_einsum:
      w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
      ret = tf.einsum("BFNH,NHD->BFD", input_tensor, w)
      ret += b
    else:
      input_tensor = tf.reshape(
          input_tensor, [batch_size * seq_len, hidden_size])
      ret = tf.matmul(input_tensor, w)
      ret += b
      ret = tf.reshape(ret, [batch_size, seq_len, hidden_size])

  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   name=None,
                   use_einsum=True):
  """A dense layer with 2D kernel.

  Args:
    input_tensor: Float tensor with rank 3.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Actication function.
    name: The name scope of this layer.
    use_einsum: Use tf.einsum.

  Returns:
    float logits Tensor.
  """
  if len(get_shape_list(input_tensor)) == 3:
    batch_size, seq_len, last_dim = get_shape_list(input_tensor)
    with tf.variable_scope(name):
      w = tf.get_variable(
          name="kernel", shape=[last_dim, output_size], initializer=initializer)
      b = tf.get_variable(
          name="bias", shape=[output_size], initializer=tf.zeros_initializer)

      if use_einsum:
        ret = tf.einsum("abc,cd->abd", input_tensor, w)
        ret += b
      else:
        input_tensor = tf.reshape(
            input_tensor, [batch_size * seq_len, last_dim])
        ret = tf.matmul(input_tensor, w)
        ret += b
        ret = tf.reshape(ret, [batch_size, seq_len, output_size])

  else:
    batch_size, last_dim = get_shape_list(input_tensor)
    with tf.variable_scope(name):
      w = tf.get_variable(
          name="kernel", shape=[last_dim, output_size], initializer=initializer)
      b = tf.get_variable(
          name="bias", shape=[output_size], initializer=tf.zeros_initializer)

      if use_einsum:
        ret = tf.einsum("ac,cd->ad", input_tensor, w)
        ret += b
      else:
        ret = tf.matmul(input_tensor, w)
        ret += b

  if activation is not None:
    return activation(ret)
  else:
    return ret


def attention_layer(query_tensor,
                    key_tensor,
                    value_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    use_einsum=True,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with tf.einsum as follows:
    Input_tensor: [BFD]
    Wq, Wk, Wv: [DNH]
    Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
    K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
    V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
    attention_scores:[BNFT] = einsum('BFNH,BTNH>BNFT', Q, K) / sqrt(H)
    attention_probs:[BNFT] = softmax(attention_scores)
    context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
    Wout:[DNH]
    Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)

  Args:
    query_tensor: float Tensor of shape [batch_size, query_seq_length,
      query_width].
    key_tensor: float Tensor of shape [batch_size, key_seq_length, key_width].
    value_tensor: float Tensor of shape [batch_size, key_seq_length, key_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    use_einsum: whether to use tf.einsum.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  from_shape = get_shape_list(query_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(key_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # `query_layer` = [B, F, N, H]
  query_layer = dense_layer_3d(query_tensor, num_attention_heads, size_per_head,
                               create_initializer(initializer_range), query_act,
                               "query", use_einsum)

  # `key_layer` = [B, T, N, H]
  key_layer = dense_layer_3d(key_tensor, num_attention_heads, size_per_head,
                             create_initializer(initializer_range), key_act,
                             "key", use_einsum)

  # `value_layer` = [B, T, N, H]
  value_layer = dense_layer_3d(value_tensor, num_attention_heads, size_per_head,
                               create_initializer(initializer_range), value_act,
                               "value", use_einsum)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  if use_einsum:
    attention_scores = tf.einsum("BFNH,BTNH->BNFT", query_layer, key_layer)
  else:
    query_layer = tf.transpose(query_layer, [0, 2, 1, 3])
    key_layer = tf.transpose(key_layer, [0, 2, 1, 3])
    # We want "BNFH,BNTH->BNFT" now.
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    if attention_mask.dtype != tf.float32:
      attention_mask = tf.cast(attention_mask, tf.float32)
    adder = tf.math.add(-10000.0, attention_mask * 10000.0)

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `context_layer` = [B, F, N, H]
  if use_einsum:
    context_layer = tf.einsum("BNFT,BTNH->BFNH", attention_probs, value_layer)
  else:
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    # We want "BNFT,BNTH->BNFH" now.
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  return context_layer, attention_scores


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      use_bottleneck=False,
                      intra_bottleneck_size=None,
                      use_bottleneck_attention=False,
                      key_query_shared_bottleneck=False,
                      num_feedforward_networks=1,
                      normalization_type="layer_norm",
                      use_einsum=False,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    use_bottleneck: whether to use bottleneck in BERT.
    intra_bottleneck_size: int, size of bottleneck.
    use_bottleneck_attention: Use attention inputs from the bottleneck
      transformation.
    key_query_shared_bottleneck: Whether to share linear transformation for
      keys and queries.
    num_feedforward_networks: int, number of ffns.
    normalization_type: string, the type of normalization_type.
    use_einsum: whether to use tf.einsum.
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if use_bottleneck:
    if intra_bottleneck_size % num_attention_heads != 0:
      raise ValueError(
          "The bottleneck size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (intra_bottleneck_size, num_attention_heads))
    attention_head_size = int(intra_bottleneck_size / num_attention_heads)
  else:
    if hidden_size % num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, num_attention_heads))
    attention_head_size = int(hidden_size / num_attention_heads)

  input_shape = get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  prev_output = input_tensor
  all_layer_outputs = []
  all_attention_maps = []
  all_layer_outputs.append(prev_output)
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      if use_bottleneck:
        with tf.variable_scope("bottleneck"):
          with tf.variable_scope("input"):
            layer_input = dense_layer_2d(
                prev_output,
                intra_bottleneck_size,
                create_initializer(initializer_range),
                None,
                name="dense",
                use_einsum=use_einsum)
            layer_input = layer_norm(layer_input, normalization_type)
          if use_bottleneck_attention:
            query_tensor = layer_input
            key_tensor = layer_input
            value_tensor = layer_input
          elif key_query_shared_bottleneck:
            with tf.variable_scope("attention"):
              shared_attention_input = tf.layers.dense(
                  prev_output,
                  intra_bottleneck_size,
                  kernel_initializer=create_initializer(initializer_range))
              shared_attention_input = layer_norm(shared_attention_input,
                                                  normalization_type)
            key_tensor = shared_attention_input
            query_tensor = shared_attention_input
            value_tensor = prev_output
          else:
            query_tensor = prev_output
            key_tensor = prev_output
            value_tensor = prev_output
      else:
        layer_input = prev_output
        query_tensor = prev_output
        key_tensor = prev_output
        value_tensor = prev_output

      if use_bottleneck:
        true_hidden_size = intra_bottleneck_size
      else:
        true_hidden_size = hidden_size

      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          attention_output, attention_map = attention_layer(
              query_tensor=query_tensor,
              key_tensor=key_tensor,
              value_tensor=value_tensor,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              use_einsum=use_einsum)
          all_attention_maps.append(attention_map)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = dense_layer_3d_proj(
              attention_output, true_hidden_size,
              num_attention_heads, attention_head_size,
              create_initializer(initializer_range), None, "dense",
              use_einsum)
          if not use_bottleneck:
            attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input,
                                        normalization_type)

      layer_input = attention_output

      if num_feedforward_networks != 1:
        for ffn_layer_idx in range(num_feedforward_networks - 1):
          with tf.variable_scope("ffn_layer_%d" % ffn_layer_idx):
            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
              intermediate_output = dense_layer_2d(
                  layer_input, intermediate_size,
                  create_initializer(initializer_range),
                  intermediate_act_fn, "dense", use_einsum)

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
              layer_output = dense_layer_2d(
                  intermediate_output, true_hidden_size,
                  create_initializer(initializer_range),
                  None, "dense", use_einsum)
              layer_output = layer_norm(layer_output + layer_input,
                                        normalization_type)
              layer_input = layer_output

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = dense_layer_2d(
            layer_input, intermediate_size,
            create_initializer(initializer_range),
            intermediate_act_fn, "dense", use_einsum)

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = dense_layer_2d(intermediate_output, true_hidden_size,
                                      create_initializer(initializer_range),
                                      None, "dense", use_einsum)
        if not use_bottleneck:
          layer_output = dropout(layer_output, hidden_dropout_prob)
          layer_output = layer_norm(layer_output + layer_input,
                                    normalization_type)
        else:
          layer_output = layer_norm(layer_output + layer_input,
                                    normalization_type)
          with tf.variable_scope("bottleneck"):
            layer_output = dense_layer_2d(layer_output, hidden_size,
                                          create_initializer(initializer_range),
                                          None, "dense", use_einsum)
            layer_output = dropout(layer_output, hidden_dropout_prob)
            layer_output = layer_norm(layer_output + prev_output,
                                      normalization_type)

        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    return all_layer_outputs, all_attention_maps
  else:
    return all_layer_outputs[-1], all_attention_maps


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
