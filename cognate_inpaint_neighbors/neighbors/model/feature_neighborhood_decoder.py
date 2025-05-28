# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""FeatureNeighborhood decoder.

Follows the model whereby the encoder output for the spelling is fed into the
decoder, and also into an auxiliary model that creates an attention vector from
the 3D [*, max_neighbors, max_spelling_len/max_pronunciation_len] vectors of
neighbor spelling/pronunciation pairs.

Shapes for Keras variables must be declared up front in order to initialize them
correctly. The shapes below follow from the construction of the graph and the
data it is applied to.
"""

from keras_interface_layer import KerasInterfaceLayer
from lingvo import compat as tf
from lingvo.core import py_utils


class BahdanauAttention(KerasInterfaceLayer):
  """Just using the definition in the NMT tutorial.

  Uses the definition from
  https://www.tensorflow.org/tutorials/text/nmt_with_attention.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = "bahdanau_attention"
    p.Define("enc_units", -1, "Number of encoding units.")
    p.Define("input_size", -1, "Size of input embeddings.")
    return p

  def __init__(self, params):
    super().__init__(params)
    # pylint: disable=invalid-name
    p = params
    # This name_scope is for checkpoint backwards-compatibility.
    with tf.name_scope(self._self_variable_scope.original_name_scope):
      self.W1 = self.AddVariable(
          tf.keras.layers.Dense(p.enc_units),
          input_shape=(None, None, p.input_size),
          keras_scope="W1")
      self.W2 = self.AddVariable(
          tf.keras.layers.Dense(p.enc_units),
          input_shape=(None, 1, p.enc_units),
          keras_scope="W2")
      self.V = self.AddVariable(
          tf.keras.layers.Dense(1),
          input_shape=(None, None, p.enc_units),
          keras_scope="V")
    # pylint: enable=invalid-name

  def __call__(self, query, values):
    """Compute attention.

    Args:
      query: query vector to compute similarity to. [batch, query_units]
      values: value vectors to attend over. [batch, seq_len, input_size]

    Returns:
      tuple of (
          context_vector: [batch, input_size]
          attention_weights: the computed attention weights [batch, seq_len, 1]
      )
    """
    # [batch, 1, query_units]
    hidden_with_time_axis = tf.expand_dims(query, axis=1)
    tanh_output = tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))
    # [batch, seq_len, 1]
    score = self.V(tanh_output)
    # attention_weights shape == [*, max_spelling_len, 1]
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights


class DotAttention(KerasInterfaceLayer):
  """Scaled dot product attention.

  See "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = "scaled_dot_attention"
    p.Define("enc_units", -1, "Number of encoding units.")
    p.Define("input_size", -1, "Size of input embeddings.")
    p.Define("scaled", True, "Scaled attention.")
    return p

  def __init__(self, params):
    super().__init__(params)
    # pylint: disable=invalid-name
    p = params
    # This name_scope is for checkpoint backwards-compatibility.
    with tf.name_scope(self._self_variable_scope.original_name_scope):
      self.Wq = self.AddVariable(
          tf.keras.layers.Dense(p.enc_units),
          input_shape=(None, p.enc_units),
          keras_scope="Wq")
      self.Wk = self.AddVariable(
          tf.keras.layers.Dense(p.enc_units),
          input_shape=(None, None, p.input_size),
          keras_scope="Wk")
      self.Wv = self.AddVariable(
          tf.keras.layers.Dense(p.input_size),
          input_shape=(None, None, p.input_size),
          keras_scope="Wv")
    # pylint: enable=invalid-name

  def __call__(self, context, inputs):
    p = self.params

    # context - [batch_size, context_size]
    # inputs - [batch_size, seq_len, input_size]

    # [batch_size, context_size] --> [batch_size, hidden_size]
    query = self.Wq(context)
    # [batch_size, seq_len, input_size] @ [input_size, hidden_size]
    # --> [batch_size, seq_len, hidden_size]
    keys = self.Wk(inputs)
    # [batch_size, seq_len, input_size] --> [batch_size, seq_len, hidden_size]
    values = self.Wv(inputs)
    # [batch_size, hidden_size] --> [batch_size, hidden_size, 1]
    query = tf.expand_dims(query, axis=2)
    # [batch_size, seq_len, hidden_size] @ [batch_size, hidden_size, 1]
    # --> [batch_size, seq_len, 1]
    logits = tf.matmul(keys, query)
    attention_weights = tf.nn.softmax(logits, axis=1)
    if p.scaled:
      attention_weights /= tf.sqrt(tf.cast(self.params.enc_units, tf.float32))

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class FeatureNeighborhoodDecoder(KerasInterfaceLayer):
  """Hanasu regression decoder.

  Includes a decoder for the main spelling to pronunciation map, as well as a
  model for the neighborhood data.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = "feature_neighborhood_decoder"
    p.Define("input", None, "Input generator Params.")
    p.Define("embedding_dim", 256, "Embedding dimension.")
    p.Define("enc_units", 256, "Number of encoder units.")
    p.Define("max_spelling_len", 20, "Maximum length of spellings.")
    p.Define("max_neighbors", 50, "Maximum number of neighbors.")
    p.Define("output_vocab_size", 256, "Size of output vocabulary.")
    p.Define("use_neighbors", True, "Whether or not to use neighbor data.")
    p.Define("start", 1, "Index of <s> in output symbol set.")
    p.Define("dot_attention", False, "Use dot product attention.")
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    # This name_scope is for checkpoint backwards-compatibility.
    with tf.name_scope(self._self_variable_scope.original_name_scope):
      self.shared_out_emb = None
      # See construction of Concat below to see why this is correct.
      gru_input_last_dim = p.embedding_dim + p.enc_units
      if p.use_neighbors:
        gru_input_last_dim += p.enc_units
      self._gru_cell = self.AddVariable(
          tf.keras.layers.GRUCell(
              p.enc_units, recurrent_initializer="glorot_uniform"),
          input_shape=(None, 1, gru_input_last_dim),
          keras_scope="{}/{}".format(p.name, "gru"))
      self._fc = self.AddVariable(
          tf.keras.layers.Dense(p.output_vocab_size),
          input_shape=(None, p.enc_units),
          keras_scope="{}/{}".format(p.name, "fc"))
      self._tf_constant_zero = tf.constant(0)
      self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

      bap = BahdanauAttention.Params()
      bap.enc_units = p.enc_units
      bap.input_size = p.enc_units
      bap.name = "main_attention"
      self.CreateChild("attention", bap)

      if p.use_neighbors:
        if p.dot_attention:
          neighbor_attention = DotAttention.Params()
        else:
          neighbor_attention = bap

        neighbor_attention.enc_units = p.enc_units
        neighbor_attention.input_size = p.enc_units
        neighbor_attention.name = "neighbor_attention"
        self.CreateChild("neighbor_attention", neighbor_attention)

  def _NeighborModelAttention(self, encoder_outputs, state):
    """Computes the attention for neighbor features.

    Combines the attention output with the neighbors.  Sets up a
    "relevance" vector (see Pundak et
    al. https://arxiv.org/pdf/1808.02480.pdf).

    If there are no neighbors we catch an error and return a pair of
    tf.constant(0)'s.

    Args:
      encoder_outputs: A NestedMap containing the encodings for neighbor
        spellings, neighbor_pronuncations.
      state: previous state of decoding.

    Returns:
      Pair of tensors, (context_vector, attention_weights) or
      (tf.constant(0), tf.constant(0)) if there are no neighbors.
    """
    p = self.params

    def sum_embeddings(t):
      reshaped = tf.reshape(
          t, (p.input.batch_size, p.max_neighbors, -1, p.enc_units))
      return tf.reduce_mean(reshaped, axis=2)

    if p.use_neighbors:
      # --> [batch_size, max_neighbors + max_neighbors, enc_units]
      enc_output = tf.concat(
          [
              # [batch_size * max_neighbors, max_spelling_len, enc_units]
              # --> [batch_size, max_neighbors, enc_units]
              sum_embeddings(encoder_outputs.neighbor_spellings_encoded),
              sum_embeddings(encoder_outputs.neighbor_pronunciations_encoded),
          ],
          axis=1)
      context_vector, attention_weights = self.neighbor_attention(
          state, enc_output)
      return context_vector, attention_weights
    else:
      return self._tf_constant_zero, self._tf_constant_zero

  def Decode(self, encoder_outputs, dec_input, state):
    """The decoder model.

    Args:
      encoder_outputs: a NestedMap containing the following fields: encoded -
        the encoding of the spelling of the target feature state - hidden state
        of the encoder output neighbor_spellings_encoded - encoding of neighbor
        spellings or tf.constant(0) neighbor_pronunciations_encoded - encoding
        of neighbor pronunciations or tf.constant(0) -
        initial prediction of [<s>] of shape [*, 1]
      dec_input: if not None, then use this instead of the dec_input in
        encoder_outputs.
      state: previous state of decoding.

    Returns:
       res: a NestedMap() containing
         predictions - of shape [*, output_vocab_size]
         state - updated hidden state
         attention_weights
         neighbor_attention_weights
    """
    p = self.params

    context_vector, attention_weights = self.attention(state,
                                                       encoder_outputs.encoded)
    (neighbor_context_vector,
     neighbor_attention_weights) = self._NeighborModelAttention(
         encoder_outputs, state)
    x = self.shared_out_emb(dec_input)  # pylint: disable=not-callable
    if p.use_neighbors:
      x = tf.concat([context_vector, neighbor_context_vector, x], axis=-1)
    else:
      x = tf.concat([context_vector, x], axis=-1)

    output, state = self._gru_cell(x, state)
    x = self._fc(output)

    # If this fails then the checkpoint contains incorrect shapes or the
    # hparams are incompatible. No idea why TF doesn't check this anymore.
    x = tf.ensure_shape(x, [None, p.output_vocab_size])

    res = py_utils.NestedMap()
    res.predictions = x
    res.state = state
    res.attention_weights = attention_weights
    res.neighbor_attention_weights = neighbor_attention_weights
    return res

  def ComputePredictions(self,
                         encoder_outputs,
                         pronunciations,
                         is_inference=False):
    """Computes the predictions from the encoder_outputs, updating losses.

    Despite the name, this function does the bulk of the decoding and loss
    computation, incrementing the loss at each time step.

    Args:
      encoder_outputs: a NestedMap consisting of outputs of the
        FeatureNeighborhoodEncoder with  encoded - encoding of the input
        spelling
        neighbor_pronunciations_encoded - encodings of the neighbor prons
        neighbor_pronunciations_encoded - encodings of the neighbor spellings
        state - encoder state to which has been added dec_input - seed output
        for the decoder [*, 1] tensor consisting of sentence start indices
        (corresponding to "<s>")
      pronunciations: NestedMap with pronunciations - [*, max_pronunciation_len]
        tensor of pronunciations
      is_inference: If False then uses teacher forcing else does autoregression.

    Returns:
      NestedMap with loss, per_sequence_losses,labels, a
      [*, max_pronunciation_len] tensor of predictions, and attention
      ([*, max_pronunciation_len, max_spelling_len]), and
      neighbor_attention ([*, max_pronunciation_len, max_neighbors])
      tensors, along with the raw batch passed through from the encoder.
    """
    p = self.params
    targets = pronunciations.pronunciations
    t_len = int(targets.get_shape().as_list()[1])
    t_idx = tf.constant(0)
    attention = tf.TensorArray(dtype=tf.float32, size=t_len)
    neighbor_attention = tf.TensorArray(dtype=tf.float32, size=t_len)

    outputs = tf.TensorArray(dtype=tf.float32, size=t_len)

    loop_cond = lambda t_idx, ts, *_: tf.less(t_idx, t_len)

    dec_input = tf.convert_to_tensor([p.start] * p.input.batch_size)
    state = encoder_outputs.state

    # pylint: disable=missing-docstring
    def loop_body(t_idx, dec_input, attention, neighbor_attention, state,
                  outputs):
      decoder_result = self.Decode(encoder_outputs, dec_input, state)

      outputs = outputs.write(t_idx, decoder_result.predictions)
      attention = attention.write(t_idx, decoder_result.attention_weights)
      neighbor_attention = neighbor_attention.write(
          t_idx,
          tf.cast(decoder_result.neighbor_attention_weights, dtype=tf.float32))

      if is_inference:
        dec_input = tf.cast(tf.argmax(decoder_result.predictions, 1), tf.int32)
      else:
        dec_input = targets[:, t_idx]
      t_idx = t_idx + 1
      state = decoder_result.state
      return t_idx, dec_input, attention, neighbor_attention, state, outputs

    _, _, attention, neighbor_attention, state, outputs = tf.while_loop(
        loop_cond,
        loop_body,
        loop_vars=[
            t_idx, dec_input, attention, neighbor_attention, state, outputs
        ])

    outputs = tf.transpose(outputs.stack(), [1, 0, 2])
    labels = tf.argmax(outputs, axis=-1)
    mask = tf.cast(
        tf.math.logical_not(tf.math.equal(targets, 0)), dtype=tf.float32)
    loss = self._loss_object(targets, outputs, sample_weight=mask)
    loss = tf.reduce_sum(loss, axis=1)
    per_sequence_losses = (loss / t_len)
    loss = tf.reduce_mean(per_sequence_losses)
    predictions = py_utils.NestedMap()
    predictions.loss = loss
    predictions.per_sequence_losses = per_sequence_losses
    predictions.labels = labels
    predictions.attention = tf.transpose(
        tf.squeeze(attention.stack()), perm=[1, 0, 2])
    if p.use_neighbors:
      predictions.neighbor_attention = tf.transpose(
          tf.squeeze(neighbor_attention.stack()), perm=[1, 0, 2])
    else:
      predictions.neighbor_attention = tf.squeeze(neighbor_attention.stack())
    # Expose this for subsequent data analysis
    predictions.batch = encoder_outputs.batch
    return predictions

  def ComputeLoss(self, predictions, pronunciations):
    """Computes loss for predictions and pronunciations.

    Args:
      predictions: NestedMap, the output of ComputePredictions
      pronunciations: a nested map containing pronunciations a [*,
        max_pronunciation_len] tensor of pronunciations.

    Returns:
      NestedMap with loss, NestedMap with per_sequence_losses.
    """
    p = self.params
    loss = py_utils.NestedMap()
    loss.loss = predictions.loss, p.input.batch_size
    per_sequence_loss = py_utils.NestedMap()
    per_sequence_loss.per_sequence_losses = (
        predictions.per_sequence_losses,
        tf.constant([1.0] * predictions.per_sequence_losses.shape[0]))
    return loss, per_sequence_loss
