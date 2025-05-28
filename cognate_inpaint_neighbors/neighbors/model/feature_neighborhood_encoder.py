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

"""Encoder for FeatureNeighborhood.

Includes encoder for the main target feature and for the neighbor features.

Shapes for Keras variables must be declared up front in order to initialize them
correctly. The shapes below follow from the construction of the graph and the
data it is applied to.
"""

from keras_interface_layer import KerasInterfaceLayer
from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import summary_utils


class FeatureNeighborhoodEncoder(KerasInterfaceLayer):
  """FeatureNeighborhood encoder.

  Computes the encoding for the main input feature and, if provided, the 3D
  neighbor spellings and neighbor pronunciations tensors.
  """

  @classmethod
  def Params(cls):
    """Configuration for the HanasuEncoder."""
    p = super().Params()
    p.Define("input", None, "Input generator Params.")
    p.Define("input_vocab_size", 256, "Input vocabulary size.")
    p.Define("output_vocab_size", 256, "Output vocabulary size.")
    p.Define("embedding_dim", 256, "Embedding dimension.")
    p.Define("enc_units", 256, "Number of encoding units.")
    p.Define("max_neighbors", 50, "Maximum number of neighbors")
    p.Define("max_pronunciation_len", 40, "Maximum length of pronunciations.")
    p.Define("max_spelling_len", 20, "Maximum length of spellings.")
    p.Define("use_neighbors", True, "Whether or not to use neighbor data.")
    p.Define("share_embeddings", False, "Share input and output embeddings.")
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    # This name_scope is for checkpoint backwards-compatibility.
    with tf.name_scope(self._self_variable_scope.original_name_scope):
      if p.share_embeddings:
        self.shared_emb = self.AddVariable(
            tf.keras.layers.Embedding(p.output_vocab_size, p.embedding_dim),
            input_shape=(p.input_vocab_size, p.max_neighbors),
            keras_scope="shared_emb")
      else:
        self._embedding = self.AddVariable(
            tf.keras.layers.Embedding(p.input_vocab_size, p.embedding_dim),
            input_shape=(p.input_vocab_size,),
            keras_scope="shared_inp_emb")
        self.shared_emb = self.AddVariable(
            tf.keras.layers.Embedding(p.output_vocab_size, p.embedding_dim),
            input_shape=(p.output_vocab_size, p.max_neighbors),
            keras_scope="shared_out_emb")
      self._gru = self.AddVariable(
          tf.keras.layers.GRU(
              p.enc_units,
              return_sequences=True,
              return_state=True,
              recurrent_initializer="glorot_uniform"),
          input_shape=(None, p.max_spelling_len, p.embedding_dim),
          keras_scope="main_gru")
      if p.use_neighbors:
        if p.share_embeddings:
          self._neighbor_spellings_embeddings = self.shared_emb
          self._neighbor_pronunciations_embeddings = self.shared_emb
        else:
          self._neighbor_spellings_embeddings = self._embedding
          self._neighbor_pronunciations_embeddings = self.shared_emb
        self._neighbor_spellings_gru = self.AddVariable(
            tf.keras.layers.GRU(
                p.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform"),
            # Multiplier in final dim since input gets reshaped from 4D to 3D.
            input_shape=(None, p.max_spelling_len, p.embedding_dim),
            keras_scope="neighbor_spellings_gru")
        self._neighbor_pronunciations_gru = self.AddVariable(
            tf.keras.layers.GRU(
                p.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform"),
            # Multiplier in final dim since input gets reshaped from 4D to 3D.
            input_shape=(None, p.max_pronunciation_len, p.embedding_dim),
            keras_scope="neighbor_pronunciations_gru")

  @property
  def supports_streaming(self):
    return False

  def FProp(self, theta, batch):
    """Encodes source as.

    Args:
      theta: A nested map object containing weights' values of this layer and
        its children layers.
      batch: A NestedMap with fields: spelling - The input spelling tensor.
        Optional fields: neighbor_spellings - [*, max_neighbors,
          max_spelling_len] int32 tensor of neighbor spellings;
          neighbor_pronunciations - [*, max_neighbors, max_pronunciation_len]
          int32 tensor of neighbor pronunciations.

    Returns:
      A NestedMap with:
        encoded: a [*, max_spelling, enc_units] tensor.
        state: a [*, enc_units] tensor for the state output of the GRU,
        batch: the original data batch
        And, optionally:
         neighbor_spellings: a [*, max_neighbors, enc_units] tensor.
         neighbor_pronunciations: a [*, max_neighbors, enc_units] tensor.
        or tf.constant(0) in each case if neighbor_spellings,
         neighbor_pronunciations or neighbor_distances are not present.
    """

    def reshape(embeddings):
      dims = embeddings.shape
      return tf.reshape(embeddings, [-1, dims[2], dims[3]])

    p = self.params
    with tf.name_scope(p.name):
      plots = []
      # --> [batch_size, max_spelling_len, embedding_dim]
      x = self._embedding(batch.spelling)
      # encoded [batch_size, max_spelling_len, embedding_dim]
      # state [batch_size, embedding_dim]
      encoded, state = self._gru(x)
      self.__AppendPlotData(plots, encoded, "encoded")
      summary_utils.PlotSequenceFeatures(
          list(reversed(plots)), "encoder", xlabel="Input Position")
      try:
        if (batch.Get("neighbor_spellings") is not None and
            batch.Get("neighbor_pronunciations") is not None):
          # [batch_size, max_neighbors, max_spelling_len] -->
          # [batch_size, max_neighbors, max_spelling_len, embedding_dim]
          neighbor_spellings_embeddings = (
              self._neighbor_spellings_embeddings(batch.neighbor_spellings))
          neighbor_pronunciations_embeddings = (
              self._neighbor_pronunciations_embeddings(
                  batch.neighbor_pronunciations))

          # [batch_size, max_spelling, max_spelling_len, enc_units] -->
          # [batch_size * max_spelling, max_spelling_len, enc_units]
          neighbor_spellings_encoded, _ = self._neighbor_spellings_gru(
              reshape(neighbor_spellings_embeddings))
          neighbor_pronunciations_encoded, _ = (
              self._neighbor_pronunciations_gru(
                  reshape(neighbor_pronunciations_embeddings)))
        else:
          neighbor_spellings_encoded = tf.constant(0)
          neighbor_pronunciations_encoded = tf.constant(0)
      except AttributeError:
        neighbor_spellings_encoded = tf.constant(0)
        neighbor_pronunciations_encoded = tf.constant(0)
      return py_utils.NestedMap(
          encoded=encoded,
          state=state,
          neighbor_spellings_encoded=neighbor_spellings_encoded,
          neighbor_pronunciations_encoded=neighbor_pronunciations_encoded,
          batch=batch)

  def __AppendPlotData(self, plots, data, name):
    dummy_pad = tf.constant(0, shape=[data.shape[0], data.shape[1]])
    plots.append(summary_utils.PrepareSequenceForPlot(data, dummy_pad, name))
