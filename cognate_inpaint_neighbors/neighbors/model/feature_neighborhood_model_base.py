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

"""FeatureNeighborhood text normalization model."""

from lingvo import compat as tf
from lingvo.core import base_model
from lingvo.core import py_utils


class FeatureNeighborhoodModelBase(base_model.BaseTask):
  """FeatureNeighborhood model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.name = "feature_neighborhood_with_neighbors"
    p.Define("embedding_dim", 256, "Embedding dimension")
    p.Define("enc_units", 256, "Number of encoder units")
    p.Define("input_symbols", None, "Input symbol table.")
    p.Define("input_vocab_size", 256, "Input vocabulary size.")
    p.Define("max_neighbors", 50, "Maximum number of neighbors")
    p.Define("max_pronunciation_len", 40, "Maximum length of pronunciations.")
    p.Define("max_spelling_len", 20, "Maximum length of spellings.")
    p.Define("output_symbols", None, "Output symbol table.")
    p.Define("output_vocab_size", 256, "Output vocabulary size.")
    p.Define("start", 1, "Index of <s> in output symbol set.")
    p.Define("use_neighbors", True, "Whether or not to use neighbor data.")
    p.Define("share_embeddings", False, "Share input and output embeddings.")
    return p

  def _shape_batch(self, batch):
    # I am not clear on why this is necessary but if you don't do it you run
    # into problems because pooowa widdle TensorFwow gets confused:
    #
    # raise ValueError("as_list() is not defined on an unknown TensorShape.")
    #
    # along with other similar problems.
    # When using TPU the shapes are set in the FeatureNeighborhoodInput class.
    if py_utils.use_tpu():
      return
    p = self.params
    batch.spelling.set_shape([p.input.batch_size, p.max_spelling_len])
    batch.pronunciation.set_shape([p.input.batch_size, p.max_pronunciation_len])
    if p.use_neighbors:
      batch.neighbor_spellings.set_shape(
          [p.input.batch_size, p.max_neighbors, p.max_spelling_len])
      batch.neighbor_pronunciations.set_shape(
          [p.input.batch_size, p.max_neighbors, p.max_pronunciation_len])

  def print_input_and_output_tensors(self, per_example_tensors):
    """Converts per_example_tensor entries to strings given symbol tables.

    Args:
      per_example_tensors: A dictionary produced by ComputePredictions()

    Returns:
      A nested map of inp, hyp, ref, ... strings
    """
    p = self.params
    strings = py_utils.NestedMap()

    def array_to_string(array, symbol_table=None):
      if not symbol_table:
        return array
      # TODO(rws): This needs to work for py3 too...ugh
      return [symbol_table.find(i) for i in array if i != 0]

    # TODO(rws): Stuff for neighbor data, but must check if p has use_neighbors
    # set...
    strings.inp = []
    strings.hyp = []
    strings.ref = []
    strings.cognate_id = []
    for b in range(p.input.batch_size):
      strings.cognate_id.append(bytes.decode(
          per_example_tensors["cognate_id"][b]))
      strings.inp.append(
          array_to_string(per_example_tensors["inp"][b, :], p.input_symbols))
      strings.ref.append(
          array_to_string(per_example_tensors["ref"][b, :], p.output_symbols))
      strings.hyp.append(
          array_to_string(per_example_tensors["hyp"][b, :], p.output_symbols))
    return strings

  def get_accuracy(self, loss, pred, target):
    p = self.params
    int_dtype = pred.dtype
    target = tf.cast(target, int_dtype)
    pad_id = int(p.input.feature_neighborhood_input.batch_opts.pad_value)
    mask = tf.cast(tf.math.not_equal(target, pad_id), int_dtype)
    pred *= mask
    num_non_zero = tf.cast(tf.reduce_sum(mask), tf.float32)
    equal = tf.math.equal(pred, target)
    loss["accuracy_per_example"] = (tf.reduce_mean(
        tf.cast(tf.reduce_all(equal, axis=1), tf.float32)), p.input.batch_size)
    equal = tf.cast(equal, tf.float32)
    equal *= tf.cast(mask, tf.float32)
    loss["accuracy_per_char"] = (tf.reduce_sum(equal) / num_non_zero,
                                 p.input.batch_size)
