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

"""Feature neighborhood model."""

import feature_neighborhood_decoder
import feature_neighborhood_encoder
import feature_neighborhood_model_base

from lingvo import compat as tf
from lingvo.core import py_utils
from lingvo.core import schedule


class FeatureNeighborhoodModel(
    feature_neighborhood_model_base.FeatureNeighborhoodModelBase):
  """FeatureNeighborhood model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    # Snarfed from srl model
    tp = p.train
    tp.lr_schedule = (
        schedule.ContinuousSchedule.Params().Set(
            initial_value=1.0,
            start_step=128000,
            half_life_steps=192000,
            min=0.01))
    tp.vn_std = 0.0
    tp.l2_regularizer_weight = 1e-6
    tp.learning_rate = 0.001
    tp.clip_gradient_norm_to_value = 0.0
    tp.grad_norm_to_clip_to_zero = 0.0
    return p

  def __init__(self, params):
    super().__init__(params)
    tf.logging.info("Initializing")

    p = params
    if p.input_symbols:
      assert p.input_symbols.num_symbols() == p.input_vocab_size
    if p.input_symbols:
      assert p.output_symbols.num_symbols() == p.output_vocab_size

    ep = feature_neighborhood_encoder.FeatureNeighborhoodEncoder.Params()
    ep.name = "feature_neighborhood_encoder"
    ep.input = p.input
    ep.input_vocab_size = p.input_vocab_size
    ep.output_vocab_size = p.output_vocab_size
    ep.embedding_dim = p.embedding_dim
    ep.enc_units = p.enc_units
    ep.max_neighbors = p.max_neighbors
    ep.max_pronunciation_len = p.max_pronunciation_len
    ep.max_spelling_len = p.max_spelling_len
    ep.use_neighbors = p.use_neighbors
    ep.share_embeddings = p.share_embeddings
    self.CreateChild("encoder", ep)
    dp = feature_neighborhood_decoder.FeatureNeighborhoodDecoder.Params()
    dp.input = p.input
    dp.embedding_dim = p.embedding_dim
    dp.enc_units = p.enc_units
    dp.max_neighbors = p.max_neighbors
    dp.max_spelling_len = p.max_spelling_len
    dp.output_vocab_size = p.output_vocab_size
    dp.use_neighbors = p.use_neighbors
    dp.start = p.start
    self.CreateChild("decoder", dp)
    self.decoder.shared_out_emb = self.encoder.shared_emb

  def _child_variable_scope_override(self):
    return {
        **super()._child_variable_scope_override(), "encoder": [],
        "decoder": []
    }

  def ComputePredictions(self, theta, input_batch):
    p = self.params
    self._shape_batch(input_batch)

    self.enc_out = self.encoder.FPropDefaultTheta(input_batch)
    pronunciations = py_utils.NestedMap()
    pronunciations.pronunciations = input_batch.pronunciation
    predictions = self.decoder.ComputePredictions(self.enc_out, pronunciations,
                                                  p.is_inference)
    self.per_example_tensors["hyp"] = predictions.labels
    self.per_example_tensors["cognate_id"] = input_batch.cognate_id
    self.per_example_tensors["inp"] = input_batch.spelling
    self.per_example_tensors["ref"] = input_batch.pronunciation
    if p.use_neighbors:  # Note that cannot return None!
      self.per_example_tensors[
          "neighbor_spellings"] = input_batch.neighbor_spellings
      self.per_example_tensors[
          "neighbor_pronunciations"] = input_batch.neighbor_pronunciations
    self.prediction_values = predictions
    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    pronunciations = py_utils.NestedMap()
    pronunciations.pronunciations = input_batch.pronunciation
    loss, per_sequence_loss = self.decoder.ComputeLoss(predictions,
                                                       pronunciations)
    inf_labels = self.decoder.ComputePredictions(
        self.enc_out, pronunciations, is_inference=True).labels
    self.get_accuracy(loss, inf_labels, input_batch.pronunciation)

    return loss, per_sequence_loss
