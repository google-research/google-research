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

"""FeatureNeighborhood transformer model."""

import feature_neighborhood_model_base

from lingvo import compat as tf
from lingvo.core import py_utils


class FeatureNeighborhoodModelTrans(
    feature_neighborhood_model_base.FeatureNeighborhoodModelBase):
  """FeatureNeighborhood transformer model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define("spell_encoder", None, "Spellings Encoder Params.")
    p.Define("pron_encoder", None, "Pronunciations Encoder Params.")
    p.Define("beam_size", 8, "Beam search beam size.")
    p.Define("use_neigh_id_emb", False,
             "Add ID embeddings to each neighbor pairs.")
    # Methods of attending to the neighbors information:
    # AVERAGE - Process all neighbours seperatly and averaging the Transformer
    # output in to a single vector before concatenating it to the enc output.
    # CONCAT - Directly concatenate all the neighbours in to one long input
    # only feasible with small neighbours.
    # CONCATAVE - Concatenate the neighbour's spelling and pron to process them
    # together in one transformer.
    p.Define("neigh_att_type", "AVERAGE", "How to attend to the neighbours.")
    p.Define(
        "also_shuffle_neighbors", True,
        "If use_neigh_id_emb and neigh_att_type=CONCAT, also "
        "shuffle neighbors.")
    p.Define("aux_dropout_prob", 0.0,
             "Dropout prob for auxiliary decoder inputs.")
    return p

  def __init__(self, params):
    super().__init__(params)
    p = params
    if p.input_symbols:
      assert p.input_symbols.num_symbols() == p.input_vocab_size
    if p.input_symbols:
      assert p.output_symbols.num_symbols() == p.output_vocab_size

    if p.share_embeddings:
      renames = [("(.*)/token_emb/(.*)", "%s/shared_emb/token_emb/%s")]
    else:
      renames = [("(.*)/(?:encoder|spell_encoder)/token_emb/(.*)",
                  "%s/shared_inp_emb/token_emb/%s"),
                 ("(.*)/(?:decoder|pron_encoder)/token_emb/(.*)",
                  "%s/shared_out_emb/token_emb/%s")]

    # Enable variable sharing.
    with py_utils.OpportunisticVariableReuseScope():
      with py_utils.VariableRenameScope(renames):
        self.CreateChild("encoder", p.encoder)
        self.CreateChild("decoder", p.decoder)

        if p.use_neighbors:
          self.CreateChild("spell_encoder", p.spell_encoder)
          if p.pron_encoder:
            self.CreateChild("pron_encoder", p.pron_encoder)

  def _GetPaddings(self, t, dtype=tf.float32):
    p = self.params
    pad_id = int(p.input.feature_neighborhood_input.batch_opts.pad_value)
    return tf.cast(tf.equal(t, pad_id), dtype)

  def _AddStartToken(self, t):
    p = self.params
    return tf.concat(
        [tf.constant(p.start, tf.int32, [p.input.batch_size, 1]), t[:, :-1]],
        axis=1)

  def _AveNeigh(self, spellings, pronunciations, theta, batch_size):
    p = self.params
    spellings = tf.reshape(spellings, (batch_size * p.max_neighbors, -1))
    pronunciations = tf.reshape(pronunciations,
                                (batch_size * p.max_neighbors, -1))

    spell_inp = py_utils.NestedMap({
        "ids": spellings,
        "paddings": self._GetPaddings(spellings, dtype=tf.int32),
    })

    pron_inp = py_utils.NestedMap({
        "ids": pronunciations,
        "paddings": self._GetPaddings(pronunciations, dtype=tf.int32),
    })

    if p.use_neigh_id_emb:
      # Add the same ID embeddings to both spelling and pron so that the
      # model knows how they pair up.
      neigh_ids = tf.range(p.max_neighbors)[:, tf.newaxis]
      spell_inp["task_ids"] = tf.tile(neigh_ids,
                                      [batch_size, p.max_spelling_len])
      pron_inp["task_ids"] = tf.tile(neigh_ids,
                                     [batch_size, p.max_pronunciation_len])

    spell_enc_out = self.spell_encoder.FProp(theta.spell_encoder, spell_inp)
    pron_enc_out = self.spell_encoder.FProp(theta.pron_encoder, pron_inp)

    spell_enc = tf.reshape(
        spell_enc_out["encoded"],
        (p.max_spelling_len, batch_size, p.max_neighbors, p.enc_units))
    spell_enc = tf.reduce_mean(spell_enc, axis=0)

    pron_enc = tf.reshape(
        pron_enc_out["encoded"],
        (p.max_pronunciation_len, batch_size, p.max_neighbors, p.enc_units))
    pron_enc = tf.reduce_mean(pron_enc, axis=0)

    spell_enc = tf.transpose(spell_enc, (1, 0, 2))
    pron_enc = tf.transpose(pron_enc, (1, 0, 2))
    padding = tf.zeros((p.max_neighbors, batch_size))

    return [spell_enc, pron_enc], [padding, padding]

  def _ConcatAveNeigh(self, spellings, pronunciations, theta, batch_size):
    p = self.params
    spellings = tf.reshape(spellings, (batch_size * p.max_neighbors, -1))
    pronunciations = tf.reshape(pronunciations,
                                (batch_size * p.max_neighbors, -1))

    # ->(batch_size * max_neighbors, max_spelling_len + max_pronunciation_len)
    neigh_info = tf.concat([spellings, pronunciations], axis=1)

    # TODO(llion): Add task ids to concatenated info?
    if p.use_neigh_id_emb:
      raise NotImplementedError()
    neigh_inp = py_utils.NestedMap({
        "ids": neigh_info,
        "paddings": self._GetPaddings(neigh_info, dtype=tf.int32),
    })

    neigh_out = self.spell_encoder.FProp(theta.spell_encoder, neigh_inp)

    neigh_enc = tf.reshape(neigh_out["encoded"],
                           (p.max_spelling_len + p.max_pronunciation_len,
                            batch_size, p.max_neighbors, p.enc_units))
    neigh_enc = tf.reduce_mean(neigh_enc, axis=0)

    neigh_enc = tf.transpose(neigh_enc, (1, 0, 2))
    padding = tf.zeros((p.max_neighbors, batch_size))

    return [neigh_enc], [padding]

  def _MemoryNeigh(self, spellings, pronunciations, enc_out, theta, batch_size):
    p = self.params
    # Take the last embedding from the encoder output as the query to the
    # neighbour lookup.
    # [batch_size, emb_size]
    # TODO(llion): Add projection?
    query = enc_out.encoded[-1, :, :]

    # Process the neighbours to get the keys
    spellings = tf.reshape(spellings, (batch_size * p.max_neighbors, -1))
    pronunciations = tf.reshape(pronunciations,
                                (batch_size * p.max_neighbors, -1))

    spell_inp = py_utils.NestedMap({
        "ids": spellings,
        "paddings": self._GetPaddings(spellings, dtype=tf.int32),
    })

    pron_inp = py_utils.NestedMap({
        "ids": pronunciations,
        "paddings": self._GetPaddings(pronunciations, dtype=tf.int32),
    })

    spell_enc_out = self.spell_encoder.FProp(theta.spell_encoder, spell_inp)
    pron_enc_out = self.spell_encoder.FProp(theta.pron_encoder, pron_inp)

    spell_enc = tf.reshape(
        spell_enc_out["encoded"],
        (p.max_spelling_len, batch_size, p.max_neighbors, p.enc_units))
    # [batch_size, max_neighbors, enc_units]
    spell_keys = spell_enc[-1, :, :, :]

    # TODO(llion): Output the neighbour directly?
    pron_entries = tf.reshape(
        pron_enc_out["encoded"],
        (p.max_pronunciation_len, batch_size, p.max_neighbors, p.enc_units))

    # Compute attention
    # [batch_size, max_neighbors, emb_size] @ [batch_size, emb_size, 1] -->
    # [batch_size, max_neighbors, 1]
    key_logits = tf.matmul(spell_keys, tf.expand_dims(query, axis=-1))
    key_prob = tf.nn.softmax(key_logits)

    # [batch_size, max_neighbors, max_pronunciation_len, enc_units]
    pron_entries = tf.transpose(pron_entries, (1, 2, 0, 3))

    weighted_pron = tf.expand_dims(key_prob, axis=-1) * pron_entries
    # --> [max_pronunciation_len, batch_size, enc_units]
    weighted_pron = tf.transpose(
        tf.reduce_sum(weighted_pron, axis=1), (1, 0, 2))
    padding = tf.zeros((p.max_pronunciation_len, batch_size))

    return [weighted_pron], [padding]

  def _GetAxiliaryNeighInputs(self, spellings, pronunciations, enc_out, theta,
                              batch_size):
    p = self.params
    if p.use_neighbors:
      if p.neigh_att_type == "CONCATAVE":
        neigh_enc, padding = self._ConcatAveNeigh(spellings, pronunciations,
                                                  theta, batch_size)
      elif p.neigh_att_type == "AVERAGE":
        neigh_enc, padding = self._AveNeigh(spellings, pronunciations, theta,
                                            batch_size)
      elif p.neigh_att_type == "MEMORY":
        assert not p.use_neigh_id_emb
        neigh_enc, padding = self._MemoryNeigh(spellings, pronunciations,
                                               enc_out, theta, batch_size)

    return neigh_enc, padding

  def ComputePredictions(self, theta, input_batch):
    p = self.params
    batch_size = p.input.batch_size
    self._shape_batch(input_batch)

    # Prepend SOS token, this is not done by the Transformer layer for you
    # since this is usually done by the input pipeline in Babelfish.
    pronunciation = self._AddStartToken(input_batch.pronunciation)

    if p.use_neighbors:
      spellings = input_batch.neighbor_spellings
      pronunciations = input_batch.neighbor_pronunciations

    inp = {
        "ids": input_batch.spelling,
    }

    if (p.use_neighbors and p.also_shuffle_neighbors and
        (p.neigh_att_type == "CONCAT" or p.use_neigh_id_emb)):
      # If we use neighbor IDs, shuffle the neighbours to stop the model
      # overfitting to the ordering of the neighbours.
      # Concat then shuffle and split so that the spelling and pronunciation
      # are shuffled the same way and the IDs are aligned.
      neighbor_info = tf.concat([spellings, pronunciations], axis=-1)
      # Transpose the max_neighbors dimension to the front and shuffle.
      neighbor_info = tf.transpose(
          tf.random.shuffle(tf.transpose(neighbor_info, (1, 2, 0))), (2, 0, 1))
      spellings, pronunciations = (neighbor_info[:, :, :p.max_spelling_len],
                                   neighbor_info[:, :, p.max_spelling_len:])

    if p.use_neighbors and p.neigh_att_type == "CONCAT":
      # Interleave and flatten the neighbours info
      # ->(batch_size, max_neighbors, max_spelling_len + max_pronunciation_len)
      neigh_info = tf.concat([spellings, pronunciations], axis=2)
      # ->(batch_size, max_neighbors*(max_spelling_len + max_pronunciation_len))
      neigh_info = tf.reshape(neigh_info, (batch_size, -1))

      inp["ids"] = tf.concat([inp["ids"], neigh_info], axis=1)

      # If we are just concatenating everything then the main encoder needs
      # neighbors IDs.
      neigh_ids = tf.range(p.max_neighbors)[:, tf.newaxis]
      neigh_ids = tf.tile(
          neigh_ids, (batch_size, p.max_spelling_len + p.max_pronunciation_len))
      neigh_ids = tf.reshape(neigh_ids, (batch_size, -1))
      # Add the ids for the main input
      main_ids = tf.tile([[p.max_neighbors]], (batch_size, p.max_spelling_len))
      inp["task_ids"] = tf.concat([main_ids, neigh_ids], axis=1)

    inp["paddings"] = self._GetPaddings(inp["ids"], dtype=tf.int32)
    enc_out = self.encoder.FProp(theta.encoder, py_utils.NestedMap(inp))

    # Auxiliary inputs that the decoder can attend to, currently can be
    # neighbour summaries.
    aux_inputs = []
    aux_paddings = []

    if p.use_neighbors and p.neigh_att_type != "CONCAT":
      neigh_enc, padding = self._GetAxiliaryNeighInputs(spellings,
                                                        pronunciations, enc_out,
                                                        theta, batch_size)

      aux_inputs.extend(neigh_enc)
      aux_paddings.extend(padding)

    if aux_inputs:
      aux_inputs = tf.concat(aux_inputs, axis=0)
      aux_paddings = tf.concat(aux_paddings, axis=0)

      if p.aux_dropout_prob and not self.do_eval:
        aux_inputs = tf.nn.dropout(
            aux_inputs,
            p.aux_dropout_prob,
            noise_shape=(aux_inputs.get_shape().as_list()[0], batch_size, 1))

      enc_out.encoded = tf.concat([enc_out.encoded, aux_inputs], axis=0)
      enc_out.padding = tf.concat([enc_out.padding, aux_paddings], axis=0)

    enc_out.embedded_inputs = None  # to verify this is not used
    predictions = self.decoder.ComputePredictions(
        theta.decoder, enc_out,
        py_utils.NestedMap({
            "ids":
                pronunciation,
            "paddings":
                self._GetPaddings(pronunciation),
            "weights":
                tf.ones_like(input_batch.pronunciation, dtype=tf.float32),
        }))

    beam_out = self.decoder.BeamSearchDecode(enc_out, p.beam_size)
    top_ids = tf.reshape(beam_out.topk_ids,
                         [batch_size, -1, p.max_pronunciation_len])
    # Just take the top beam decodings
    top_ids = top_ids[:, 0, :]

    if p.is_inference:
      self.BuildInferenceInfo(top_ids, input_batch.pronunciation, enc_out)
      self.per_example_tensors["beam_scores"] = beam_out.topk_scores

    self.per_example_tensors["hyp"] = top_ids
    self.per_example_tensors["cognate_id"] = input_batch.cognate_id
    self.per_example_tensors["inp"] = input_batch.spelling
    self.per_example_tensors["ref"] = input_batch.pronunciation
    if p.use_neighbors:  # Note that cannot return None!
      self.per_example_tensors[
          "neighbor_spellings"] = input_batch.neighbor_spellings
      self.per_example_tensors[
          "neighbor_pronunciations"] = input_batch.neighbor_pronunciations
    self.prediction_values = predictions
    predictions.batch = input_batch

    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    loss, per_sequence_loss = self.GetDecoderLoss(theta, predictions,
                                                  input_batch.pronunciation)

    self.get_accuracy(loss, self.per_example_tensors["hyp"],
                      input_batch.pronunciation)

    return loss, per_sequence_loss

  def GetDecoderLoss(self, theta, predictions, pronunciation):
    return self.decoder.ComputeLoss(
        theta.decoder, predictions,
        py_utils.NestedMap({
            "labels": pronunciation,
            "paddings": self._GetPaddings(pronunciation),
            "weights": tf.ones_like(pronunciation, dtype=tf.float32),
        }))

  def GetSequenceInfo(self, ids, enc_out):
    inp_ids = self._AddStartToken(ids)
    dummy_pred = self.decoder.ComputePredictions(
        self.theta.decoder, enc_out,
        py_utils.NestedMap({
            "ids": inp_ids,
            "paddings": self._GetPaddings(inp_ids),
            "weights": tf.ones_like(inp_ids, dtype=tf.float32),
        }))
    # What's that? You thought 'softmax_input' in dummy_pred were the logits?
    # Don't be silly.
    # Let's pass what we have through the loss layer to really get the logits.
    # and don't forget this magic line!
    self.decoder.params.per_example_tensors = True
    _, per_example_tensors = self.GetDecoderLoss(self.theta, dummy_pred,
                                                 inp_ids)

    mask = tf.transpose(1 - self._GetPaddings(ids))
    logits = per_example_tensors["logits"]

    log_p = tf.nn.log_softmax(logits)
    prob = tf.exp(log_p)
    entropy = -tf.reduce_sum(log_p * prob, axis=-1) * mask
    ave_entropy = tf.reduce_sum(entropy, axis=0) / tf.reduce_sum(mask, axis=0)

    return logits, ave_entropy, dummy_pred.attention["probs"]

  def BuildInferenceInfo(self, top_ids, ref_ids, enc_out):
    """Build the inference visualization and analysis tensors.

    Args:
      top_ids: Top ids decoded from beam search.
      ref_ids: Target ids from the maps data.
      enc_out: Results from the encoder.FProp.
    """
    # The logits and attention tensors from the beam search are inaccessable
    # due to being stuck inside a tf.while_loop, so what we do is take
    # the beam search decoded ids and pass them though a training decoder to
    # fetch them.
    logits, ave_entropy, att_prob = self.GetSequenceInfo(top_ids, enc_out)
    _, ref_ave_entropy, _ = self.GetSequenceInfo(ref_ids, enc_out)
    self.per_example_tensors["logits"] = logits
    self.per_example_tensors["ave_entropy"] = ave_entropy
    self.per_example_tensors["ref_ave_entropy"] = ref_ave_entropy
    self.per_example_tensors["attention"] = att_prob
