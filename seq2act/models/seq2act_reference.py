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

"""The reference models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
import tensorflow.compat.v1 as tf
from seq2act.layers import area_utils
from seq2act.layers import common_embed


def span_embedding(encoder_input_length, area_encodings, spans, hparams):
  """Computes the embedding for each span. (TODO: liyang): comment shapes."""
  with tf.control_dependencies([tf.assert_equal(tf.rank(area_encodings), 3)]):
    area_indices = area_utils.area_range_to_index(
        area_range=tf.reshape(spans, [-1, 2]), length=encoder_input_length,
        max_area_width=hparams.max_span)
  return area_utils.batch_gather(
      area_encodings, tf.reshape(area_indices,
                                 [tf.shape(spans)[0], tf.shape(spans)[1]]))


def span_average_embed(area_encodings, spans, embed_scope):
  """Embeds a span of tokens using averaging.

  Args:
    area_encodings: [batch_size, length, depth].
    spans: [batch_size, ref_len, 2].
    embed_scope: the variable scope for embedding.
  Returns:
    the average embeddings in the shape of [batch_size, ref_lengths, depth].
  """
  ref_len = common_layers.shape_list(spans)[1]
  starts_ends = tf.reshape(spans, [-1, 2])
  depth = common_layers.shape_list(area_encodings)[-1]
  length = common_layers.shape_list(area_encodings)[1]
  area_encodings = tf.reshape(
      tf.tile(tf.expand_dims(area_encodings, 1), [1, ref_len, 1, 1]),
      [-1, length, depth])
  area_ranges = starts_ends[:, 1] - starts_ends[:, 0]
  max_num_tokens = tf.reduce_max(area_ranges)
  def _fetch_embeddings(area_encoding_and_range):
    """Fetches embeddings for the range."""
    area_encoding = area_encoding_and_range[0]
    area_range = area_encoding_and_range[1]
    start = area_range[0]
    end = area_range[1]
    embeddings = area_encoding[start:end, :]
    em_len = area_range[1] - area_range[0]
    embeddings = tf.pad(embeddings, [[0, max_num_tokens - em_len], [0, 0]],
                        constant_values=0.0)
    return embeddings
  # [batch_size * ref_len, max_num_tokens, depth]
  area_embeddings = tf.map_fn(_fetch_embeddings,
                              [area_encodings, starts_ends],
                              dtype=tf.float32, infer_shape=False)
  # To give a fixed dimension
  area_embeddings = tf.reshape(area_embeddings,
                               [-1, max_num_tokens, depth])
  emb_sum = tf.reduce_sum(tf.abs(area_embeddings), axis=-1)
  non_paddings = tf.not_equal(emb_sum, 0.0)
  # [batch_size * ref_len, depth]
  area_embeddings = common_embed.average_bag_of_embeds(
      area_embeddings, non_paddings, use_bigrams=True,
      bigram_embed_scope=embed_scope, append_start_end=True)
  area_embeddings = tf.reshape(area_embeddings, [-1, ref_len, depth])
  return area_embeddings


def _prepare_decoder_input(area_encoding, decoder_nonpadding,
                           features, hparams,
                           embed_scope=None):
  """Prepare the input for the action decoding.

  Args:
    area_encoding: the encoder output in shape of [batch_size, area_len, depth].
    decoder_nonpadding: the nonpadding mask for the decoding seq.
    features: a dictionary of tensors in the shape of [batch_size, seq_length].
    hparams: the hyperparameters.
    embed_scope: the embedding scope.
  Returns:
    decoder_input: decoder input in shape of
        [batch_size, num_steps, latent_depth]
    decoder_self_attention_bias: decoder attention bias.
  """
  with tf.variable_scope("prepare_decoder_input", reuse=tf.AUTO_REUSE):
    shape = common_layers.shape_list(features["task"])
    batch_size = shape[0]
    encoder_input_length = shape[1]
    depth = common_layers.shape_list(area_encoding)[-1]
    if hparams.span_aggregation == "sum":
      verb_embeds = span_embedding(encoder_input_length,
                                   area_encoding, features["verb_refs"],
                                   hparams)
      object_embeds = span_embedding(encoder_input_length,
                                     area_encoding, features["obj_refs"],
                                     hparams)
      input_embeds = span_embedding(encoder_input_length,
                                    area_encoding, features["input_refs"],
                                    hparams)
      non_input_embeds = tf.tile(tf.expand_dims(tf.expand_dims(
          tf.get_variable(name="non_input_embeds",
                          shape=[depth]),
          0), 0), [batch_size, tf.shape(features["input_refs"])[1], 1])
      input_embeds = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(features["input_refs"][:, :, 1],
                                      features["input_refs"][:, :, 0]), 2),
              [1, 1, tf.shape(input_embeds)[-1]]),
          non_input_embeds,
          input_embeds)
    elif hparams.span_aggregation == "mean":
      area_encoding = area_encoding[:, :encoder_input_length, :]
      verb_embeds = span_average_embed(area_encoding, features["verb_refs"],
                                       embed_scope)
      object_embeds = span_average_embed(area_encoding, features["obj_refs"],
                                         embed_scope)
      input_embeds = span_average_embed(area_encoding, features["input_refs"],
                                        embed_scope)
    else:
      raise ValueError("Unrecognized span aggregation method %s" % (
          hparams.span_aggregation))
    embeds = verb_embeds + object_embeds + input_embeds
    embeds = tf.multiply(
        tf.expand_dims(decoder_nonpadding, 2), embeds)
    start_embed = tf.tile(tf.expand_dims(tf.expand_dims(
        tf.get_variable(name="start_step_embed",
                        shape=[depth]), 0), 0),
                          [batch_size, 1, 1])
    embeds = tf.concat([start_embed, embeds], axis=1)
    embeds = embeds[:, :-1, :]
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(
            common_layers.shape_list(features["verb_refs"])[1]))
    if hparams.pos == "timing":
      decoder_input = common_attention.add_timing_signal_1d(embeds)
    elif hparams.pos == "emb":
      decoder_input = common_attention.add_positional_embedding(
          embeds, hparams.max_length, "targets_positional_embedding",
          None)
    else:
      decoder_input = embeds
    return decoder_input, decoder_self_attention_bias


def encode_decode_task(features, hparams, train, attention_weights=None):
  """Model core graph for the one-shot action.

  Args:
    features: a dictionary contains "inputs" that is a tensor in shape of
        [batch_size, num_tokens], "verb_id_seq" that is in shape of
        [batch_size, num_actions], "object_spans" and "param_span" tensor
        in shape of [batch_size, num_actions, 2]. 0 is used as padding or
        non-existent values.
    hparams: the general hyperparameters for the model.
    train: the train mode.
    attention_weights: the dict to keep attention weights for analysis.
  Returns:
    loss_dict: the losses for training.
    prediction_dict: the predictions for action tuples.
    areas: the area encodings of the task.
    scope: the embedding scope.
  """
  del train
  input_embeddings, scope = common_embed.embed_tokens(
      features["task"],
      hparams.task_vocab_size,
      hparams.hidden_size, hparams)
  with tf.variable_scope("encode_decode", reuse=tf.AUTO_REUSE):
    encoder_nonpadding = tf.minimum(tf.to_float(features["task"]), 1.0)
    input_embeddings = tf.multiply(
        tf.expand_dims(encoder_nonpadding, 2),
        input_embeddings)
    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            input_embeddings, None, hparams, features=None))
    encoder_input = tf.nn.dropout(
        encoder_input,
        keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
    if hparams.instruction_encoder == "transformer":
      encoder_output = transformer.transformer_encoder(
          encoder_input,
          self_attention_bias,
          hparams,
          save_weights_to=attention_weights,
          make_image_summary=not common_layers.is_xla_compiled())
    else:
      raise ValueError("Unsupported instruction encoder %s" % (
          hparams.instruction_encoder))
    span_rep = hparams.get("span_rep", "area")
    area_encodings, area_starts, area_ends = area_utils.compute_sum_image(
        encoder_output, max_area_width=hparams.max_span)
    current_shape = tf.shape(area_encodings)
    if span_rep == "area":
      area_encodings, _, _ = area_utils.compute_sum_image(
          encoder_output, max_area_width=hparams.max_span)
    elif span_rep == "basic":
      area_encodings = area_utils.compute_alternative_span_rep(
          encoder_output, input_embeddings, max_area_width=hparams.max_span,
          hidden_size=hparams.hidden_size, advanced=False)
    elif span_rep == "coref":
      area_encodings = area_utils.compute_alternative_span_rep(
          encoder_output, input_embeddings, max_area_width=hparams.max_span,
          hidden_size=hparams.hidden_size, advanced=True)
    else:
      raise ValueError("xyz")
    areas = {}
    areas["encodings"] = area_encodings
    areas["starts"] = area_starts
    areas["ends"] = area_ends
    with tf.control_dependencies([tf.print("encoder_output",
                                           tf.shape(encoder_output)),
                                  tf.assert_equal(current_shape,
                                                  tf.shape(area_encodings),
                                                  summarize=100)]):
      paddings = tf.cast(tf.less(self_attention_bias, -1), tf.int32)
    padding_sum, _, _ = area_utils.compute_sum_image(
        tf.expand_dims(tf.squeeze(paddings, [1, 2]), 2),
        max_area_width=hparams.max_span)
    num_areas = common_layers.shape_list(area_encodings)[1]
    area_paddings = tf.reshape(tf.minimum(tf.to_float(padding_sum), 1.0),
                               [-1, num_areas])
    areas["bias"] = area_paddings
    decoder_nonpadding = tf.to_float(
        tf.greater(features["verb_refs"][:, :, 1],
                   features["verb_refs"][:, :, 0]))
    if hparams.instruction_encoder == "lstm":
      hparams_decoder = copy.copy(hparams)
      hparams_decoder.set_hparam("pos", "none")
    else:
      hparams_decoder = hparams
    decoder_input, decoder_self_attention_bias = _prepare_decoder_input(
        area_encodings, decoder_nonpadding, features, hparams_decoder,
        embed_scope=scope)
    decoder_input = tf.nn.dropout(
        decoder_input, keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
    if hparams.instruction_decoder == "transformer":
      decoder_output = transformer.transformer_decoder(
          decoder_input=decoder_input,
          encoder_output=encoder_output,
          decoder_self_attention_bias=decoder_self_attention_bias,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          hparams=hparams_decoder)
    else:
      raise ValueError("Unsupported instruction encoder %s" % (
          hparams.instruction_encoder))
    return decoder_output, decoder_nonpadding, areas, scope


def predict_refs(logits, starts, ends):
  """Outputs the refs based on area predictions."""
  with tf.control_dependencies([
      tf.assert_equal(tf.rank(logits), 3),
      tf.assert_equal(tf.rank(starts), 2),
      tf.assert_equal(tf.rank(ends), 2)]):
    predicted_areas = tf.argmax(logits, -1)
  return area_utils.area_to_refs(starts, ends, predicted_areas)


def compute_logits(features, hparams, train):
  """Computes reference logits and auxiliary information.

  Args:
    features: the feature dict.
    hparams: the hyper-parameters.
    train: whether it is in the train mode.
  Returns:
    a dict that contains:
    input_logits: [batch_size, num_steps, 2]
    verb_area_logits: [batch_size, num_steps, num_areas]
    obj_area_logits: [batch_size, num_steps, num_areas]
    input_area_logits: [batch_size, num_steps, num_areas]
    verb_hidden: [batch_size, num_steps, hidden_size]
    obj_hidden: [batch_size, num_steps, hidden_size]
    areas: a dict that contains area representation of the source sentence.
  """
  latent_state, _, areas, embed_scope = encode_decode_task(
      features, hparams, train)
  task_encoding = areas["encodings"]
  task_encoding_bias = areas["bias"]
  def _output(latent_state, hparams, name):
    """Output layer."""
    with tf.variable_scope("latent_to_" + name, reuse=tf.AUTO_REUSE):
      hidden = tf.layers.dense(latent_state, units=hparams.hidden_size)
      hidden = common_layers.apply_norm(
          hidden, hparams.norm_type, hparams.hidden_size,
          epsilon=hparams.norm_epsilon)
      return tf.nn.relu(hidden)
  with tf.variable_scope("output_layer", values=[latent_state, task_encoding,
                                                 task_encoding_bias],
                         reuse=tf.AUTO_REUSE):
    with tf.control_dependencies([tf.assert_equal(tf.rank(latent_state), 3)]):
      verb_hidden = _output(latent_state, hparams, "verb")
      object_hidden = _output(latent_state, hparams, "object")
      verb_hidden = tf.nn.dropout(
          verb_hidden,
          keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
      object_hidden = tf.nn.dropout(
          object_hidden,
          keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
    with tf.variable_scope("verb_refs", reuse=tf.AUTO_REUSE):
      verb_area_logits = area_utils.query_area(
          tf.layers.dense(verb_hidden, units=hparams.hidden_size,
                          name="verb_query"),
          task_encoding, task_encoding_bias)
    with tf.variable_scope("object_refs", reuse=tf.AUTO_REUSE):
      obj_area_logits = area_utils.query_area(
          tf.layers.dense(object_hidden, units=hparams.hidden_size,
                          name="obj_query"),
          task_encoding, task_encoding_bias)
    with tf.variable_scope("input_refs", reuse=tf.AUTO_REUSE):
      input_logits = tf.layers.dense(
          _output(latent_state, hparams, "input"), units=2)
      input_area_logits = area_utils.query_area(
          _output(latent_state, hparams, "input_refs"),
          task_encoding, task_encoding_bias)
    references = {}
    references["input_logits"] = input_logits
    references["verb_area_logits"] = verb_area_logits
    references["obj_area_logits"] = obj_area_logits
    references["input_area_logits"] = input_area_logits
    references["verb_hidden"] = verb_hidden
    references["object_hidden"] = object_hidden
    references["areas"] = areas
    references["decoder_output"] = latent_state
    references["embed_scope"] = embed_scope
    if hparams.freeze_reference_model:
      for key in ["input_logits", "verb_area_logits", "obj_area_logits",
                  "input_area_logits", "verb_hidden", "object_hidden",
                  "decoder_output"]:
        references[key] = tf.stop_gradient(references[key])
      for key in references["areas"]:
        references["areas"][key] = tf.stop_gradient(references["areas"][key])
    return references


def compute_losses(loss_dict, features, references, hparams):
  """Compute the loss based on the logits and labels."""
  # Commented code can be useful for examining seq lengths distribution
  # srcs, _, counts = tf.unique_with_counts(features["data_source"])
  # lengths = tf.reduce_sum(tf.to_int32(
  #    tf.greater(features["verb_refs"][:, :, 1],
  #               features["verb_refs"][:, :, 0])), -1) - 1
  # lengths, _, len_counts = tf.unique_with_counts(lengths)
  # with tf.control_dependencies([
  #    tf.print("sources", srcs, counts, lengths, len_counts, summarize=1000)]):
  input_mask = tf.to_float(
      tf.greater(features["verb_refs"][:, :, 1],
                 features["verb_refs"][:, :, 0]))
  input_loss = tf.reduce_mean(
      tf.losses.sparse_softmax_cross_entropy(
          labels=tf.to_int32(
              tf.greater(features["input_refs"][:, :, 1],
                         features["input_refs"][:, :, 0])),
          logits=references["input_logits"],
          reduction=tf.losses.Reduction.NONE) * input_mask)
  encoder_input_length = common_layers.shape_list(features["task"])[1]
  verb_area_loss = area_utils.area_loss(
      logits=references["verb_area_logits"], ranges=features["verb_refs"],
      length=encoder_input_length,
      max_area_width=hparams.max_span)
  object_area_loss = area_utils.area_loss(
      logits=references["obj_area_logits"],
      ranges=features["obj_refs"],
      length=encoder_input_length,
      max_area_width=hparams.max_span)
  input_area_loss = area_utils.area_loss(
      logits=references["input_area_logits"], ranges=features["input_refs"],
      length=encoder_input_length,
      max_area_width=hparams.max_span)
  loss_dict["reference_loss"] = (
      input_loss + verb_area_loss + object_area_loss + input_area_loss)
  loss_dict["input_loss"] = input_loss
  loss_dict["verb_refs_loss"] = verb_area_loss
  loss_dict["obj_refs_loss"] = object_area_loss
  loss_dict["input_refs_loss"] = input_area_loss
  return loss_dict["reference_loss"]


def compute_predictions(prediction_dict, references):
  """Predict the action tuple based on the logits."""
  prediction_dict["input"] = tf.argmax(references["input_logits"], -1)
  prediction_dict["verb_refs"] = predict_refs(references["verb_area_logits"],
                                              references["areas"]["starts"],
                                              references["areas"]["ends"])
  prediction_dict["obj_refs"] = predict_refs(references["obj_area_logits"],
                                             references["areas"]["starts"],
                                             references["areas"]["ends"])
  prediction_dict["input_refs"] = predict_refs(
      references["input_area_logits"],
      references["areas"]["starts"],
      references["areas"]["ends"]) * tf.to_int32(tf.expand_dims(
          prediction_dict["input"], 2))
