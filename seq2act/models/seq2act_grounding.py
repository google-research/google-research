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

"""The grounding models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
from seq2act.layers import area_utils
from seq2act.layers import common_embed
from seq2act.layers import encode_screen
from seq2act.models import seq2act_reference


def encode_screen_ffn(features, hparams, embed_scope):
  """Encodes a screen with feed forward neural network.

  Args:
    features: the feature dict.
    hparams: the hyperparameter.
    embed_scope: the name scope.
  Returns:
    encoder_outputs: a Tensor of shape
        [batch_size, num_steps, max_object_count, hidden_size]
    obj_mask: A tensor of shape
        [batch_size, num_steps, max_object_count]
  """
  object_embed, obj_mask, obj_bias = encode_screen.prepare_encoder_input(
      features=features, hparams=hparams,
      embed_scope=embed_scope)
  for layer in range(hparams.num_hidden_layers):
    with tf.variable_scope(
        "encode_screen_ff_layer_%d" % layer, reuse=tf.AUTO_REUSE):
      object_embed = tf.layers.dense(object_embed, units=hparams.hidden_size)
      object_embed = common_layers.apply_norm(
          object_embed, hparams.norm_type, hparams.hidden_size,
          epsilon=hparams.norm_epsilon)
      object_embed = tf.nn.relu(object_embed)
      object_embed = tf.nn.dropout(
          object_embed,
          keep_prob=1.0 - hparams.layer_prepostprocess_dropout)
      object_embed = object_embed * tf.expand_dims(obj_mask, 3)
  return object_embed, obj_bias


def compute_logits(features, references, hparams):
  """Grounds using the predicted references.

  Args:
    features: the feature dict.
    references: the dict that keeps the reference results.
    hparams: the hyper-parameters.
  Returns:
    action_logits: [batch_size, num_steps, num_actions]
    object_logits: [batch_size, num_steps, max_num_objects]
  """
  lang_hidden_layers = hparams.num_hidden_layers
  pos_embed = hparams.pos
  hparams.set_hparam("num_hidden_layers", hparams.screen_encoder_layers)
  hparams.set_hparam("pos", "none")
  with tf.variable_scope("compute_grounding_logits", reuse=tf.AUTO_REUSE):
    # Encode objects
    if hparams.screen_encoder == "gcn":
      screen_encoding, _, screen_encoding_bias = (
          encode_screen.gcn_encoder(
              features, hparams, references["embed_scope"],
              discretize=False))
    elif hparams.screen_encoder == "transformer":
      screen_encoding, _, screen_encoding_bias = (
          encode_screen.transformer_encoder(
              features, hparams, references["embed_scope"]))
    elif hparams.screen_encoder == "mlp":
      screen_encoding, screen_encoding_bias = encode_screen_ffn(
          features, hparams, references["embed_scope"])
    else:
      raise ValueError(
          "Unsupported encoder: %s" % hparams.screen_encoder)
    # Compute query
    if hparams.compute_verb_obj_separately:
      verb_hidden, object_hidden = _compute_query_embedding(
          features, references, hparams, references["embed_scope"])
    else:
      verb_hidden = references["verb_hidden"]
      object_hidden = references["object_hidden"]
    # Predict actions
    with tf.variable_scope("compute_action_logits", reuse=tf.AUTO_REUSE):
      action_logits = tf.layers.dense(
          verb_hidden, units=hparams.action_vocab_size)
    # Predict objects
    obj_logits, consumed_logits = _compute_object_logits(
        hparams,
        object_hidden,
        screen_encoding,
        screen_encoding_bias)
    hparams.set_hparam("num_hidden_layers", lang_hidden_layers)
    hparams.set_hparam("pos", pos_embed)
    return action_logits, obj_logits, consumed_logits


def _compute_object_logits(hparams, object_hidden,
                           screen_encoding, screen_encoding_bias):
  """The output layer for a specific domain."""
  with tf.variable_scope("compute_object_logits", reuse=tf.AUTO_REUSE):
    if hparams.alignment == "cosine_similarity":
      object_hidden = tf.layers.dense(
          object_hidden, units=hparams.hidden_size)
      screen_encoding = tf.layers.dense(
          screen_encoding, units=hparams.hidden_size)
      norm_screen_encoding = tf.math.l2_normalize(screen_encoding, axis=-1)
      norm_obj_hidden = tf.math.l2_normalize(object_hidden, axis=-1)
      align_logits = tf.matmul(norm_screen_encoding,
                               tf.expand_dims(norm_obj_hidden, 3))
    elif hparams.alignment == "scaled_cosine_similarity":
      object_hidden = tf.layers.dense(
          object_hidden, units=hparams.hidden_size)
      screen_encoding = tf.reshape(
          screen_encoding,
          common_layers.shape_list(
              screen_encoding)[:-1] + [hparams.hidden_size])
      screen_encoding = tf.layers.dense(
          screen_encoding, units=hparams.hidden_size)
      norm_screen_encoding = tf.math.l2_normalize(screen_encoding, axis=-1)
      norm_obj_hidden = tf.math.l2_normalize(object_hidden, axis=-1)
      dot_products = tf.matmul(norm_screen_encoding,
                               tf.expand_dims(norm_obj_hidden, 3))
      align_logits = tf.layers.dense(dot_products, units=1)
    elif hparams.alignment == "dot_product_attention":
      object_hidden = tf.layers.dense(
          object_hidden, units=hparams.hidden_size)
      align_logits = tf.matmul(screen_encoding,
                               tf.expand_dims(object_hidden, 3))
    elif hparams.alignment == "mlp_attention":
      batch_size = tf.shape(screen_encoding)[0]
      num_steps = tf.shape(screen_encoding)[1]
      num_objects = tf.shape(screen_encoding)[2]
      tiled_object_hidden = tf.tile(tf.expand_dims(object_hidden, 2),
                                    [1, 1, num_objects, 1])
      align_feature = tf.concat([tiled_object_hidden, screen_encoding], axis=-1)
      align_feature = tf.reshape(
          align_feature,
          [batch_size, num_steps, num_objects, hparams.hidden_size * 2])
      with tf.variable_scope("align", reuse=tf.AUTO_REUSE):
        align_hidden = tf.layers.dense(align_feature, units=hparams.hidden_size)
        align_hidden = common_layers.apply_norm(
            align_hidden, hparams.norm_type, hparams.hidden_size,
            epsilon=hparams.norm_epsilon)
        align_hidden = tf.nn.tanh(align_hidden)
        align_logits = tf.layers.dense(align_hidden, units=1)
    else:
      raise ValueError("Unsupported alignment: %s" % hparams.alignment)

    obj_logits = tf.squeeze(align_logits, [3]) + screen_encoding_bias
    # [batch_size, num_steps]
    batch_size = common_layers.shape_list(obj_logits)[0]
    num_steps = common_layers.shape_list(obj_logits)[1]
    # [batch_size * num_steps, 1]
    batch_indices = tf.to_int64(tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_steps]),
        [-1, 1]))
    step_indices = tf.to_int64(tf.reshape(
        tf.tile(tf.expand_dims(tf.range(num_steps), 0), [batch_size, 1]),
        [-1, 1]))
    object_indices = tf.reshape(tf.argmax(obj_logits, -1), [-1, 1])
    indices = tf.concat([batch_indices, step_indices, object_indices], -1)
    # [batch_size, num_steps, depth]
    depth = tf.shape(screen_encoding)[-1]
    best_logits = tf.reshape(
        tf.gather_nd(screen_encoding, indices=indices),
        [batch_size, num_steps, depth])
    consumed_logits = tf.layers.dense(
        tf.reshape(tf.concat([object_hidden, best_logits], -1),
                   [batch_size, num_steps, hparams.hidden_size * 2]),
        2)
    with tf.control_dependencies([tf.assert_equal(
        tf.reduce_all(tf.math.is_nan(consumed_logits)), False,
        data=[tf.shape(best_logits), best_logits,
              tf.constant("screen_encoding"), screen_encoding,
              tf.constant("indices"), indices],
        summarize=10000, message="consumed_logits_nan")]):
      consumed_logits = tf.identity(consumed_logits)
    return obj_logits, consumed_logits


def _compute_query_embedding(features, references, hparams, embed_scope=None):
  """Computes lang embeds for verb and object from predictions.

  Args:
    features: a dictionary contains "inputs" that is a tensor in shape of
        [batch_size, num_tokens], "verb_id_seq" that is in shape of
        [batch_size, num_actions], "object_spans" and "param_span" tensor
        in shape of [batch_size, num_actions, 2]. 0 is used as padding or
        non-existent values.
    references: the dict that keeps the reference results.
    hparams: the general hyperparameters for the model.
    embed_scope: the embedding variable scope.
  Returns:
    verb_embeds: a Tensor of shape
        [batch_size, num_steps, depth]
    object_embeds:
        [batch_size, num_steps, depth]
  """
  pred_verb_refs = seq2act_reference.predict_refs(
      references["verb_area_logits"],
      references["areas"]["starts"], references["areas"]["ends"])
  pred_obj_refs = seq2act_reference.predict_refs(
      references["obj_area_logits"],
      references["areas"]["starts"], references["areas"]["ends"])
  input_embeddings, _ = common_embed.embed_tokens(
      features["task"], hparams.task_vocab_size, hparams.hidden_size, hparams,
      embed_scope=references["embed_scope"])
  if hparams.obj_text_aggregation == "sum":
    area_encodings, _, _ = area_utils.compute_sum_image(
        input_embeddings, max_area_width=hparams.max_span)
    shape = common_layers.shape_list(features["task"])
    encoder_input_length = shape[1]
    verb_embeds = seq2act_reference.span_embedding(
        encoder_input_length, area_encodings, pred_verb_refs, hparams)
    object_embeds = seq2act_reference.span_embedding(
        encoder_input_length, area_encodings, pred_obj_refs, hparams)
  elif hparams.obj_text_aggregation == "mean":
    verb_embeds = seq2act_reference.span_average_embed(
        input_embeddings, pred_verb_refs, embed_scope, hparams)
    object_embeds = seq2act_reference.span_average_embed(
        input_embeddings, pred_obj_refs, embed_scope, hparams)
  else:
    raise ValueError("Unrecognized query aggreggation %s" % (
        hparams.span_aggregation))
  return verb_embeds, object_embeds


def compute_losses(loss_dict, features, action_logits, obj_logits,
                   consumed_logits):
  """Compute the loss based on the logits and labels."""
  valid_obj_mask = tf.to_float(tf.greater(features["verbs"], 1))
  action_losses = tf.losses.sparse_softmax_cross_entropy(
      labels=features["verbs"],
      logits=action_logits,
      reduction=tf.losses.Reduction.NONE) * valid_obj_mask
  action_loss = tf.reduce_mean(action_losses)
  object_losses = tf.losses.sparse_softmax_cross_entropy(
      labels=features["objects"],
      logits=obj_logits,
      reduction=tf.losses.Reduction.NONE) * valid_obj_mask
  object_loss = tf.reduce_mean(object_losses)
  if "consumed" in features:
    consumed_loss = tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(
            labels=features["consumed"],
            logits=consumed_logits,
            reduction=tf.losses.Reduction.NONE) * valid_obj_mask)
  else:
    consumed_loss = 0.0
  loss_dict["grounding_loss"] = action_loss + object_loss + consumed_loss
  loss_dict["verbs_loss"] = action_loss
  loss_dict["objects_loss"] = object_loss
  loss_dict["verbs_losses"] = action_losses
  loss_dict["object_losses"] = object_losses
  loss_dict["consumed_loss"] = consumed_loss
  return loss_dict["grounding_loss"]


def compute_predictions(prediction_dict, action_logits, obj_logits,
                        consumed_logits):
  """Predict the action tuple based on the logits."""
  prediction_dict["verbs"] = tf.argmax(action_logits, -1)
  prediction_dict["objects"] = tf.argmax(obj_logits, -1)
  prediction_dict["consumed"] = tf.argmax(consumed_logits, -1)
