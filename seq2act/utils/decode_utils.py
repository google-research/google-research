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

"""decode_utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
from seq2act.layers import area_utils


class ActionTypes(Enum):
  """The action types and ids of Android actions."""
  CLICK = 2
  INPUT = 3
  SWIPE = 4
  CHECK = 5
  UNCHECK = 6
  LONG_CLICK = 7
  OTHERS = 8


def verb_refs_to_lengths(task, verb_refs, include_eos=True):
  """Computes the length of a sequence."""
  eos_positions = tf.to_int32(tf.expand_dims(
      tf.where(tf.equal(task, 1))[:, 1], 1))
  seq_mask = tf.logical_not(tf.cast(tf.cumsum(tf.to_int32(
      tf.logical_and(
          tf.equal(verb_refs[:, :, 0], eos_positions),
          tf.equal(verb_refs[:, :, 1], eos_positions + 1))), axis=-1), tf.bool))
  lengths = tf.reduce_sum(tf.to_float(seq_mask), axis=-1)
  if include_eos:
    lengths = lengths + 1
  return lengths


def compute_seq_metrics(label_dict, feature_dict, debug=False, mask=None):
  """Compute the reference accuracy."""
  gt_lengths = verb_refs_to_lengths(label_dict["task"],
                                    label_dict["verb_refs"], include_eos=False)
  pred_lengths = verb_refs_to_lengths(feature_dict["task"],
                                      feature_dict["verb_refs"],
                                      include_eos=False)
  gt_actions = tf.concat([
      tf.expand_dims(label_dict["verbs"], 2),
      tf.expand_dims(label_dict["objects"], 2),
      label_dict["input_refs"]], axis=-1)
  pr_actions = tf.concat([
      tf.expand_dims(feature_dict["verbs"], 2),
      tf.expand_dims(feature_dict["objects"], 2),
      feature_dict["input_refs"]], axis=-1)
  complete_act_acc, partial_act_acc = sequence_accuracy(
      gt_actions, pr_actions, gt_lengths, pred_lengths,
      debug=debug, name="act")
  gt_refs = tf.concat([
      label_dict["verb_refs"],
      label_dict["obj_refs"],
      label_dict["input_refs"]], axis=-1)
  pr_refs = tf.concat([
      feature_dict["verb_refs"],
      feature_dict["obj_refs"],
      feature_dict["input_refs"]], axis=-1)
  if mask is not None:
    mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
    gt_refs = gt_refs * mask
    pr_refs = pr_refs * mask
    pred_lengths = gt_lengths
  with tf.control_dependencies([tf.print(
      "mask", gt_refs, pr_refs, summarize=100)]):
    complete_refs_acc, partial_refs_acc = sequence_accuracy(
        gt_refs, pr_refs, gt_lengths, pred_lengths,
        debug=debug, name="ref")
  refs_metrics = {}
  refs_metrics["complete_acts_acc"] = complete_act_acc
  refs_metrics["partial_acts_acc"] = partial_act_acc
  refs_metrics["complete_refs_acc"] = complete_refs_acc
  refs_metrics["partial_refs_acc"] = partial_refs_acc
  refs_metrics["gt_seq"] = gt_actions
  refs_metrics["pred_seq"] = pr_actions
  return refs_metrics


def unify_input_ref(pred_verbs, pred_input_ref):
  """Changes the input ref to zero according if pred_verbs are not input."""
  pred_verbs = tf.expand_dims(pred_verbs, axis=-1)
  same_dim_verbs = tf.concat([pred_verbs, pred_verbs], axis=-1)
  zero_refs = tf.zeros_like(pred_input_ref)
  return tf.where(
      tf.equal(same_dim_verbs, ActionTypes.INPUT.value), pred_input_ref,
      zero_refs)


def sequence_accuracy(gt_seqs, decode_seqs, gt_seq_lengths, pr_seq_lengths,
                      debug=False, name=""):
  """Computes the complete and the partial sequence accuracy."""
  gt_shape = common_layers.shape_list(gt_seqs)
  pr_shape = common_layers.shape_list(decode_seqs)
  batch_size = gt_shape[0]
  depth = gt_shape[-1]
  gt_len = gt_shape[1]
  pr_len = pr_shape[1]
  max_len = tf.maximum(gt_len, pr_len)
  gt_seqs = tf.pad(gt_seqs,
                   [[0, 0], [0, max_len - gt_len], [0, 0]])
  decode_seqs = tf.pad(decode_seqs,
                       [[0, 0], [0, max_len - pr_len], [0, 0]])
  gt_seqs = tf.where(
      tf.tile(
          tf.expand_dims(tf.sequence_mask(gt_seq_lengths, maxlen=max_len), 2),
          [1, 1, depth]),
      gt_seqs,
      tf.fill(tf.shape(gt_seqs), -1))
  decode_seqs = tf.where(
      tf.tile(
          tf.expand_dims(tf.sequence_mask(pr_seq_lengths, maxlen=max_len), 2),
          [1, 1, depth]),
      decode_seqs,
      tf.fill(tf.shape(decode_seqs), -1))
  # [batch_size, decode_length]
  corrects = tf.reduce_all(tf.equal(gt_seqs, decode_seqs), -1)
  correct_mask = tf.reduce_all(corrects, -1)
  # [batch_size]
  if debug:
    incorrect_mask = tf.logical_not(correct_mask)
    incorrect_gt = tf.boolean_mask(gt_seqs, incorrect_mask)
    incorrect_pr = tf.boolean_mask(decode_seqs, incorrect_mask)
    with tf.control_dependencies([tf.print(name + "_mismatch",
                                           incorrect_gt,
                                           incorrect_pr,
                                           summarize=1000)]):
      correct_mask = tf.identity(correct_mask)
  correct_seqs = tf.to_float(correct_mask)
  total_correct_seqs = tf.reduce_sum(correct_seqs)
  mean_complete_accuracy = total_correct_seqs / tf.to_float(batch_size)
  # Compute partial accuracy
  errors = tf.logical_not(corrects)
  errors = tf.cast(tf.cumsum(tf.to_float(errors), axis=-1), tf.bool)
  # [batch_size]
  correct_steps = tf.reduce_sum(tf.to_float(tf.logical_not(errors)), axis=-1)
  mean_partial_accuracy = tf.reduce_mean(
      tf.div(tf.minimum(correct_steps, gt_seq_lengths), gt_seq_lengths))
  return mean_complete_accuracy, mean_partial_accuracy


def _advance(step, beam_log_probs, previous_refs,
             area_logits, areas, batch_size, beam_size, append_refs=True,
             condition=None):
  """Advance one element in the tuple for a decoding step.

  Args:
    step: the current decoding step.
    beam_log_probs: [batch_size * beam_size]
    previous_refs: [batch_size * beam_size, input_length - 1, 2]
    area_logits: [batch_size * beam_size, num_areas]
    areas: the areas.
    batch_size: the batch size.
    beam_size: the beam_size.
    append_refs: returning references or ids.
    condition: conditional probability mask in shape [batch_size * beam_size].
  Returns:
    beam_log_probs: [batch_size * beam_size]
    references in shape of [batch_size * beam_size, input_length, 2] or
        ids in shape of [batch_size * beam_size]
  """
  with tf.control_dependencies([
      tf.equal(tf.shape(beam_log_probs), (batch_size * beam_size,))]):
    num_expansions = tf.minimum(beam_size, tf.shape(area_logits)[-1])
    # [batch_size * beam_size, num_expansions]
    area_log_probs = common_layers.log_prob_from_logits(area_logits)
    if condition is not None:
      area_log_probs = area_log_probs * tf.to_float(
          tf.expand_dims(condition, 1))
    top_area_log_probs, top_area_ids = tf.nn.top_k(
        area_log_probs, k=num_expansions)
  if append_refs:
    # [batch_size * beam_size, num_expansions, 2]
    refs = area_utils.area_to_refs(areas["starts"], areas["ends"],
                                   top_area_ids)
    if condition is not None:
      refs = refs * tf.expand_dims(tf.expand_dims(condition, 1), 2)
    refs = tf.reshape(refs, [batch_size, beam_size, num_expansions, 1, 2])
    if step > 0:
      previous_refs = tf.reshape(
          previous_refs, [batch_size, beam_size, 1, step, 2])
      previous_refs = tf.tile(previous_refs, [1, 1, num_expansions, 1, 1])
      new_refs = tf.concat([previous_refs, refs], axis=3)
    else:
      new_refs = refs
    new_refs = tf.reshape(
        new_refs, [batch_size * beam_size * num_expansions, step + 1, 2])
  # [batch_size, beam_size * num_expansions]
  log_probs = tf.reshape(tf.expand_dims(beam_log_probs, 1) + top_area_log_probs,
                         [batch_size, beam_size * num_expansions])
  # [batch_size, beam_size]
  beam_log_probs, beam_indices = tf.nn.top_k(log_probs, k=beam_size)
  beam_indices = tf.reshape(beam_indices, [-1])
  beam_log_probs = tf.reshape(beam_log_probs, [batch_size * beam_size])
  indices = tf.reshape(
      tf.tile(tf.expand_dims(tf.range(batch_size) * beam_size * num_expansions,
                             axis=1), [1, beam_size]), [-1]) + beam_indices
  if append_refs:
    new_refs = tf.gather(new_refs, indices=indices)
  else:
    new_refs = tf.gather(tf.reshape(top_area_ids, [-1]), indices=indices)
  return beam_log_probs, new_refs


def decode_one_step(step, live_beams, eos_positions,
                    compute_logits,
                    beam_log_probs, batch_size, beam_size,
                    features, areas, hparams,
                    use_cache, cache,
                    mode=tf.estimator.ModeKeys.EVAL,
                    always_consumed=True):
  """decode one step."""
  # features: [batch_size * beam_size, step + 1, ...]
  # [batch_size * beam_size, num_areas]
  action_logits, object_logits, consumed_logits, references = compute_logits(
      features, hparams, mode, use_cache, cache)
  input_logits = references["input_logits"]
  verb_area_logits = references["verb_area_logits"]
  obj_area_logits = references["obj_area_logits"]
  input_area_logits = references["input_area_logits"]
  cache = {}
  cache["input_logits"] = input_logits
  cache["verb_area_logits"] = verb_area_logits
  cache["obj_area_logits"] = obj_area_logits
  cache["input_area_logits"] = input_area_logits
  # step + 1
  input_length = tf.shape(features["verb_refs"])[1]
  output_length = tf.shape(verb_area_logits)[1]
  with tf.control_dependencies([
      tf.assert_equal(input_length, output_length)]):
    # Decode consumed
    beam_log_probs, is_ref_consumed = _advance(
        step,
        beam_log_probs,
        previous_refs=None,
        area_logits=consumed_logits[:, -1, :], areas=None,
        batch_size=batch_size, beam_size=beam_size, append_refs=False,
        condition=tf.to_int32(live_beams))
    if always_consumed:
      use_cache = tf.zeros_like(use_cache)
    else:
      use_cache = 1 - is_ref_consumed
    # Decode actions and objects greedy
    _, action = tf.nn.top_k(action_logits[:, -1, :])
    features["verbs"] = tf.concat([
        features["verbs"][:, :step],
        action, features["verbs"][:, step + 1:]], axis=1)
    features["verbs"] = tf.where(tf.equal(use_cache, 1),
                                 # Emit CLICK (2) if not consumed
                                 tf.fill(tf.shape(features["verbs"]), 2),
                                 features["verbs"])
    _, obj = tf.nn.top_k(object_logits[:, -1, :])
    features["objects"] = tf.concat([
        features["objects"][:, :step],
        obj, features["objects"][:, step + 1:]], axis=1)
    # Decode verb refs
    beam_log_probs, new_refs = _advance(
        step,
        beam_log_probs,
        previous_refs=features["verb_refs"][:, :-1, :],
        area_logits=verb_area_logits[:, -1, :], areas=areas,
        batch_size=batch_size, beam_size=beam_size,
        condition=tf.to_int32(live_beams))
    features["verb_refs"] = new_refs
    live_beams = tf.logical_and(
        live_beams,
        tf.not_equal(new_refs[:, -1, 0], eos_positions))
    # Decode object refs
    beam_log_probs, new_refs = _advance(
        step,
        beam_log_probs,
        previous_refs=features["obj_refs"][:, :-1, :],
        area_logits=obj_area_logits[:, -1, :], areas=areas,
        batch_size=batch_size, beam_size=beam_size,
        condition=tf.to_int32(live_beams))
    features["obj_refs"] = new_refs
    # Decode input refs
    beam_log_probs, need_inputs = _advance(
        step,
        beam_log_probs,
        previous_refs=None,
        area_logits=input_logits[:, -1, :], areas=None,
        batch_size=batch_size, beam_size=beam_size, append_refs=False,
        condition=tf.to_int32(live_beams))
    beam_log_probs, new_refs = _advance(
        step,
        beam_log_probs,
        previous_refs=features["input_refs"][:, :-1, :],
        area_logits=input_area_logits[:, -1, :], areas=areas,
        batch_size=batch_size, beam_size=beam_size,
        condition=tf.to_int32(live_beams) * need_inputs)
    features["input_refs"] = new_refs
    return beam_log_probs, live_beams, use_cache, cache


def _expand_to_beam(features, beam_size):
  shape_list = common_layers.shape_list(features)
  batch_size = shape_list[0]
  features = tf.expand_dims(features, axis=1)
  tile_dims = [1] * features.shape.ndims
  tile_dims[1] = beam_size
  shape_list[0] = batch_size * beam_size
  features = tf.reshape(tf.tile(features, tile_dims), shape_list)
  return features


def _recover_shape(features, beam_size):
  shape_list = common_layers.shape_list(features)
  batch_size = shape_list.pop(0) // beam_size
  shape_list = [batch_size, beam_size] + shape_list
  features = tf.reshape(features, shape_list)
  return features


def decode_n_step(compute_logits, features, areas,
                  hparams, n=20, beam_size=1, top_beam=True):
  """Decode for n steps.

  Args:
    compute_logits: the callback function for computing the logits.
    features: a dictionary of features.
    areas: the dict of area index mapping, with each tensor in the shape of
      [batch_size, num_areas].
    hparams: the hyperparameters.
    n: the number of steps to decode.
    beam_size: the beam size for beach search.
    top_beam: whether to return the results from the top beam only.
  """
  print(features)
  use_obj_dom_dist = ("obj_dom_dist" in features)
  batch_size = tf.shape(features["task"])[0]
  beam_log_probs = tf.fill([batch_size * beam_size], 0.)
  live_beams = tf.fill([batch_size * beam_size], True)
  use_cache = tf.fill([batch_size * beam_size], 0)
  cache = {}
  for step in range(n):
    if step == 0:
      features["verb_refs"] = tf.zeros([batch_size, 1, 2], tf.int32)
      features["obj_refs"] = tf.zeros([batch_size, 1, 2], tf.int32)
      features["input_refs"] = tf.zeros([batch_size, 1, 2], tf.int32)
      for key in features:
        features[key] = _expand_to_beam(features[key], beam_size)
      areas["starts"] = _expand_to_beam(areas["starts"], beam_size)
      areas["ends"] = _expand_to_beam(areas["ends"], beam_size)
      # Backup the screen features
      def pad_to_match(feature, target_length, rank, constant_values):
        """Pad the feature to the decode length."""
        padding_list = []
        target_length = tf.maximum(target_length, tf.shape(feature)[1])
        for r in range(rank):
          if r == 1:
            padding_list.append([0, target_length - tf.shape(feature)[1]])
          else:
            padding_list.append([0, 0])
        return tf.pad(feature, padding_list, constant_values=constant_values,
                      name="pad_to_match")
      features["backup_obj_text"] = pad_to_match(features["obj_text"], n, 4, 0)
      features["backup_obj_type"] = pad_to_match(features["obj_type"], n, 3, -1)
      features["backup_obj_clickable"] = pad_to_match(
          features["obj_clickable"], n, 3, 0)
      features["backup_obj_screen_pos"] = pad_to_match(
          features["obj_screen_pos"], n, 4, 0)
      features["backup_obj_dom_pos"] = pad_to_match(features["obj_dom_pos"],
                                                    n, 4, 0)
      if use_obj_dom_dist:
        features["backup_obj_dom_dist"] = pad_to_match(features["obj_dom_dist"],
                                                       n, 4, 0)
      # Set the screen features
      features["obj_text"] = features["obj_text"][:, :1]
      features["obj_type"] = features["obj_type"][:, :1]
      features["obj_clickable"] = features["obj_clickable"][:, :1]
      features["obj_screen_pos"] = features["obj_screen_pos"][:, :1]
      features["obj_dom_pos"] = features["obj_dom_pos"][:, :1]
      if use_obj_dom_dist:
        features["obj_dom_dist"] = features["obj_dom_dist"][:, :1]
    else:
      features["verb_refs"] = tf.pad(features["verb_refs"],
                                     [[0, 0], [0, 1], [0, 0]],
                                     name="pad_verb_refs")
      features["obj_refs"] = tf.pad(features["obj_refs"],
                                    [[0, 0], [0, 1], [0, 0]],
                                    name="pad_obj_refs")
      features["input_refs"] = tf.pad(features["input_refs"],
                                      [[0, 0], [0, 1], [0, 0]],
                                      name="pad_input_refs")
      # Fill in the screen information
      features["obj_text"] = features["backup_obj_text"][:, :step + 1]
      features["obj_type"] = features["backup_obj_type"][:, :step + 1]
      features["obj_clickable"] = features["backup_obj_clickable"][:, :step + 1]
      features["obj_screen_pos"] = (
          features["backup_obj_screen_pos"][:, :step + 1])
      features["obj_dom_pos"] = (
          features["backup_obj_dom_pos"][:, :step + 1])
      if use_obj_dom_dist:
        features["obj_dom_dist"] = (
            features["backup_obj_dom_dist"][:, :step + 1])
    eos_positions = tf.to_int32(tf.where(tf.equal(features["task"], 1))[:, 1])
    beam_log_probs, live_beams, use_cache, cache = decode_one_step(
        step, live_beams, eos_positions,
        compute_logits,
        beam_log_probs,
        batch_size, beam_size, features,
        areas, hparams,
        use_cache=use_cache, cache=cache,
        always_consumed=True)
  for key in features:
    features[key] = _recover_shape(features[key], beam_size)
    if top_beam:
      features[key] = features[key][:, 0]
      if key in ["obj_type", "obj_clickable"]:
        features[key] = tf.pad(
            features[key], [[0, 0],
                            [0, n - tf.shape(features[key])[1]], [0, 0]],
            constant_values=-1 if key.endswith("type") else 0,
            name="pad_type_clickable")
      elif key in ["obj_text", "obj_screen_pos", "obj_dom_pos", "obj_dom_dist"]:
        features[key] = tf.pad(features[key],
                               [[0, 0], [0, n - tf.shape(features[key])[1]],
                                [0, 0], [0, 0]],
                               name="pad_rest_screen_features")
