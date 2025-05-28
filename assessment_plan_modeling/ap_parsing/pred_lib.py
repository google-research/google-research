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

"""Utilities for the prediction and evaluation of model outputs."""

import copy
from typing import Dict, Iterable, List, Optional

from absl import logging
import numpy as np
import seqeval.metrics
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import constants


def load_model(model_path):
  loaded_model = tf.saved_model.load(model_path)
  inference_model = loaded_model.signatures["serving_default"]
  logging.info("Model signature %s", inference_model.pretty_printed_signature())
  return inference_model


def offset_labeled_token_span(
    labeled_token_span,
    offset):
  new_labeled_token_span = copy.deepcopy(labeled_token_span)
  new_labeled_token_span.start_token += offset
  new_labeled_token_span.end_token += offset
  return new_labeled_token_span


def predict_single(
    model,
    features):
  """Predicts labeled token spans for the sample using the given model.

  Args:
    model: Keras model, assumed to take the features given and output logits.
    features: Dict of tensors with the features the model is expecting.

  Returns:
    The predicted labeled token spans.
  """
  model_input_keys = model.structured_input_signature[1]
  model_inputs = {k: features[k] for k in model_input_keys}
  model_outputs = model(**model_inputs)

  fragment_type_ids = tf.squeeze(
      model_outputs["fragment_type/predict_ids"]).numpy()
  action_item_logits = tf.squeeze(
      model_outputs["action_item_type/logits"]).numpy()

  labeled_token_spans = token_labels_to_labeled_token_spans(
      fragment_type_ids, action_item_logits)
  return labeled_token_spans


def token_labels_to_labeled_token_spans(
    fragment_type_ids, action_item_logits
):
  """Converts token level labels to labeled token spans.

  Fragments are assumed to be separated by tokens belonging to no fragment.
  Action item type is calculated as the most frequent label in the span (non-0).

  Args:
    fragment_type_ids: Fragment type as a per token int label.
    action_item_logits: Optional action item type as numpy array with per token
      logits.

  Returns:
    LabeledTokenSpans aggregated from the labels.
  """
  labeled_token_spans = []

  ### Get token_spans from fragment_type/predict_ids BIO labels ###
  fragment_type_pred_labels = [
      constants.CLASS_NAMES["fragment_type"][x] for x in fragment_type_ids
  ]

  fragment_seqeval_spans = seqeval.metrics.sequence_labeling.get_entities(
      fragment_type_pred_labels)

  for span_type, start_token, end_token in fragment_seqeval_spans:
    labeled_token_spans.append(
        ap_parsing_lib.LabeledTokenSpan(
            start_token=start_token,
            end_token=end_token + 1,  # seqeval is end-inclusive
            span_type=ap_parsing_lib.LabeledSpanType(
                constants.FRAGMENT_TYPE_TO_ENUM[span_type])))

  if isinstance(action_item_logits, np.ndarray):
    #### Add action items type as maximum likelihood ####
    # not-set and other artificially discarded by setting to low value.
    action_item_logits[:, [0, -1]] = -1e9
    log_probs = tf.nn.log_softmax(action_item_logits, -1).numpy()

    for lt in labeled_token_spans:
      if lt.span_type == ap_parsing_lib.LabeledSpanType.ACTION_ITEM:
        # sum probabilities along sequence for each type, argmax for type.
        # approx. equivalent to viterbi on a CRF with an identity transition
        # matrix.
        lt.action_item_type = ap_parsing_lib.ActionItemType(
            log_probs[lt.start_token:lt.end_token].sum(-2).argmax(-1).item())

  return labeled_token_spans
