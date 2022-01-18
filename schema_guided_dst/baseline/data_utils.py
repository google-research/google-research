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

"""Dataset reader and tokenization-related utilities for baseline model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import re

import tensorflow.compat.v1 as tf

from schema_guided_dst import schema
from schema_guided_dst.baseline.bert import tokenization

# Dimension of the embedding for intents, slots and categorical slot values in
# the schema. Should be equal to BERT's hidden_size.
EMBEDDING_DIMENSION = 768
STR_DONTCARE = "dontcare"
# The maximum total input sequence length after WordPiece tokenization.
DEFAULT_MAX_SEQ_LENGTH = 128

# These are used to represent the status of slots (off, active, dontcare) and
# intents (off, active) in dialogue state tracking.
STATUS_OFF = 0
STATUS_ACTIVE = 1
STATUS_DONTCARE = 2

# Name of the file containing all predictions and their corresponding frame
# metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """
  pass


def load_dialogues(dialog_json_filepaths):
  """Obtain the list of all dialogues from specified json files."""
  dialogs = []
  for dialog_json_filepath in sorted(dialog_json_filepaths):
    with tf.io.gfile.GFile(dialog_json_filepath) as f:
      dialogs.extend(json.load(f))
  return dialogs


class Dstc8DataProcessor(object):
  """Data generator for dstc8 dialogues."""

  def __init__(self,
               dstc8_data_dir,
               dataset_config,
               vocab_file,
               do_lower_case,
               max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
               log_data_warnings=False):
    self.dstc8_data_dir = dstc8_data_dir
    self._log_data_warnings = log_data_warnings
    self._dataset_config = dataset_config
    # BERT tokenizer
    self._tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length

  @property
  def dataset_config(self):
    return self._dataset_config

  def get_dialog_examples(self, dataset):
    """Return a list of `InputExample`s of the data splits' dialogues.

    Args:
      dataset: str. can be "train", "dev", or "test".

    Returns:
      examples: a list of `InputExample`s.
    """
    dialog_paths = [
        os.path.join(self.dstc8_data_dir, dataset,
                     "dialogues_{:03d}.json".format(i))
        for i in self._dataset_config.file_ranges[dataset]
    ]
    dialogs = load_dialogues(dialog_paths)
    schema_path = os.path.join(self.dstc8_data_dir, dataset, "schema.json")
    schemas = schema.Schema(schema_path)

    examples = []
    for dialog_idx, dialog in enumerate(dialogs):
      tf.logging.log_every_n(tf.logging.INFO, "Processed %d dialogs.", 1000,
                             dialog_idx)
      examples.extend(
          self._create_examples_from_dialog(dialog, schemas, dataset))
    return examples

  def _create_examples_from_dialog(self, dialog, schemas, dataset):
    """Create examples for every turn in the dialog."""
    dialog_id = dialog["dialogue_id"]
    prev_states = {}
    examples = []
    for turn_idx, turn in enumerate(dialog["turns"]):
      # Generate an example for every frame in every user turn.
      if turn["speaker"] == "USER":
        user_utterance = turn["utterance"]
        user_frames = {f["service"]: f for f in turn["frames"]}
        if turn_idx > 0:
          system_turn = dialog["turns"][turn_idx - 1]
          system_utterance = system_turn["utterance"]
          system_frames = {f["service"]: f for f in system_turn["frames"]}
        else:
          system_utterance = ""
          system_frames = {}
        turn_id = "{}-{}-{:02d}".format(dataset, dialog_id, turn_idx)
        turn_examples, prev_states = self._create_examples_from_turn(
            turn_id, system_utterance, user_utterance, system_frames,
            user_frames, prev_states, schemas)
        examples.extend(turn_examples)
    return examples

  def _get_state_update(self, current_state, prev_state):
    state_update = dict(current_state)
    for slot, values in current_state.items():
      if slot in prev_state and prev_state[slot][0] in values:
        # Remove the slot from state if its value didn't change.
        state_update.pop(slot)
    return state_update

  def _create_examples_from_turn(self, turn_id, system_utterance,
                                 user_utterance, system_frames, user_frames,
                                 prev_states, schemas):
    """Creates an example for each frame in the user turn."""
    system_tokens, system_alignments, system_inv_alignments = (
        self._tokenize(system_utterance))
    user_tokens, user_alignments, user_inv_alignments = (
        self._tokenize(user_utterance))
    states = {}
    base_example = InputExample(
        dataset_config=self._dataset_config,
        max_seq_length=self._max_seq_length,
        is_real_example=True,
        tokenizer=self._tokenizer,
        log_data_warnings=self._log_data_warnings)
    base_example.example_id = turn_id
    base_example.add_utterance_features(system_tokens, system_inv_alignments,
                                        user_tokens, user_inv_alignments)
    examples = []
    for service, user_frame in user_frames.items():
      # Create an example for this service.
      example = base_example.make_copy_with_utterance_features()
      example.example_id = "{}-{}".format(turn_id, service)
      example.service_schema = schemas.get_service_schema(service)
      system_frame = system_frames.get(service, None)
      state = user_frame["state"]["slot_values"]
      state_update = self._get_state_update(state, prev_states.get(service, {}))
      states[service] = state
      # Populate features in the example.
      example.add_categorical_slots(state_update)
      # The input tokens to bert are in the format [CLS] [S1] [S2] ... [SEP]
      # [U1] [U2] ... [SEP] [PAD] ... [PAD]. For system token indices a bias of
      # 1 is added for the [CLS] token and for user tokens a bias of 2 +
      # len(system_tokens) is added to account for [CLS], system tokens and
      # [SEP].
      user_span_boundaries = self._find_subword_indices(
          state_update, user_utterance, user_frame["slots"], user_alignments,
          user_tokens, 2 + len(system_tokens))
      if system_frame is not None:
        system_span_boundaries = self._find_subword_indices(
            state_update, system_utterance, system_frame["slots"],
            system_alignments, system_tokens, 1)
      else:
        system_span_boundaries = {}
      example.add_noncategorical_slots(state_update, user_span_boundaries,
                                       system_span_boundaries)
      example.add_requested_slots(user_frame)
      example.add_intents(user_frame)
      examples.append(example)
    return examples, states

  def _find_subword_indices(self, slot_values, utterance, char_slot_spans,
                            alignments, subwords, bias):
    """Find indices for subwords corresponding to slot values."""
    span_boundaries = {}
    for slot, values in slot_values.items():
      # Get all values present in the utterance for the specified slot.
      value_char_spans = {}
      for slot_span in char_slot_spans:
        if slot_span["slot"] == slot:
          value = utterance[slot_span["start"]:slot_span["exclusive_end"]]
          start_tok_idx = alignments[slot_span["start"]]
          end_tok_idx = alignments[slot_span["exclusive_end"] - 1]
          if 0 <= start_tok_idx < len(subwords):
            end_tok_idx = min(end_tok_idx, len(subwords) - 1)
            value_char_spans[value] = (start_tok_idx + bias, end_tok_idx + bias)
      for v in values:
        if v in value_char_spans:
          span_boundaries[slot] = value_char_spans[v]
          break
    return span_boundaries

  def _tokenize(self, utterance):
    """Tokenize the utterance using word-piece tokenization used by BERT.

    Args:
      utterance: A string containing the utterance to be tokenized.

    Returns:
      bert_tokens: A list of tokens obtained by word-piece tokenization of the
        utterance.
      alignments: A dict mapping indices of characters corresponding to start
        and end positions of words (not subwords) to corresponding indices in
        bert_tokens list.
      inverse_alignments: A list of size equal to bert_tokens. Each element is a
        tuple containing the index of the starting and inclusive ending
        character of the word corresponding to the subword. This list is used
        during inference to map word-piece indices to spans in the original
        utterance.
    """
    utterance = tokenization.convert_to_unicode(utterance)
    # After _naive_tokenize, spaces and punctuation marks are all retained, i.e.
    # direct concatenation of all the tokens in the sequence will be the
    # original string.
    tokens = _naive_tokenize(utterance)
    # Filter out empty tokens and obtain aligned character index for each token.
    alignments = {}
    char_index = 0
    bert_tokens = []
    # These lists store inverse alignments to be used during inference.
    bert_tokens_start_chars = []
    bert_tokens_end_chars = []
    for token in tokens:
      if token.strip():
        subwords = self._tokenizer.tokenize(token)
        # Store the alignment for the index of starting character and the
        # inclusive ending character of the token.
        alignments[char_index] = len(bert_tokens)
        bert_tokens_start_chars.extend([char_index] * len(subwords))
        bert_tokens.extend(subwords)
        # The inclusive ending character index corresponding to the word.
        inclusive_char_end = char_index + len(token) - 1
        alignments[inclusive_char_end] = len(bert_tokens) - 1
        bert_tokens_end_chars.extend([inclusive_char_end] * len(subwords))
      char_index += len(token)
    inverse_alignments = list(
        zip(bert_tokens_start_chars, bert_tokens_end_chars))
    return bert_tokens, alignments, inverse_alignments

  def get_num_dialog_examples(self, dataset):
    """Get the number of dilaog examples in the data split.

    Args:
      dataset: str. can be "train", "dev", or "test".

    Returns:
      example_count: int. number of examples in the specified dataset.
    """
    example_count = 0
    dialog_paths = [
        os.path.join(self.dstc8_data_dir, dataset,
                     "dialogues_{:03d}.json".format(i))
        for i in self._dataset_config.file_ranges[dataset]
    ]
    dst_set = load_dialogues(dialog_paths)
    for dialog in dst_set:
      for turn in dialog["turns"]:
        if turn["speaker"] == "USER":
          example_count += len(turn["frames"])
    return example_count


class InputExample(object):
  """An example for training/inference."""

  def __init__(self,
               dataset_config,
               max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
               service_schema=None,
               example_id="NONE",
               is_real_example=False,
               tokenizer=None,
               log_data_warnings=False):
    """Constructs an InputExample.

    Args:
      dataset_config: DataConfig object denoting the config of the dataset.
      max_seq_length: The maximum length of the sequence. Sequences longer than
        this value will be truncated.
      service_schema: A ServiceSchema object wrapping the schema for the service
        corresponding to this example.
      example_id: Unique identifier for the example.
      is_real_example: Indicates if an example is real or used for padding in a
        minibatch.
      tokenizer: A tokenizer object that has convert_tokens_to_ids and
        convert_ids_to_tokens methods. It must be non-None when
        is_real_example=True.
      log_data_warnings: If True, warnings generted while processing data are
        logged. This is useful for debugging data processing.
    """
    self.service_schema = service_schema
    self.example_id = example_id
    self.is_real_example = is_real_example
    self._max_seq_length = max_seq_length
    self._tokenizer = tokenizer
    self._log_data_warnings = log_data_warnings
    self._dataset_config = dataset_config
    if self.is_real_example and self._tokenizer is None:
      raise ValueError("Must specify tokenizer when input is a real example.")

    # The id of each subword in the vocabulary for BERT.
    self.utterance_ids = [0] * self._max_seq_length
    # Denotes the identity of the sequence. Takes values 0 (system utterance)
    # and 1 (user utterance).
    self.utterance_segment = [0] * self._max_seq_length
    # Mask which takes the value 0 for padded tokens and 1 otherwise.
    self.utterance_mask = [0] * self._max_seq_length
    # Start and inclusive end character indices in the original utterance
    # corresponding to the tokens. This is used to obtain the character indices
    # from the predicted subword indices during inference.
    # NOTE: A positive value indicates the character indices in the user
    # utterance whereas a negative value indicates the character indices in the
    # system utterance. The indices are offset by 1 to prevent ambiguity in the
    # 0 index, which could be in either the user or system utterance by the
    # above convention. Now the 0 index corresponds to padded tokens.
    self.start_char_idx = [0] * self._max_seq_length
    self.end_char_idx = [0] * self._max_seq_length

    # Number of categorical slots present in the service.
    self.num_categorical_slots = 0
    # The status of each categorical slot in the service.
    self.categorical_slot_status = [STATUS_OFF
                                   ] * dataset_config.max_num_cat_slot
    # Number of values taken by each categorical slot.
    self.num_categorical_slot_values = [0] * dataset_config.max_num_cat_slot
    # The index of the correct value for each categorical slot.
    self.categorical_slot_values = [0] * dataset_config.max_num_cat_slot

    # Number of non-categorical slots present in the service.
    self.num_noncategorical_slots = 0
    # The status of each non-categorical slot in the service.
    self.noncategorical_slot_status = [STATUS_OFF
                                      ] * dataset_config.max_num_noncat_slot
    # The index of the starting subword corresponding to the slot span for a
    # non-categorical slot value.
    self.noncategorical_slot_value_start = [
        0
    ] * dataset_config.max_num_noncat_slot
    # The index of the ending (inclusive) subword corresponding to the slot span
    # for a non-categorical slot value.
    self.noncategorical_slot_value_end = [0
                                         ] * dataset_config.max_num_noncat_slot

    # Total number of slots present in the service. All slots are included here
    # since every slot can be requested.
    self.num_slots = 0
    # Takes value 1 if the corresponding slot is requested, 0 otherwise.
    self.requested_slot_status = [STATUS_OFF] * (
        dataset_config.max_num_cat_slot + dataset_config.max_num_noncat_slot)

    # Total number of intents present in the service.
    self.num_intents = 0
    # Takes value 1 if the intent is active, 0 otherwise.
    self.intent_status = [STATUS_OFF] * dataset_config.max_num_intent

  @property
  def readable_summary(self):
    """Get a readable dict that summarizes the attributes of an InputExample."""
    seq_length = sum(self.utterance_mask)
    utt_toks = self._tokenizer.convert_ids_to_tokens(
        self.utterance_ids[:seq_length])
    utt_tok_mask_pairs = list(
        zip(utt_toks, self.utterance_segment[:seq_length]))
    active_intents = [
        self.service_schema.get_intent_from_id(idx)
        for idx, s in enumerate(self.intent_status)
        if s == STATUS_ACTIVE
    ]
    if len(active_intents) > 1:
      raise ValueError(
          "Should not have multiple active intents in a single service.")
    active_intent = active_intents[0] if active_intents else ""
    slot_values_in_state = {}
    for idx, s in enumerate(self.categorical_slot_status):
      if s == STATUS_ACTIVE:
        value_id = self.categorical_slot_values[idx]
        slot_values_in_state[self.service_schema.get_categorical_slot_from_id(
            idx)] = self.service_schema.get_categorical_slot_value_from_id(
                idx, value_id)
      elif s == STATUS_DONTCARE:
        slot_values_in_state[self.service_schema.get_categorical_slot_from_id(
            idx)] = STR_DONTCARE
    for idx, s in enumerate(self.noncategorical_slot_status):
      if s == STATUS_ACTIVE:
        slot = self.service_schema.get_non_categorical_slot_from_id(idx)
        start_id = self.noncategorical_slot_value_start[idx]
        end_id = self.noncategorical_slot_value_end[idx]
        # Token list is consisted of the subwords that may start with "##". We
        # remove "##" to reconstruct the original value. Note that it's not a
        # strict restoration of the original string. It's primarily used for
        # debugging.
        # ex. ["san", "j", "##ose"] --> "san jose"
        readable_value = " ".join(utt_toks[start_id:end_id + 1]).replace(
            " ##", "")
        slot_values_in_state[slot] = readable_value
      elif s == STATUS_DONTCARE:
        slot = self.service_schema.get_non_categorical_slot_from_id(idx)
        slot_values_in_state[slot] = STR_DONTCARE

    summary_dict = {
        "utt_tok_mask_pairs": utt_tok_mask_pairs,
        "utt_len": seq_length,
        "num_categorical_slots": self.num_categorical_slots,
        "num_categorical_slot_values": self.num_categorical_slot_values,
        "num_noncategorical_slots": self.num_noncategorical_slots,
        "service_name": self.service_schema.service_name,
        "active_intent": active_intent,
        "slot_values_in_state": slot_values_in_state
    }
    return summary_dict

  def add_utterance_features(self, system_tokens, system_inv_alignments,
                             user_tokens, user_inv_alignments):
    """Add utterance related features input to bert.

    Note: this method modifies the system tokens and user_tokens in place to
    make their total length <= the maximum input length for BERT model.

    Args:
      system_tokens: a list of strings which represents system utterance.
      system_inv_alignments: a list of tuples which denotes the start and end
        charater of the tpken that a bert token originates from in the original
        system utterance.
      user_tokens: a list of strings which represents user utterance.
      user_inv_alignments: a list of tuples which denotes the start and end
        charater of the token that a bert token originates from in the original
        user utterance.
    """
    # Make user-system utterance input (in BERT format)
    # Input sequence length for utterance BERT encoder
    max_utt_len = self._max_seq_length

    # Modify lengths of sys & usr utterance so that length of total utt
    # (including [CLS], [SEP], [SEP]) is no more than max_utt_len
    is_too_long = truncate_seq_pair(system_tokens, user_tokens, max_utt_len - 3)
    if is_too_long and self._log_data_warnings:
      tf.logging.info("Utterance sequence truncated in example id - %s.",
                      self.example_id)

    # Construct the tokens, segment mask and valid token mask which will be
    # input to BERT, using the tokens for system utterance (sequence A) and
    # user utterance (sequence B).
    utt_subword = []
    utt_seg = []
    utt_mask = []
    start_char_idx = []
    end_char_idx = []

    utt_subword.append("[CLS]")
    utt_seg.append(0)
    utt_mask.append(1)
    start_char_idx.append(0)
    end_char_idx.append(0)

    for subword_idx, subword in enumerate(system_tokens):
      utt_subword.append(subword)
      utt_seg.append(0)
      utt_mask.append(1)
      st, en = system_inv_alignments[subword_idx]
      start_char_idx.append(-(st + 1))
      end_char_idx.append(-(en + 1))

    utt_subword.append("[SEP]")
    utt_seg.append(0)
    utt_mask.append(1)
    start_char_idx.append(0)
    end_char_idx.append(0)

    for subword_idx, subword in enumerate(user_tokens):
      utt_subword.append(subword)
      utt_seg.append(1)
      utt_mask.append(1)
      st, en = user_inv_alignments[subword_idx]
      start_char_idx.append(st + 1)
      end_char_idx.append(en + 1)

    utt_subword.append("[SEP]")
    utt_seg.append(1)
    utt_mask.append(1)
    start_char_idx.append(0)
    end_char_idx.append(0)

    utterance_ids = self._tokenizer.convert_tokens_to_ids(utt_subword)

    # Zero-pad up to the BERT input sequence length.
    while len(utterance_ids) < max_utt_len:
      utterance_ids.append(0)
      utt_seg.append(0)
      utt_mask.append(0)
      start_char_idx.append(0)
      end_char_idx.append(0)
    self.utterance_ids = utterance_ids
    self.utterance_segment = utt_seg
    self.utterance_mask = utt_mask
    self.start_char_idx = start_char_idx
    self.end_char_idx = end_char_idx

  def make_copy_with_utterance_features(self):
    """Make a copy of the current example with utterance features."""
    new_example = InputExample(
        dataset_config=self._dataset_config,
        max_seq_length=self._max_seq_length,
        service_schema=self.service_schema,
        example_id=self.example_id,
        is_real_example=self.is_real_example,
        tokenizer=self._tokenizer,
        log_data_warnings=self._log_data_warnings)
    new_example.utterance_ids = list(self.utterance_ids)
    new_example.utterance_segment = list(self.utterance_segment)
    new_example.utterance_mask = list(self.utterance_mask)
    new_example.start_char_idx = list(self.start_char_idx)
    new_example.end_char_idx = list(self.end_char_idx)
    return new_example

  def add_categorical_slots(self, state_update):
    """Add features for categorical slots."""
    categorical_slots = self.service_schema.categorical_slots
    self.num_categorical_slots = len(categorical_slots)
    for slot_idx, slot in enumerate(categorical_slots):
      values = state_update.get(slot, [])
      # Add categorical slot value features.
      slot_values = self.service_schema.get_categorical_slot_values(slot)
      self.num_categorical_slot_values[slot_idx] = len(slot_values)
      if not values:
        self.categorical_slot_status[slot_idx] = STATUS_OFF
      elif values[0] == STR_DONTCARE:
        self.categorical_slot_status[slot_idx] = STATUS_DONTCARE
      else:
        self.categorical_slot_status[slot_idx] = STATUS_ACTIVE
        self.categorical_slot_values[slot_idx] = (
            self.service_schema.get_categorical_slot_value_id(slot, values[0]))

  def add_noncategorical_slots(self, state_update, system_span_boundaries,
                               user_span_boundaries):
    """Add features for non-categorical slots."""
    noncategorical_slots = self.service_schema.non_categorical_slots
    self.num_noncategorical_slots = len(noncategorical_slots)
    for slot_idx, slot in enumerate(noncategorical_slots):
      values = state_update.get(slot, [])
      if not values:
        self.noncategorical_slot_status[slot_idx] = STATUS_OFF
      elif values[0] == STR_DONTCARE:
        self.noncategorical_slot_status[slot_idx] = STATUS_DONTCARE
      else:
        self.noncategorical_slot_status[slot_idx] = STATUS_ACTIVE
        # Add indices of the start and end tokens for the first encountered
        # value. Spans in user utterance are prioritized over the system
        # utterance. If a span is not found, the slot value is ignored.
        if slot in user_span_boundaries:
          start, end = user_span_boundaries[slot]
        elif slot in system_span_boundaries:
          start, end = system_span_boundaries[slot]
        else:
          # A span may not be found because the value was cropped out or because
          # the value was mentioned earlier in the dialogue. Since this model
          # only makes use of the last two utterances to predict state updates,
          # it will fail in such cases.
          if self._log_data_warnings:
            tf.logging.info(
                "Slot values %s not found in user or system utterance in "
                "example with id - %s.", str(values), self.example_id)
          continue
        self.noncategorical_slot_value_start[slot_idx] = start
        self.noncategorical_slot_value_end[slot_idx] = end

  def add_requested_slots(self, frame):
    all_slots = self.service_schema.slots
    self.num_slots = len(all_slots)
    for slot_idx, slot in enumerate(all_slots):
      if slot in frame["state"]["requested_slots"]:
        self.requested_slot_status[slot_idx] = STATUS_ACTIVE

  def add_intents(self, frame):
    all_intents = self.service_schema.intents
    self.num_intents = len(all_intents)
    for intent_idx, intent in enumerate(all_intents):
      if intent == frame["state"]["active_intent"]:
        self.intent_status[intent_idx] = STATUS_ACTIVE


def _create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f


# Modified from run_classifier.file_based_convert_examples_to_features in the
# public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def file_based_convert_examples_to_features(dial_examples, dataset_config,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(dial_examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d", ex_index, len(dial_examples))

    if isinstance(example, PaddingInputExample):
      ex = InputExample(dataset_config=dataset_config)
    else:
      ex = example

    features = collections.OrderedDict()

    features["example_id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[ex.example_id.encode("utf-8")]))
    features["is_real_example"] = _create_int_feature([int(ex.is_real_example)])
    features["service_id"] = _create_int_feature([ex.service_schema.service_id])

    features["utt"] = _create_int_feature(ex.utterance_ids)
    features["utt_seg"] = _create_int_feature(ex.utterance_segment)
    features["utt_mask"] = _create_int_feature(ex.utterance_mask)

    features["cat_slot_num"] = _create_int_feature([ex.num_categorical_slots])
    features["cat_slot_status"] = _create_int_feature(
        ex.categorical_slot_status)
    features["cat_slot_value_num"] = _create_int_feature(
        ex.num_categorical_slot_values)
    features["cat_slot_value"] = _create_int_feature(ex.categorical_slot_values)

    features["noncat_slot_num"] = _create_int_feature(
        [ex.num_noncategorical_slots])
    features["noncat_slot_status"] = _create_int_feature(
        ex.noncategorical_slot_status)
    features["noncat_slot_value_start"] = _create_int_feature(
        ex.noncategorical_slot_value_start)
    features["noncat_slot_value_end"] = _create_int_feature(
        ex.noncategorical_slot_value_end)
    features["noncat_alignment_start"] = _create_int_feature(ex.start_char_idx)
    features["noncat_alignment_end"] = _create_int_feature(ex.end_char_idx)

    features["req_slot_num"] = _create_int_feature([ex.num_slots])
    features["req_slot_status"] = _create_int_feature(ex.requested_slot_status)

    features["intent_num"] = _create_int_feature([ex.num_intents])
    features["intent_status"] = _create_int_feature(ex.intent_status)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(tf_example.SerializeToString())
  writer.close()


def normalize_list_length(input_list, target_len, padding_unit):
  """Post truncate or pad the input list in place to be of target length.

  Args:
    input_list: the list whose length will be normalized to `target_len` by post
      truncation or padding.
    target_len: the target length which `input_list` should be.
    padding_unit: when the length of `input_list` is smaller than target_len, we
      append a sequence of `padding_unit`s at the end of the input_list so that
      the length of input_list will be `target_len`.
  """

  if len(input_list) < target_len:
    input_list.extend(
        [padding_unit for _ in range(target_len - len(input_list))])
  elif len(input_list) > target_len:
    del input_list[target_len:]
  assert len(input_list) == target_len


def _naive_tokenize(s):
  """Tokenize a string, separating words, spaces and punctuations."""
  # Spaces and punctuation marks are all retained, i.e. direct concatenation
  # of all the tokens in the sequence will be the original string.
  seq_tok = [tok for tok in re.split(r"([^a-zA-Z0-9])", s) if tok]
  return seq_tok


def _get_token_char_range(utt_tok):
  """Get starting and end character positions of each token in utt_tok."""
  char_pos = 0
  # List of (start_char_pos, end_char_pos) for each token in utt_tok.
  utt_char_range = []
  for tok in utt_tok:
    start = char_pos
    end = start + len(tok) - 1
    utt_char_range.append((start, end))
    char_pos = end + 1
  return utt_char_range


def _get_token_label(utt_char_range, start_char_pos, exclusive_end_char_pos):
  """Get position of token according to char range of each tokens."""
  end_char_pos = exclusive_end_char_pos - 1
  slot_at_boundary = True
  for idx, (start, end) in enumerate(utt_char_range):
    if start <= start_char_pos <= end:
      if start != start_char_pos:
        slot_at_boundary = False
      start_tok_pos = idx
    if start <= end_char_pos <= end:
      if end != end_char_pos:
        slot_at_boundary = False
      end_tok_pos = idx
  assert start_tok_pos <= end_tok_pos
  return start_tok_pos, end_tok_pos, slot_at_boundary


# Modified from run_classifier._truncate_seq_pair in the public bert model repo.
# https://github.com/google-research/bert/blob/master/run_classifier.py.
def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncate a seq pair in place so that their total length <= max_length."""
  is_too_long = False
  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    is_too_long = True
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
  return is_too_long
