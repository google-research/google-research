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

"""For running inference with a Felix tagger/insertion model.

Throughout this file we use the terms predict and realize. Prediction involves
making a decision (what tag to apply or what each token should point to).
Realization takes this decision and applies it to the input sentence to produce
the output sentence. In the simplest case, prediction involves selecting which
tokens should be deleted, and the realization code will then produce a new
sentence without these deleted tokens.
"""
from typing import Mapping, Optional, Sequence, Tuple, cast

from absl import logging
import numpy as np
from official.nlp.bert import configs
import tensorflow as tf

from felix import beam_search
from felix import example_builder_for_felix_insert
from felix import felix_constants as constants
from felix import felix_models
from felix import insertion_converter
from felix import preprocess
from felix import utils


class FelixPredictor:
  """Class for computing and realizing predictions with Felix."""

  def __init__(
      self,
      bert_config_tagging,
      bert_config_insertion,
      model_tagging_filepath,
      model_insertion_filepath,
      label_map_file,
      sequence_length = 128,
      max_predictions = 20,
      do_lowercase = True,
      vocab_file = None,
      use_open_vocab = False,
      is_pointing = False,
      insert_after_token = True,
      special_glue_string_for_joining_sources = ' ',
  ):
    """Initializes an instance of FelixPredictor.

    Args:
        bert_config_tagging: The config file for the tagging model.
        bert_config_insertion: The config file for the insertion model. Only
          needed in the case of use_open_vocab (Felix/FelixInsert).
        model_tagging_filepath: The file location of the tagging model.  If not
          provided in a randomly initialized model is used (not recommended).
        model_insertion_filepath: The file location of the insertion model. Only
          needed in the case of use_open_vocab (Felix/FelixInsert)  and if not
          provided  a randomly initialized model is used (not recommended).
        label_map_file: Label map file path.
        sequence_length: Maximum length of a sequence.
        max_predictions: Maximum number of predictions for insertion model.
        do_lowercase: If the input text is lowercased.
        vocab_file: BERT vocab file.
        use_open_vocab: If it is an open vocab model (felix/felixInsert).
          Currently only True is supported.
        is_pointing: if the tagging model uses a pointing mechanism. Currently
          only True is supported.
        insert_after_token: Whether to insert tokens after rather than before
          the current token.
        special_glue_string_for_joining_sources: String that is used to join
          multiple source strings of a given example into one string.
    """
    if not use_open_vocab:
      raise ValueError('Currently only use_open_vocab=True is supported')
    self._bert_config_insertion = bert_config_insertion
    self._bert_config_tagging = bert_config_tagging
    if model_tagging_filepath is None:
      logging.warning(
          'No filepath is provided for tagging model, a randomly initialized '
          'model will be used!')
    if bert_config_insertion is None and use_open_vocab:
      raise ValueError('bert_config_insertion needs to be provided.')
    if model_insertion_filepath is None and not use_open_vocab:
      logging.warning(
          'No filepath is provided for insertion model, a randomly initialized '
          'model will be used!')
    self._model_tagging_filepath = model_tagging_filepath
    self._model_insertion_filepath = model_insertion_filepath
    self._tagging_model = None
    self._insertion_model = None

    self._sequence_length = sequence_length
    self._max_predictions = max_predictions
    self._use_open_vocab = use_open_vocab
    self._is_pointing = is_pointing
    self._do_lowercase = do_lowercase

    self._builder = preprocess.initialize_builder(
        use_pointing=self._is_pointing,
        use_open_vocab=self._use_open_vocab,
        label_map_file=label_map_file,
        max_seq_length=self._sequence_length,
        max_predictions_per_seq=self._max_predictions,
        do_lower_case=self._do_lowercase,
        vocab_file=vocab_file,
        insert_after_token=insert_after_token,
        special_glue_string_for_sources=special_glue_string_for_joining_sources,
        max_mask=20)
    self._inverse_label_map = {
        tag_id: tag for tag, tag_id in self._builder.label_map.items()
    }

  def predict_end_to_end_batch(
      self, batch
      ):
    """Takes in a batch of source sentences and runs Felix on them.

    Args:
      batch: Source inputs, where each input is composed of multiple source
        utterances.

    Returns:
      taggings_outputs: Intermediate realized output of the tagging model.
      insertion_outputs: The final realized output of the model after running
      the tagging and insertion model.
    """
    taggings_outputs = self._predict_and_realize_batch(
        batch, is_insertion=False)
    insertion_outputs = self._predict_and_realize_batch(
        taggings_outputs, is_insertion=True)
    return taggings_outputs, insertion_outputs

  def _load_model(self, is_insertion=True):
    """Loads either an insertion or tagging model for inference."""

    def _get_fake_loss_fn():
      """"A fake loss function for inference."""

      def _fake_loss_fn(unused_labels, unused_losses, **unused_args):
        return 0.0

      return _fake_loss_fn

    model_filepath = None
    if is_insertion:
      model, _ = felix_models.get_insertion_model(
          self._bert_config_insertion,
          self._sequence_length,
          max_predictions_per_seq=self._max_predictions,
          is_training=False)
      model_filepath = self._model_insertion_filepath
    else:
      model, _ = felix_models.get_tagging_model(
          self._bert_config_tagging,
          self._sequence_length,
          use_pointing=self._is_pointing,
          pointing_weight=0,
          is_training=False)
      model_filepath = self._model_tagging_filepath
    if model_filepath:
      checkpoint = tf.train.Checkpoint(model=model)
      latest_checkpoint_file = tf.train.latest_checkpoint(model_filepath)
      checkpoint.restore(latest_checkpoint_file)
    model.compile(loss=_get_fake_loss_fn())
    if is_insertion:
      self._insertion_model = model
    else:
      self._tagging_model = model

  def _predict_and_realize_batch(
      self,
      source_senteces,
      is_insertion):
    """Run tagging inference on a batch and return the realizations."""

    batch_dictionaries, batch_list = self._convert_source_sentences_into_batch(
        source_senteces, is_insertion=is_insertion)
    predictions = self._predict_batch(batch_list, is_insertion=is_insertion)
    if is_insertion:
      realizations = self._realize_insertion_batch(batch_dictionaries,
                                                   predictions)
    else:
      realizations = self._realize_tagging_batch(batch_dictionaries,
                                                 predictions)
    return realizations

  def _convert_source_sentences_into_batch(
      self,
      source_sentences,
      is_insertion):
    """Converts source sentence into a batch."""
    batch_dictionaries = []
    for  source_sentence in source_sentences:
      if is_insertion:
        # Note source_sentence is the output from the tagging model and
        # therefore already tokenized.
        example = utils.build_feed_dict(
            source_sentence.split(' '),
            self._builder.tokenizer,
            max_seq_length=self._sequence_length,
            max_predictions_per_seq=self._max_predictions)

        assert example is not None, (
            f'Source sentence {source_sentence} returned None when '
            'converting to insertion example.')
        # Previously the code produced an output with a batch size of 1, this
        # dimension is removed, as we do arbitrary batching in this code now.
        example = dict(example)
        for k, v in example.items():
          example[k] = v[0]
        # Note masked_lm_ids and masked_lm_weights are filled with zeros.
        batch_dictionaries.append({
            'input_word_ids': np.array(example['input_ids']),
            'input_mask': np.array(example['input_mask']),
            'input_type_ids': np.array(example['segment_ids']),
            'masked_lm_positions': np.array(example['masked_lm_positions']),
            'masked_lm_ids': np.array(example['masked_lm_ids']),
            'masked_lm_weights': np.array(example['masked_lm_weights'])
        })
      else:
        example, _ = self._builder.build_bert_example([source_sentence],
                                                      target=None,
                                                      is_test_time=True)
        assert example is not None, (f'Tagging could not convert '
                                     f'{source_sentence}.')
        dict_element = {
            'input_word_ids': np.array(example.features['input_ids']),
            'input_mask': np.array(example.features['input_mask']),
            'input_type_ids': np.array(example.features['segment_ids']),
        }
        batch_dictionaries.append(dict_element)
    # Convert from a list of dictionaries to dictionary of lists.
    batch_list = ({
        k: np.array([dic[k] for dic in batch_dictionaries
                    ]) for k in batch_dictionaries[0]
    })
    return batch_dictionaries, batch_list

  def _predict_batch(self, source_batch, is_insertion):
    """Produce output from tensorflow model."""
    if is_insertion:
      if self._insertion_model is None:
        self._load_model(is_insertion=True)
      predictions = self._insertion_model(source_batch, training=False)
      # Go from a probability distribution to a vocab item.
      return np.argmax(predictions, axis=-1)

    if self._tagging_model is None:
      self._load_model(is_insertion=False)
    if self._is_pointing:
      tag_logits, pointing_logits = self._tagging_model(
          source_batch, training=False)
      # Convert two lists into a single list of tuples.
      return list(zip(tag_logits, pointing_logits))
    else:
      tag_logits = self._tagging_model(source_batch, training=False)
      return tag_logits

  def _realize_insertion_batch(self, source_batch, prediction_batch):
    """Produces the realized predicitions for a batch from the tagging model."""
    realizations = []
    for source, predicted_tokens in zip(source_batch, prediction_batch):
      sequence_length = sum(source['input_mask'])
      realization = self._realize_insertion_single(source['input_word_ids'],
                                                   sequence_length - 1,
                                                   predicted_tokens)

      realization = ' '.join(realization)
      realizations.append(realization)
    return realizations

  def _realize_insertion_single(self, input_word_ids, end_index,
                                predicted_tokens):
    """Realizes the predictions from the insertion model."""
    tokens = self._builder.tokenizer.convert_ids_to_tokens(
        input_word_ids[:end_index + 1])
    current_mask = 0
    new_tokens = []
    in_deletion_bracket = False
    for token in tokens:
      if token.lower() == constants.DELETE_SPAN_END:
        in_deletion_bracket = False
        continue
      elif in_deletion_bracket:
        continue
      elif token.lower() == constants.DELETE_SPAN_START:
        in_deletion_bracket = True
        continue

      if token.lower() == constants.MASK.lower():
        new_tokens.append(
            self._builder.tokenizer.convert_ids_to_tokens(
                [predicted_tokens[current_mask]])[0])
        current_mask += 1
      else:
        new_tokens.append(token)
    return new_tokens

  def _realize_tagging_batch(self, source_batch, prediction_batch):
    """Produces the realized predictions for a batch from the tagging model."""
    realizations = []
    for source, prediction in zip(source_batch, prediction_batch):
      end_index = sum(source['input_mask']) - 1

      if self._is_pointing:
        tag_logits, pointing_logits = prediction
        realization = self._realize_tagging_single(source['input_word_ids'],
                                                   end_index, tag_logits,
                                                   pointing_logits)
      else:
        tag_logits = prediction
        realization = self._realize_tagging_wo_pointing_single(
            source['input_word_ids'], end_index, tag_logits)

      # Copy source sentence if prediction has failed.
      if realization is None:
        realization = self._builder.tokenizer.convert_ids_to_tokens(
            source['input_word_ids'][:end_index + 1])

      realization = ' '.join(realization)
      realizations.append(realization)
    return realizations

  def _realize_tagging_single(self,
                              input_word_ids,
                              last_token_index,
                              tag_logits,
                              point_logits,
                              beam_size = 15):
    """Returns realized prediction for a given source using beam search.

    Args:
      input_word_ids:  Source token ids.
      last_token_index: The index, in the input_word_ids, of the last token (not
        including padding tokens).
      tag_logits: Tag logits  [vocab size, sequence_length].
      point_logits: Point logits  [sequence_length, sequence_length] .
      beam_size: The size of the beam.

    Returns:
      Realized predictions including deleted tokens. It is possible that beam
      search fails (producing malformed output), in this case return None.
    """
    # Need to help type checker.
    self._inverse_label_map = cast(Mapping[int, str], self._inverse_label_map)

    predicted_tags = list(np.argmax(tag_logits, axis=1))
    non_deleted_indexes = set(
        i for i, tag in enumerate(predicted_tags[:last_token_index + 1])
        if self._inverse_label_map[int(tag)] not in constants.DELETED_TAGS)
    source_tokens = self._builder.tokenizer.convert_ids_to_tokens(
        list(input_word_ids))
    sep_indexes = set([
        i for i, token in enumerate(source_tokens)
        if token.lower() == constants.SEP.lower() and i in non_deleted_indexes
    ])

    best_sequence = beam_search.beam_search_single_tagging(
        list(point_logits), non_deleted_indexes, sep_indexes, beam_size,
        last_token_index, self._sequence_length)
    if best_sequence is None:
      return None

    return self._realize_beam_search(input_word_ids, best_sequence,
                                     predicted_tags, last_token_index + 1)

  def _realize_beam_search(self, source_token_ids,
                           ordered_source_indexes,
                           tags,
                           source_length):
    """Returns realized prediction using indexes and tags.

    TODO: Refactor this function to share code with
    `_create_masked_source` from insertion_converter.py to reduce code
    duplication and to ensure that the insertion example creation is consistent
    between preprocessing and prediction.

    Args:
      source_token_ids: List of source token ids.
      ordered_source_indexes: The order in which the kept tokens should be
        realized.
      tags: a List of tags.
      source_length: How long is the source input (excluding padding).

    Returns:
      Realized predictions (with deleted tokens).
    """
    # Need to help type checker.
    self._inverse_label_map = cast(Mapping[int, str], self._inverse_label_map)

    source_token_ids_set = set(ordered_source_indexes)
    out_tokens = []
    out_tokens_with_deletes = []
    for j, index in enumerate(ordered_source_indexes):
      token = self._builder.tokenizer.convert_ids_to_tokens(
          [source_token_ids[index]])
      out_tokens += token
      tag = self._inverse_label_map[tags[index]]
      if self._use_open_vocab:
        out_tokens_with_deletes += token
        # Add the predicted MASK tokens.
        number_of_masks = insertion_converter.get_number_of_masks(tag)
        # Can not add phrases after last token.
        if j == len(ordered_source_indexes) - 1:
          number_of_masks = 0
        masks = [constants.MASK] * number_of_masks
        out_tokens += masks
        out_tokens_with_deletes += masks

        # Find the deleted tokens, which appear after the current token.
        deleted_tokens = []
        for i in range(index + 1, source_length):
          if i in source_token_ids_set:
            break
          deleted_tokens.append(source_token_ids[i])
        # Bracket the deleted tokens, between unused0 and unused1.
        if deleted_tokens:
          deleted_tokens = [constants.DELETE_SPAN_START] + list(
              self._builder.tokenizer.convert_ids_to_tokens(deleted_tokens)) + [
                  constants.DELETE_SPAN_END
              ]
          out_tokens_with_deletes += deleted_tokens
      # Add the predicted phrase.
      elif '|' in tag:
        pos_pipe = tag.index('|')
        added_phrase = tag[pos_pipe + 1:]
        out_tokens.append(added_phrase)

    if not self._use_open_vocab:
      out_tokens_with_deletes = out_tokens
    assert (
        out_tokens_with_deletes[0] == (constants.CLS)
    ), (f' {out_tokens_with_deletes} did not start/end with the correct tokens '
        f'{constants.CLS}, {constants.SEP}')
    return out_tokens_with_deletes

  def _realize_tagging_wo_pointing_single(
      self, input_word_ids, last_token_index,
      tag_logits):
    """Returns realized prediction for a given source for FelixInsert.

    TODO: Add special handling for [SEP] tokens like done above for the
    full Felix model.

    Args:
      input_word_ids:  Source token ids.
      last_token_index: The index, in the input_word_ids, of the last token (not
        including padding tokens).
      tag_logits: Tag logits  [vocab size, sequence_length].

    Returns:
      Realized predictions including deleted tokens.
    """
    # Need to help type checker.
    self._inverse_label_map = cast(Mapping[int, Tuple[str, int]],
                                   self._inverse_label_map)
    self._builder = cast(
        example_builder_for_felix_insert.FelixInsertExampleBuilder,
        self._builder)

    input_tokens = self._builder.tokenizer.convert_ids_to_tokens(input_word_ids)
    predicted_tags = list(np.argmax(tag_logits, axis=1))[:last_token_index + 1]
    label_tuples = [self._inverse_label_map[int(tag)] for tag in predicted_tags]
    tokens = self._builder.build_insertion_tokens(input_tokens, label_tuples)
    if tokens is None:
      return None
    return tokens[0]
