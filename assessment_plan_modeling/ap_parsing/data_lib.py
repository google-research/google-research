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

"""Utilities for transforming data into model input for AP Parsing."""

import copy
import dataclasses
import enum
import hashlib
import itertools
import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from absl import logging
import apache_beam as beam
from apache_beam.dataframe import convert as df_convert
from apache_beam.dataframe import io as df_io
import numpy as np
import tensorflow as tf

from assessment_plan_modeling.ap_parsing import ap_parsing_lib
from assessment_plan_modeling.ap_parsing import ap_parsing_utils
from assessment_plan_modeling.ap_parsing import ap_problems_action_items_annotator
from assessment_plan_modeling.ap_parsing import augmentation_lib as aug_lib
from assessment_plan_modeling.ap_parsing import constants
from assessment_plan_modeling.ap_parsing import tokenizer_lib
from assessment_plan_modeling.note_sectioning import note_section_lib


class Partition(enum.IntEnum):
  NONRATED = 0  # All non-rated are train-only.
  TRAIN = 1
  VAL = 2
  TEST = 3


@dataclasses.dataclass
class Note:
  note_id: int
  subject_id: int
  text: str
  category: str


@dataclasses.dataclass
class APData:
  """AP section based container for data generated in the beam pipeline."""
  note_id: str
  subject_id: str
  ap_text: str = ""
  char_offset: int = 0
  tokens: List[tokenizer_lib.Token] = dataclasses.field(default_factory=list)
  labeled_char_spans: List[ap_parsing_lib.LabeledCharSpan] = dataclasses.field(
      default_factory=list)
  labels: Dict[str, List[int]] = dataclasses.field(default_factory=dict)
  token_features: Dict[str, List[int]] = dataclasses.field(default_factory=dict)
  partition: Partition = Partition.NONRATED
  augmentation_name: str = ""  # Name of the augmentation applied,
  # empty if no augmentation was applied.

  @property
  def is_rated(self):
    return self.partition != Partition.NONRATED


class ApplyAugmentations(beam.DoFn):
  """DoFn to apply augmentations to ap sections.

  The DoFn recieves possible augmentations via the augmentation config.
  This configs specifies the different augmentation sequences and the
  probabilities to sample each of these sequences.

  The DoFn emits both the original section and a number of augmented views.
  The number of augmented viewes is either given via n_augmentations or sampled
  from a poisson distribution paramaterized by the augmentation config. Poisson
  sampling takes precedence.
  """

  def __init__(self):
    self.n_augmentations_dist = beam.metrics.Metrics.distribution(
        "augmentations", "n_augmentations")
    self.total_augmented_counter = beam.metrics.Metrics.counter(
        "augmentations", "total_augmented")

  def process(self,
              element,
              augmentation_config,
              random_seed = 0):
    key, ap_data = element

    # yield non augmented:
    yield element

    # Apply augmentations to train only:
    if ap_data.partition < Partition.VAL:
      seed = hash_key(key, random_seed)
      rng = np.random.default_rng(seed=seed)

      n_augmentations = augmentation_config.get_n_augmentations(rng)

      # Update distribution.
      self.n_augmentations_dist.update(n_augmentations)

      for _ in range(n_augmentations):
        structured_ap = aug_lib.StructuredAP.build(ap_data.ap_text,
                                                   ap_data.labeled_char_spans)
        aug_seq = rng.choice(
            augmentation_config.augmentation_sequences,
            p=augmentation_config.augmentation_sample_probabilities)

        structured_ap = aug_lib.apply_augmentations(
            structured_ap, aug_seq, seed=seed)

        # Update counter.
        self.total_augmented_counter.inc()
        beam.metrics.Metrics.counter("augmentations", aug_seq.name).inc()

        # Construct new record with augmented text.
        new_ap_data = dataclasses.replace(ap_data)
        new_ap_data.ap_text, new_ap_data.labeled_char_spans = structured_ap.compile(
        )
        new_ap_data.tokens = tokenizer_lib.tokenize(new_ap_data.ap_text)
        new_ap_data.augmentation_name = aug_seq.name

        yield (key, new_ap_data)


class ProcessAPData(beam.DoFn):
  """Processes notes with ratings for section based feature and label extraction."""

  def __init__(self, filter_inorganic_threshold = 10):
    self.filter_inorganic_threshold = filter_inorganic_threshold

  def process(
      self, element,
      section_markers):
    (key, values) = element

    assert (len(values["notes"]) == 1), f"not one note per note id {key}"
    note = values["notes"][0]

    ap_data = APData(note_id=str(key), subject_id=str(note.subject_id))

    def yield_sections(
        ap_data,
        note_ratings = None
    ):
      for section in extract_ap_sections(note.text, section_markers):
        # Copy per section.
        cur_ap_data = dataclasses.replace(ap_data)
        cur_ap_data.ap_text = note.text[section.char_start:section.char_end]
        cur_ap_data.tokens = tokenizer_lib.tokenize(cur_ap_data.ap_text)
        cur_ap_data.char_offset = section.char_start

        # Filter inorganic AP sections by word counts of non CAPS tokens >= 10
        if filter_inorganic(
            cur_ap_data.tokens, threshold=self.filter_inorganic_threshold):
          continue

        if cur_ap_data.is_rated:
          cur_ap_data.labeled_char_spans = process_rating_labels(
              note_ratings, section)
        else:
          cur_ap_data.partition = Partition.NONRATED
          cur_ap_data.labeled_char_spans = annotate_ap(cur_ap_data.ap_text)

        cur_ap_data.labeled_char_spans = (
            ap_parsing_utils.normalize_labeled_char_spans_iterable(
                cur_ap_data.labeled_char_spans, cur_ap_data.tokens))

        yield (f"{cur_ap_data.note_id}|{section.char_start}", cur_ap_data)

    if values["ratings"]:
      assert len(values["note_partition"]) == len(values["ratings"]), (
          "all ratings should have one associated partition [train, val, "
          f"test], got {values['note_partition']}")

      for str_partition, note_ratings in zip(values["note_partition"],
                                             values["ratings"]):
        cur_ap_data = dataclasses.replace(
            ap_data, partition=Partition[str_partition.upper()])
        yield from yield_sections(cur_ap_data, note_ratings)
    else:
      yield from yield_sections(ap_data)


class ProcessFeaturesAndLabels(beam.DoFn):
  """Beam DoFn for generating features and labels for the model."""

  def __init__(self, vocab, max_seq_length = None):
    self._max_seq_length = max_seq_length
    self._vocab = vocab
    self.over_max_seq_length = beam.metrics.Metrics.counter(
        "features_and_labels", "length_over_max")
    self.seq_length_dist = beam.metrics.Metrics.distribution(
        "features_and_labels", "seq_length")

  def process(self, element):
    """Transform text and labels to a machine learning ready form.

    Beam DoFn to generate a DataAndLabels of the features to be used in
    training.
    Yields key values tuples where the key is the label source.
    Args:
      element: Keyed APData

    Yields:
      key, value pair of the APData with the added features.
    """
    (key, ap_data) = element

    # Get Features:
    token_features = generate_token_features(ap_data.tokens, self._vocab)

    # Get labels:
    labels = generate_model_labels(ap_data.labeled_char_spans, ap_data.tokens)

    # Filter whitespaces:
    token_mask = [t.token_text == " " for t in ap_data.tokens]
    tokens = self.filter_whitespaces(ap_data.tokens, token_mask)
    self.seq_length_dist.update(len(tokens))

    if self._max_seq_length and len(tokens) > self._max_seq_length:
      self.over_max_seq_length.inc()
      logging.log_every_n_seconds(logging.INFO,
                                  "AP section too long: %d tokens", 3,
                                  len(tokens))
      return

    token_features = self.filter_whitespaces_dict(token_features, token_mask)
    labels = self.filter_whitespaces_dict(labels, token_mask)

    # Pad.
    token_features = self.pad(token_features, value=0)
    labels = self.pad(labels, value=-1)

    ap_data.tokens = tokens
    ap_data.token_features = token_features
    ap_data.labels = labels

    yield (key, ap_data)

  def filter_whitespaces(self, token_level_seq,
                         token_mask):
    return [x for x, b in zip(token_level_seq, token_mask) if not b]

  def filter_whitespaces_dict(self, token_level_dict,
                              token_mask):
    return {
        k: self.filter_whitespaces(v, token_mask)
        for k, v in token_level_dict.items()
    }

  def pad(self, token_level_dict,
          value):

    def _pad(l):
      if self._max_seq_length:
        return l + list(
            np.full(self._max_seq_length - len(l), fill_value=value, dtype=int))
      return l

    return {k: _pad(v) for k, v in token_level_dict.items()}


class OneNoteIdPerRatedSubjectId(beam.DoFn):
  """Get a single note per patient if any notes of that patient are rated."""

  def process(self, element,
              seed):
    """Picks one note per subject.

    Given a GroupBy list of APData, returns a random note by a given priority:
    1. If any are rated, picks one at random.
    2. Return all notes.

    Supposedly rating is done appropriately and so only a single note fits
    condition 1.

    Args:
      element: Output of a GroupBy subject id, a tuple of subject_id and a list
        of APData.
      seed: Random seed for permuting the list.

    Yields:
      Values from the list according to the aforementioned rational.
    """
    subject_id, values = element
    # Values are kept as some serialized format,
    # iterating over them to unpack them:
    values = list(values)

    def get_note_ids(values):
      random.seed(seed + int(subject_id))
      random.shuffle(values)

      is_rated = [x.is_rated for x in values]
      if any(is_rated):
        return [values[np.argmax(is_rated)].note_id]

      return [x.note_id for x in values]

    if values:
      note_ids = set(get_note_ids(values))
      yield from [x for x in values if x.note_id in note_ids]


def generate_model_labels(
    labels,
    tokens):
  """Converts labeled character spans to token labels.

  Args:
    labels: Labels as character level labeled spans.
    tokens: Tokens of the text.

  Returns:
    Converted labels as as dict of {label name: token level labels}.
  """
  # Transform enum to BIO - each enum value maps to two except 0:
  # 0    1     2     3   ... n
  # 0    1 2   3 4   5 6 ... 2n-1 2n
  get_b = lambda val: (val * 2) - 1
  get_i = lambda val: val * 2

  fragment_type_labels = np.zeros(len(tokens), dtype=int)
  action_item_type_labels = np.zeros(len(tokens), dtype=int)
  for label in ap_parsing_utils.normalize_labeled_char_spans_iterable(
      labels, tokens):

    token_start, token_end = ap_parsing_utils.char_span_to_token_span(
        tokens, (label.start_char, label.end_char))

    # Fragment type labels:
    fragment_type_labels[token_start:token_end] = get_b(
        label.span_type)  # B-span
    if token_end > token_start + 1:  # More than 1 token
      fragment_type_labels[token_start + 1:token_end] = get_i(
          label.span_type)  # I-span

    # Action item type labels:
    if label.span_type == ap_parsing_lib.LabeledSpanType.ACTION_ITEM:
      action_item_type_labels[token_start:token_end] = label.action_item_type

  # No action items labels, mask all.
  if not np.any(action_item_type_labels):
    action_item_type_labels[:] = -1

  return {
      "fragment_type": fragment_type_labels.tolist(),
      "action_item_type": action_item_type_labels.tolist()
  }


def generate_token_features(tokens,
                            vocab):
  full_vocab = [*constants.RESERVED_TOKENS, *vocab]
  id_map = {x: i for i, x in enumerate(full_vocab)}

  return {
      "token_ids":
          list([
              id_map.get(token.token_text.lower(), full_vocab.index("[UNK]"))
              for token in tokens
          ]),
      "token_type": [token.token_type.value for token in tokens],
      "is_upper": [int(token.token_text.isupper()) for token in tokens],
      "is_title": [int(token.token_text.istitle()) for token in tokens],
  }


def convert_to_tf_examples(element,
                           debug_output = False):
  """Convert dict of values to tf examples.

  Args:
    element: Key value tuple where value is a dict of features.
    debug_output: Whether to output debug information in addition to model
      inputs in the tf examples.

  Returns:
    tf.Example of features and metadata.
  """

  def _byte_feature(values):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[x.encode("utf-8") for x in values]))

  def _int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

  _, ap_data = element
  features = {
      "note_id": _int_feature([int(ap_data.note_id)]),
      "char_offset": _int_feature([ap_data.char_offset]),
      "seq_length": _int_feature([len(ap_data.tokens)])
  }
  if debug_output:
    features.update({
        "augmentation_name": _byte_feature([ap_data.augmentation_name]),
        "partition": _byte_feature([ap_data.partition.name]),
        "tokens": _byte_feature([x.token_text for x in ap_data.tokens]),
        "ap_text": _byte_feature([ap_data.ap_text]),
    })
  for k, v in ap_data.token_features.items():
    features[k] = _int_feature(v)
  for k, v in ap_data.labels.items():
    features[k] = _int_feature(v)

  return tf.train.Example(features=tf.train.Features(feature=features))


def convert_and_save_tf_examples(features_and_labels,
                                 output_path,
                                 debug_output = False):
  """Beam PTransform taking features and labels and saving them as tf examples.

  Args:
    features_and_labels: PCollection of APData holding the features and labels.
    output_path: base folder for output, the function creates 3 sharded tfrecord
      files for the train, val and test sets.
    debug_output: whether to output debug information in addition to model
      inputs in the tf examples.
  """

  def _save_fn(pcoll, filename):
    _ = (
        pcoll
        | f"Reshuffle({filename})" >> beam.Reshuffle()
        | f"ToCSV({filename})" >>
        beam.Map(lambda x: "{},{}".format(x[1].subject_id, x[1].note_id))
        | f"SaveCSV({filename})" >> beam.io.WriteToText(
            os.path.join(output_path, filename + ".csv"),
            header="subject_id,note_id"))

    _ = (
        pcoll
        | f"ConvertToTFExamples({filename})" >> beam.Map(
            convert_to_tf_examples, debug_output=debug_output)
        | f"SaveTFRecords({filename})" >> beam.io.WriteToTFRecord(
            os.path.join(output_path, filename + ".tfrecord"),
            coder=beam.coders.ProtoCoder(tf.train.Example)))

  train_set, val_set, test_set = (
      features_and_labels
      | "SplitTrainValTest" >> beam.Partition(
          lambda x, n_part: max(x[1].partition.value - 1, 0), 3))

  _save_fn(train_set, "train_set")
  _save_fn(val_set, "val_set")
  _save_fn(test_set, "test_set")

  # Partition by rating and augmentation status and save stratification.
  def _split_by_status(element, n_part):
    del n_part
    _, ap_data = element
    return 2 * (ap_data.is_rated) + bool(ap_data.augmentation_name)

  by_status = (
      train_set | "SplitByStatus" >> beam.Partition(_split_by_status, 4))

  for i, rated_status in enumerate(["nonrated", "rated"]):
    for j, aug_status in enumerate(["nonaugmented", "augmented"]):
      stratus = f"train_{rated_status}_{aug_status}"
      _save_fn(by_status[i * 2 + j], stratus)


def extract_ap_sections(
    note_text,
    section_markers):
  """Extract AP sections from note.

  Args:
    note_text: Text to parse.
    section_markers: Dict[marker: str,[section_types: str]] of text markers
      denoting section starts.

  Yields:
    Section objects.
  """

  section_finder = note_section_lib.SectionFinder(section_markers)
  sections = section_finder.find_sections(note_text)
  for section in sections:
    if "assessment and plan" in section.section_types:
      yield section


def filter_inorganic(tokens,
                     threshold = 10):
  is_word = lambda token: token.token_type == tokenizer_lib.TokenType.WORD
  word_count = sum(map(is_word, tokens))
  caps_words_count = sum(
      map(lambda token: (is_word(token) and token.token_text.isupper()),
          tokens))
  return (word_count - caps_words_count) <= threshold


def filter_by_labels(element,
                     non_rated_threshold = 10):
  """Filters notes by their labeling status.

  All rated notes are kept, nonrated notes are kept if the regex parser
  identifies more labels than the specified threshold.

  Args:
    element: APData containing the labels.
    non_rated_threshold: Total number of labels required for nonrated notes.

  Returns:
    bool, should this note be kept.
  """

  values = element[1]
  n = len(values.labeled_char_spans)
  return (values.is_rated and n > 0) or (n > non_rated_threshold)


def process_rating_labels(
    rating_labels,
    section):
  """Converts rating labels to section based indexing, filtering out of bounds labels."""

  def adjust_labeled_char_span(
      labeled_char_spans,
      section
  ):
    # Check if labeled span is contained in section:
    for labeled_char_span in labeled_char_spans:
      if (labeled_char_span.start_char >= section.char_start and
          labeled_char_span.end_char <= section.char_end):
        new_labeled_char_span = offset_labeled_char_span(
            labeled_char_span, -section.char_start)
        yield new_labeled_char_span

  return list(adjust_labeled_char_span(rating_labels, section))


def offset_labeled_char_span(labeled_char_span,
                             offset):
  new_labeled_char_span = copy.deepcopy(labeled_char_span)
  new_labeled_char_span.start_char += offset
  new_labeled_char_span.end_char += offset
  return new_labeled_char_span


def annotate_ap(ap_text):
  """Runs AP annotator and returns clusters as LabeledSpans."""
  annotator = ap_problems_action_items_annotator.APProblemsActionItemsAnnotator(
  )
  clusters = annotator(ap_text)
  labeled_char_spans = list(
      itertools.chain(*[
          ap_problems_action_items_annotator
          .problem_cluster_to_labeled_char_spans(cluster)
          for cluster in clusters
      ]))
  return labeled_char_spans


def hash_key(k, random_seed):
  return int(
      hashlib.sha256(f"{random_seed}-{k}".encode("utf-8")).hexdigest(), 16)


def downsample(pcoll, n,
               random_seed):
  """Deterministic PCollection downsampling using sha256."""

  return (pcoll
          |
          "HashKey" >> beam.Map(lambda x: (str(hash_key(x[0], random_seed)), x)
                               ).with_output_types(Tuple[str, Tuple[str, Dict]])
          | "Downsample" >> beam.combiners.Top.Of(
              n, key=lambda x: x[0]).with_output_types(
                  List[Tuple[str, Tuple[str, Dict]]])
          | "Unpack" >> beam.FlatMap(lambda x: x)
          | "DropHashKey" >> beam.Values())


def read_vocab(path):
  return [
      line.rstrip(b"\n").decode("unicode-escape")
      for line in tf.io.gfile.GFile(path, "rb").readlines()
  ]


def read_csv_as_pcoll(pipeline, path):
  label = os.path.basename(path)
  raw_df = (pipeline | f"ReadCSV{label}" >> df_io.read_csv(path))
  return df_convert.to_pcollection(
      raw_df, pipeline=pipeline, label=f"ToPColl{label}")


def read_notes(pipeline,
               input_note_events):
  notes = read_csv_as_pcoll(pipeline, input_note_events)
  notes = (notes | "ConvertNote" >> beam.Map(convert_note))
  return notes


def read_raw_ratings(pipeline,
                     input_ratings):
  return read_csv_as_pcoll(pipeline, input_ratings)


def read_filter_notes(pipeline,
                      input_note_events):
  notes = read_notes(pipeline, input_note_events)
  return (
      notes
      | "FilterNotes" >> beam.Filter(lambda note: "physician" in note.category)
      | "AddKey" >> beam.Map(lambda note: (str(note.note_id), note)))


def convert_note(note_event):
  """Converts note events from CSV to Note object."""
  return Note(
      note_id=note_event.ROW_ID,
      subject_id=note_event.SUBJECT_ID,
      text=note_event.TEXT,
      category=note_event.CATEGORY.strip().lower())


def convert_ratings(
    csv_labeled_char_span):
  """Converts labeled spans from CSV to LabeledCharSpan object."""
  labeled_char_span = ap_parsing_lib.LabeledCharSpan(
      start_char=csv_labeled_char_span.char_start,
      end_char=csv_labeled_char_span.char_end,
      span_type=ap_parsing_lib.LabeledSpanType[csv_labeled_char_span.span_type])
  if csv_labeled_char_span.action_item_type:
    labeled_char_span.action_item_type = ap_parsing_lib.ActionItemType[
        csv_labeled_char_span.action_item_type]

  return (str(csv_labeled_char_span.note_id), labeled_char_span)
