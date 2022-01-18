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

# Lint as: python3
"""Create tf.Example and vocab for widget captioning model."""

import csv
import os
import re
from typing import Any, Callable, Dict, List, Generator

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam import runners
import nltk
import six
import tensorflow as tf

from widget_caption import create_tf_example_fn

FLAGS = flags.FLAGS

flags.DEFINE_string('task', None,
                    'Task name, could be CREATE_VOCAB or CREATE_TF_EXAMPLE.')
flags.DEFINE_string('dataset_paths', None,
                    'List of dataset paths, separated by comma.')
flags.DEFINE_string('csv_file_path', None, 'CSV label file path.')
flags.DEFINE_string('word_vocab_path', None, 'Word vocab file path.')
flags.DEFINE_integer('max_token_per_label', 10, 'Max tokens per caption.')
flags.DEFINE_integer('max_label_per_node', 4, 'Max captions per UI node.')
flags.DEFINE_string('output_vocab_path', None, 'Output vocab file path.')
flags.DEFINE_string('output_tfexample_path', None, 'Path to output tf.Example.')


def caption_tokenizer():
  """Creates a tokenizer for widget captioning with default configuration.

  This tokenizer is used by the widget captioning project in various places. So
  we hardcode the configuration here to ensure consistency.

  Returns:
    A tokenizer with configuration for widget captioning project.
  """
  return Tokenizer(
      lowercase_text=True,
      remove_punctuation=True,
      remove_nonascii_character=True,
      max_token_length=30)


def _get_ascii_token(token):
  """Removes non-ASCII characters in the token."""
  chars = []
  for char in token:
    # Try to encode the character with ASCII encoding. If there is an encoding
    # error, it's not an ASCII character and can be skipped.
    try:
      char.encode('ascii')
    except UnicodeEncodeError:
      continue
    chars.append(char)

  return ''.join(chars)


class Tokenizer(object):
  """Tokenizer using NLTK with a few additional options."""

  # Pattern for recognizing non-punctuation words.
  _ALPHANUMERIC_PATTERN = re.compile(r'[a-zA-Z0-9]')

  def __init__(self,
               lowercase_text = False,
               remove_punctuation = False,
               remove_nonascii_character = False,
               max_token_length = -1):
    """Constructor.

    Args:
      lowercase_text: If True, convert text to lower case before tokenization.
      remove_punctuation: If True, remove punctuation in the tokens.
      remove_nonascii_character: If True, remove non-ascii characters within a
        token.
      max_token_length: Remove tokens with length larger than this value if it's
        positive.
    """
    self._lowercase_text = lowercase_text
    self._remove_punctuation = remove_punctuation
    self._max_token_length = max_token_length
    self._remove_nonascii_character = remove_nonascii_character

  def tokenize(self, text):
    """Toeknize text into a list of tokens.

    Args:
      text: Input text.

    Returns:
      A list of tokens.
    """
    text = text.strip()

    # Lowercase and tokenize text.
    if self._lowercase_text:
      text = text.lower()

    tokens = nltk.word_tokenize(text)

    # Remove punctuation.
    if self._remove_punctuation:
      tokens = [t for t in tokens if self._ALPHANUMERIC_PATTERN.search(t)]

    # Remove non-ASICII characters within the tokens.
    if self._remove_nonascii_character:
      tokens = [_get_ascii_token(t) for t in tokens]
      tokens = [t for t in tokens if t]

    # Remove long tokens.
    if self._max_token_length > 0:
      tokens = [t for t in tokens if len(t) <= self._max_token_length]
    return tokens


def _get_filepath_prefix(dataset_path):
  """Generates file prefixes from a dataset path."""
  # Get all the json files under the dataset folder.
  json_pattern = os.path.join(dataset_path, '*.json')
  prefixes = []
  for file in tf.io.gfile.glob(json_pattern):
    prefix = file[:-5]
    # We assume the file name without suffix is the screen id.
    screen_id = os.path.basename(prefix)
    prefixes.append((screen_id, prefix))
  return prefixes


def _get_csv_labels(csv_file_path):
  """Creates a {screenId: {nodeId: [captions]}} mapping."""
  labels = {}
  with tf.io.gfile.GFile(csv_file_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      captions = row['captions'].split('|')
      if row['screenId'] not in labels:
        labels[row['screenId']] = {}
      labels[row['screenId']][row['nodeId']] = captions

  return labels


def _merge_prefix_and_labels(prefixes, labels):
  """Merges file prefixes and mturk labels into a dict with screen id as key."""
  screen_id_to_labels = {}

  for screen_id, captions in labels.items():
    if screen_id not in screen_id_to_labels:
      screen_id_to_labels[screen_id] = {}
    screen_id_to_labels[screen_id]['labels'] = captions

  for screen_id, prefix in prefixes:
    if screen_id in screen_id_to_labels:
      screen_id_to_labels[screen_id]['prefix'] = prefix

  return screen_id_to_labels.values()


def _generate_merged_prefix_and_labels(dataset_paths, csv_file_path):
  """Generates file prefix and MTurk labels for each screen."""
  # Get file prefixes.
  prefixes = []
  for path in dataset_paths:
    prefixes += _get_filepath_prefix(path)

  # Get MTurk labels.
  labels = _get_csv_labels(csv_file_path)

  # Merge the above two results into a dict keyed by screen id.
  merged = _merge_prefix_and_labels(prefixes, labels)
  return merged


class CreateTokenFn(beam.DoFn):
  """Reads a view hierarchy json file and yields tokens."""

  def __init__(self):
    """Constructor."""
    self._screen_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'screen')

  def start_bundle(self):
    # Creates tokenizer used by the captioning model.
    self._tokenizer = caption_tokenizer()

  def process(self, screen_info):
    """Emits tokens.

    Args:
      screen_info: A dict containing screen information.

    Yields:
      Tokens.
    """
    self._screen_counter.inc(1)
    prefix = screen_info['prefix']
    labels = screen_info['labels']
    json_path = prefix + '.json'
    for text in create_tf_example_fn.extract_token(json_path, labels,
                                                   self._tokenizer):
      yield text


class CreateTFExampleFn(beam.DoFn):
  """Reads view hierarchy json and image and yields tf.Example."""

  def __init__(self, word_vocab_path, max_token_per_label, max_label_per_node):
    """Constructor.

    Args:
      word_vocab_path: Path to word vocab.
      max_token_per_label: Max tokens for each caption/label.
      max_label_per_node: Max captions/labels for each UI element.
    """
    self._screen_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'screen')
    self._example_counter = beam.metrics.Metrics.counter(
        self.__class__, 'example')
    self._word_vocab_path = word_vocab_path
    self._max_token_per_label = max_token_per_label
    self._max_label_per_node = max_label_per_node

  def start_bundle(self):
    self._word_vocab = {}

    # Initialize word vocab.
    with tf.io.gfile.GFile(self._word_vocab_path) as f:
      for index, word in enumerate(f):
        word = word.strip()
        self._word_vocab[word] = index

    self._tokenizer = caption_tokenizer()

  def process(self, screen_info):
    """Emits serialized tf.Example proto.

    Args:
      screen_info: A dict containing screen information.

    Yields:
      A serizlied tf.Example.
    """
    self._screen_counter.inc(1)
    prefix = screen_info['prefix']
    labels = screen_info['labels']
    screen_id = prefix.split('/')[-1]
    example = create_tf_example_fn.create_tf_example(prefix, labels,
                                                     self._tokenizer,
                                                     self._word_vocab,
                                                     self._max_token_per_label,
                                                     self._max_label_per_node)
    if not example:
      return

    self._example_counter.inc(1)
    del screen_id
    yield example


class CreateRawTextFn(beam.DoFn):
  """Reads a view hierarchy json file and MTurk labels and yields raw text."""

  def __init__(self, vocab_tokenizer):
    """Constructor."""
    self._screen_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'screen')
    self._tokenizer = vocab_tokenizer

  def process(self, screen_info):
    """Emits raw text.

    Args:
      screen_info: A dict containing screen information.

    Yields:
      Raw text.
    """
    self._screen_counter.inc(1)
    prefix = screen_info['prefix']
    labels = screen_info['labels']
    json_path = prefix + '.json'
    for text in create_tf_example_fn.extract_raw_text(json_path, labels):
      for token in self._tokenizer.tokenize(text):
        yield token


def create_pipeline(task, dataset_paths, csv_file_path,
                    word_vocab_path, max_token_per_label,
                    max_label_per_node, output_vocab_path,
                    output_tfexample_path):
  """Runs the end-to-end beam pipeline."""

  # Get file prefix and MTurk labels for each screen.
  merged = _generate_merged_prefix_and_labels(
      dataset_paths.split(','), csv_file_path)

  def vocab_pipeline(root):
    """Pipeline for vocab generation ."""
    _ = (
        root | 'CreateCollection' >> beam.Create(merged)
        | 'CreateToken' >> beam.ParDo(CreateTokenFn())
        | 'CountTokens' >> beam.combiners.Count.PerElement()
        | 'FormatCount' >> beam.Map(lambda kv: '{}\t{}'.format(kv[0], kv[1]))
        | 'WriteToFile' >> beam.io.WriteToText(output_vocab_path))

  def tf_example_pipeline(root):
    """Pipeline for tf.Example generation."""
    _ = (
        root | 'CreateCollection' >> beam.Create(merged)
        | 'GenerateTFExample' >> beam.ParDo(
            CreateTFExampleFn(word_vocab_path, max_token_per_label,
                              max_label_per_node))
        | 'WriteToFile' >> beam.io.WriteToTFRecord(
            output_tfexample_path,
            coder=beam.coders.ProtoCoder(tf.train.Example)))

  if task == 'CREATE_VOCAB':
    return vocab_pipeline
  elif task == 'CREATE_TF_EXAMPLE':
    return tf_example_pipeline
  else:
    raise ValueError('Task must be CREATE_VOCAB or CREATE_TF_EXAMPLE.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  pipeline = create_pipeline(FLAGS.task, FLAGS.dataset_paths,
                             FLAGS.csv_file_path, FLAGS.word_vocab_path,
                             FLAGS.max_token_per_label,
                             FLAGS.max_label_per_node, FLAGS.output_vocab_path,
                             FLAGS.output_tfexample_path)
  runners.DataflowRunner().run_pipeline(pipeline)


if __name__ == '__main__':
  app.run(main)
