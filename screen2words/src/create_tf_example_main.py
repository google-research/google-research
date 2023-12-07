# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Create tf.Example and vocab for screen2words model."""
import json
import csv
import re
from typing import Callable, List, Generator, Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam import runners
from apache_beam.options.pipeline_options import PipelineOptions
import nltk
import tensorflow as tf

from src import create_tf_example_fn

FLAGS = flags.FLAGS

flags.DEFINE_string('task', 'CREATE_VOCAB',
                    'Task name, could be CREATE_VOCAB or CREATE_TF_EXAMPLE.')
flags.DEFINE_string('dataset_paths', None,
                    'List of dataset paths, separated by comma.')
flags.DEFINE_string('screen_summaries_path', None, 'Label file path with ScreenId to MTurk mappings.')
flags.DEFINE_string('word_vocab_path', None, 'Word vocab file path.')
flags.DEFINE_integer('max_token_per_label', 10,
                     'Max amount of tokens each label could have.')
flags.DEFINE_integer('max_label_per_screen', 5,
                     'Max amount of labels each screen has.')
flags.DEFINE_string('output_vocab_path', '/tmp/word_vocab.txt',
                    'Output vocab file path.')
flags.DEFINE_string('output_tfexample_path', None, 'Path to output tf.Example.')

BBOX_MAX_W = 360
BBOX_MAX_H = 640


def _generate_screen_id_and_captions_pair(file_path):
  """Generates pair of screen id and MTurk labels for each screen."""
  with tf.io.gfile.GFile(file_path) as f:
    if file_path.endswith('.csv'):
      csv_content = csv.reader(f, delimiter=',')
      header = next(csv_content) # remove header
      screens = {row[0]:row[1] for row in csv_content}
    else:
      screens = json.load(f)
  return list(screens.items())


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


def caption_tokenizer():
  """Creates a tokenizer for screen summary with default configuration.

  Returns:
    A tokenizer with configuration for screen summary.
  """
  return Tokenizer(
      lowercase_text=True,
      remove_punctuation=True,
      remove_nonascii_character=True,
      max_token_length=30)


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
    """Tokenize text into a list of tokens.

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


class CreateTokenFn(beam.DoFn):
  """Reads a view hierarchy json file and yields tokens."""

  def __init__(self, dataset_path):
    """Constructor."""
    self._screen_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'screen')
    self.dataset_path = dataset_path

  def start_bundle(self):
    # Creates tokenizer used by the model.
    self._tokenizer = caption_tokenizer()

  def process(self, labels):
    """Emits tokens and phrases.

    Args:
      labels: A pair of <screen id, mturk labels>. Labels including captions and
        labeller-annotated attention bbx.

    Yields:
      Tokens and phrases.
    """
    self._screen_counter.inc(1)
    screen_id, mtruk_labels = labels
    logging.debug(
      f"Processing screen {screen_id} with label '{f'{mtruk_labels[:10]}...' if len(mtruk_labels) > 10 else mtruk_labels}'")
    json_path = self.dataset_path + screen_id + '.json'
    for text in create_tf_example_fn.extract_token(json_path, screen_id,
                                                   mtruk_labels,
                                                   self._tokenizer):
      yield text.encode()


class CreateTFExampleFn(beam.DoFn):
  """Reads view hierarchy json and image and yields tf.Example."""

  def __init__(self, dataset_path, word_vocab_path, max_token_per_label,
               max_label_per_screen):
    """Constructor.

    Args:
      dataset_path: Path to rico dataset.
      word_vocab_path: Path to word vocab.
      max_token_per_label: Max tokens for each caption/label.
      max_label_per_screen: Max captions/labels for each screen.
    """
    self._screen_counter = beam.metrics.Metrics.counter(self.__class__,
                                                        'screen')
    self._example_counter = beam.metrics.Metrics.counter(
        self.__class__, 'example')
    self._word_vocab_path = word_vocab_path
    self._max_token_per_label = max_token_per_label
    self._max_label_per_screen = max_label_per_screen
    self._dataset_path = dataset_path

  def start_bundle(self):
    self._word_vocab = {}

    # Initialize word/phrase vocab and phrase type mapping.
    with tf.io.gfile.GFile(self._word_vocab_path) as f:
      for index, word in enumerate(f):
        word = word.strip()
        self._word_vocab[word] = index

    self._tokenizer = caption_tokenizer()

  def process(self, labels):
    """Emits serialized tf.Example proto.

    Args:
      labels: A pair of <screen id, mturk labels>. Labels including captions and
        labeller-annotated attention bbx.

    Yields:
      A serialized tf.Example.
    """
    self._screen_counter.inc(1)

    screen_id, mturk_labels = labels
    prefix = self._dataset_path + screen_id

    example = create_tf_example_fn.create_tf_example(prefix, mturk_labels,
                                                     self._tokenizer,
                                                     self._word_vocab,
                                                     self._max_token_per_label,
                                                     self._max_label_per_screen)
    if not example:
      return

    self._example_counter.inc(1)
    yield example


def create_pipeline(task, dataset_path, screen_summaries_path,
                    word_vocab_path, max_token_per_label,
                    max_label_per_screen, output_vocab_path,
                    output_tfexample_path):

  if not tf.io.gfile.isdir(dataset_path):
    raise Exception(f'Folder for dataset_path="{dataset_path}" does not exist!')

  if not tf.io.gfile.exists(screen_summaries_path):
    raise Exception(f'File for screen_summaries_path="{screen_summaries_path}" does not exist!')

  if word_vocab_path is None:
    raise Exception(f'Path for word_vocab_path="{word_vocab_path}" is not set!')

  if output_tfexample_path is None:
    raise Exception(f'Path for output_tfexample_path="{output_tfexample_path}" is not set!')

  """Runs the end-to-end beam pipeline."""

  # Get file prefix and MTurk labels for each screen.
  merged = _generate_screen_id_and_captions_pair(screen_summaries_path)

  with beam.Pipeline() as p:
    """Pipeline for vocab generation ."""
    vocab_pipeline = (
      p | 'CreateCollection' >> beam.Create(merged)
        | 'CreateToken' >> beam.ParDo(CreateTokenFn(dataset_path))
        | 'CountTokens' >> beam.combiners.Count.PerElement()
        | 'FormatCount' >>
        beam.Map(lambda kv: '{}\t{}'.format(kv[0].decode(), kv[1]))
        | 'WriteToFile' >> beam.io.WriteToText(output_vocab_path))

  with beam.Pipeline() as p:
    """Pipeline for tf.Example generation."""
    tf_example_pipeline = (
      p | 'CreateCollection' >> beam.Create(merged)
        | 'GenerateTFExample' >> beam.ParDo(
            CreateTFExampleFn(dataset_path, word_vocab_path,
                              max_token_per_label, max_label_per_screen))
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
                             FLAGS.screen_summaries_path, FLAGS.word_vocab_path,
                             FLAGS.max_token_per_label,
                             FLAGS.max_label_per_screen,
                             FLAGS.output_vocab_path,
                             FLAGS.output_tfexample_path)

  runners.DataflowRunner().run_pipeline(pipeline, PipelineOptions())


if __name__ == '__main__':
  app.run(main)
