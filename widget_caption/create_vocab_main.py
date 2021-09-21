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

# Lint as: python3
"""Create word vocab for widget captioning model."""

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('token_count_file', None, 'Beam pipeline output file path.')
flags.DEFINE_integer('min_word_frequency', 0, 'Minimum word frequency.')
flags.DEFINE_string('output_word_vocab_file', None,
                    'Output word vocab file path.')

# Common tokens used by the word vocab.
_COMMON_WORDS = ('<PADDING>', '<EOS>', '<UNK>', '<START>')


def _create_word_vocab(token_count_file, min_word_frequency, output_vocab_file):
  """Creates word vocab from beam pipeline output."""
  tokens = []
  with tf.io.gfile.GFile(token_count_file) as f:
    for line in f:
      text_type, text, count = line.strip().split('\t')
      # Only uses tokens with `token` annotation from beam pipeline output.
      if text_type != 'token':
        continue
      tokens.append((text, int(count)))

  # Remove tokens with frequency lower than min_word_frequency.
  tokens = [t for t in tokens if t[1] >= min_word_frequency]

  # Sort tokens and write to the file.
  tokens = sorted(tokens, key=lambda kv: kv[1], reverse=True)
  with tf.io.gfile.GFile(output_vocab_file, 'w') as f:
    for word in _COMMON_WORDS:
      f.write(word + '\n')
    for token, _ in tokens:
      f.write(token + '\n')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  _create_word_vocab(FLAGS.token_count_file, FLAGS.min_word_frequency,
                     FLAGS.output_word_vocab_file)


if __name__ == '__main__':
  app.run(main)
