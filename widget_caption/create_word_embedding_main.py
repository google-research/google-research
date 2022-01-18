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
"""Generate word embeddings using glove model."""
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf  # tf
import tensorflow_hub as hub

FLAGS = flags.FLAGS

flags.DEFINE_string('word_vocab_file', None, 'Word vocab file path.')
flags.DEFINE_string('output_embedding_file', None,
                    'Output embedding file path.')


def _create_embedding(word_vocab_file, output_embedding_file):
  """Creates word embedding for words in the vocab file."""
  embeddings = []
  word_count = 0
  oov_count = 0
  with tf.Graph().as_default():
    embed = hub.Module('@tf-text/glove_embeddings/wiki_gigaword/6b_300d/1')
    word_input = tf.placeholder(dtype=tf.string)
    output = embed(word_input)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())

      with tf.io.gfile.GFile(word_vocab_file) as f:
        for line in f:
          word_count += 1
          word = line.strip()
          # Use all 0 for the padding token.
          if word == '<PADDING>':
            embeddings.append((word, [0] * 300))
            continue
          # Run the glove model to get word embedding.
          embedding = sess.run(output, feed_dict={word_input: [word]})[0]

          # For OOV word, glove model return all 0 values.
          if sum(embedding) == 0:
            # For each OOV word, randomly generate a vector for it.
            embedding = np.random.uniform(0, 0.1, 300)
            oov_count += 1

          embeddings.append((word, embedding))

  logging.info('Word count %s, OOV count %s', word_count, oov_count)

  # Write word embedding to output file.
  with tf.io.gfile.GFile(output_embedding_file, 'w') as f:
    for word, embedding in embeddings:
      embedding = [str(v) for v in embedding]
      embedding_text = ' '.join(embedding)
      f.write('{} {}\n'.format(word, embedding_text))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  _create_embedding(FLAGS.word_vocab_file, FLAGS.output_embedding_file)


if __name__ == '__main__':
  app.run(main)
