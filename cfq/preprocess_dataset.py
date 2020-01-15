# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Preprocesses a specific split of the CFQ dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import string
from typing import Any, Dict, List, Text, Tuple

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', None, 'Path to the JSON file containing '
                    'the dataset.')

flags.DEFINE_string('split_path', None, 'Path to the JSON file containing '
                    'split information.')

flags.DEFINE_string('save_path', None, 'Path to the directory where to '
                    'save the files to.')

flags.mark_flag_as_required('save_path')

flags.register_validator('dataset_path', os.path.exists, 'Dataset not found.')
flags.register_validator('split_path', os.path.exists, 'Split not found.')

Dataset = Dict[Text, List[Tuple[Text, Text]]]


def load_json(path):
  logging.info(f'Reading json from {path} into memory...')
  with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  logging.info(f'Successfully loaded json data from {path} into memory.')
  return data


def tokenize_punctuation(text):
  text = map(lambda c: f' {c} ' if c in string.punctuation else c, text)
  return ' '.join(''.join(text).split())


def preprocess_sparql(query):
  """Do various preprocessing on the SPARQL query."""
  # Tokenize braces.
  query = query.replace('count(*)', 'count ( * )')

  tokens = []
  for token in query.split():
    # Replace 'ns:' prefixes.
    if token.startswith('ns:'):
      token = token[3:]
    # Replace mid prefixes.
    if token.startswith('m.'):
      token = 'm_' + token[2:]
    tokens.append(token)

  return ' '.join(tokens).replace('\\n', ' ')


def get_encode_decode_pair(sample):
  # Apply some simple preprocessing on the tokenizaton, which improves the
  # performance of the models significantly.
  encode_text = tokenize_punctuation(sample['questionPatternModEntities'])
  decode_text = preprocess_sparql(sample['sparqlPatternModEntities'])
  return (encode_text, decode_text)


def get_dataset(samples, split):
  """Creates a dataset by taking @split from @samples."""
  logging.info('Retrieving splits...')
  split_names = ['train', 'dev', 'test']
  idx_names = [f'{s}Idxs' for s in split_names]
  dataset = collections.defaultdict(list)
  if not set(idx_names) <= split.keys():
    logging.fatal(f'Invalid split: JSON should contain fields {idx_names}.')
    return dataset
  for split_name, idx_name in zip(split_names, idx_names):
    logging.info(
        f'  Retrieving {split_name} ({len(split[idx_name])} instances)')
    for idx in split[idx_name]:
      dataset[split_name].append(get_encode_decode_pair(samples[idx]))

  size_str = ', '.join(f'{s}={len(dataset[s])}' for s in split_names)
  logging.info(f'Finished retrieving splits. Size: {size_str}')
  return dataset


def write_dataset(dataset, save_path):
  """Saves the given dataset into the given location."""
  if not dataset:
    logging.info('No dataset to write.')
    return
  logging.info(f'Writing dataset to {save_path}')
  for split_name, list_of_input_output_pairs in dataset.items():
    folder_name = os.path.join(save_path, split_name)
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    encode_name = os.path.join(folder_name, f'{split_name}_encode.txt')
    decode_name = os.path.join(folder_name, f'{split_name}_decode.txt')
    with open(
        encode_name, 'w', encoding='utf8') as encode_f, open(
            decode_name, 'w', encoding='utf8') as decode_f:
      for pair in list_of_input_output_pairs:
        encode_f.write(pair[0] + '\n')
        decode_f.write(pair[1] + '\n')
  logging.info(f'Dataset written to {save_path}')


def write_token_vocab(words, save_path):
  """"Writes token vocabulary from @words to @save_path."""
  # Sort tokens by frequency and then lexically to break ties.
  words_with_counts = words.most_common()
  words_with_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
  vocab_path = os.path.join(save_path, 'vocab.cfq.tokens')

  with open(vocab_path, 'w') as f:
    # Tensor2tensor needs these additional tokens.
    f.write('<pad>\n<EOS>\n<OOV>\n')
    for word, _ in words_with_counts:
      f.write(f'{word}\n')
  logging.info(f'Token vocabulary written to {vocab_path} ({len(words)} '
               'distinct tokens).')


def get_lines(path, filename):
  with open(os.path.join(path, 'train', filename), 'r') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
  return lines


def get_token_vocab(path):
  words = collections.Counter()
  lines = get_lines(path, 'train_encode.txt')
  lines.extend(get_lines(path, 'train_decode.txt'))
  for line in lines:
    words.update(line.split(' '))
  return words


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  write_dataset(
      get_dataset(load_json(FLAGS.dataset_path), load_json(FLAGS.split_path)),
      FLAGS.save_path)
  write_token_vocab(get_token_vocab(FLAGS.save_path), FLAGS.save_path)


if __name__ == '__main__':
  app.run(main)
