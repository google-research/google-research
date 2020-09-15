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

# Lint as: python3
"""Utils for preprocessing the CFQ dataset."""

import collections
import json
import os
import re
import string
from typing import Any, Dict, List, Tuple

from absl import logging

from tensorflow.compat.v1.io import gfile

Dataset = Dict[str, List[Tuple[str, str]]]

_QUESTION_FIELD = 'questionPatternModEntities'
_QUERY_FIELD = 'sparqlPatternModEntities'


def _scrub_json(content):
  """Reduce JSON by filtering out only the fields of interest."""
  # Loading of json data with the standard Python library is very inefficient:
  # For the 4GB dataset file it requires more than 40GB of RAM and takes 3min.
  # There are more efficient libraries but in order to avoid additional
  # dependencies we use a simple (perhaps somewhat brittle) regexp to reduce
  # the content to only what is needed. This takes 1min to execute but
  # afterwards loading requires only 500MB or RAM and is done in 2s.
  regex = re.compile(
      r'("%s":\s*"[^"]*").*?("%s":\s*"[^"]*")' %
      (_QUESTION_FIELD, _QUERY_FIELD), re.DOTALL)
  return '[' + ','.join([
      '{' + m.group(1) + ',' + m.group(2) + '}' for m in regex.finditer(content)
  ]) + ']'


def load_json(path, scrub = False):
  logging.info('Reading json from %s into memory...', path)
  with gfile.GFile(path) as f:
    if scrub:
      data = json.loads(_scrub_json(f.read()))
    else:
      data = json.load(f)
  logging.info('Successfully loaded json data from %s into memory.', path)
  return data


def load_scan(path):
  """Read original scan task data and convert into CFQ-style json format."""
  logging.info('Reading SCAN tasks from %s.', path)

  def parse(infile):
    for line in infile.read().split('\n'):
      if not line.startswith('IN: '):
        continue
      commands, actions = line[len('IN: '):].strip().split(' OUT: ', 1)
      yield {_QUESTION_FIELD: commands, _QUERY_FIELD: actions}

  return list(parse(gfile.GFile(path)))


def load_dataset(path):
  """Load dataset from .json or SCAN task format."""
  if path[-5:] == '.json':
    return load_json(path, scrub=True)
  else:
    return load_scan(path)


def tokenize_punctuation(text):
  text = map(lambda c: ' %s ' % c if c in string.punctuation else c, text)
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
    logging.fatal('Invalid split: JSON should contain fields %s.', idx_names)
    return dataset
  for split_name, idx_name in zip(split_names, idx_names):
    logging.info(
        '  Retrieving %s (%s instances)', split_name, len(split[idx_name]))
    for idx in split[idx_name]:
      dataset[split_name].append(get_encode_decode_pair(samples[idx]))

  size_str = ', '.join('%s=%s' %(s, len(dataset[s])) for s in split_names)
  logging.info('Finished retrieving splits. Size: %s', size_str)
  return dataset


def write_dataset(dataset, save_path):
  """Saves the given dataset into the given location."""
  if not dataset:
    logging.info('No dataset to write.')
    return
  logging.info('Writing dataset to %s', save_path)
  for split_name, list_of_input_output_pairs in dataset.items():
    folder_name = os.path.join(save_path, split_name)
    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    encode_name = os.path.join(folder_name, '%s_encode.txt' % split_name)
    decode_name = os.path.join(folder_name, '%s_decode.txt' % split_name)
    with gfile.GFile(encode_name,
                     'w') as encode_f, gfile.GFile(decode_name,
                                                   'w') as decode_f:
      for pair in list_of_input_output_pairs:
        encode_f.write(pair[0] + '\n')
        decode_f.write(pair[1] + '\n')
  logging.info('Dataset written to %s', save_path)


def write_token_vocab(words,
                      save_path,
                      problem = 'cfq'):
  """"Writes token vocabulary from @words to @save_path."""
  # Sort tokens by frequency and then lexically to break ties.
  words_with_counts = words.most_common()
  words_with_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
  vocab_path = os.path.join(save_path, 'vocab.%s.tokens' % problem)

  with gfile.GFile(vocab_path, 'w') as f:
    # Tensor2tensor needs these additional tokens.
    f.write('<pad>\n<EOS>\n<OOV>\n')
    for word, _ in words_with_counts:
      f.write(word + '\n')
  logging.info('Token vocabulary written to %s (%s distinct tokens).',
               vocab_path, len(words))


def get_lines(path, filename):
  with gfile.GFile(os.path.join(path, 'train', filename)) as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
  return lines


def get_token_vocab(path):
  words = collections.Counter()
  lines = get_lines(path, 'train_encode.txt')
  lines.extend(get_lines(path, 'train_decode.txt'))
  for line in lines:
    words.update(line.split(' '))
  return words
