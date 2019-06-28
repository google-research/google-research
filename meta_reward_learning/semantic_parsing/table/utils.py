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

"""Utility functions."""
import re
from babel import numbers
import numpy as np
from six import string_types, text_type
from meta_reward_learning.semantic_parsing.table.wtq \
import evaluator

KEYS_TO_REMOVE = {
    'wtq':
        set([u'nt-1956', u'nt-6492', u'nt-7165', u'nt-10189']),
    'wikisql':
        set([
            u'train-10537', u'train-10540', u'train-25185', u'train-25186',
            u'dev-3756', u'test-14330'
        ])
}


def average_token_embedding(tks, model, embedding_size=300):
  arrays = []
  for tk in tks:
    if tk in model:
      arrays.append(model[tk])
    else:
      arrays.append(np.zeros(embedding_size))
  return np.average(np.vstack(arrays), axis=0)


def get_embedding_for_constant(value, model, embedding_size=300):
  if isinstance(value, list):
    # Use zero embeddings for values from the question to
    # avoid overfitting
    return np.zeros(embedding_size)
  elif value[:2] == 'r.':
    value_split = value.split('-')
    type_str = value_split[-1]
    type_embedding = average_token_embedding([type_str], model)
    value_split = value_split[:-1]
    value = '-'.join(value_split)
    raw_tks = value[2:].split('_')
    tks = []
    for tk in raw_tks:
      valid_tks = find_tk_in_model(tk, model)
      tks += valid_tks
    val_embedding = average_token_embedding(tks or raw_tks, model)
    return (val_embedding + type_embedding) / 2
  else:
    raise NotImplementedError('Unexpected value: {}'.format(value))


def find_tk_in_model(tk, model):
  special_tk_dict = {'-lrb-': '(', '-rrb-': ')'}
  if tk in model:
    return [tk]
  elif tk in special_tk_dict:
    return [special_tk_dict[tk]]
  elif tk.upper() in model:
    return [tk.upper()]
  elif tk[:1].upper() + tk[1:] in model:
    return [tk[:1].upper() + tk[1:]]
  elif re.search('\\/', tk):
    tks = tk.split('\\\\/')
    if len(tks) == 1:
      return []
    valid_tks = []
    for tk in tks:
      valid_tk = find_tk_in_model(tk, model)
      if valid_tk:
        valid_tks += valid_tk
    return valid_tks
  else:
    return []


# WikiSQL evaluation utility functions.
def wikisql_normalize(val):
  """Normalize the val for wikisql experiments."""
  if (isinstance(val, float) or isinstance(val, int)):
    return val
  elif isinstance(val, string_types):
    try:
      val = numbers.parse_decimal(val, locale='en_US')
    except numbers.NumberFormatError:
      val = val.lower()
    return val
  else:
    return None


def wikisql_process_answer(answer):
  processed_answer = []
  for a in answer:
    normalized_val = wikisql_normalize(a)
    # Ignore None value and normalize the rest, keep the
    # order.
    if normalized_val is not None:
      processed_answer.append(normalized_val)
  return processed_answer


def wikisql_score(prediction, answer):
  prediction = wikisql_process_answer(prediction)
  if prediction == answer:
    return 1.0
  else:
    return 0.0


# WikiTableQuestions evaluation function.
def wtq_score(prediction, answer):
  processed_answer = evaluator.target_values_map(*answer)
  correct = evaluator.check_prediction([text_type(p) for p in prediction],
                                       processed_answer)
  if correct:
    return 1.0
  else:
    return 0.0


def remove_long_contexts(dataset):
  """Removes the context with large number of tokens in the question."""
  # This was done due to the practical reason of reducing the memory
  # consumption of the unrolled graph to fit into GPU
  # Hack to check whether the dataset was wikisql or wikitable
  keys_to_remove = []
  name = dataset[0]['id']
  if name.startswith('train-') or name.startswith('dev-') or name.startswith(
      'test-'):
    keys_to_remove = KEYS_TO_REMOVE['wikisql']
  elif name.startswith('nt-') or name.startswith('nu-'):
    keys_to_remove = KEYS_TO_REMOVE['wtq']
  new_dataset = [d for d in dataset if d['id'] not in keys_to_remove]
  return new_dataset
