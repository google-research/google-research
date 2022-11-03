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

"""CFQ and SCAN tasks for T5."""
import string

import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf
import tensorflow_datasets as tfds

TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask


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


def cfq_preprocess(dataset):
  """Select input/target features and add prefix to input."""

  def compute_inputs_and_targets(inputs, targets):
    inputs = tf.compat.as_text(inputs.numpy())
    inputs = tokenize_punctuation(inputs)
    targets = tf.compat.as_text(targets.numpy())
    targets = preprocess_sparql(targets)

    return inputs, targets

  def map_fn(x):
    inputs, targets = tf.py_function(
        compute_inputs_and_targets,
        inp=[x['question'], x['query']],
        Tout=[tf.string, tf.string])
    return {
        # The reshape is necessary as otherwise the tensor has unknown rank.
        'inputs': tf.reshape(inputs, shape=[]),
        'targets': tf.reshape(targets, shape=[])
    }

  return dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


for split in ['mcd1', 'mcd2', 'mcd3', '2m', 'random']:
  TaskRegistry.add(
      f'cfq_{split}',
      TfdsTask,
      tfds_name=f'cfq/{split}:1.2.0',
      text_preprocessor=cfq_preprocess,
      postprocess_fn=t5_postprocessors.lower_text,
      metric_fns=[t5_metrics.sequence_accuracy])


def scan_preprocess(dataset):
  """Select input/target features and add prefix to input."""

  def scan_map(sample):
    return {
        'inputs':
            tf.strings.join(['executescancommand:', sample['commands']], ' '),
        'targets':
            sample['actions']
    }

  return dataset.map(scan_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


for split in tfds.builder('scan').builder_configs.keys():
  TaskRegistry.add(
      f'scan_{split}',
      TfdsTask,
      tfds_name=f'scan/{split}:1.1.1',
      text_preprocessor=scan_preprocess,
      postprocess_fn=t5_postprocessors.lower_text,
      metric_fns=[t5_metrics.sequence_accuracy])
