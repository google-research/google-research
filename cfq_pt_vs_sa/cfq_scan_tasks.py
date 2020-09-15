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
"""CFQ and SCAN tasks for T5."""
import t5.data
from t5.data import postprocessors as t5_postprocessors
from t5.evaluation import metrics as t5_metrics
import tensorflow as tf
import tensorflow_datasets as tfds

from cfq import preprocess

TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask


def cfq_preprocess(dataset):
  """Select input/target features and add prefix to input."""

  def cfq_map(sample):
    inputs = preprocess.tokenize_punctuation(sample['question']).split()
    targets = preprocess.preprocess_sparql(sample['query']).split()
    return {
        'inputs':
            tf.strings.join(['semanticparse: ', inputs], ' '),
        'targets':
            tf.strings.join(targets, ' ')
    }

  return dataset.map(cfq_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)


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
