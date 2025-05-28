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

"""Canary-related tasks with public NLP datasets.

Add mixture of tasks with canary data and NLP datasets.
"""

import functools
from typing import Dict

import seqio
from t5.evaluation import metrics
import tensorflow as tf

from learn_to_forget.transformers import constants


@seqio.map_over_dataset
def example_sparse_to_dense(
    example):
  """Converts all sparse features into dense ones."""
  example = dict(example)
  for key in list(example):
    value = example[key]
    if isinstance(value, tf.SparseTensor):
      example[key] = tf.sparse.to_dense(value)
  return example


def add_wmt_canary_tasks(
    clean_task_name, file_patterns
):
  """Adds wmt tasks to the registry and mixes them with the task.

  Args:
    clean_task_name: the name of the task to add canaries to.
    file_patterns: the file pattern corresponding to canaries.
  """
  task_names = []

  for repeats in file_patterns:
    canary_task_name = 'wmt_obliterate_dataset_{}:1.0.0'.format(repeats)
    task_names.append(canary_task_name)
    seqio.TaskRegistry.add(
        name=canary_task_name,
        source=seqio.TFExampleDataSource(
            reader_cls=tf.data.TFRecordDataset,
            split_to_filepattern={
                'train': constants.WMT_TFDS_DATA_PATTERN.format(repeats),
                'validation': constants.WMT_TFDS_DATA_PATTERN.format(repeats),
                'test': constants.WMT_TFDS_TEST_DATA_PATTERN.format(repeats),
            },
            feature_description={
                'inputs': tf.io.FixedLenFeature([], dtype=tf.string),
                'targets': tf.io.FixedLenFeature([], dtype=tf.string),
            },
            # This value should match the number of sampled examples
            # (see `prepare_in_distribution.py`, line 25)
            # TODO(teobaluta) add a better way of parsing this value rather
            # than hard-coding it.
            num_input_examples={
                'train': 512 * repeats,
                'validation': 512,
                'test': 512,
            }),
        preprocessors=[
            example_sparse_to_dense,
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[
            metrics.bleu, metrics.sequence_accuracy, metrics.edit_distance
        ],
        output_features=constants.OUTPUT_FEATURES)

  # The canary datasets are constructed such that each split already has
  # the repeats as per the config.
  # Compute the ratio such that we use the number of repetitions in the
  tasks = [clean_task_name]

  for task_name in task_names:
    tasks.append(task_name)

  seqio.MixtureRegistry.add(
      name='wmt_t2t_de_en_v003-wmt_canary',
      tasks=tasks,
      default_rate=functools.partial(
          seqio.mixing_rate_num_examples, temperature=1.0))


# Add the tasks corresponding to the validation set
# These correspond to the number of repetitions when generating the dataset
# (see `prepare_in_distribution.py`, line 26)
# TODO(teobaluta) add a better way of parsing this value rather
# than hard-coding it.
wmt_file_patterns = [1, 10, 100]
add_wmt_canary_tasks('wmt_t2t_de_en_v003', wmt_file_patterns)
