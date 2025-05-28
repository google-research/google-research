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

import seqio
from t5.data import preprocessors
from t5.evaluation import metrics
import tensorflow as tf

from learn_to_forget.transformers import constants


def add_canary_tasks(
    clean_task_name, file_patterns
):
  """Adds canary tasks to the registry and mixes them with the task.

  Args:
    clean_task_name: the name of the task to add canaries to.
    file_patterns: the file pattern corresponding to canaries.
  """
  task_names = []

  for length, repeats in file_patterns:
    canary_task_name = 'canary_dataset_{}_{}:1.0.0'.format(length, repeats)
    task_names.append(canary_task_name)
    seqio.TaskRegistry.add(
        name=canary_task_name,
        source=seqio.TFExampleDataSource(
            reader_cls=tf.data.TFRecordDataset,
            split_to_filepattern={
                'train': constants.CANARY_TFDS_DATA_PATTERN.format(
                    length, repeats
                ),
                'validation': constants.CANARY_TFDS_DATA_PATTERN.format(
                    length, repeats
                ),
                'test': constants.CANARY_TFDS_DATA_PATTERN.format(
                    length, repeats
                ),
            },
            feature_description={
                'inputs': tf.io.FixedLenFeature([], dtype=tf.string),
                'targets': tf.io.FixedLenFeature([], dtype=tf.string),
                'num_repetitions': tf.io.FixedLenFeature((), dtype=tf.int64),
            },
            # `512` are the number of canaries generated based on the config.ini
            # and `10000` are the number of reference points for computing the
            # exposure.
            # (in `generate_canary_tfds.py`, `num_secrets` and `num_references`)
            # TODO(teobaluta) add a better way of parsing this value rather
            # than hard-coding it.
            num_input_examples={
                'train': 512 * repeats,
                'validation': 512 * repeats,
                'test': 10000,
            }
        ),
        preprocessors=[
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        metric_fns=[metrics.sequence_accuracy],
        output_features=constants.OUTPUT_FEATURES
    )

  # The canary datasets are constructed such that each split already has
  # the repeats as per the config.
  tasks = [clean_task_name]

  for task_name in task_names:
    tasks.append(task_name)

  seqio.MixtureRegistry.add(
      name='wmt_t2t_de_en_v003-canary',
      tasks=tasks,
      default_rate=functools.partial(
          seqio.mixing_rate_num_examples, temperature=1.0))


# TODO(teobaluta) For outside Google, set the TFDS_DATA_DIR and pass that to the
# `train.py` script
# Translation task from German to English
seqio.TaskRegistry.add(
    name='wmt_t2t_de_en_v003',
    source=seqio.TfdsDataSource(tfds_name='wmt_t2t_translate/de-en:1.0.0'),
    preprocessors=[
        functools.partial(
            preprocessors.translate,
            source_language='de', target_language='en'),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.bleu, metrics.sequence_accuracy],
    output_features=constants.OUTPUT_FEATURES
)


# TODO(teobaluta) read this from the dataset metadata?
# These value correspond to the length of the canaries and the number of
# repetitions.
canary_file_patterns = [(10, 1), (10, 10), (10, 100)]
add_canary_tasks('wmt_t2t_de_en_v003', canary_file_patterns)
