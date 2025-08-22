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

"""File for registering all seqio datasets in the TaskRegistry.

For a general overview, see https://github.com/google/seqio.
"""

import functools

import seqio
from t5.data import preprocessors as t5_preprocessors
import tensorflow_datasets as tfds

from imp.max.data.datasets import seqio_utils

TaskRegistry = seqio.TaskRegistry


TaskRegistry.add(
    'max.c4.en.bert',
    source=seqio.TfdsDataSource(tfds_name='c4/en:3.0.1'),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey, key_map={
                'inputs': 'text',
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio_utils.mask_tokens_for_bert,
        seqio_utils.trim_to_sequence_length_bert,
        seqio_utils.token_mask_bert,
    ],
    output_features=seqio_utils.SEQIO_OUTPUT_FEATURES_WITH_WEIGHTS_BERT,
    metric_fns=[])

TaskRegistry.add(
    'max.c4.en.t5',
    source=seqio.TfdsDataSource(tfds_name='c4/en:3.0.1'),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey, key_map={
                'inputs': 'text',
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        t5_preprocessors.span_corruption,
        seqio_utils.prepend_bos_token,
        seqio_utils.trim_to_sequence_length,
        seqio_utils.token_mask,
    ],
    output_features=seqio_utils.SEQIO_OUTPUT_FEATURES_WITH_WEIGHTS_T5,
    metric_fns=[])

TaskRegistry.add(
    'max.wikipedia.en.bert',
    source=seqio.TfdsDataSource(tfds_name='wikipedia/20220620.en:1.0.0'),
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey, key_map={
                'inputs': 'text',
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio_utils.mask_tokens_for_bert,
        seqio_utils.trim_to_sequence_length_bert,
        seqio_utils.token_mask_bert,
    ],
    output_features=seqio_utils.SEQIO_OUTPUT_FEATURES_WITH_WEIGHTS_BERT,
    metric_fns=[])
