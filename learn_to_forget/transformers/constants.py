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

"""Constants for paths."""

import immutabledict
import seqio

# Use the T5 public vocabulary
VOCABULARY = seqio.SentencePieceVocabulary(
    'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
)


OUTPUT_FEATURES = immutabledict.immutabledict({
    'inputs': seqio.Feature(vocabulary=VOCABULARY),
    'targets': seqio.Feature(vocabulary=VOCABULARY)
})

CANARY_TFDS_PATH = ('')
WMT_VAL_TFDS_PATH = ('')

CANARY_TFDS_DATA_PATTERN = (''
                            'train_data-{}_{}.tfrecord')

CANARY_TFDS_TEST_DATA_PATTERN = (''
                                 'test_data-{}_{}.tfrecord')

WMT_TFDS_DATA_PATTERN = (''
                         'train_data-wmt_{}.tfrecord')
# Using the training examples to measure the sequence accuracy
WMT_TFDS_TEST_DATA_PATTERN = (''
                              'train_data-wmt_{}.tfrecord')

TFDS_DATA_DIR = ('')

