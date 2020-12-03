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
"""Input pipeline for the MLPerf WMT de-en dataset.

This version of the WMT17 dataset uses a shared 32K subword tokenization
and training data packed into 256-length sequences with additional
segmentation masks and sub-sequence position data.  The evaluation
data is from WMT14 and not packed, and has a max length of 97.

Generation script instructions can be found at:
https://github.com/mlperf/training_results_v0.6/tree/master/Google/benchmarks/transformer/implementations/tpu-v3-32-transformer/dataset_preproc

We load the WMT train and eval datasets in their entirety into host memory.
"""

import os
from absl import flags
import jax
import tensorflow.compat.v2 as tf

# MLPerf Dataset Constants.
# Packed WMT17 training data.
MAX_TRAIN_LEN = 256  # multiple sequences are packed into this length.
N_TRAIN = 566340  # number of (packed) training tfrecords.
TRAIN_KEYS = ['inputs', 'inputs_position', 'inputs_segmentation',
              'targets', 'targets_position', 'targets_segmentation']
# Truncated WMT14 eval data.
MAX_EVAL_LEN = 97  # no eval sequences are longer than this.
N_EVAL = 3003  # number of eval tfrecords.
EVAL_KEYS = ['inputs', 'targets']
# Default data paths.
TRAIN_PATH = None
EVAL_PATH = None
VOCAB_PATH = None

ROOT = 'gs://jax-demo-data/wmt-mlperf/'
# Packed Training Data:
TRAIN_PATH = ROOT + 'translate_ende_wmt32k_packed-train*'
# Evaluation Data -- Not Packed:
EVAL_PATH = ROOT + 'translate_ende_wmt32k-dev*'
VOCAB_PATH = ROOT + 'vocab.translate_ende_wmt32k.32768.subwords'


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'train_data_path', default=TRAIN_PATH,
    help='Path to packed training dataset tfrecords.')

flags.DEFINE_string(
    'eval_data_path', default=EVAL_PATH,
    help='Path to unpacked evaluation dataset tfrecords.')

flags.DEFINE_string(
    'vocab_path', default=VOCAB_PATH,
    help='Path to shared vocabulary file.')


def length_filter(max_len):
  def filter_fn(batch):
    l = tf.maximum(tf.shape(batch['inputs'])[0],
                   tf.shape(batch['targets'])[0])
    return tf.less(l, max_len + 1)
  return filter_fn


def pad_up_to(t, max_in_dims):
  s = tf.shape(t)
  paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
  return tf.pad(t, paddings, 'CONSTANT', constant_values=0)


def get_wmt_dataset(batch_size, train, shuffle_size=16384):
  """Get the train or eval split of WMT as a tf.data.Dataset."""
  keys = TRAIN_KEYS if train else EVAL_KEYS

  def parse_function(example_proto):
    return tf.io.parse_single_example(
        example_proto,
        {k: tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
         for k in keys})

  def cast_to_int32(x):
    return {k: tf.dtypes.cast(x[k], tf.int32) for k in keys}

  def pad(x):
    return {k: pad_up_to(x[k], [MAX_TRAIN_LEN if train else MAX_EVAL_LEN,])
            for k in keys}

  file_pattern = os.path.join(
      FLAGS.train_data_path if train else FLAGS.eval_data_path)
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  dataset = dataset.shard(jax.host_count(), jax.host_id())
  concurrent_files = min(10, 1024 // jax.host_count())
  dataset = dataset.interleave(
      tf.data.TFRecordDataset, concurrent_files, 1, concurrent_files)

  dataset = dataset.map(parse_function, num_parallel_calls=32)
  dataset = dataset.map(cast_to_int32, num_parallel_calls=32)
  if train:
    # Filter out rare long, unpacked single-examples.
    dataset = dataset.filter(length_filter(MAX_TRAIN_LEN))

  dataset = dataset.map(pad, num_parallel_calls=32)
  if train:
    dataset = dataset.cache().shuffle(shuffle_size).repeat()
  dataset = dataset.batch(batch_size, drop_remainder=train)
  if not train:
    dataset = dataset.cache().repeat()
  dataset = dataset.prefetch(1024)

  options = tf.data.Options()
  options.experimental_deterministic = False
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_threading.private_threadpool_size = 48
  dataset = dataset.with_options(options)

  return dataset
