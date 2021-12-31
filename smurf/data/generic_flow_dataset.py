# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Data loader for generic flow datasets."""

import os

import numpy as np
import tensorflow as tf

from smurf.data import data_utils
# pylint:disable=unused-import
from smurf.data.data_utils import evaluate
from smurf.data.data_utils import list_eval_keys


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 gt_flow_shape=None,
                 seed=41):
  """Make a dataset for training or evaluating SMURF.

  Args:
    path: string, in the format of 'some/path/dir1,dir2,dir3' to load all files
      in some/path/dir1, some/path/dir2, and some/path/dir3.
    mode: string, one of ['train', 'eval', 'test'] to switch between loading
      training data, evaluation data, and test data, which right now all return
      the same data.
    seq_len: int length of sequence to return. Currently only 2 is supported.
    shuffle_buffer_size: int, size of the shuffle buffer; no shuffling if 0.
    height: int, height for reshaping the images (only if mode==train)
    width: int, width for reshaping the images (only if mode==train)
    resize_gt_flow: bool, indicates if ground truth flow should be resized
      during traing or not (only relevant for supervised training)
    gt_flow_shape: list, if not None sets a fixed size for ground truth flow
      tensor, e.g. [384,512,2]
    seed: int, controls the shuffling of the data shards.

  Returns:
    A tf.dataset of image sequences and ground truth flow for training
    (see parse functions above). The dataset still requires batching
    and prefetching before using it to make an iterator.
  """
  if ',' in path:
    paths = []
    l = path.split(',')
    paths.append(l[0])
    for subpath in l[1:]:
      subpath_length = len(subpath.split('/'))
      basedir = '/'.join(l[0].split('/')[:-subpath_length])
      paths.append(os.path.join(basedir, subpath))
  else:
    paths = [path]
  # Generate list of filenames.
  if seq_len != 2:
    raise ValueError('for_eval only compatible with seq_len == 2.')
  # Generate list of filenames.
  # pylint:disable=g-complex-comprehension
  files = [
      os.path.join(d, f)
      for d in paths
      for f in tf.io.gfile.listdir(d)
  ]

  if 'train' in mode:
    rgen = np.random.RandomState(seed=seed)
    rgen.shuffle(files)

  num_files = len(files)

  ds = tf.data.Dataset.from_tensor_slices(files)
  if shuffle_buffer_size:
    ds = ds.shuffle(num_files)
  # Create a nested dataset.
  ds = ds.map(tf.data.TFRecordDataset)
  # Parse each element of the subsequences and unbatch the result
  # Do interleave rather than flat_map because it is much faster.
  include_flow = 'eval' in mode or 'sup' in mode
  # pylint:disable=g-long-lambda
  ds = ds.interleave(
      lambda x: x.map(
          lambda y: data_utils.parse_data(
              y, include_flow=include_flow, height=height, width=width,
              resize_gt_flow=resize_gt_flow, gt_flow_shape=gt_flow_shape),
          num_parallel_calls=tf.data.experimental.AUTOTUNE),
      cycle_length=min(1 if 'movie' in mode else 10, num_files),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle_buffer_size:
    # Shuffle image pairs.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Put repeat after shuffle for better mixing.
  if 'train' in mode:
    ds = ds.repeat()
  # Prefetch a number of batches because reading new ones can take much longer
  # when they are from new files.
  ds = ds.prefetch(10)

  return ds
