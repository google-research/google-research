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

"""Functions for dealing with data.

Getting datasets, creating "replicated" datasets (stacking several samples along
the channel dimension), pre-processing.
"""

import enum

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


class DatasetSplit(enum.Enum):
  TRAIN = 0
  VALID = 1
  TEST = 2
  TRAIN_FULL = 3


def get_dataset(dataset_name, split_type, validation_percent=20,
                batch_size=None, preprocess_fn=None):
  """Load the specified split of the given tfds dataset, with some extra optons.

  Args:
    dataset_name: a string specifying the name and other parameters of the
      dataset to be loaded. Currently has to be of the form DATASETNAME.N,
      where N is how many to replicate the dataset to create a multi-label
      setup.
    split_type: a DatasetSplit object specifying which split to load
    validation_percent: the percentage of the training set to be used for
      validation
    batch_size: mini-batch size
    preprocess_fn: function used to pre-process the images
  Returns:
    ds: tf.data.Dataset instance, with the requested dataset
  """

  if split_type == DatasetSplit.TRAIN:
    split = "train[{}%:]".format(validation_percent)
  elif split_type == DatasetSplit.VALID:
    split = "train[:{}%]".format(validation_percent)
  elif split_type == DatasetSplit.TEST:
    split = "test"
  elif split_type == DatasetSplit.TRAIN_FULL:
    split = "train"
  else:
    raise ValueError("Unknown split_type {}".format(split_type))

  ds = tfds.load(name=dataset_name, split=split,
                 as_dataset_kwargs={"shuffle_files": False}
                ).shuffle(1000, seed=17)

  if split_type in [DatasetSplit.TRAIN, DatasetSplit.TRAIN_FULL]:
    ds = ds.repeat()

  ds = ds.map(preprocess_fn, num_parallel_calls=batch_size)

  return (ds.batch(
      batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
