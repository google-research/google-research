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

"""Data related functions for packing emnist dataset.

parse_data: extract client data from emnist and add ids to tuples of numpy
arrays.
pack_dataset: pack the numpy arrays into tf.data.Dataset.
"""
import numpy as np
import tensorflow as tf

NUM_EMNIST_CLASSES = 62
MAX_DATA_SIZE = 700000
SPLIT_SIZE = 0.1
SHUFFLE_SIZE = 10000
PARSE_DATA_BATCH_SIZE = 256


def pack_dataset(data_tuple, mode, batch_size=256, with_dist=False):
  """Packs the arrays into tf.data.Dataset.

  Args:
    data_tuple: tuples of numpy array return from parse_data() as inputs.
      It follows the orders:
      For with_dist is True:
        Input images, client ids, label distributions, labels
      For with_dist is False:
        Input images, client ids, labels
    mode: training mode of test mode.
    batch_size: batch size for the dataset.
    with_dist: using label distributions as inputs.

  Returns:
    A tf.data.Dataset
  """
  if with_dist:
    x, idx, dist, y = data_tuple
    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_x': x,
        'input_id': idx,
        'input_dist': dist,
    }, y))
  else:
    x, idx, y = data_tuple
    dataset = tf.data.Dataset.from_tensor_slices(({
        'input_x': x,
        'input_id': idx
    }, y))
  if mode == 'train':
    dataset = dataset.shuffle(SHUFFLE_SIZE)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  return dataset


def count_dataset(dataset):
  cnt = 0
  for _ in iter(dataset):
    cnt = cnt + 1
  return int(cnt)


def get_local_y_dist(client_dataset):
  dist = np.zeros((1, NUM_EMNIST_CLASSES))
  for x in client_dataset:
    y = x['label'].numpy()
    dist[y] += 1

  return np.array(dist).reshape((1, -1)) / np.sum(dist)


def parse_data(emnist_train,
               emnist_test,
               client_ids,
               cliend_encodings,
               with_dist=False):
  """Packs the client dataset into tuples of arrays with client ids.

  Args:
    emnist_train: the tff clientdata object of the training sets.
    emnist_test: the tff clientdata object of the test sets.
    client_ids: client ids to extract.
    cliend_encodings: a dictionary encoding client string id to number.
    with_dist: using label distributions as inputs or not.

  Returns:
    Three tuples of numpy arrays:
    The training set for fine-tuning, the smaller split of the training set,
    the test set, each is a tuple of the following np.array:

    Input images, input ids, label distributions, labels if with_dist is True
    Input images, input ids, labels if with_dist is False
  """
  def merge_clients(emnist, split_size=1):
    # Cache in the memory for faster training iterations
    train_x, train_id, train_y = np.zeros((MAX_DATA_SIZE, 28, 28, 1)), np.zeros(
        (MAX_DATA_SIZE)), np.zeros((MAX_DATA_SIZE))
    cnt = 0

    if with_dist:
      train_dist = np.zeros((MAX_DATA_SIZE, NUM_EMNIST_CLASSES))

    client_num_list = []
    for client_id in client_ids:
      ds = emnist.create_tf_dataset_for_client(client_id)
      client_id = cliend_encodings[client_id]
      ds_np = ds.batch(PARSE_DATA_BATCH_SIZE)

      if with_dist:
        y_dist = get_local_y_dist(ds)

      client_num = 0
      for x in ds_np:
        y = x['label']
        x = tf.expand_dims(x['pixels'], axis=-1)
        if split_size < 1:
          split_num = int(len(y)*split_size)
          ids = np.random.choice(np.arange(len(y)), split_num)
          y = tf.gather(y, ids)
          x = tf.gather(x, ids)

        num = len(y)
        idx = np.array([client_id]*num)

        train_x[cnt:cnt+num] = x
        train_y[cnt:cnt+num] = y
        train_id[cnt:cnt+num] = idx

        if with_dist:
          train_dist[cnt:cnt+num] = np.tile(y_dist, [num, 1])

        cnt += num
        client_num += num
      client_num_list.append(client_num)

    train_x = train_x[:cnt]
    train_y = train_y[:cnt]
    train_id = train_id[:cnt]

    if with_dist:
      train_dist = train_dist[:cnt]
      return train_x, train_id, train_dist, train_y
    else:
      return train_x, train_id, train_y

  return merge_clients(emnist_train), merge_clients(
      emnist_train, split_size=SPLIT_SIZE), merge_clients(emnist_test)
