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

"""Dataset loader for various datasets."""

import collections
import os
from typing import Any, Dict, List, Optional, Text, Tuple

from absl import logging
import numpy as np
import tensorflow as tf
from multiple_user_representations.synthetic_data import util as synthetic_data_util


def prep_next_item_dataset(
    k,
    item_sequence,
    user_negative_items,
    candidate_sample_prob=None,
):
  """Converts raw user-item dataset to a dataset with a dictionary of features.

  The produced dataset uses the kth item from the end as the next item to be
  predicted and items before that as the input sequence. Typically, `k` will be
  1 or 2 depending on the data split_type (user/step). When using validation and
  split_type=step, `k` will 3 for training data, 2 for validation data and 1 for
  testing data.

  Args:
    k: Integer index denoting the kth item from the end.
    item_sequence: The complete user item history.
    user_negative_items: Negative item samples used for evaluation. See
      evaluation in the proposal doc
      (http://shortn/_PO6OdvUuAs#bookmark=id.4dm5yebd3f21) for details.
    candidate_sample_prob: The candidate sampling probability. This is used for
      correcting the sample bias of in-batch negatives.

  Returns:
    A dictionary containing the following features in the dataset:
    user_item_sequence: List of user item sequences.
    next_item: List of next items for each user.
    user_negative_items: List of negative items for each user. This is used to
      compute metrics for real datasets.
    candidate_sample_prob: The candidate sampling probability.
  """

  user_item_input = {
      'user_item_sequence': item_sequence[:-k],
      'next_item': item_sequence[-k],
  }

  if user_negative_items is not None:
    user_item_input['user_negative_items'] = user_negative_items
  if candidate_sample_prob is not None:
    user_item_input['candidate_sampling_probability'] = candidate_sample_prob

  return user_item_input


def _prep_candidate_sampling_probability_for_training(
    user_item_seq,
    split_type,
    use_validation,
    fraction_head = 0.2
):
  """Prepares candidate_sampling_probability for training with batch softmax.

  Args:
    user_item_seq: List[int] or np.ndarray of item sequences used for training.
    split_type: The split_type used for training the model. The axis along which
      sampling probability is computed depends on the split_type.
    use_validation: If true, the target_item index is 3rd last from the end for
      during training. If false the target_item index will be 2nd last from end.
    fraction_head: The fraction of items that are considered head items.

  Returns:
    candidate_sampling_probability: Array of probability for candidates.
    head_items: Top fraction_head*100% items with the highest frequency.
    item_count_weight: Dictionary mapping item__id to (frequency, item_weight)
      in train set. This is used for iterative density weighting of items. The
      object defines the initial state of item_weights and the sample_weight is
      updated when using density weighting.
  """

  # Determine target_item index for training data.
  if split_type == 'user':
    axis = -1
  elif use_validation:
    axis = -3
  else:
    axis = -2

  if isinstance(user_item_seq, np.ndarray):
    next_item_train_arr = user_item_seq[:, axis]
  else:
    next_item_train_arr = []
    for item_seq in user_item_seq:
      next_item_train_arr.append(item_seq[axis])
    next_item_train_arr = np.array(next_item_train_arr)

  items, indices, counts = np.unique(
      next_item_train_arr, return_inverse=True, return_counts=True)
  num_head_items = int(fraction_head * len(items))
  head_items = items[counts.argsort()[-num_head_items:]]
  probs = counts * 1.0 / np.sum(counts)
  candidate_sampling_probability = probs[indices]

  assert len(items) == len(probs), 'Arr len mismatch: {:d} & {:d}.'.format(
      len(items), len(probs))
  return (candidate_sampling_probability, head_items,
          dict(zip(items, zip(counts, probs))))


def _add_head_item_feature(dataset,
                           head_items,
                           batch_size = 100):
  """Adds `is_item_head` binary feature to input features.

  Args:
    dataset: The train/valid/test dataset. The dataset should contain
      `next_item` key.
    head_items: List of items that belong to the head of the item distribution.
    batch_size: The number of examples to transform using the map fn.

  Returns:
    output_dataset: An updated version of dataset containing an additional input
      field called `is_item_head` indicating whether next_item belongs to the
      head of the distribution.
  """

  head_items = tf.convert_to_tensor(head_items, dtype=tf.int32)

  def add_head_item_indicator(features):

    next_item = tf.expand_dims(
        tf.cast(features['next_item'], tf.int32), axis=-1)
    features['is_head_item'] = tf.cast(
        tf.math.reduce_any(tf.math.equal(next_item, head_items), axis=1),
        tf.float32)

    return features

  return dataset.batch(batch_size).map(add_head_item_indicator).unbatch()


def load_dataset(dataset_name,
                 dataset_path,
                 use_validation = False,
                 split_type='step'):
  """Returns the tf Dataset for training and testing.

  Args:
    dataset_name: Name of the dataset used to call the relevant load data fn.
    dataset_path: Path to the dataset directory.
    use_validation: If true, also returns validation dataset.
    split_type: Type of train-test split. Can be `user` or `step`. For
      split_type = `user`, the train and test sets have different users, and for
      split_type = `step`, the training split has [1:n-2] as the input sequence
      and the testing split has [1:n-1] as the input sequence.

  Returns:
    A dictionary with the following keys:
    train_dataset: Train dataset.
    test_dataset: Test dataset.
    validation_dataset: Validation dataset (only if use_validation=True).
    item_dataset: Item dataset.
    max_seq_size: Maximum sequence size.
    num_items: Number of items.

  Raises:
    ValueError: If dataset not found or if split_type is not valid.
  """

  output_dataset = dict()

  if dataset_name == 'conditional_synthetic':

    dataset = synthetic_data_util.load_data(dataset_path)
    user_item_seq = dataset['user_item_sequences']
    max_seq_size = user_item_seq.shape[1] - 1
    # We don't use negative sampling for evaluating the synthetic dataset.
    user_negative_items = None
  elif dataset_name == 'amazon_review_category' or dataset_name == 'movielens':

    dataset = load_preprocessed_real_data(dataset_path, stride=5)
    user_item_seq = dataset['user_item_sequences']
    user_negative_items = dataset['user_negative_items']
    max_seq_size = dataset['max_seq_size'] - 1
    # `mask_zero=True` indicates that item_id '0' is used as padding and it
    # needs to be masked when applying attention.
    output_dataset['mask_zero'] = True
  else:
    raise ValueError('Dataset {} not found.'.format(dataset_name))

  logging.info('Number of sequences: %d', len(user_item_seq))
  all_items = sorted(dataset['items'])
  item_dataset = tf.data.Dataset.from_tensor_slices(all_items)
  num_users = len(user_item_seq)

  prep_dataset_k1 = lambda *data: prep_next_item_dataset(1, *data)
  prep_dataset_k2 = lambda *data: prep_next_item_dataset(2, *data)
  prep_dataset_k3 = lambda *data: prep_next_item_dataset(3, *data)

  if split_type == 'user':
    if use_validation:
      num_training_examples = int(0.8 * num_users)
      num_validation_examples = int(0.1 * num_users)
    else:
      num_training_examples = int(0.9 * num_users)
      num_validation_examples = 0

    train_valid_index = num_training_examples + num_validation_examples
    train_example_seqs = user_item_seq[:num_training_examples]
    valid_example_seqs = user_item_seq[num_training_examples:train_valid_index]
    test_example_seqs = user_item_seq[train_valid_index:]

    candidate_sample_probs, head_items, item_count_probs = _prep_candidate_sampling_probability_for_training(
        train_example_seqs, 'user', use_validation)
    train_user_negative_items = None
    valid_user_negative_items = None
    test_user_negative_items = None

    if user_negative_items is not None:
      train_user_negative_items = user_negative_items[:num_training_examples]
      valid_user_negative_items = user_negative_items[
          num_training_examples:train_valid_index]
      test_user_negative_items = user_negative_items[train_valid_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_example_seqs, train_user_negative_items, candidate_sample_probs))
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (valid_example_seqs, valid_user_negative_items))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_example_seqs, test_user_negative_items))

    train_dataset = train_dataset.map(prep_dataset_k1)
    test_dataset = test_dataset.map(prep_dataset_k1)
    if use_validation:
      valid_dataset = valid_dataset.map(prep_dataset_k1)

  elif split_type == 'step':
    num_training_examples = num_users
    candidate_sample_probs, head_items, item_count_probs = _prep_candidate_sampling_probability_for_training(
        user_item_seq, 'step', use_validation)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (user_item_seq, user_negative_items, candidate_sample_probs))
    val_candidate_sample_probs, _, _ = _prep_candidate_sampling_probability_for_training(
        user_item_seq, 'step', False)
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (user_item_seq, user_negative_items, val_candidate_sample_probs))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (user_item_seq, user_negative_items))

    if use_validation:
      train_dataset = train_dataset.map(prep_dataset_k3)
      valid_dataset = valid_dataset.map(prep_dataset_k2)
      test_dataset = test_dataset.map(prep_dataset_k1)
    else:
      train_dataset = train_dataset.map(prep_dataset_k2)
      test_dataset = test_dataset.map(prep_dataset_k1)

  else:
    raise ValueError(f"""{split_type} is not a valid split_type. Accepted \
                     values are `step` and `user`.""")
  output_dataset['train_dataset'] = _add_head_item_feature(
      train_dataset, head_items)
  output_dataset['test_dataset'] = _add_head_item_feature(
      test_dataset, head_items)
  output_dataset['item_dataset'] = item_dataset
  output_dataset['max_seq_size'] = max_seq_size
  output_dataset['num_items'] = len(all_items)
  output_dataset['num_users'] = num_users
  output_dataset['item_count_probs'] = item_count_probs
  if use_validation:
    output_dataset['valid_dataset'] = _add_head_item_feature(
        valid_dataset, head_items)

  return output_dataset


def load_preprocessed_real_data(dataset_dir,
                                max_seq_size = 30,
                                stride = 1):
  """Loads preprocessed real data (Amazon category dataset and MovieLens).

  Args:
    dataset_dir: Path to preprocessed data directory. The dir should have the
      user_item_mapped.txt file and user_neg_items.txt. The item ids should
      begin with "1" since "0" is reserved for sequence padding. The
      user_item_time.txt file should contain <user_id> <item_id> <timestamp> in
      each line sorted by timestamp, and the user_neg_items.txt file should
      contain negative items for each user, where each line is:
      <user_id>: <item_id_1>...<item_id_N>, where N is the total negative items
        sampled.
    max_seq_size: Max sequence size. Sequence of length less than max_seq_size,
      is padded with zeros from left.
    stride: Stride to use when splitting the item sequence, This is used when
      the sequence length is greater than max_seq_size.

  Returns:
    dataset: A dictionary containing the dataset.
  """

  logging.info('Loading data from %s.', dataset_dir)
  num_users = 0
  num_items = 0
  user_list = collections.defaultdict(list)
  user_neg_item_dict = collections.defaultdict(list)

  dataset_path = os.path.join(dataset_dir, 'user_item_mapped.txt')
  negative_items_path = os.path.join(dataset_dir, 'user_neg_items.txt')

  with tf.io.gfile.GFile(dataset_path, 'r') as fin:
    for line in fin:

      user_id, item_id, _ = line.rstrip().split(' ')
      user_id = int(user_id)
      item_id = int(item_id)
      num_users = max(user_id, num_users)
      num_items = max(item_id, num_items)
      user_list[user_id].append(item_id)

  logging.info('Num users: %d, Num items: %d', num_users, num_items)
  for user in user_list:
    item_seq = user_list[user]
    item_seq_len = len(item_seq)
    if item_seq_len < max_seq_size:
      padded_item_seq = [0] * (max_seq_size - item_seq_len)
      padded_item_seq.extend(item_seq)
    else:
      padded_item_seq = item_seq

    user_list[user] = padded_item_seq

  dataset = {}

  with tf.io.gfile.GFile(negative_items_path, 'r') as fin:
    for line in fin:
      user_id, items = line.rstrip().split(':')
      user_id = int(user_id)
      user_neg_item_dict[user_id] = list(map(int, items.strip().split(' ')))

  user_neg_items = []
  user_item_sequences = []
  for user_id in sorted(user_neg_item_dict.keys()):

    item_seq = user_list[user_id]
    if len(item_seq) > max_seq_size:
      # Split the sequence into multiple sequences.
      split = 0
      while (split * stride + max_seq_size) < len(item_seq):
        start_ix = split * stride
        user_item_sequences.append(item_seq[start_ix:start_ix + max_seq_size])
        # Since the user is the same, reuse user_neg_item_dict[user_id].
        user_neg_items.append(user_neg_item_dict[user_id])
        split += 1
      # The last split is the max_seq_size items from the end.
      user_item_sequences.append(item_seq[(len(item_seq) - max_seq_size):])
    else:
      user_item_sequences.append(user_list[user_id])
    user_neg_items.append(user_neg_item_dict[user_id])

  dataset['user_negative_items'] = user_neg_items
  dataset['user_item_sequences'] = user_item_sequences
  dataset['num_items'] = num_items
  dataset['items'] = range(0, num_items + 1)  # 0 is padding.
  dataset['max_seq_size'] = max_seq_size
  logging.info('Data loaded from %s.', dataset_dir)

  return dataset
