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

"""Convert sanitized data to tf dataset."""
import dataclasses
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf

Array = np.ndarray
OUTPUT_USER_KEY = "uid"
OUTPUT_ITEM_KEY = "sid"


@dataclasses.dataclass
class InputMatrix:
  """Represents a sparse input matrix. Used for batching of input data."""
  indices: np.ndarray
  values: np.ndarray
  num_rows: int
  num_cols: int
  # Weights of the examples in the loss function
  weights: Optional[np.ndarray]
  # Regularization weights (one per row) for frequency-regularization
  row_reg: Optional[np.ndarray]

  def to_sparse_tensor(self):
    return tf.SparseTensor(
        indices=self.indices,
        values=self.values,
        dense_shape=[self.num_rows, self.num_cols])

  def batch_ls(self, batch_size):
    """Batches the data for least squares solvers.

    Args:
      batch_size: the number of rows per batch.

    Returns:
      A dataset with these fields:
        shifted_indices: indices of the ratings, where the rows are shifted to
          start at 0.
        values: values of the ratings.
        weights: weights of the ratings.
        row_reg: the regularization weights, of shape [batch_size].
        update_indices: the original row indices, of shape [batch_size].
    """
    row_ids = self.indices[:, 0]
    def make_batch(start_row, end_row):
      mask = np.greater_equal(row_ids, start_row) & np.less(row_ids, end_row)
      indices = self.indices[mask]
      shifted_indices = indices - np.array([start_row, 0])
      update_indices = np.arange(start_row, end_row)
      batch = dict(
          shifted_indices=shifted_indices,
          update_indices=update_indices,
          values=self.values[mask],
          num_rows=end_row - start_row)
      if self.weights is not None:
        batch["weights"] = self.weights[mask]
      if self.row_reg is not None:
        batch["row_reg"] = self.row_reg[start_row:end_row]
      return batch
    if not batch_size:
      endpoints = [0, self.num_rows]
    else:
      endpoints = [i*batch_size for i in range(self.num_rows//batch_size + 1)]
      if self.num_rows % batch_size:
        endpoints.append(self.num_rows)
    intervals = list(zip(endpoints, endpoints[1:]))
    def gen():
      for start, end in intervals:
        yield make_batch(start, end)
    output_signature = dict(
        shifted_indices=tf.TensorSpec([None, 2], tf.int64),
        update_indices=tf.TensorSpec([None], tf.int64),
        values=tf.TensorSpec([None], tf.float32),
        num_rows=tf.TensorSpec([], tf.int64))
    if self.weights is not None:
      output_signature["weights"] = tf.TensorSpec([None], tf.float32)
    if self.row_reg is not None:
      output_signature["row_reg"] = tf.TensorSpec([None], tf.float32)
    return tf.data.Dataset.from_generator(
        gen, output_signature=output_signature)

  def batch_gd(
      self,
      user_axis,
      batch_size,
      random_seed = 1):
    """Batches the data for gradient descent solvers.

    Args:
      user_axis: axis of the user in the input_data. Should be set as 0 if the
        rows represent users, and 1 if the columns represent users.
      batch_size: the batch size of the dataset.
      random_seed: seed for random generator.

    Returns:
      A tf.Dataset of the form ({"uid": user_ids, "sid": item_ids}}, rating,
      weight).
    """
    if user_axis not in [0, 1]:
      raise ValueError("user_axis must be 0 or 1")
    uids = self.indices[:, user_axis]
    sids = self.indices[:, 1-user_axis]
    values = self.values
    weights = self.weights
    if weights is None:
      weights = np.ones(len(values), dtype=np.float32)
    data_size = len(values)
    if batch_size > data_size:
      raise ValueError(f"{batch_size=} cannot be larger than the size of the "
                       f"data ({data_size})")
    def generator():
      rng = np.random.default_rng(random_seed)
      while True:
        perm = np.tile(rng.permutation(data_size), 2)
        num_batches = data_size // batch_size
        if data_size % batch_size:
          num_batches += 1
        for i in range(num_batches):
          indices = perm[i*batch_size: (i+1)*batch_size]
          yield ({OUTPUT_USER_KEY: uids[indices],
                  OUTPUT_ITEM_KEY: sids[indices]},
                 values[indices],
                 weights[indices])
    return tf.data.Dataset.from_generator(
        generator,
        output_types=({OUTPUT_USER_KEY: tf.int64, OUTPUT_ITEM_KEY: tf.int64},
                      tf.float32, tf.float32),
        output_shapes=({OUTPUT_USER_KEY: (batch_size,),
                        OUTPUT_ITEM_KEY: (batch_size,)},
                       (batch_size,),
                       (batch_size,)))

  def batch_gd_by_user(
      self,
      user_axis,
      num_examples_per_user,
      num_users_per_batch,
      random_seed = 1):
    """Batches the data for gradient descent solvers, grouped by users.

    Suppose we have n users with data {D_1,..., D_n}. For each batch, we want to
    randomly sample `num_users_per_batch` users, and take randomly
    `num_examples_per_user` examples from each them.

    User-grouping is useful for user-level privacy.

    Args:
      user_axis: axis of the user in the input_data. Should be set as 0 if the
        rows represent users, and 1 if the columns represent users.
      num_examples_per_user: the number of examples taken from each user
        when form one batch.
      num_users_per_batch: the number of users in each batch.
      random_seed: seed for random generator.

    Returns:
      A tf.Dataset of the form ({"uid": user_ids, "sid": item_ids}}, rating,
      weight). The batch size is  `num_examples_per_user * num_users_per_batch`.

    Raises:
      ValueError if `num_users_per_batch` is larger than the number of users.
    """
    if user_axis not in [0, 1]:
      raise ValueError("user_axis must be 0 or 1")
    uids = self.indices[:, user_axis]
    sids = self.indices[:, 1-user_axis]
    values = self.values
    weights = self.weights
    if weights is None:
      weights = np.ones(len(values), dtype=np.float32)
    # Sort data
    indices = np.argsort(uids)
    uids = uids[indices]
    sids = sids[indices]
    values = values[indices]
    weights = weights[indices]
    # Compute sizes
    user_sizes = np.zeros(max(uids) + 1, dtype=np.int32)
    for uid in uids:
      user_sizes[uid] += 1
    offsets = np.concatenate([[0], np.cumsum(user_sizes)])
    (nonempty_users,) = np.where(user_sizes)
    nusers = len(nonempty_users)
    if num_users_per_batch > nusers:
      raise ValueError(
          f"num_users_per_batch ({num_users_per_batch}) should not be larger "
          f"than the number of users ({nusers}).")
    # Restrict to non-empty users, because for an empty users we cannot yield
    # any examples, so including them would change the batch size.
    def generator():
      """Combines data for a batch of users."""
      rng = np.random.default_rng(random_seed)
      def user_gen(uid, rng):
        """Yields num_examples_per_user sampled indices for a given user.

        If the user has fewer items than `num_examples_per_user`, then some
        items will be sampled more than once.

        Args:
          uid: the user id.
          rng: the random number generator.
        """
        user_size = user_sizes[uid]
        samples = (
            offsets[uid] +
            np.tile(np.arange(user_size), num_examples_per_user//user_size))
        remaining_to_sample = num_examples_per_user % user_size
        if remaining_to_sample:
          perm = offsets[uid] + np.tile(rng.permutation(user_size), 2)
          i = 0
          while True:
            yield np.concatenate([samples, perm[i:i+remaining_to_sample]])
            i = (i + remaining_to_sample) % user_size
        else:
          while True:
            yield samples
      user_gens = {uid: user_gen(uid, rng) for uid in nonempty_users}
      while True:
        rng.shuffle(nonempty_users)
        # Tile to avoid incomplete batch.
        # Note that we already check num_users_per_batch <= nusers.
        shuffled_uids = np.tile(nonempty_users, 2)
        i = 0
        while i < nusers:
          samples = np.concatenate([
              next(user_gens[uid])
              for uid in shuffled_uids[i : i + num_users_per_batch]
          ])
          i += num_users_per_batch
          yield ({OUTPUT_USER_KEY: uids[samples],
                  OUTPUT_ITEM_KEY: sids[samples]},
                 values[samples],
                 weights[samples])

    batch_size = num_examples_per_user * num_users_per_batch
    return tf.data.Dataset.from_generator(
        generator,
        output_types=({OUTPUT_USER_KEY: tf.int64, OUTPUT_ITEM_KEY: tf.int64},
                      tf.float32, tf.float32),
        output_shapes=({OUTPUT_USER_KEY: (batch_size,),
                        OUTPUT_ITEM_KEY: (batch_size,)},
                       (batch_size,),
                       (batch_size,)))


def df_to_input_matrix(
    df,
    num_rows,
    num_cols,
    row_key = "uid",
    col_key = "sid",
    value_key = None,
    sort = True):
  """Creates an InputMatrix from a pd.DataFrame.

  Args:
    df: the DataFrame. Must contain row_key and col_key.
    num_rows: the number of rows.
    num_cols: the number of columns.
    row_key: the dataframe key to use for the row ids.
    col_key: the dataframe key to use for the column ids.
    value_key: uses this field for the values. If None, fills with ones.
    sort: whether to sort the indices.

  Returns:
    An InputMatrix.
  """
  if sort:
    df = df.sort_values([row_key, col_key])
  # convert to int to handle empty DataFrames
  row_ids = df[row_key].values.astype(np.int64)
  col_ids = df[col_key].values.astype(np.int64)
  indices = np.stack([row_ids, col_ids], axis=1)
  if value_key:
    if value_key not in df:
      raise ValueError(f"key {value_key} is missing from the DataFrame")
    values = df[value_key].values
  else:
    values = np.ones(len(df))
  values = values.astype(np.float32)
  return InputMatrix(
      indices=indices,
      values=values,
      weights=None,
      row_reg=None,
      num_rows=num_rows,
      num_cols=num_cols)
