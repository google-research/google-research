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

"""Synthetic data preparation utilities."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm


tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers


_BAG_SIZE = flags.DEFINE_integer('bag_size', default=10, help='Bag size.')
_WRITE_DIR = flags.DEFINE_string(
    'write_dir',
    default='mir_uai24/datasets/synthetic',
    help='Data write directory.',
)

N_FEATURES = 32
N_BAGS = 10000
BUFFER = 1
# OVERLAP_SAMPLING_TEMP: {<bag_size>: {<overlap_percent>: <temp>, ...}, ...}
# pylint: disable=bad-whitespace
OVERLAP_SAMPLING_TEMP = {
    2  : {5: 2.0, 10: 2.4, 15: 2.7, 20: 3.2, 25: 5.0},
    5  : {5: 2.3, 10: 2.8, 15: 3.3, 20: 3.8, 25: 4.4},
    10 : {5: 2.7, 10: 3.7, 15: 4.9, 20: 6.5, 25: 10.3}
}
# pylint: enable=bad-whitespace


def get_uncorrelated_bag_data():
  instance_dist = tfd.MultivariateNormalDiag(
      loc=tf.zeros(N_FEATURES), scale_diag=tf.ones(N_FEATURES))
  instance_data = instance_dist.sample(
      _BAG_SIZE.value*N_BAGS*BUFFER, seed=0).numpy()
  bag_data = instance_data.reshape(N_BAGS*BUFFER, _BAG_SIZE.value*N_FEATURES)
  return bag_data


def make_correlated_bag_data(bag_data):
  """Makes correlated bag data using a randomly sampled Cholesky matrix."""

  all_scale = tf.eye(N_FEATURES*_BAG_SIZE.value)[None, :, :]
  all_scale = tf.repeat(all_scale, axis=0, repeats=N_BAGS*BUFFER)

  scale_mask = np.zeros((
      N_FEATURES*_BAG_SIZE.value, N_FEATURES*_BAG_SIZE.value), dtype=bool)
  for i in range(N_FEATURES*_BAG_SIZE.value):
    for j in range(i+N_FEATURES, N_FEATURES*_BAG_SIZE.value, N_FEATURES):
      scale_mask[j, i] = True
  scale_mask = tf.convert_to_tensor(scale_mask, dtype=tf.bool)[None, :, :]
  scale_mask = tf.repeat(scale_mask, axis=0, repeats=N_BAGS*BUFFER)
  tf.random.set_seed(0)
  uniform = tf.random.uniform((
      N_BAGS*BUFFER, N_FEATURES*_BAG_SIZE.value, N_FEATURES*_BAG_SIZE.value))
  all_scale = tf.where(scale_mask, uniform, all_scale)
  return tf.matmul(all_scale, bag_data[:, :, None]).numpy(
      ).reshape(N_BAGS*BUFFER, _BAG_SIZE.value, N_FEATURES)


def get_labels(correlated_bag_data):
  """Computes noisy labels for the correlated bag data using a polynomial model."""

  correlated_instance_data = correlated_bag_data.reshape(
      N_BAGS*_BAG_SIZE.value*BUFFER, N_FEATURES)
  poly = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
  correlated_instance_data_poly = poly.fit_transform(correlated_instance_data)

  coef = tfd.Uniform(low=-0.1, high=0.1).sample(
      (N_FEATURES*(N_FEATURES+1))//2).numpy()
  coef = np.concatenate(
      [tfd.Uniform(low=-1, high=1).sample(N_FEATURES).numpy(), coef])
  y = (correlated_instance_data_poly * coef[None]).sum(axis=1)
  y_noisy = tfd.Normal(
      loc=y, scale=tf.ones(N_BAGS*_BAG_SIZE.value*BUFFER)).sample().numpy()
  y_noisy_bags = y_noisy.reshape(N_BAGS*BUFFER, _BAG_SIZE.value)
  return y_noisy_bags


def make_dataframe(
    correlated_bag_data, y_noisy_bags):
  """Makes a dataframe from the correlated bag data and noisy labels."""

  instance_ids = np.arange(N_BAGS*_BAG_SIZE.value*BUFFER).reshape(
      N_BAGS*BUFFER, _BAG_SIZE.value)
  random_state = np.random.RandomState(0)
  bags = {}
  instance_id_map = {}

  instance_i = 0
  for instance_i in range(N_BAGS*BUFFER):
    prime_i = random_state.choice(instance_ids[instance_i])
    bags[prime_i] = instance_ids[instance_i]
    instance_id_map[prime_i] = instance_i

  for instances in bags.values():
    for orig_instance_i in instances:
      if orig_instance_i not in instance_id_map:
        instance_i += 1
        instance_id_map[orig_instance_i] = instance_i

  df = []
  for i, prime_i in enumerate(bags):
    assert i == instance_id_map[prime_i]
    row = {'bag_id': instance_id_map[prime_i]}
    row['instance_id'] = np.array(
        list(map(lambda x: instance_id_map[x], bags[prime_i])))
    row['bag_id_X_instance_id'] = np.arange(
        instance_id_map[prime_i]*_BAG_SIZE.value,
        (instance_id_map[prime_i]+1)*_BAG_SIZE.value)

    for feature_i in range(N_FEATURES):
      row[str(feature_i)] = correlated_bag_data[i, :, feature_i]
    row['y'] = y_noisy_bags[i]
    df.append(row)
  df = pd.DataFrame(df)
  return df


def create_splits(
    df,
):
  """Creates train, val, test, and train_instance splits."""

  train_df = df[: int(N_BAGS * 0.8)].copy()
  val_df = df[int(N_BAGS * 0.8) : int(N_BAGS * 0.9)].copy()
  test_df = df[int(N_BAGS * 0.9) : N_BAGS].copy()

  val_df = val_df.explode(
      column=[str(feature_i) for feature_i in range(N_FEATURES)]
      + ['instance_id', 'bag_id_X_instance_id', 'y']
  )
  val_df = val_df[val_df.bag_id == val_df.instance_id]
  val_df.reset_index(drop=True, inplace=True)
  val_df.bag_id_X_instance_id = val_df.index
  val_df.bag_id = val_df.index
  val_df.instance_id = val_df.index

  test_df = test_df.explode(
      column=[str(feature_i) for feature_i in range(N_FEATURES)]
      + ['instance_id', 'bag_id_X_instance_id', 'y'])
  test_df = test_df[test_df.bag_id == test_df.instance_id]
  test_df.reset_index(drop=True, inplace=True)
  test_df.bag_id_X_instance_id = test_df.index
  test_df.bag_id = test_df.index
  test_df.instance_id = test_df.index

  train_df_instance = train_df.explode(
      column=[str(feature_i) for feature_i in range(N_FEATURES)]
      + ['instance_id', 'bag_id_X_instance_id', 'y'])
  train_df_instance = train_df_instance[
      train_df_instance.bag_id == train_df_instance.instance_id]
  train_df_instance.reset_index(drop=True, inplace=True)
  train_df_instance.bag_id_X_instance_id = train_df_instance.index
  train_df_instance.bag_id = train_df_instance.index
  train_df_instance.instance_id = train_df_instance.index

  return train_df, val_df, test_df, train_df_instance


def get_overlap_candidates(df):
  train_x_candidates = pd.concat(
      [df[: int(N_BAGS * 0.8)].copy(), df[N_BAGS:].copy()])
  train_x_candidates = train_x_candidates.explode(
      column=[str(feature_i) for feature_i in range(N_FEATURES)]
      + ['instance_id', 'bag_id_X_instance_id', 'y']
  )
  train_y_candidates = train_x_candidates['y'].to_numpy()
  train_x_candidates = train_x_candidates[
      [str(feature_i) for feature_i in range(N_FEATURES)]].to_numpy()
  train_x_candidates = train_x_candidates.astype(np.float32)
  return train_x_candidates, train_y_candidates


def get_log_probs(train_x_candidates):
  """Computes log probs for every possible pair of training instances."""

  log_probs = []
  assert train_x_candidates.shape[0] % 1000 == 0
  for i in range(0, train_x_candidates.shape[0], 1000):
    log_probs.append(
        tfd.MultivariateNormalDiag(
            loc=train_x_candidates[None], scale_diag=tf.ones(N_FEATURES)
        )
        .log_prob(train_x_candidates[i : i + 1000, None, :])
        .numpy()
    )
  log_probs = np.concatenate(log_probs, axis=0).astype(np.float64)
  return log_probs


def make_dataframe_with_overlaps(
    train_x_candidates,
    train_y_candidates,
    log_probs,
    overlap_percent
):
  """Makes a dataframe with overlapping bags with a desired overlap percentage."""

  random_state = np.random.RandomState(0)
  n_total_bags = train_x_candidates.shape[0] // _BAG_SIZE.value
  temperature = OVERLAP_SAMPLING_TEMP[_BAG_SIZE.value][overlap_percent]
  sampled_instances = (
      tfd.Categorical(logits=log_probs / temperature)
      .sample(seed=0)
      .numpy()
      .reshape(n_total_bags, _BAG_SIZE.value)
  )
  bags = {}
  instance_id_map = {}
  instance_overlaps = np.zeros(log_probs.shape[0])
  candidate_bag_i = 0
  instance_i = 0
  prime_instances = set()

  with tqdm.tqdm(total=int(N_BAGS*0.8)) as pbar:
    while len(bags) < int(N_BAGS*0.8) and candidate_bag_i < n_total_bags:
      bag = set()
      unnormalized_probs = np.exp(log_probs[candidate_bag_i]/temperature)
      probs = unnormalized_probs/unnormalized_probs.sum()
      for j in range(_BAG_SIZE.value):
        new_instance = sampled_instances[candidate_bag_i, j]
        while new_instance in bag:
          new_instance = random_state.choice(
              a=np.arange(log_probs.shape[1]), p=probs)
        instance_overlaps[new_instance] += 1
        bag.add(new_instance)
      candidate_bag_i += 1
      non_prime_instances = bag - prime_instances
      if not non_prime_instances:
        continue
      prime_i = random_state.choice(list(non_prime_instances))
      prime_instances.add(prime_i)
      bags[prime_i] = bag
      instance_id_map[prime_i] = instance_i
      instance_i += 1
      pbar.update(1)
  if len(bags) < int(N_BAGS * 0.8):
    raise ValueError(
        'Not enough candidate bags to create overlap. '
        'Please increase the buffer.'
    )

  for instances in bags.values():
    for orig_instance_i in instances:
      if orig_instance_i not in instance_id_map:
        instance_i += 1
        instance_id_map[orig_instance_i] = instance_i
  logging.info('Overlap percent: %f', (instance_overlaps > 1).mean())
  overlap_df = []
  for i, prime_i in enumerate(bags):
    assert i == instance_id_map[prime_i]
    row = {'bag_id': instance_id_map[prime_i]}
    row['instance_id'] = np.array(
        list(map(lambda x: instance_id_map[x], bags[prime_i]))
    )
    row['bag_id_X_instance_id'] = np.arange(
        instance_id_map[prime_i] * _BAG_SIZE.value,
        (instance_id_map[prime_i] + 1) * _BAG_SIZE.value,
    )

    for feature_i in range(N_FEATURES):
      row[str(feature_i)] = train_x_candidates[list(bags[prime_i]), feature_i]
    row['y'] = train_y_candidates[list(bags[prime_i])]
    overlap_df.append(row)
  overlap_df = pd.DataFrame(overlap_df)
  return overlap_df


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  tf.random.set_seed(0)

  bag_data = get_uncorrelated_bag_data()
  correlated_bag_data = make_correlated_bag_data(bag_data)
  y_noisy_bags = get_labels(correlated_bag_data)
  df = make_dataframe(correlated_bag_data, y_noisy_bags)
  train_df, val_df, test_df, train_df_instance = create_splits(df)

  os.makedirs(
      os.path.join(_WRITE_DIR.value, f'bag_size_{_BAG_SIZE.value}'),
      exist_ok=True
  )
  train_df.to_feather(
      os.path.join(
          _WRITE_DIR.value,
          f'bag_size_{_BAG_SIZE.value}',
          'bags_overlap_0_train.ftr',
      )
  )
  val_df.to_feather(
      os.path.join(
          _WRITE_DIR.value, f'bag_size_{_BAG_SIZE.value}', 'overlap0_val.ftr')
  )
  test_df.to_feather(
      os.path.join(
          _WRITE_DIR.value, f'bag_size_{_BAG_SIZE.value}', 'overlap0_test.ftr')
  )
  train_df_instance.to_feather(
      os.path.join(
          _WRITE_DIR.value,
          f'bag_size_{_BAG_SIZE.value}',
          'instance_overlap_0_train.ftr',
      )
  )

  train_x_candidates, train_y_candidates = get_overlap_candidates(df)
  logging.info('Computing log probs to sample overlapping bags ...')
  log_probs = get_log_probs(train_x_candidates)

  for overlap_percent in OVERLAP_SAMPLING_TEMP[_BAG_SIZE.value]:
    logging.info('Makeing overlap percent: %d', overlap_percent)
    overlap_df = make_dataframe_with_overlaps(
        train_x_candidates, train_y_candidates, log_probs, overlap_percent)
    logging.info('Writing ...')
    overlap_df.to_feather(
        os.path.join(
            _WRITE_DIR.value,
            f'bag_size_{_BAG_SIZE.value}',
            f'bags_overlap_{overlap_percent}_train.ftr',
        )
    )


if __name__ == '__main__':
  app.run(main)
