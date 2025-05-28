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

"""Utils for TGB experiments."""

import json
import os
import pickle
import random
import sys
from typing import Any

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf



def save_pkl(obj, fname):
  r"""save a python object as a pickle file"""
  with tf.io.gfile.GFile(fname, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname):
  r"""load a python object from a pickle file"""
  with tf.io.gfile.GFile(fname, 'rb') as handle:
    return pickle.load(handle)


def set_random_seed(seed):
  r"""setting random seed for reproducibility"""
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]


def save_results(new_results, filename):
  r"""save (new) results into a json file

  :param: new_results (dictionary): a dictionary of new results to be saved
  :filename: the name of the file to save the (new) results
  """

  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)

  with tf.io.gfile.GFile(filename, 'w') as json_file:
    json.dump(new_results, json_file, indent=4)


def _load_batch_stats(batch_filename):
  r"""Load structural features for a given batch."""
  batch_index = int(batch_filename.split('_')[-1].split('.')[0])
  with tf.io.gfile.GFile(batch_filename, 'rb') as f:
    batch_stats = pickle.load(f)
    batch_stats['batch_index'] = batch_index
  for feature_key, feature_values in batch_stats.items():
    if feature_key in ['src', 'dst', 'batch_index']:
      continue
    for node_id in sorted(feature_values):
      node_features = feature_values.pop(node_id)
      feature_values[int(node_id)] = node_features
  return batch_stats


def _compute_total_structural_feature_mean_std(
    structural_features,
    feature_dim,
    structural_feats_list,
):
  r"""Compute the dimension-wise mean and standard deviation from the train data."""

  features_sum = np.zeros(feature_dim)
  features_squared_sum = np.zeros(feature_dim)
  total_samples = 0
  for unused_batch_index, batch_features in structural_features.items():
    present_structural_feats_list = [
        feat_name for feat_name in structural_feats_list
        if feat_name in batch_features
    ]
    assert present_structural_feats_list
    for node in batch_features[present_structural_feats_list[0]]:
      node_feat = np.concatenate([
          batch_features[feature_key][node]
          for feature_key in present_structural_feats_list
      ])
      features_sum += node_feat
      features_squared_sum += node_feat**2
      total_samples += 1
  features_mean = features_sum / total_samples
  features_std = np.sqrt(
      features_squared_sum / total_samples - features_mean**2
  )
  features_std[np.where(features_std == 0)] = 0.0001
  return list(features_mean), list(features_std)


def _compute_total_structural_feature_dimension(
    structural_features,
):
  r"""Compute the total structural feature dimension."""
  total_feature_dimension = 0
  batch_count = 0
  for batch_index, batch_features in sorted(structural_features.items()):
    batch_total_feature_dimension = 0
    for feature_key, feature_value in batch_features.items():
      if feature_key in ['src', 'dst']:
        continue
      feature_dimension = 0
      for idx, (node_id, node_features) in enumerate(feature_value.items()):
        if idx == 0:
          feature_dimension = len(node_features)
        if len(node_features) != feature_dimension:
          raise ValueError(
              f'Structural feature dimension mismatch for node {node_id}.'
          )
      batch_total_feature_dimension += feature_dimension
    if batch_count == 0:
      total_feature_dimension = batch_total_feature_dimension
    if batch_total_feature_dimension != total_feature_dimension:
      raise ValueError(
          f'Structural feature dimension mismatch for batch {batch_index}.'
      )
    batch_count += 1
  return total_feature_dimension


def load_structural_features(
    dataset_root,
    data_name,
    community,
    split,
    num_workers = 10,
    structural_feature_file_tag = '',
    structural_feats_list = [],
):
  r"""Load structural features for a given dataset split.

  Structural feature filenames are formatted like
    {data}_{community}_{split}{tag_str}_structural_features_{batch_index}.pkl
  where tag_str is equivalent to _{structural_feature_file_tag}.

  Returns:
    structural_features: a dict of structural features for each batch.
    feature dimension: the total structural feature dimension.
  """
  batches_root = os.path.join(dataset_root, 'structural_features_by_batch')
  filename_prefix = data_name + '_' + community + '_' + split
  if structural_feature_file_tag:
    filename_prefix += '_' + structural_feature_file_tag
  batch_filenames = tf.io.gfile.glob(
      os.path.join(
          batches_root,
          filename_prefix + '_structural_features_*',
      )
  )
  logging.info(
      f'Loading structural features from {len(batch_filenames)} batches.'
  )

  structural_feature_dicts = []
  if 'parallel' not in sys.modules:
    for batch_filename in batch_filenames:
      structural_feature_dicts.append(_load_batch_stats(batch_filename))
  structural_features = {}
  for batch_stats in structural_feature_dicts:
    batch_index = batch_stats['batch_index']
    structural_features[batch_index] = batch_stats
    del structural_features[batch_index]['batch_index']
  feature_dim = _compute_total_structural_feature_dimension(structural_features)
  if split == 'train':
    feature_mean, feature_std = _compute_total_structural_feature_mean_std(
        structural_features, feature_dim, structural_feats_list
    )
  else:
    feature_mean, feature_std = [], []
  return structural_features, feature_dim, feature_mean, feature_std


def _get_structural_measurement_filepath_from_model_path(
    model_path,
    measurement_name,
):
  model_path_basename = os.path.basename(model_path)
  model_id = model_path_basename.split('.')[0]
  return os.path.join(
      os.path.dirname(model_path), f'{model_id}---{measurement_name}.json'
  )


def save_structural_feature_measurement(
    model_path,
    measurement_name,
    measurement,
):
  sf_filepath = _get_structural_measurement_filepath_from_model_path(
      model_path, measurement_name
  )
  with tf.io.gfile.GFile(sf_filepath, 'w') as f:
    f.write(json.dumps(measurement))


def load_structural_feature_measurement(
    model_path,
    measurement_name,
):
  sf_filepath = _get_structural_measurement_filepath_from_model_path(
      model_path, measurement_name
  )
  measurement = []
  with tf.io.gfile.GFile(sf_filepath, 'r') as f:
    measurement.extend(json.loads(f.read()))
  return measurement
