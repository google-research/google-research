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

"""Utilities for synthetic data."""

import pandas as pd

from mir_uai24 import enum_utils


BAG_SIZE = 5
OVERLAP_PERCENT = 0

TRAIN_BAGS_DATA_PATH = (f'mir_uai24/datasets/synthetic/bag_size_{BAG_SIZE}/'
                        'bags_overlap_{OVERLAP_PERCENT}_train.ftr')
TRAIN_INSTANCE_DATA_PATH = (f'mir_uai24/datasets/synthetic/bag_size_{BAG_SIZE}/'
                            'instance_overlap0_train.ftr')
VAL_DATA_PATH = (f'mir_uai24/datasets/synthetic/bag_size_{BAG_SIZE}/'
                 'overlap0_val.ftr')
TEST_DATA_PATH = (f'mir_uai24/datasets/synthetic/bag_size_{BAG_SIZE}/'
                  'overlap0_test.ftr')


FEATURES = list(map(str, range(32)))


def get_info(
    return_bags_df = False
):
  """Returns synthetic data info.

  Args:
    return_bags_df: Whether to return the loaded bags dataframe.

  Returns:
    If return_bags_df is True, returns a tuple of dataset info and bags
    dataframe. Otherwise, returns dataset info object.
  """
  features = []
  for key in FEATURES:
    features.append(
        enum_utils.Feature(
            key=key,
            type=enum_utils.FeatureType.REAL
        ))
  bags_df = pd.read_feather(TRAIN_BAGS_DATA_PATH)
  n_instances = len(bags_df)*BAG_SIZE

  dataset_info = enum_utils.DatasetInfo(
      bag_id='bag_id',
      instance_id='instance_id',
      bag_id_x_instance_id='bag_id_X_instance_id',
      bag_size=BAG_SIZE,
      n_instances=n_instances,
      features=features,
      label='y',
      memberships=enum_utils.DatasetMembershipInfo(
          instances=dict(), bags=dict()))

  if return_bags_df:
    return dataset_info, bags_df
  return dataset_info
