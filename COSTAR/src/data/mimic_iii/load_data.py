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

"""Load mimic-iii data."""

import copy
import logging
from typing import List

import numpy as np
import omegaconf
import pandas as pd

deepcopy = copy.deepcopy
ListConfig = omegaconf.ListConfig
logger = logging.getLogger(__name__)


def process_static_features(
    static_features, drop_first=False
):
  """Global standard normalisation of static features & one hot encoding.

  Args:
      static_features: pd.DataFrame with unprocessed static features
      drop_first: Dropping first class of one-hot-encoded features

  Returns:
      pd.DataFrame with pre-processed static features
  """
  processed_static_features = []
  for feature in static_features.columns:
    if isinstance(static_features[feature].iloc[0], float):
      mean = np.mean(static_features[feature])
      std = np.std(static_features[feature])
      processed_static_features.append((static_features[feature] - mean) / std)
    else:
      one_hot = pd.get_dummies(static_features[feature], drop_first=drop_first)
      processed_static_features.append(one_hot.astype(float))

  static_features = pd.concat(processed_static_features, axis=1)
  return static_features


def load_mimic3_data_processed(
    data_path,
    min_seq_length = None,
    max_seq_length = None,
    treatment_list = None,
    outcome_list = None,
    vital_list = None,
    static_list = None,
    max_number = None,
    data_seed = 100,
    drop_first=False,
    **kwargs,
):
  """Load and pre-process MIMIC-3 hourly averaged dataset.

  (for real-world experiments)

  Args:
    data_path: Path with MIMIC-3 dataset (HDFStore)
    min_seq_length: Min sequence lenght in cohort
    max_seq_length: Max sequence lenght in cohort
    treatment_list: List of treaments
    outcome_list: List of outcomes
    vital_list: List of vitals (time-varying covariates)
    static_list: List of static features
    max_number: Maximum number of patients in cohort
    data_seed: Seed for random cohort patient selection
    drop_first: Dropping first class of one-hot-encoded features
    **kwargs: kwargs

  Returns:
    tuple of DataFrames and params (treatments, outcomes, vitals,
  static_features, outcomes_unscaled, scaling_params)
  """

  logger.info('%s', f'Loading MIMIC-III dataset from {data_path}.')
  _ = kwargs

  h5 = pd.HDFStore(data_path, 'r')
  if treatment_list is None:
    treatment_list = ['vaso', 'vent']
  if outcome_list is None:
    outcome_list = ['diastolic blood pressure', 'oxygen saturation']
  else:
    outcome_list = ListConfig(
        [outcome.replace('_', ' ') for outcome in outcome_list]
    )
  if vital_list is None:
    vital_list = [
        'heart rate',
        'red blood cell count',
        'sodium',
        'mean blood pressure',
        'systemic vascular resistance',
        'glucose',
        'chloride urine',
        'glascow coma scale total',
        'hematocrit',
        'positive end-expiratory pressure set',
        'respiratory rate',
        'prothrombin time pt',
        'cholesterol',
        'hemoglobin',
        'creatinine',
        'blood urea nitrogen',
        'bicarbonate',
        'calcium ionized',
        'partial pressure of carbon dioxide',
        'magnesium',
        'anion gap',
        'phosphorous',
        'venous pvo2',
        'platelets',
        'calcium urine',
    ]
  if static_list is None:
    static_list = ['gender', 'ethnicity', 'age']

  treatments = h5['/interventions'][treatment_list]
  all_vitals = h5['/vitals_labs_mean'][outcome_list + vital_list]
  static_features = h5['/patients'][static_list]

  treatments = treatments.droplevel(['hadm_id', 'icustay_id'])
  all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
  column_names = []
  for column in all_vitals.columns:
    if isinstance(column, str):
      column_names.append(column)
    else:
      column_names.append(column[0])
  all_vitals.columns = column_names
  static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

  # Filling NA
  all_vitals = all_vitals.fillna(method='ffill')
  all_vitals = all_vitals.fillna(method='bfill')

  # Filtering longer then min_seq_length and cropping to max_seq_length
  user_sizes = all_vitals.groupby('subject_id').size()
  filtered_users = (
      user_sizes.index[user_sizes >= min_seq_length]
      if min_seq_length is not None
      else user_sizes.index
  )
  if max_number is not None and max_number > 0:
    np.random.seed(data_seed)
    filtered_users = np.random.choice(
        filtered_users, size=max_number, replace=False
    )
  treatments = treatments.loc[filtered_users]
  all_vitals = all_vitals.loc[filtered_users]
  if max_seq_length is not None:
    treatments = treatments.groupby('subject_id').head(max_seq_length)
    all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
  static_features = static_features[static_features.index.isin(filtered_users)]
  logger.info('%s', f'Number of patients filtered: {len(filtered_users)}.')

  # Global scaling (same as with semi-synthetic)
  outcomes_unscaled = all_vitals[outcome_list].copy()
  mean = np.mean(all_vitals, axis=0)
  std = np.std(all_vitals, axis=0)
  all_vitals = (all_vitals - mean) / std

  # Splitting outcomes and vitals
  outcomes = all_vitals[outcome_list].copy()
  vitals = all_vitals[vital_list].copy()
  static_features = process_static_features(
      static_features, drop_first=drop_first
  )
  scaling_params = {
      'output_means': mean[outcome_list].to_numpy(),
      'output_stds': std[outcome_list].to_numpy(),
  }

  h5.close()
  return (
      treatments,
      outcomes,
      vitals,
      static_features,
      outcomes_unscaled,
      scaling_params,
  )


def load_mimic3_data_raw(
    data_path,
    min_seq_length = None,
    max_seq_length = None,
    max_number = None,
    vital_list = None,
    static_list = None,
    data_seed = 100,
    drop_first=False,
    **kwargs,
):
  """Load MIMIC-3 hourly averaged dataset.

  without preprocessing (for semi-synthetic experiments)

  Args:
    data_path: Path with MIMIC-3 dataset (HDFStore)
    min_seq_length: Min sequence lenght in cohort
    max_seq_length: Max sequence length in cohort
    max_number: Maximum number of patients in cohort
    vital_list: List of vitals (time-varying covariates)
    static_list: List of static features
    data_seed: Seed for random cohort patient selection
    drop_first: Dropping first class of one-hot-encoded features
    **kwargs: kwargs

  Returns:
    Tuple of DataFrames (all_vitals, static_features)
  """
  logger.info('%s', f'Loading MIMIC-III dataset from {data_path}.')
  _ = kwargs

  h5 = pd.HDFStore(data_path, 'r')
  if vital_list is None:
    vital_list = [
        'heart rate',
        'red blood cell count',
        'sodium',
        'mean blood pressure',
        'systemic vascular resistance',
        'glucose',
        'chloride urine',
        'glascow coma scale total',
        'hematocrit',
        'positive end-expiratory pressure set',
        'respiratory rate',
        'prothrombin time pt',
        'cholesterol',
        'hemoglobin',
        'creatinine',
        'blood urea nitrogen',
        'bicarbonate',
        'calcium ionized',
        'partial pressure of carbon dioxide',
        'magnesium',
        'anion gap',
        'phosphorous',
        'venous pvo2',
        'platelets',
        'calcium urine',
    ]
  if static_list is None:
    static_list = ['gender', 'ethnicity', 'age']

  all_vitals = h5['/vitals_labs_mean'][vital_list]
  static_features = h5['/patients'][static_list]

  all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
  column_names = []
  for column in all_vitals.columns:
    if isinstance(column, str):
      column_names.append(column)
    else:
      column_names.append(column[0])
  all_vitals.columns = column_names
  static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

  # Filling NA
  all_vitals = all_vitals.fillna(method='ffill')
  all_vitals = all_vitals.fillna(method='bfill')

  # Filtering longer then min_seq_length and cropping to max_seq_length
  user_sizes = all_vitals.groupby('subject_id').size()
  filtered_users = (
      user_sizes.index[user_sizes >= min_seq_length]
      if min_seq_length is not None
      else user_sizes.index
  )
  if max_number is not None and max_number > 0:
    np.random.seed(data_seed)
    filtered_users = np.random.choice(
        filtered_users, size=max_number, replace=False
    )
  all_vitals = all_vitals.loc[filtered_users]
  static_features = static_features.loc[filtered_users]
  if max_seq_length is not None:
    all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
  logger.info('%s', f'Number of patients filtered: {len(filtered_users)}.')

  # Global Mean-Std Normalisation
  mean = np.mean(all_vitals, axis=0)
  std = np.std(all_vitals, axis=0)
  all_vitals = (all_vitals - mean) / std

  static_features = process_static_features(
      static_features, drop_first=drop_first
  )

  h5.close()
  return all_vitals, static_features


def load_mimic3_data_raw_age_domain(
    data_path,
    min_seq_length = None,
    max_seq_length = None,
    max_number = None,
    vital_list = None,
    static_list = None,
    data_seed = 100,
    drop_first=False,
    src_age_domains=None,
    tgt_age_domains=None,
    **kwargs,
):
  """Load MIMIC-3 hourly averaged dataset.

  without preprocessing (for semi-synthetic experiments)

  Args:
    data_path: Path with MIMIC-3 dataset (HDFStore)
    min_seq_length: Min sequence lenght in cohort
    max_seq_length: Max sequence length in cohort
    max_number: Maximum number of patients in cohort
    vital_list: List of vitals (time-varying covariates)
    static_list: List of static features
    data_seed: Seed for random cohort patient selection
    drop_first: Dropping first class of one-hot-encoded features
    src_age_domains: source age domain
    tgt_age_domains: target age domain
    **kwargs: kwargs

  Returns:
    Tuple of DataFrames (all_vitals, static_features)
  """
  logger.info('%s', f'Loading MIMIC-III dataset from {data_path}.')
  _, _, _ = max_number, data_seed, kwargs

  h5 = pd.HDFStore(data_path, 'r')
  if vital_list is None:
    vital_list = [
        'heart rate',
        'red blood cell count',
        'sodium',
        'mean blood pressure',
        'systemic vascular resistance',
        'glucose',
        'chloride urine',
        'glascow coma scale total',
        'hematocrit',
        'positive end-expiratory pressure set',
        'respiratory rate',
        'prothrombin time pt',
        'cholesterol',
        'hemoglobin',
        'creatinine',
        'blood urea nitrogen',
        'bicarbonate',
        'calcium ionized',
        'partial pressure of carbon dioxide',
        'magnesium',
        'anion gap',
        'phosphorous',
        'venous pvo2',
        'platelets',
        'calcium urine',
    ]
  if static_list is None:
    static_list = ['gender', 'ethnicity', 'age']

  all_vitals = h5['/vitals_labs_mean'][vital_list]
  static_features = h5['/patients'][static_list]

  all_vitals = all_vitals.droplevel(['hadm_id', 'icustay_id'])
  column_names = []
  for column in all_vitals.columns:
    if isinstance(column, str):
      column_names.append(column)
    else:
      column_names.append(column[0])
  all_vitals.columns = column_names
  static_features = static_features.droplevel(['hadm_id', 'icustay_id'])

  # Filling NA
  all_vitals = all_vitals.fillna(method='ffill')
  all_vitals = all_vitals.fillna(method='bfill')

  # Filtering longer then min_seq_length and cropping to max_seq_length
  user_sizes = all_vitals.groupby('subject_id').size()
  filtered_users = (
      user_sizes.index[user_sizes >= min_seq_length]
      if min_seq_length is not None
      else user_sizes.index
  )
  all_vitals = all_vitals.loc[filtered_users]
  static_features = static_features.loc[filtered_users]
  if max_seq_length is not None:
    all_vitals = all_vitals.groupby('subject_id').head(max_seq_length)
  logger.info('%s', f'Number of patients filtered: {len(filtered_users)}.')

  # Global Mean-Std Normalisation
  mean = np.mean(all_vitals, axis=0)
  std = np.std(all_vitals, axis=0)
  all_vitals = (all_vitals - mean) / std

  ages = deepcopy(static_features['age'])
  static_features = process_static_features(
      static_features, drop_first=drop_first
  )

  # Group 0: [20, 45]; Group 1: [46, 65]; Group 2: [66, 85]; Group 3: [85, inf)
  age_bins = [19, 45, 65, 85, np.inf]
  age_groups = pd.cut(ages, bins=age_bins).cat.codes
  src_users = age_groups[age_groups.isin(src_age_domains)].index
  tgt_users = age_groups[age_groups.isin(tgt_age_domains)].index

  src_vitals, src_static_features = (
      all_vitals.loc[src_users],
      static_features.loc[src_users],
  )
  tgt_vitals, tgt_static_features = (
      all_vitals.loc[tgt_users],
      static_features.loc[tgt_users],
  )

  h5.close()
  return src_vitals, src_static_features, tgt_vitals, tgt_static_features
