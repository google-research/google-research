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

"""Semi synthetic mimic-iii data generation."""

import copy
import gzip
import itertools
import logging
import os
import pickle as pkl
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
from src import ROOT_PATH
from src.data.dataset_collection import SyntheticDatasetCollection
from src.data.mimic_iii.load_data import load_mimic3_data_raw
from src.data.mimic_iii.load_data import load_mimic3_data_raw_age_domain
from src.data.mimic_iii.real_dataset import MIMIC3RealDataset
from src.data.mimic_iii.utils import sigmoid
from src.data.mimic_iii.utils import SplineTrendsMixture
import tqdm

deepcopy = copy.deepcopy
Parallel = joblib.Parallel
delayed = joblib.delayed
train_test_split = sklearn.model_selection.train_test_split
tqdm = tqdm.tqdm
logger = logging.getLogger(__name__)


class SyntheticOutcomeGenerator:
  """Generator of synthetic outcome."""

  def __init__(
      self,
      exogeneous_vars,
      exog_dependency,
      exog_weight,
      endo_dependency,
      endo_rand_weight,
      endo_spline_weight,
      outcome_name,
  ):
    """Init.

    Args:
      exogeneous_vars: List of time-varying covariates
      exog_dependency: Callable function of exogeneous_vars (f_Z)
      exog_weight: alpha_f
      endo_dependency: Callable function of endogenous variables (g)
      endo_rand_weight: alpha_g
      endo_spline_weight: alpha_S
      outcome_name: Name of the outcome variable j

    Returns:
      outcome generator
    """
    self.exogeneous_vars = exogeneous_vars
    self.exog_dependency = exog_dependency
    self.exog_weight = exog_weight
    self.endo_rand_weight = endo_rand_weight
    self.endo_spline_weight = endo_spline_weight
    self.endo_dependency = endo_dependency
    self.outcome_name = outcome_name

  def simulate_untreated(
      self, all_vitals, static_features
  ):
    """Simulate untreated outcomes (Z).

    Args:
        all_vitals: Time-varying covariates (as exogeneous vars)
        static_features: Static covariates (as exogeneous vars)

    Returns:
        data
    """
    _ = static_features
    logger.info('%s', f'Simulating untreated outcome {self.outcome_name}')
    user_sizes = all_vitals.groupby(level='subject_id', sort=False).size()

    # Exogeneous dependency
    all_vitals[f'{self.outcome_name}_exog'] = (
        self.exog_weight
        * self.exog_dependency(all_vitals[self.exogeneous_vars].values)
    )

    # Endogeneous dependency + B-spline trend
    time_range = np.arange(0, user_sizes.max())
    y_endo_rand = self.endo_dependency(time_range, len(user_sizes))
    y_endo_splines = SplineTrendsMixture(
        n_patients=len(user_sizes), max_time=user_sizes.max()
    )(time_range)
    y_endo_full = (
        self.endo_rand_weight * y_endo_rand
        + self.endo_spline_weight * y_endo_splines
    )

    arr = []
    for i, l in enumerate(user_sizes):
      for value in y_endo_full[i, :l]:
        arr.append(value)
    all_vitals[f'{self.outcome_name}_endo'] = np.array(arr).reshape(-1, 1)

    # Untreated outcome
    all_vitals[f'{self.outcome_name}_untreated'] = (
        all_vitals[f'{self.outcome_name}_exog']
        + all_vitals[f'{self.outcome_name}_endo']
    )

    # Placeholder for treated outcome
    all_vitals[f'{self.outcome_name}'] = all_vitals[
        f'{self.outcome_name}_untreated'
    ].copy()


class SyntheticTreatment:
  """Generator of synthetic treatment."""

  def __init__(
      self,
      confounding_vars,
      confounder_outcomes,
      confounding_dependency,
      window,
      conf_outcome_weight,
      conf_vars_weight,
      bias,
      full_effect,
      effect_window,
      treatment_name,
      post_nonlinearity = None,
  ):
    """Init.

    Args:
      confounding_vars: Confounding time-varying covariates (from all_vitals)
      confounder_outcomes: Confounding previous outcomes
      confounding_dependency: Callable function of confounding_vars (f_Y)
      window: Window of averaging of confounding previous outcomes (T_l)
      conf_outcome_weight: gamma_Y
      conf_vars_weight: gamma_X
      bias: constant bias
      full_effect: beta
      effect_window: w_l
      treatment_name: Name of treatment l
      post_nonlinearity: Post non-linearity after sigmoid
    """
    self.confounding_vars = confounding_vars
    self.confounder_outcomes = confounder_outcomes
    self.confounding_dependency = confounding_dependency
    self.treatment_name = treatment_name
    self.post_nonlinearity = post_nonlinearity

    # Parameters
    self.window = window
    self.conf_outcome_weight = conf_outcome_weight
    self.conf_vars_weight = conf_vars_weight
    self.bias = bias

    self.full_effect = full_effect
    self.effect_window = effect_window

  def treatment_proba(self, patient_df, t):
    """Calculates propensity score for patient_df and time-step t.

    Args:
        patient_df: DataFrame of patient
        t: Time-step

    Returns:
        propensity score
    """
    t_start = max(0, t - self.window)

    agr_range = np.arange(t_start, t + 1)
    avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
    x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
    f_x = self.confounding_dependency(x)
    treat_proba = sigmoid(
        self.bias
        + self.conf_outcome_weight * avg_y
        + self.conf_vars_weight * f_x
    ).flatten()
    if self.post_nonlinearity is not None:
      treat_proba = self.post_nonlinearity(treat_proba)
    return treat_proba

  def get_treated_outcome(
      self, patient_df, t, outcome_name, treat_proba=1.0, treat=True
  ):
    """Calculate future outcome under treatment, applied at the time-step t.

    Args:
        patient_df: DataFrame of patient
        t: Time-step
        outcome_name: Name of the outcome variable j
        treat_proba: Propensity scores of treatment
        treat: Treatment application flag

    Returns:
        Effect window, treated outcome
    """
    scaled_effect = self.full_effect * treat_proba

    t_stop = min(
        max(patient_df.index.get_level_values('hours_in')),
        t + self.effect_window,
    )
    treatment_range = np.arange(t + 1, t_stop + 1)
    treatment_range_rel = treatment_range - t

    future_outcome = patient_df.loc[treatment_range, outcome_name]
    if treat:
      future_outcome += scaled_effect / treatment_range_rel**0.5
    return treatment_range, future_outcome

  @staticmethod
  def combine_treatments(
      treatment_ranges, treated_future_outcomes, treat_flags
  ):
    """Min combining of different treatment effects.

    Args:
        treatment_ranges: List of effect windows w_l
        treated_future_outcomes: Future outcomes under each individual treatment
        treat_flags: Treatment application flags

    Returns:
        Combined effect window, combined treated outcome
    """
    treated_future_outcomes = pd.concat(treated_future_outcomes, axis=1)
    if treat_flags.any():  # Min combining all the treatments
      common_treatment_range = [
          set(treatment_range)
          for i, treatment_range in enumerate(treatment_ranges)
          if treat_flags[i]
      ]
      common_treatment_range = set.union(*common_treatment_range)
      common_treatment_range = sorted(list(common_treatment_range))
      treated_future_outcomes = treated_future_outcomes.loc[
          common_treatment_range
      ]
      treated_future_outcomes['agg'] = np.nanmin(
          treated_future_outcomes.iloc[:, treat_flags].values, axis=1
      )
    else:  # No treatment is applied
      common_treatment_range = treatment_ranges[0]
      treated_future_outcomes['agg'] = treated_future_outcomes.iloc[
          :, 0
      ]  # Taking untreated outcomes
    return common_treatment_range, treated_future_outcomes['agg']


class MIMIC3SyntheticDataset(MIMIC3RealDataset):
  """Pytorch-style semi-synthetic MIMIC-III dataset."""

  def __init__(
      self,
      all_vitals,
      static_features,
      synthetic_outcomes,
      synthetic_treatments,
      treatment_outcomes_influence,
      subset_name,
      mode='factual',
      projection_horizon = None,
      treatments_seq = None,
      n_treatments_seq = None,
      gt_causal_prediction_for=None,
      factual_resample_num=1,
      data_gen_n_jobs=8,
  ):
    """Init.

    Args:
      all_vitals: DataFrame with vitals (time-varying covariates); multiindex by
        (patient_id, timestep)
      static_features: DataFrame with static features
      synthetic_outcomes: List of SyntheticOutcomeGenerator
      synthetic_treatments: List of SyntheticTreatment
      treatment_outcomes_influence: dict with treatment-outcomes influences
      subset_name: train / val / test
      mode: factual / counterfactual_one_step / counterfactual_treatment_seq
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      treatments_seq: Fixed (non-random) treatment sequecne for
        multiple-step-ahead prediction
      n_treatments_seq: Number of random trajectories after rolling origin in
        test subset
      gt_causal_prediction_for: do prediction of the specified outcome with its
        ground-truth causes only (used for evaluating the improvement of knowing
        causes)
      factual_resample_num: factual resample num
      data_gen_n_jobs: data gen n jobs

    Returns:
      dataset
    """

    self.subset_name = subset_name
    self.all_vitals = all_vitals.copy()
    vital_cols = all_vitals.columns
    self.vital_cols = vital_cols
    self.synthetic_outcomes = synthetic_outcomes
    self.synthetic_treatments = synthetic_treatments
    self.treatment_outcomes_influence = treatment_outcomes_influence

    prev_treatment_cols = [
        f'{treatment.treatment_name}_prev'
        for treatment in self.synthetic_treatments
    ]
    prev_treatment_proba_cols = [
        f'{treatment.treatment_name}_prev_proba'
        for treatment in self.synthetic_treatments
    ]
    outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]

    # Sampling untreated outcomes
    for outcome in self.synthetic_outcomes:
      outcome.simulate_untreated(self.all_vitals, static_features)
    # Placeholders
    for treatment in self.synthetic_treatments:
      self.all_vitals[f'{treatment.treatment_name}_prev'] = 0.0
    self.all_vitals['fact'] = np.nan
    self.all_vitals.loc[(slice(None), 0), 'fact'] = (
        1.0  # First observation is always factual
    )
    user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

    # Treatment application
    seeds = np.random.randint(np.iinfo(np.int32).max, size=len(static_features))
    # par = Parallel(n_jobs=multiprocessing.cpu_count() - 10, backend='loky')
    # par = Parallel(n_jobs=4, backend='loky')
    # par = Parallel(n_jobs=multiprocessing.cpu_count() // 2, backend='loky')
    par = Parallel(n_jobs=data_gen_n_jobs, backend='loky')
    logger.info(
        '%s', f'Simulating {mode} treatments and applying them to outcomes.'
    )
    resampled_dfs = None
    unexploded_all_vitals = []
    if mode == 'factual':
      self.all_vitals = par(
          delayed(self.treat_patient_factually)(patient_ix, seed)
          for patient_ix, seed in tqdm(
              zip(static_features.index, seeds), total=len(static_features)
          )
      )
    elif mode == 'factual_resampling':
      all_vitals_resamples = par(
          delayed(self.treat_patient_factually_resample)(
              patient_ix, seed, factual_resample_num
          )
          for patient_ix, seed in tqdm(
              zip(static_features.index, seeds), total=len(static_features)
          )
      )
      self.all_vitals, resampled_dfs = list(zip(*all_vitals_resamples))
    elif (
        mode == 'counterfactual_treatment_seq'
        or mode == 'counterfactual_one_step'
    ):
      self.treatments_seq = treatments_seq
      if mode == 'counterfactual_one_step':
        treatment_options = [0.0, 1.0]
        self.treatments_seq = np.array(
            list(
                itertools.product(
                    *([treatment_options] * len(prev_treatment_cols))
                )
            )
        )
        self.treatments_seq = self.treatments_seq[:, None, :]
        self.cf_start = 0
      else:
        self.cf_start = 1  # First time-step needed for encoder

      assert (
          projection_horizon is not None and n_treatments_seq is not None
      ) or self.treatments_seq is not None

      if self.treatments_seq is not None:
        self.projection_horizon = self.treatments_seq.shape[1]
        self.n_treatments_seq = self.treatments_seq.shape[0]
      else:
        self.projection_horizon = projection_horizon
        self.n_treatments_seq = n_treatments_seq

      exploded_all_nonvitals = par(
          delayed(self.treat_patient_counterfactually)(patient_ix, seed)
          for patient_ix, seed in tqdm(
              zip(static_features.index, seeds), total=len(static_features)
          )
      )

      logger.info('Concatenating all the trajectories together.')
      exploded_all_nonvitals = [
          pd.concat(
              cf_patient_df, keys=range(len(cf_patient_df)), names=['traj']
          )
          for cf_patient_df in exploded_all_nonvitals
      ]

      logger.info('Keep unexploded vitals')
      unexploded_all_vitals = []
      for patient_ix, _ in tqdm(
          zip(static_features.index, seeds), total=len(static_features)
      ):
        unexploded_all_vitals.append(
            self.all_vitals.loc[patient_ix][self.vital_cols]
        )

      self.all_vitals = exploded_all_nonvitals
    else:
      raise NotImplementedError()

    ids_in_unexploded = None
    if mode == 'factual':
      self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index)
      # Padding with nans
      self.all_vitals = (
          self.all_vitals.unstack(fill_value=np.nan, level=0)
          .stack(dropna=False)
          .swaplevel(0, 1)
          .sort_index()
      )
      static_features = static_features.sort_index()
      static_features = static_features.values
    elif mode == 'factual_resampling':
      resample_nums = [len(x) for x in resampled_dfs]
      orig_index = static_features.index.tolist()
      resample_index = []
      for resample_num, orig_ix in zip(resample_nums, orig_index):
        resample_index.extend([orig_ix] * resample_num)
      all_vitals_index = static_features.index.tolist() + resample_index

      self.all_vitals = list(self.all_vitals)
      for resampled_df_list in resampled_dfs:
        self.all_vitals.extend(resampled_df_list)

      self.all_vitals = pd.concat(
          self.all_vitals, keys=list(range(len(self.all_vitals)))
      )
      self.all_vitals = (
          self.all_vitals.unstack(fill_value=np.nan, level=0)
          .stack(dropna=False)
          .swaplevel(0, 1)
          .sort_index()
      )
      sf_index = all_vitals_index
      static_features = static_features.loc[sf_index]
      static_features = static_features.values
    elif (
        mode == 'counterfactual_one_step'
        or mode == 'counterfactual_treatment_seq'
    ):
      self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index)
      self.all_vitals = (
          self.all_vitals.unstack(fill_value=np.nan, level=2)
          .stack(dropna=False)
          .sort_index()
      )

      self.unexploded_all_vitals = pd.concat(
          unexploded_all_vitals, keys=static_features.index
      )
      self.unexploded_all_vitals = (
          self.unexploded_all_vitals.unstack(fill_value=np.nan, level=0)
          .stack(dropna=False)
          .swaplevel(0, 1)
          .sort_index()
      )

      static_features_exploaded = pd.merge(
          self.all_vitals.groupby(['subject_id', 'traj']).head(1),
          static_features,
          on='subject_id',
      )
      static_features = static_features_exploaded[
          static_features.columns
      ].values

      subject_ids = self.unexploded_all_vitals.index.unique(
          level='subject_id'
      ).to_numpy()
      unexploded_ids = list(range(len(subject_ids)))
      ids_subject_ids = pd.DataFrame(
          {'subject_id': subject_ids, 'unexploded_id': unexploded_ids}
      ).set_index('subject_id')
      ids_in_unexploded = pd.merge(
          self.all_vitals.groupby(['subject_id', 'traj']).head(1),
          ids_subject_ids,
          on='subject_id',
      )
      ids_in_unexploded = ids_in_unexploded['unexploded_id'].values

    # Conversion to np arrays
    max_user_sizes = max(user_sizes)
    treatments = (
        self.all_vitals[prev_treatment_cols]
        .fillna(0.0)
        .values.reshape((-1, max(user_sizes), len(prev_treatment_cols)))
    )
    unexploded_vitals = None
    if vital_cols[0] in self.all_vitals.columns:
      vitals = (
          self.all_vitals[vital_cols]
          .fillna(0.0)
          .values.reshape((-1, max(user_sizes), len(vital_cols)))
      )
    else:
      vitals = None
      unexploded_vitals = (
          self.unexploded_all_vitals[vital_cols]
          .fillna(0.0)
          .values.reshape((-1, max(user_sizes), len(vital_cols)))
      )
    outcomes_unscaled = (
        self.all_vitals[outcome_cols]
        .fillna(0.0)
        .values.reshape((-1, max(user_sizes), len(outcome_cols)))
    )
    active_entries = (~self.all_vitals.isna().all(1)).astype(float)
    active_entries = active_entries.values.reshape((-1, max(user_sizes), 1))
    user_sizes = np.squeeze(active_entries.sum(1))

    if vitals is not None:
      logger.info('%s', f'Shape of exploded vitals: {vitals.shape}.')
    else:
      logger.info(
          '%s',
          f'Only save unexploded vitals with shape: {unexploded_vitals.shape}!',
      )

    self.data = {
        'sequence_lengths': user_sizes - 1,
        'prev_treatments': treatments[:, :-1, :],
        'current_treatments': treatments[:, 1:, :],
        'static_features': static_features,
        'active_entries': active_entries[:, 1:, :],
        'unscaled_outputs': outcomes_unscaled[:, 1:, :],
        'prev_unscaled_outputs': outcomes_unscaled[:, :-1, :],
    }
    if vitals is not None:
      self.data.update({
          'vitals': vitals[:, 1:, :],
          'next_vitals': vitals[:, 2:, :],
      })
    else:
      self.data.update({
          'unexploded_vitals': unexploded_vitals[:, 1:, :],
          'unexploded_next_vitals': unexploded_vitals[:, 2:, :],
          'id_in_unexploded': ids_in_unexploded,
      })

    if prev_treatment_proba_cols[0] in self.all_vitals.columns:
      treatments_proba = (
          self.all_vitals[prev_treatment_proba_cols]
          .fillna(0.0)
          .values.reshape((-1, max_user_sizes, len(prev_treatment_proba_cols)))
      )
      self.data.update({
          'prev_treatments_proba': treatments_proba[:, :-1, :].astype(float),
          'current_treatments_proba': treatments_proba[:, 1:, :].astype(float),
      })

    self.feature_names = {
        'vitals': vital_cols.tolist(),
        'treatments': [
            f'{treatment.treatment_name}'
            for treatment in self.synthetic_treatments
        ],
        'outputs': outcome_cols,
    }

    self.mask_features(gt_causal_prediction_for)

    self.processed = False  # Need for normalisation of newly generated outcomes
    self.processed_sequential = False
    self.processed_autoregressive = False

    self.norm_const = 1.0

  def mask_features(self, gt_causal_prediction_for):
    self.gt_causal_prediction_for = gt_causal_prediction_for
    if gt_causal_prediction_for is not None:
      # assert gt_causal_prediction_for in self.feature_names['outputs']
      logger.info(
          '%s',
          'Masking features as causes for {}'.format(gt_causal_prediction_for),
      )
      pad_noise = False
      vital_causes = []
      treatments_causes = []
      output_names = []
      extra_outputs_causes = []

      if gt_causal_prediction_for == 'y1':
        output_names = ['y1']
        vital_causes = [
            'heart rate',
            'glucose',
            'sodium',
            'blood urea nitrogen',
            'systemic vascular resistance',
            'bicarbonate',
            'anion gap',
        ]
        treatments_causes = ['t1', 't2']
        extra_outputs_causes = ['y2']
      elif gt_causal_prediction_for == 'y1_all':
        output_names = ['y1']
        vital_causes = self.feature_names['vitals']
        treatments_causes = self.feature_names['treatments']
        extra_outputs_causes = ['y2']
      elif gt_causal_prediction_for == 'y1_padnoise':
        output_names = ['y1']
        vital_causes = [
            'heart rate',
            'glucose',
            'sodium',
            'blood urea nitrogen',
            'systemic vascular resistance',
            'bicarbonate',
            'anion gap',
        ]
        treatments_causes = ['t1', 't2']
        extra_outputs_causes = ['y2']
        pad_noise = True
      elif gt_causal_prediction_for == 'y2':
        output_names = ['y2']
        vital_causes = [
            'heart rate',
            'glucose',
            'anion gap',
            'systemic vascular resistance',
            'bicarbonate',
            'chloride urine',
            'sodium',
            'magnesium',
        ]
        treatments_causes = ['t2', 't3']
        extra_outputs_causes = ['y1']
      elif gt_causal_prediction_for == 'y2_all':
        output_names = ['y2']
        vital_causes = self.feature_names['vitals']
        treatments_causes = self.feature_names['treatments']
        extra_outputs_causes = ['y1']
      elif gt_causal_prediction_for == 'y2_padnoise':
        output_names = ['y2']
        vital_causes = [
            'heart rate',
            'glucose',
            'anion gap',
            'systemic vascular resistance',
            'bicarbonate',
            'chloride urine',
            'sodium',
            'magnesium',
        ]
        treatments_causes = ['t2', 't3']
        extra_outputs_causes = ['y1']
        pad_noise = True
      elif gt_causal_prediction_for == 'y1_y1y2':
        output_names = ['y1']
        vital_causes = [
            'heart rate',
            'glucose',
            'sodium',
            'blood urea nitrogen',
            'systemic vascular resistance',
            'bicarbonate',
            'anion gap',
            'chloride urine',
            'magnesium',
        ]
        treatments_causes = ['t1', 't2', 't3']
        extra_outputs_causes = ['y2']
      elif gt_causal_prediction_for == 'y2_y1y2':
        output_names = ['y2']
        vital_causes = [
            'heart rate',
            'glucose',
            'sodium',
            'blood urea nitrogen',
            'systemic vascular resistance',
            'bicarbonate',
            'anion gap',
            'chloride urine',
            'magnesium',
        ]
        treatments_causes = ['t1', 't2', 't3']
        extra_outputs_causes = ['y1']
      elif gt_causal_prediction_for == 'y1y2_y1y2':
        output_names = ['y1', 'y2']
        vital_causes = [
            'heart rate',
            'glucose',
            'sodium',
            'blood urea nitrogen',
            'systemic vascular resistance',
            'bicarbonate',
            'anion gap',
            'chloride urine',
            'magnesium',
        ]
        treatments_causes = ['t1', 't2', 't3']
        extra_outputs_causes = []
      elif gt_causal_prediction_for == 'y1y2_all':
        output_names = ['y1', 'y2']
        vital_causes = self.feature_names['vitals']
        treatments_causes = self.feature_names['treatments']
        extra_outputs_causes = []
      elif gt_causal_prediction_for == 'y1_acd_maxf1':
        output_names = ['y1']
        vital_causes = [
            'heart rate',
            'red blood cell count',
            'sodium',
            'systemic vascular resistance',
            'glucose',
            'chloride urine',
            'glascow coma scale total',
            'hematocrit',
            'cholesterol',
            'hemoglobin',
            'bicarbonate',
            'partial pressure of carbon dioxide',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
        ]
        treatments_causes = ['t1', 't2', 't3']
        extra_outputs_causes = ['y2']
      elif gt_causal_prediction_for == 'y2_acd_maxf1':
        output_names = ['y2']
        vital_causes = [
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
            'cholesterol',
            'hemoglobin',
            'bicarbonate',
            'partial pressure of carbon dioxide',
            'magnesium',
            'anion gap',
            'phosphorous',
            'venous pvo2',
            'platelets',
        ]
        treatments_causes = ['t1', 't2', 't3']
        extra_outputs_causes = ['y1']
      self.gt_causes = {
          'vital': vital_causes,
          'treatments': treatments_causes,
          'extra_outputs': extra_outputs_causes,
      }

      vital_indices = [
          self.feature_names['vitals'].index(f) for f in vital_causes
      ]
      treatments_indices = [
          self.feature_names['treatments'].index(f) for f in treatments_causes
      ]
      outputs_indices = [
          self.feature_names['outputs'].index(f) for f in output_names
      ]
      extra_outputs_indices = [
          self.feature_names['outputs'].index(f) for f in extra_outputs_causes
      ]
      self.gt_causes_indices = {
          'vital': vital_indices,
          'treatments': treatments_indices,
          'outputs': outputs_indices,
          'extra_outputs': extra_outputs_indices,
      }

      noise_rng = np.random.RandomState(seed=42)

      def filter_features(arr, f_indices, pad_noise):
        if not f_indices:
          return None
        if pad_noise:
          ret = noise_rng.uniform(low=-1.0, high=1.0, size=arr.shape)
          ret[Ellipsis, f_indices] = arr[Ellipsis, f_indices]
          return ret
        else:
          return arr[Ellipsis, f_indices]

      data_to_update = {
          'sequence_lengths': self.data['sequence_lengths'],
          'prev_treatments': filter_features(
              self.data['prev_treatments'], treatments_indices, pad_noise
          ),
          'current_treatments': filter_features(
              self.data['current_treatments'], treatments_indices, pad_noise
          ),
          'static_features': np.zeros(
              (self.data['static_features'].shape[0], 1), dtype=np.float32
          ),
          'active_entries': self.data['active_entries'],
          'unscaled_outputs': filter_features(
              self.data['unscaled_outputs'], outputs_indices, pad_noise=False
          ),
          'prev_unscaled_outputs': filter_features(
              self.data['prev_unscaled_outputs'],
              outputs_indices,
              pad_noise=False,
          ),
      }
      if 'vitals' in self.data:
        data_to_update.update({
            'vitals': np.concatenate(
                [
                    x
                    for x in [
                        filter_features(
                            self.data['vitals'], vital_indices, pad_noise
                        ),
                        filter_features(
                            self.data['prev_unscaled_outputs'],
                            extra_outputs_indices,
                            pad_noise=False,
                        ),
                    ]
                    if x is not None
                ],
                axis=-1,
            ),
            'next_vitals': np.concatenate(
                [
                    x
                    for x in [
                        filter_features(
                            self.data['next_vitals'], vital_indices, pad_noise
                        ),
                        filter_features(
                            self.data['prev_unscaled_outputs'][:, 1:],
                            extra_outputs_indices,
                            pad_noise=False,
                        ),
                    ]
                    if x is not None
                ],
                axis=-1,
            ),
        })
      else:
        data_to_update.update({
            'unexploded_vitals': filter_features(
                self.data['unexploded_vitals'], vital_indices, pad_noise
            ),
            'extra_outputs_in_vitals': filter_features(
                self.data['prev_unscaled_outputs'],
                extra_outputs_indices,
                pad_noise=False,
            ),
            'unexploded_next_vitals': filter_features(
                self.data['unexploded_next_vitals'],
                vital_indices,
                pad_noise,
            ),
            'extra_next_outputs_in_vitals': filter_features(
                self.data['prev_unscaled_outputs'][:, 1:],
                extra_outputs_indices,
                pad_noise=False,
            ),
            'id_in_unexploded': self.data['id_in_unexploded'],
        })
      self.data = data_to_update

  def plot_timeseries(self, n_patients=5, mode='factual'):
    """Plotting patient trajectories.

    Args:
        n_patients: Number of trajectories
        mode: factual / counterfactual

    Returns:
        plot
    """
    fig, ax = plt.subplots(
        nrows=4 * len(self.synthetic_outcomes) + len(self.synthetic_treatments),
        ncols=1,
        figsize=(15, 30),
    )
    for i, patient_ix in enumerate(
        self.all_vitals.index.levels[0][:n_patients]
    ):
      ax_ind = 0
      factuals = self.all_vitals.fillna(0.0).fact.astype(bool)
      for outcome in self.synthetic_outcomes:
        outcome_name = outcome.outcome_name
        ax[ax_ind].plot(
            self.all_vitals[factuals]
            .loc[patient_ix, f'{outcome_name}_exog']
            .groupby('hours_in')
            .head(1)
            .values
        )
        ax[ax_ind + 1].plot(
            self.all_vitals[factuals]
            .loc[patient_ix, f'{outcome_name}_endo']
            .groupby('hours_in')
            .head(1)
            .values
        )
        ax[ax_ind + 2].plot(
            self.all_vitals[factuals]
            .loc[patient_ix, f'{outcome_name}_untreated']
            .groupby('hours_in')
            .head(1)
            .values
        )
        if mode == 'factual':
          ax[ax_ind + 3].plot(
              self.all_vitals.loc[patient_ix, outcome_name].values
          )
        elif mode == 'counterfactual':
          # color = next(ax[ax_ind + 3]._get_lines.prop_cycler)['color']
          color = 'k'
          ax[ax_ind + 3].plot(
              self.all_vitals[factuals]
              .loc[patient_ix, outcome_name]
              .groupby('hours_in')
              .head(1)
              .index.get_level_values(1),
              self.all_vitals[factuals]
              .loc[patient_ix, outcome_name]
              .groupby('hours_in')
              .head(1)
              .values,
              color=color,
          )
          ax[ax_ind + 3].scatter(
              self.all_vitals.loc[
                  patient_ix, outcome_name
              ].index.get_level_values(1),
              self.all_vitals.loc[patient_ix, outcome_name].values,
              color=color,
              s=2,
          )

        ax[ax_ind].set_title(f'{outcome_name}_exog')
        ax[ax_ind + 1].set_title(f'{outcome_name}_endo')
        ax[ax_ind + 2].set_title(f'{outcome_name}_untreated')
        ax[ax_ind + 3].set_title(f'{outcome_name}')
        ax_ind += 4

      for treatment in self.synthetic_treatments:
        treatment_name = treatment.treatment_name
        ax[ax_ind].plot(
            self.all_vitals[factuals]
            .loc[patient_ix, f'{treatment_name}_prev']
            .groupby('hours_in')
            .head(1)
            .values
            + 2 * i
        )
        ax[ax_ind].set_title(f'{treatment_name}')
        ax_ind += 1

    fig.suptitle(f'Time series from {self.subset_name}', fontsize=16)
    plt.show()

  def _sample_treatments_from_factuals(
      self, patient_df, t, rng=np.random.RandomState(None)
  ):
    """Sample treatment for patient_df and time-step t.

    Args:
        patient_df: DataFrame of patient
        t: Time-step
        rng: Random numbers generator (for parallelizing)

    Returns:
        Propensity scores, sampled treatments
    """
    factual_patient_df = patient_df[patient_df.fact.astype(bool)]
    treat_probas = {
        treatment.treatment_name: treatment.treatment_proba(
            factual_patient_df, t
        )
        for treatment in self.synthetic_treatments
    }
    treatment_sample = {
        treatment_name: rng.binomial(1, treat_proba)[0]
        for treatment_name, treat_proba in treat_probas.items()
    }
    return treat_probas, treatment_sample

  def _combined_treating(
      self,
      patient_df,
      t,
      outcome,
      treat_probas,
      treat_flags,
  ):
    """Combing application of treatments.

    Args:
        patient_df: DataFrame of patient
        t: Time-step
        outcome: Outcome to treat
        treat_probas: Propensity scores
        treat_flags: Treatment application flags

    Returns:
        Combined effect window, combined treated outcome
    """
    treatment_ranges, treated_future_outcomes = [], []
    influencing_treatments = self.treatment_outcomes_influence[
        outcome.outcome_name
    ]
    influencing_treatments = [
        treatment
        for treatment in self.synthetic_treatments
        if treatment.treatment_name in influencing_treatments
    ]

    for treatment in influencing_treatments:
      treatment_range, treated_future_outcome = treatment.get_treated_outcome(
          patient_df,
          t,
          outcome.outcome_name,
          treat_probas[treatment.treatment_name],
          bool(treat_flags[treatment.treatment_name]),
      )

      treatment_ranges.append(treatment_range)
      treated_future_outcomes.append(treated_future_outcome)

    common_treatment_range, future_outcomes = (
        SyntheticTreatment.combine_treatments(
            treatment_ranges,
            treated_future_outcomes,
            np.array(
                [
                    bool(treat_flags[treatment.treatment_name])
                    for treatment in influencing_treatments
                ]
            ),
        )
    )
    return common_treatment_range, future_outcomes

  def treat_patient_factually(self, patient_ix, seed = None):
    """Generate factually treated outcomes.

    Args:
        patient_ix: Index of patient
        seed: Random seed

    Returns:
        DataFrame of patient
    """
    patient_df = self.all_vitals.loc[patient_ix].copy()
    rng = np.random.RandomState(seed)
    prev_treatment_cols = [
        f'{treatment.treatment_name}_prev'
        for treatment in self.synthetic_treatments
    ]
    prev_treatment_proba_cols = [
        f'{treatment.treatment_name}_prev_proba'
        for treatment in self.synthetic_treatments
    ]

    for t in range(len(patient_df)):
      # Sampling treatments, based on previous factual outcomes
      treat_probas, treat_flags = self._sample_treatments_from_factuals(
          patient_df, t, rng
      )

      if t < max(patient_df.index.get_level_values('hours_in')):
        # Setting factuality flags
        patient_df.loc[t + 1, 'fact'] = 1.0

        # Setting factual sampled treatments
        patient_df.loc[t + 1, prev_treatment_cols] = {
            f'{t}_prev': v for t, v in treat_flags.items()
        }

        # also save probas
        patient_df.loc[t + 1, prev_treatment_proba_cols] = {
            f'{t}_prev_proba': v for t, v in treat_probas.items()
        }

        # Treatments applications
        if sum(treat_flags.values()) > 0:
          # Treating each outcome separately
          for outcome in self.synthetic_outcomes:
            common_treatment_range, future_outcomes = self._combined_treating(
                patient_df, t, outcome, treat_probas, treat_flags
            )
            patient_df.loc[
                common_treatment_range, f'{outcome.outcome_name}'
            ] = future_outcomes

    return patient_df

  def treat_patient_factually_resample(
      self, patient_ix, seed = None, resample_num = 1
  ):
    """Generate factually treated outcomes.

    Args:
        patient_ix: Index of patient
        seed: Random seed
        resample_num: number of resamples

    Returns:
        DataFrame of patient
    """
    patient_df = self.all_vitals.loc[patient_ix].copy()
    rng = np.random.RandomState(seed)
    prev_treatment_cols = [
        f'{treatment.treatment_name}_prev'
        for treatment in self.synthetic_treatments
    ]
    prev_treatment_proba_cols = [
        f'{treatment.treatment_name}_prev_proba'
        for treatment in self.synthetic_treatments
    ]

    resampled_dfs = []
    hist_matching_dfs = []

    for t in range(len(patient_df)):
      input_patient_df = deepcopy(patient_df)

      # Sampling treatments, based on previous factual outcomes
      treat_probas, treat_flags = self._sample_treatments_from_factuals(
          patient_df, t, rng
      )

      if t < max(patient_df.index.get_level_values('hours_in')):
        # Setting factuality flags
        patient_df.loc[t + 1, 'fact'] = 1.0

        # Setting factual sampled treatments
        patient_df.loc[t + 1, prev_treatment_cols] = {
            f'{t}_prev': v for t, v in treat_flags.items()
        }

        # also save probas
        patient_df.loc[t + 1, prev_treatment_proba_cols] = {
            f'{t}_prev_proba': v for t, v in treat_probas.items()
        }

        # Treatments applications
        if sum(treat_flags.values()) > 0:
          # Treating each outcome separately
          for outcome in self.synthetic_outcomes:
            common_treatment_range, future_outcomes = self._combined_treating(
                patient_df, t, outcome, treat_probas, treat_flags
            )
            patient_df.loc[
                common_treatment_range, f'{outcome.outcome_name}'
            ] = future_outcomes

      rng_backup = deepcopy(rng)

      new_sample_needed = max(resample_num - 1 - len(hist_matching_dfs), 0)
      updated_hist_matching_dfs = []
      for i in range(len(hist_matching_dfs) + new_sample_needed):
        if i < len(hist_matching_dfs):
          resampled_df = hist_matching_dfs[i]
        else:
          resampled_df = deepcopy(input_patient_df)
        treat_probas, treat_flags = self._sample_treatments_from_factuals(
            resampled_df, t, rng
        )
        if t < max(input_patient_df.index.get_level_values('hours_in')):
          # Setting factuality flags
          resampled_df.loc[t + 1, 'fact'] = 1.0
          # Setting factual sampled treatments
          resampled_df.loc[t + 1, prev_treatment_cols] = {
              f'{t}_prev': v for t, v in treat_flags.items()
          }
          # also save probas
          resampled_df.loc[t + 1, prev_treatment_proba_cols] = {
              f'{t}_prev_proba': v for t, v in treat_probas.items()
          }
          # Treatments applications
          if sum(treat_flags.values()) > 0:
            # Treating each outcome separately
            for outcome in self.synthetic_outcomes:
              common_treatment_range, future_outcomes = self._combined_treating(
                  resampled_df, t, outcome, treat_probas, treat_flags
              )
              resampled_df.loc[
                  common_treatment_range, f'{outcome.outcome_name}'
              ] = future_outcomes

          matching_last_hist = True
          for prev_treatment_col in prev_treatment_cols:
            if (
                patient_df.loc[t + 1, prev_treatment_col]
                != resampled_df.loc[t + 1, prev_treatment_col]
            ):
              matching_last_hist = False
              break

          if matching_last_hist:
            updated_hist_matching_dfs.append(resampled_df)
          else:
            resampled_dfs.append(resampled_df)
      hist_matching_dfs = updated_hist_matching_dfs

      rng = rng_backup

    resampled_dfs.extend(hist_matching_dfs)

    return patient_df, resampled_dfs

  def treat_patient_counterfactually(self, patient_ix, seed = None):
    """Generate counterfactually treated outcomes.

    Args:
        patient_ix: Index of patient
        seed: Random seed

    Returns:
        DataFrame of patient
    """
    patient_df = self.all_vitals.loc[patient_ix].copy()
    rng = np.random.RandomState(seed)
    prev_treatment_cols = [
        f'{treatment.treatment_name}_prev'
        for treatment in self.synthetic_treatments
    ]
    treatment_options = [0.0, 1.0]  #

    cf_patient_dfs = []

    for t in range(len(patient_df)):
      if self.treatments_seq is None:  # sampling random treatment trajectories
        possible_treatments = rng.choice(
            treatment_options,
            (
                self.n_treatments_seq,
                self.projection_horizon,
                len(self.synthetic_treatments),
            ),
        )
      else:  # using pre-defined trajectories
        possible_treatments = self.treatments_seq

      if t + self.projection_horizon <= max(
          patient_df.index.get_level_values('hours_in')
      ):
        # Counterfactual treatment treatment trajectories
        if t >= self.cf_start:
          possible_patient_dfs = [
              patient_df.copy().loc[: t + self.projection_horizon]
              for _ in range(possible_treatments.shape[0])
          ]

          for time_ind in range(self.projection_horizon):
            for traj_ind, possible_treatment in enumerate(
                possible_treatments[:, time_ind, :]
            ):
              future_treat_probas, _ = self._sample_treatments_from_factuals(
                  possible_patient_dfs[traj_ind], t + time_ind, rng
              )
              future_treat_flags = {
                  treatment.treatment_name: possible_treatment[j]
                  for j, treatment in enumerate(self.synthetic_treatments)
              }

              # Setting treatment trajectories and factualities
              possible_patient_dfs[traj_ind].loc[
                  t + 1 + time_ind, prev_treatment_cols
              ] = {f'{t}_prev': v for t, v in future_treat_flags.items()}

              # Setting pseudo-factuality to ones
              # (needed for proper future_treat_probas)
              possible_patient_dfs[traj_ind].at[t + 1 + time_ind, 'fact'] = 1.0

              # Treating each outcome separately
              for outcome in self.synthetic_outcomes:
                common_treatment_range, future_outcomes = (
                    self._combined_treating(
                        possible_patient_dfs[traj_ind],
                        t + time_ind,
                        outcome,
                        future_treat_probas,
                        future_treat_flags,
                    )
                )
                possible_patient_dfs[traj_ind].loc[
                    common_treatment_range, outcome.outcome_name
                ] = future_outcomes

          # Setting pseudo-factuality to zero
          for possible_patient_df in possible_patient_dfs:
            possible_patient_df.loc[t + 1 :, 'fact'] = 0.0

          # do not save vitals: taking too much RAM!
          for df_i in range(len(possible_patient_dfs)):
            possible_patient_dfs[df_i].drop(
                columns=self.vital_cols, inplace=True
            )

          cf_patient_dfs.extend(possible_patient_dfs)

        # Factual treatment sampling & Application
        treat_probas, treat_flags = self._sample_treatments_from_factuals(
            patient_df, t, rng
        )

        # Setting factuality
        patient_df.loc[t + 1, 'fact'] = 1.0

        # Setting factual sampled treatments
        patient_df.loc[t + 1, prev_treatment_cols] = {
            f'{t}_prev': v for t, v in treat_flags.items()
        }

        # Treating each outcome separately
        if sum(treat_flags.values()) > 0:
          # Treating each outcome separately
          for outcome in self.synthetic_outcomes:
            common_treatment_range, future_outcomes = self._combined_treating(
                patient_df, t, outcome, treat_probas, treat_flags
            )
            patient_df.loc[common_treatment_range, outcome.outcome_name] = (
                future_outcomes
            )

    return cf_patient_dfs

  def get_scaling_params(self):
    outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
    logger.info('Performing normalisation.')
    scaling_params = {
        'output_means': self.all_vitals[outcome_cols].mean(0).to_numpy(),
        'output_stds': self.all_vitals[outcome_cols].std(0).to_numpy(),
    }
    return scaling_params

  def process_data(self, scaling_params):
    """Pre-process dataset for one-step-ahead prediction.

    Args:
        scaling_params: dict of standard normalization parameters (calculated
          with train subset)

    Returns:
        data
    """
    if not self.processed:
      logger.info(
          '%s', f'Processing {self.subset_name} dataset before training'
      )

      output_means = scaling_params['output_means']
      output_stds = scaling_params['output_stds']
      self.data['outputs'] = (
          self.data['unscaled_outputs'] - output_means
      ) / output_stds
      self.data['prev_outputs'] = (
          self.data['prev_unscaled_outputs'] - output_means
      ) / output_stds

      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      self.scaling_params = scaling_params
      self.processed = True
    else:
      logger.info('%s', f'{self.subset_name} Dataset already processed')

    return self.data


class MIMIC3SyntheticDatasetCollection(SyntheticDatasetCollection):
  """Dataset collection.

  (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
  """

  def __init__(
      self,
      path,
      synth_outcomes_list,
      synth_treatments_list,
      treatment_outcomes_influence,
      min_seq_length = None,
      max_seq_length = None,
      max_number = None,
      seed = 100,
      data_seed = 100,
      split=None,
      projection_horizon = 4,
      autoregressive=True,
      n_treatments_seq = None,
      gt_causal_prediction_for=None,
      factual_resample_num=1,
      same_subjs_train_test=False,
      data_gen_n_jobs=8,
      **kwargs,
  ):
    """Init.

    Args:
      path: Path with MIMIC-3 dataset (HDFStore)
      synth_outcomes_list: List of SyntheticOutcomeGenerator
      synth_treatments_list: List of SyntheticTreatment
      treatment_outcomes_influence: dict with treatment-outcomes influences
      min_seq_length: Min sequence lenght in cohort
      max_seq_length: Max sequence lenght in cohort
      max_number: Maximum number of patients in cohort
      seed: Seed for sampling random functions
      data_seed: Seed for random cohort patient selection
      split: Ratio of train / val / test split
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      autoregressive: is auto-regressive
      n_treatments_seq: Number of random trajectories after rolling origin in
        test subset
      gt_causal_prediction_for: do prediction of the specified outcome with its
        ground-truth causes only (used for evaluating the improvement of knowing
        causes)
      factual_resample_num: factual resample num
      same_subjs_train_test: use same subjects in train and test
      data_gen_n_jobs: jobs used in data generation
      **kwargs: kwargs

    Returns:
      data
    """
    super(MIMIC3SyntheticDatasetCollection, self).__init__()
    if not split:
      split = {'val': 0.2, 'test': 0.2}
    self.seed = seed
    np.random.seed(seed)
    self.max_seq_length = max_seq_length
    all_vitals, static_features = load_mimic3_data_raw(
        ROOT_PATH + '/' + path,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        max_number=max_number,
        data_seed=data_seed,
        **kwargs,
    )

    # Train/val/test random_split
    static_features, static_features_test = train_test_split(
        static_features, test_size=split['test'], random_state=seed
    )
    all_vitals, all_vitals_test = (
        all_vitals.loc[static_features.index],
        all_vitals.loc[static_features_test.index],
    )

    if split['val'] > 0.0:
      static_features_train, static_features_val = train_test_split(
          static_features,
          test_size=split['val'] / (1 - split['test']),
          random_state=2 * seed,
      )
      all_vitals_train, all_vitals_val = (
          all_vitals.loc[static_features_train.index],
          all_vitals.loc[static_features_val.index],
      )
    else:
      static_features_train = static_features
      all_vitals_train = all_vitals
      all_vitals_val = None
      static_features_val = None

    if same_subjs_train_test:
      self.train_f = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'train',
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )
      if split['val'] > 0.0:
        self.val_f = MIMIC3SyntheticDataset(
            all_vitals_val,
            static_features_val,
            synth_outcomes_list,
            synth_treatments_list,
            treatment_outcomes_influence,
            'val',
            gt_causal_prediction_for=gt_causal_prediction_for,
            data_gen_n_jobs=data_gen_n_jobs,
        )
      self.test_cf_one_step = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_one_step',
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

      self.test_cf_treatment_seq = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_treatment_seq',
          projection_horizon,
          n_treatments_seq=n_treatments_seq,
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )
    else:
      if factual_resample_num == 1:
        self.train_f = MIMIC3SyntheticDataset(
            all_vitals_train,
            static_features_train,
            synth_outcomes_list,
            synth_treatments_list,
            treatment_outcomes_influence,
            'train',
            gt_causal_prediction_for=gt_causal_prediction_for,
            data_gen_n_jobs=data_gen_n_jobs,
        )
      else:
        self.train_f = MIMIC3SyntheticDataset(
            all_vitals_train,
            static_features_train,
            synth_outcomes_list,
            synth_treatments_list,
            treatment_outcomes_influence,
            'train',
            'factual_resampling',
            gt_causal_prediction_for=gt_causal_prediction_for,
            factual_resample_num=factual_resample_num,
            data_gen_n_jobs=data_gen_n_jobs,
        )

      if split['val'] > 0.0:
        self.val_f = MIMIC3SyntheticDataset(
            all_vitals_val,
            static_features_val,
            synth_outcomes_list,
            synth_treatments_list,
            treatment_outcomes_influence,
            'val',
            gt_causal_prediction_for=gt_causal_prediction_for,
            data_gen_n_jobs=data_gen_n_jobs,
        )
      self.test_cf_one_step = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_one_step',
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

      self.test_cf_treatment_seq = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_treatment_seq',
          projection_horizon,
          n_treatments_seq=n_treatments_seq,
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

    self.projection_horizon = projection_horizon
    self.autoregressive = autoregressive
    self.has_vitals = True
    self.train_scaling_params = self.train_f.get_scaling_params()

  def save_data_in_crn_format(self, savepath):
    pickle_map = {
        'num_time_steps': self.max_seq_length,
        'training_data': self.train_f.data,
        'validation_data': self.val_f.data,
        'test_data': self.test_cf_one_step.data,
        #   'test_data_factuals': test_data_factuals,
        'test_data_seq': self.test_cf_treatment_seq.data,
        # 'test_factual_data': self.test_f_factual,
        'scaling_data': self.train_scaling_params,
    }
    logger.info('%s', 'Saving pickle map to {}'.format(savepath))
    if not os.path.exists(savepath):
      os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(
        savepath, 'mimic_synthetic_seed-{}.pt'.format(self.seed)
    )
    with gzip.open(filepath, 'wb') as f:
      pkl.dump(pickle_map, f)

  def save_to_pkl(self, savepath):
    logger.info('%s', 'Saving dataset collection to {}'.format(savepath))
    if not os.path.exists(savepath):
      os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, 'dataset_collection.pt')
    with gzip.open(filepath, 'wb') as f:
      pkl.dump(self, f)
    finish_flag = os.path.join(savepath, 'finished.txt')
    with open(finish_flag, 'w') as f:
      f.write('finished')


class MIMIC3SyntheticDatasetAgeDomainCollection(SyntheticDatasetCollection):
  """Dataset collection.

  (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
  """

  def __init__(
      self,
      path,
      synth_outcomes_list,
      synth_treatments_list,
      treatment_outcomes_influence,
      min_seq_length = None,
      max_seq_length = None,
      max_number = None,
      seed = 100,
      data_seed = 100,
      split=None,
      projection_horizon = 4,
      autoregressive=True,
      n_treatments_seq = None,
      gt_causal_prediction_for=None,
      factual_resample_num=1,
      src_age_domains=None,
      tgt_age_domains=None,
      few_shot_sample_num=0.01,
      use_few_shot=False,
      data_gen_n_jobs=8,
      use_src_test=False,
      **kwargs,
  ):
    """Init.

    Args:
      path: Path with MIMIC-3 dataset (HDFStore)
      synth_outcomes_list: List of SyntheticOutcomeGenerator
      synth_treatments_list: List of SyntheticTreatment
      treatment_outcomes_influence: dict with treatment-outcomes influences
      min_seq_length: Min sequence lenght in cohort
      max_seq_length: Max sequence lenght in cohort
      max_number: Maximum number of patients in cohort
      seed: Seed for sampling random functions
      data_seed: Seed for random cohort patient selection
      split: Ratio of train / val / test split
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      autoregressive: is autoregressive
      n_treatments_seq: Number of random trajectories after rolling origin in
        test subset
      gt_causal_prediction_for: do prediction of the specified outcome with its
        ground-truth causes only (used for evaluating the improvement of knowing
        causes)
      factual_resample_num: factual resample num
      src_age_domains: source domain age
      tgt_age_domains: target domain age
      few_shot_sample_num: number of few-shot samples
      use_few_shot: use few-shot
      data_gen_n_jobs: jobs used in data generation
      use_src_test: use test data from source domain
      **kwargs: kwargs

    Returns:
      dataset collection
    """
    super(MIMIC3SyntheticDatasetAgeDomainCollection, self).__init__()
    if not split:
      split = {'val': 0.2, 'test': 0.2}
    self.seed = seed
    np.random.seed(seed)
    self.max_seq_length = max_seq_length
    src_vitals, src_static_features, tgt_vitals, tgt_static_features = (
        load_mimic3_data_raw_age_domain(
            ROOT_PATH + '/' + path,
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            max_number=max_number,
            data_seed=data_seed,
            src_age_domains=src_age_domains,
            tgt_age_domains=tgt_age_domains,
            **kwargs,
        )
    )
    if max_number is None or max_number <= 0:
      max_number = np.inf

    src_max_number = min(max_number, len(src_static_features))
    train_num = int(src_max_number * (1 - split['tr_test'] - split['tr_val']))
    val_num = int(src_max_number * split['tr_val'])

    tgt_max_number = min(max_number, len(tgt_static_features))
    test_te_num = int(tgt_max_number * split['te_test'])
    test_val_num = int(tgt_max_number * split['te_val'])

    test_tr_num = 0
    if few_shot_sample_num <= 0:
      test_tr_num = int(
          tgt_max_number * (1 - split['te_val'] - split['te_test'])
      )
    elif isinstance(few_shot_sample_num, float):
      test_tr_num = int(tgt_max_number * few_shot_sample_num)
    elif isinstance(few_shot_sample_num, int):
      test_tr_num = few_shot_sample_num

    logger.info(
        '%s',
        'src_max_number: {}; tgt_max_number: {}'.format(
            src_max_number, tgt_max_number
        ),
    )
    logger.info('%s', 'src train: {}, src val: {}'.format(train_num, val_num))
    logger.info(
        '%s',
        'tgt train: {}, tgt val: {}, tgt test: {}'.format(
            test_tr_num, test_val_num, test_te_num
        ),
    )

    train_val_static_features = src_static_features.loc[
        np.random.choice(
            src_static_features.index, train_num + val_num, replace=False
        )
    ]
    static_features_train, static_features_val = train_test_split(
        train_val_static_features, test_size=val_num, random_state=2 * seed
    )
    all_vitals_train, all_vitals_val = (
        src_vitals.loc[static_features_train.index],
        src_vitals.loc[static_features_val.index],
    )

    static_features_test_trval, static_features_test = train_test_split(
        tgt_static_features, test_size=test_te_num, random_state=3 * seed
    )
    all_vitals_test_trval, all_vitals_test = (
        tgt_vitals.loc[static_features_test_trval.index],
        tgt_vitals.loc[static_features_test.index],
    )
    static_features_test_tr, static_features_test_val = train_test_split(
        static_features_test_trval,
        test_size=test_val_num,
        random_state=4 * seed,
    )
    all_vitals_test_tr, all_vitals_test_val = (
        all_vitals_test_trval.loc[static_features_test_tr.index],
        all_vitals_test_trval.loc[static_features_test_val.index],
    )

    static_features_test_tr = static_features_test_tr.loc[
        np.random.choice(
            static_features_test_tr.index, test_tr_num, replace=False
        )
    ]
    all_vitals_test_tr = all_vitals_test_tr.loc[static_features_test_tr.index]

    self.test_train_f = MIMIC3SyntheticDataset(
        all_vitals_test_tr,
        static_features_test_tr,
        synth_outcomes_list,
        synth_treatments_list,
        treatment_outcomes_influence,
        'train',
        gt_causal_prediction_for=gt_causal_prediction_for,
        data_gen_n_jobs=data_gen_n_jobs,
    )
    self.test_val_f = MIMIC3SyntheticDataset(
        all_vitals_test_val,
        static_features_test_val,
        synth_outcomes_list,
        synth_treatments_list,
        treatment_outcomes_influence,
        'val',
        gt_causal_prediction_for=gt_causal_prediction_for,
        data_gen_n_jobs=data_gen_n_jobs,
    )
    self.src_train_f = MIMIC3SyntheticDataset(
        all_vitals_train,
        static_features_train,
        synth_outcomes_list,
        synth_treatments_list,
        treatment_outcomes_influence,
        'train',
        gt_causal_prediction_for=gt_causal_prediction_for,
        data_gen_n_jobs=data_gen_n_jobs,
    )

    self.src_val_f = MIMIC3SyntheticDataset(
        all_vitals_val,
        static_features_val,
        synth_outcomes_list,
        synth_treatments_list,
        treatment_outcomes_influence,
        'val',
        gt_causal_prediction_for=gt_causal_prediction_for,
        data_gen_n_jobs=data_gen_n_jobs,
    )
    if use_few_shot:
      self.train_f, self.val_f = self.test_train_f, self.test_val_f
    else:
      self.train_f, self.val_f = self.src_train_f, self.src_val_f

    if not use_src_test:
      self.test_cf_one_step = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_one_step',
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

      self.test_cf_treatment_seq = MIMIC3SyntheticDataset(
          all_vitals_test,
          static_features_test,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_treatment_seq',
          projection_horizon,
          n_treatments_seq=n_treatments_seq,
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

    else:
      # we have to use val as src_test
      # since src_test is not reserved in previous training
      # for fair comparison we compare the best perf of every method on val
      self.test_cf_one_step = MIMIC3SyntheticDataset(
          all_vitals_val,
          static_features_val,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_one_step',
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

      self.test_cf_treatment_seq = MIMIC3SyntheticDataset(
          all_vitals_val,
          static_features_val,
          synth_outcomes_list,
          synth_treatments_list,
          treatment_outcomes_influence,
          'test',
          'counterfactual_treatment_seq',
          projection_horizon,
          n_treatments_seq=n_treatments_seq,
          gt_causal_prediction_for=gt_causal_prediction_for,
          data_gen_n_jobs=data_gen_n_jobs,
      )

    # if not os.path.exists(cached_data_dir):
    #     os.makedirs(cached_data_dir)
    # with gzip.open(cached_data_path, 'wb') as f:
    #     pkl.dump({
    #         'train_f': self.train_f, 'val_f': self.val_f,
    #         'test_cf_one_step': self.test_cf_one_step,
    #         'test_cf_treatment_seq': self.test_cf_treatment_seq
    #     }, f)

    self.projection_horizon = projection_horizon
    self.autoregressive = autoregressive
    self.has_vitals = True
    self.train_scaling_params = self.src_train_f.get_scaling_params()

  def save_data_in_crn_format(self, savepath):
    pickle_map = {
        'num_time_steps': self.max_seq_length,
        'training_data': self.train_f.data,
        'validation_data': self.val_f.data,
        'test_data': self.test_cf_one_step.data,
        #   'test_data_factuals': test_data_factuals,
        'test_data_seq': self.test_cf_treatment_seq.data,
        # 'test_factual_data': self.test_f_factual,
        'scaling_data': self.train_scaling_params,
    }
    logger.info('%s', 'Saving pickle map to {}'.format(savepath))
    if not os.path.exists(savepath):
      os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(
        savepath, 'mimic_synthetic_seed-{}.pt'.format(self.seed)
    )
    with gzip.open(filepath, 'wb') as f:
      pkl.dump(pickle_map, f)

  def save_to_pkl(self, savepath):
    logger.info('%s', 'Saving dataset collection to {}'.format(savepath))
    if not os.path.exists(savepath):
      os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(savepath, 'dataset_collection.pt')
    with gzip.open(filepath, 'wb') as f:
      pkl.dump(self, f)
    finish_flag = os.path.join(savepath, 'finished.txt')
    with open(finish_flag, 'w') as f:
      f.write('finished')
