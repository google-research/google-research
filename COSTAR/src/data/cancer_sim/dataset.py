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

"""Cancer sim dataset."""

import collections
import copy
import gzip
import logging
import os
import pickle as pkl

import numpy as np
from src.data.cancer_sim.cancer_simulation import generate_params
from src.data.cancer_sim.cancer_simulation import get_scaling_params
from src.data.cancer_sim.cancer_simulation import simulate_counterfactual_1_step
from src.data.cancer_sim.cancer_simulation import simulate_counterfactuals_treatment_seq
from src.data.cancer_sim.cancer_simulation import simulate_factual
from src.data.cancer_sim.cancer_simulation import TUMOUR_DEATH_THRESHOLD
from src.data.dataset_collection import SyntheticDatasetCollection
import torch

DefaultDict = collections.defaultdict
deepcopy = copy.deepcopy
Dataset = torch.utils.data.Dataset

logger = logging.getLogger(__name__)


class SyntheticCancerDataset(Dataset):
  """Pytorch-style Dataset of Tumor Growth Simulator datasets."""

  def __init__(
      self,
      chemo_coeff,
      radio_coeff,
      num_patients,
      window_size,
      seq_length,
      subset_name,
      mode='factual',
      projection_horizon = None,
      seed=None,
      lag = 0,
      cf_seq_mode = 'sliding_treatment',
      treatment_mode = 'multiclass',
      reuse_params=None,
  ):
    """Synthetic cancer sim dataset.

    Args:
      chemo_coeff: Confounding coefficient of chemotherapy
      radio_coeff: Confounding coefficient of radiotherapy
      num_patients: Number of patients in dataset
      window_size: Used for biased treatment assignment
      seq_length: Max length of time series
      subset_name: train / val / test
      mode: factual / counterfactual_one_step / counterfactual_treatment_seq
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      seed: Random seed
      lag: Lag for treatment assignment window
      cf_seq_mode: sliding_treatment / random_trajectories
      treatment_mode: multiclass / multilabel
      reuse_params: reuse_params
    """

    if seed is not None:
      np.random.seed(seed)

    self.chemo_coeff = chemo_coeff
    self.radio_coeff = radio_coeff
    self.window_size = window_size
    self.num_patients = num_patients
    if reuse_params is None:
      self.params = generate_params(
          num_patients,
          chemo_coeff=chemo_coeff,
          radio_coeff=radio_coeff,
          window_size=window_size,
          lag=lag,
      )
    else:
      self.params = reuse_params
    self.subset_name = subset_name

    if mode == 'factual':
      self.data = simulate_factual(self.params, seq_length)
    elif mode == 'counterfactual_one_step':
      self.data = simulate_counterfactual_1_step(self.params, seq_length)
    elif mode == 'counterfactual_treatment_seq':
      assert projection_horizon is not None
      self.data = simulate_counterfactuals_treatment_seq(
          self.params, seq_length, projection_horizon, cf_seq_mode
      )
    self.processed = False
    self.processed_sequential = False
    self.processed_autoregressive = False
    self.treatment_mode = treatment_mode
    self.exploded = False

    self.norm_const = TUMOUR_DEATH_THRESHOLD

  def __getitem__(self, index):
    result = {}
    if 'augmented_lengths_csum' in self.data:
      real_index = (
          np.searchsorted(
              self.data['augmented_lengths_csum'], index, side='right'
          )
          - 1
      )
      future_past_split = (
          index - self.data['augmented_lengths_csum'][real_index] + 1
      )
      result['future_past_split'] = torch.tensor(future_past_split).long()
      index = real_index
      result.update(
          {
              k: torch.as_tensor(v[index]).to(torch.get_default_dtype())
              for k, v in self.data.items()
              if hasattr(v, '__len__')
              and len(v) == self.data['current_covariates'].shape[0]
          }
      )
    else:
      result = {
          k: torch.as_tensor(v[index]).to(torch.get_default_dtype())
          for k, v in self.data.items()
          if hasattr(v, '__len__') and len(v) == len(self)
      }
    if hasattr(self, 'encoder_r'):
      if 'original_index' in self.data:
        result.update(
            {
                'encoder_r': torch.as_tensor(
                    self.encoder_r[int(result['original_index'])]
                ).to(torch.get_default_dtype())
            }
        )
      else:
        result.update(
            {
                'encoder_r': torch.as_tensor(self.encoder_r[index]).to(
                    torch.get_default_dtype()
                )
            }
        )
    return result

  def __len__(self):
    if 'augmented_lengths_csum' in self.data:
      return self.data['augmented_lengths_csum'][-1]
    else:
      return self.data['current_covariates'].shape[0]

  def get_scaling_params(self):
    return get_scaling_params(self.data)

  def few_shot_sampling(self, few_shot_sample_num, seed):
    assert not self.processed
    full_length = len(self.data['sequence_lengths'])
    rng = np.random.RandomState(seed)
    few_shot_indices = rng.choice(
        full_length, few_shot_sample_num, replace=False
    )
    for k in self.data:
      if self.data[k].shape[0] == full_length:
        self.data[k] = self.data[k][few_shot_indices]

  def process_data(self, scaling_params):
    """Pre-process dataset for one-step-ahead prediction.

    Args:
        scaling_params: dict of standard normalization parameters (calculated
          with train subset)

    Returns:
        dict self.data
    """
    if not self.processed:
      logger.info(
          '%s', f'Processing {self.subset_name} dataset before training'
      )

      mean, std = scaling_params

      horizon = 1
      offset = 1

      mean['chemo_application'] = 0
      mean['radio_application'] = 0
      std['chemo_application'] = 1
      std['radio_application'] = 1

      input_means = mean[[
          'cancer_volume',
          'patient_types',
          'chemo_application',
          'radio_application',
      ]].values.flatten()
      input_stds = std[[
          'cancer_volume',
          'patient_types',
          'chemo_application',
          'radio_application',
      ]].values.flatten()

      # Continuous values
      cancer_volume = (
          self.data['cancer_volume'] - mean['cancer_volume']
      ) / std['cancer_volume']
      patient_types = (
          self.data['patient_types'] - mean['patient_types']
      ) / std['patient_types']

      patient_types = np.stack(
          [patient_types for _ in range(cancer_volume.shape[1])], axis=1
      )

      # Binary application
      chemo_application = self.data['chemo_application']
      radio_application = self.data['radio_application']
      sequence_lengths = self.data['sequence_lengths']

      # Convert prev_treatments to one-hot encoding

      treatments = np.concatenate(
          [
              chemo_application[:, :-offset, np.newaxis],
              radio_application[:, :-offset, np.newaxis],
          ],
          axis=-1,
      )

      if self.treatment_mode == 'multiclass':
        one_hot_treatments = np.zeros(
            shape=(treatments.shape[0], treatments.shape[1], 4)
        )
        for patient_id in range(treatments.shape[0]):
          for timestep in range(treatments.shape[1]):
            if (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 0
            ):
              one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 0
            ):
              one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 1
            ):
              one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 1
            ):
              one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

        one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

        self.data['prev_treatments'] = one_hot_previous_treatments
        self.data['current_treatments'] = one_hot_treatments

      elif self.treatment_mode == 'multilabel':
        self.data['prev_treatments'] = treatments[:, :-1, :]
        self.data['current_treatments'] = treatments

      current_covariates = np.concatenate(
          [
              cancer_volume[:, :-offset, np.newaxis],
              patient_types[:, :-offset, np.newaxis],
          ],
          axis=-1,
      )
      outputs = cancer_volume[:, horizon:, np.newaxis]

      output_means = mean[['cancer_volume']].values.flatten()[
          0
      ]  # because we only need scalars here
      output_stds = std[['cancer_volume']].values.flatten()[0]

      # Add active entires
      active_entries = np.zeros(outputs.shape)

      for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

      self.data['current_covariates'] = current_covariates
      self.data['outputs'] = outputs
      self.data['active_entries'] = active_entries

      self.data['unscaled_outputs'] = (
          outputs * std['cancer_volume'] + mean['cancer_volume']
      )

      self.scaling_params = {
          'input_means': input_means,
          'inputs_stds': input_stds,
          'output_means': output_means,
          'output_stds': output_stds,
      }

      # Unified data format
      self.data['prev_outputs'] = current_covariates[:, :, :1]
      self.data['static_features'] = current_covariates[:, 0, 1:]
      zero_init_treatment = np.zeros(
          shape=[
              current_covariates.shape[0],
              1,
              self.data['prev_treatments'].shape[-1],
          ]
      )
      self.data['prev_treatments'] = np.concatenate(
          [zero_init_treatment, self.data['prev_treatments']], axis=1
      )

      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      self.processed = True
    else:
      x_keys = [
          'vitals',
          'prev_outputs',
          'static_features_t',
          'current_treatments',
      ]
      x_keys = [x for x in x_keys if x in self.data]

      for x_key in x_keys:
        self.data[x_key] = np.nan_to_num(self.data[x_key])
        logger.info('%s', f'{self.subset_name} Dataset already processed')

    return self.data

  def explode_trajectories(self, projection_horizon):
    assert self.processed

    logger.info(
        '%s',
        f'Exploding {self.subset_name} dataset before testing (multiple'
        ' sequences)',
    )

    outputs = self.data['outputs']
    prev_outputs = self.data['prev_outputs']
    sequence_lengths = self.data['sequence_lengths']
    # vitals = self.data['vitals']
    # next_vitals = self.data['next_vitals']
    active_entries = self.data['active_entries']
    current_treatments = self.data['current_treatments']
    previous_treatments = self.data['prev_treatments']
    static_features = self.data['static_features']
    if 'stabilized_weights' in self.data:
      stabilized_weights = self.data['stabilized_weights']
    else:
      stabilized_weights = None

    num_patients, max_seq_length, _ = outputs.shape
    num_seq2seq_rows = num_patients * max_seq_length

    seq2seq_previous_treatments = np.zeros(
        (num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1])
    )
    seq2seq_current_treatments = np.zeros(
        (num_seq2seq_rows, max_seq_length, current_treatments.shape[-1])
    )
    seq2seq_static_features = np.zeros(
        (num_seq2seq_rows, static_features.shape[-1])
    )
    seq2seq_outputs = np.zeros(
        (num_seq2seq_rows, max_seq_length, outputs.shape[-1])
    )
    seq2seq_prev_outputs = np.zeros(
        (num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1])
    )
    seq2seq_active_entries = np.zeros(
        (num_seq2seq_rows, max_seq_length, active_entries.shape[-1])
    )
    seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
    if 'stabilized_weights' in self.data:
      seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))
    else:
      seq2seq_stabilized_weights = None

    total_seq2seq_rows = 0  # we use this to shorten any trajectories later

    for i in range(num_patients):
      sequence_length = int(sequence_lengths[i])

      for t in range(
          projection_horizon, sequence_length
      ):  # shift outputs back by 1
        seq2seq_active_entries[total_seq2seq_rows, : (t + 1), :] = (
            active_entries[i, : (t + 1), :]
        )
        if 'stabilized_weights' in self.data:
          seq2seq_stabilized_weights[total_seq2seq_rows, : (t + 1)] = (
              stabilized_weights[i, : (t + 1)]
          )
        seq2seq_previous_treatments[total_seq2seq_rows, : (t + 1), :] = (
            previous_treatments[i, : (t + 1), :]
        )
        seq2seq_current_treatments[total_seq2seq_rows, : (t + 1), :] = (
            current_treatments[i, : (t + 1), :]
        )
        seq2seq_outputs[total_seq2seq_rows, : (t + 1), :] = outputs[
            i, : (t + 1), :
        ]
        seq2seq_prev_outputs[total_seq2seq_rows, : (t + 1), :] = prev_outputs[
            i, : (t + 1), :
        ]
        seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
        seq2seq_static_features[total_seq2seq_rows] = static_features[i]

        total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_previous_treatments = seq2seq_previous_treatments[
        :total_seq2seq_rows, :, :
    ]
    seq2seq_current_treatments = seq2seq_current_treatments[
        :total_seq2seq_rows, :, :
    ]
    seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
    # seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
    # seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    if 'stabilized_weights' in self.data:
      seq2seq_stabilized_weights = seq2seq_stabilized_weights[
          :total_seq2seq_rows
      ]

    new_data = {
        'prev_treatments': seq2seq_previous_treatments,
        'current_treatments': seq2seq_current_treatments,
        'static_features': seq2seq_static_features,
        'prev_outputs': seq2seq_prev_outputs,
        'outputs': seq2seq_outputs,
        # 'vitals': seq2seq_vitals,
        # 'next_vitals': seq2seq_next_vitals,
        'unscaled_outputs': (
            seq2seq_outputs * self.scaling_params['output_stds']
            + self.scaling_params['output_means']
        ),
        'sequence_lengths': seq2seq_sequence_lengths,
        'active_entries': seq2seq_active_entries,
    }
    if 'stabilized_weights' in self.data:
      new_data['stabilized_weights'] = seq2seq_stabilized_weights

    self.data = new_data
    self.exploded = True

    data_shapes = {k: v.shape for k, v in self.data.items()}
    logger.info(
        '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
    )

  def process_sequential(
      self, encoder_r, projection_horizon, save_encoder_r=False
  ):
    """Pre-process dataset for multiple-step-ahead prediction.

    explodes dataset to a larger one with rolling origin

    Args:
        encoder_r: Representations of encoder
        projection_horizon: Projection horizon
        save_encoder_r: Save all encoder representations (for cross-attention of
          EDCT)

    Returns:
        exploded dataset
    """

    assert self.processed

    if not self.processed_sequential:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before training (multiple'
          ' sequences)',
      )

      outputs = self.data['outputs']
      sequence_lengths = self.data['sequence_lengths']
      active_entries = self.data['active_entries']
      current_treatments = self.data['current_treatments']
      previous_treatments = self.data['prev_treatments'][
          :, 1:, :
      ]  # Without zero_init_treatment
      current_covariates = self.data['current_covariates']
      stabilized_weights = (
          self.data['stabilized_weights']
          if 'stabilized_weights' in self.data
          else None
      )

      num_patients, seq_length, _ = outputs.shape

      num_seq2seq_rows = num_patients * seq_length

      seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
      seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, seq_length))
      seq2seq_original_index = np.zeros((num_seq2seq_rows,))
      seq2seq_previous_treatments = np.zeros(
          (num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1])
      )
      seq2seq_current_treatments = np.zeros(
          (num_seq2seq_rows, projection_horizon, current_treatments.shape[-1])
      )
      seq2seq_current_covariates = np.zeros(
          (num_seq2seq_rows, projection_horizon, current_covariates.shape[-1])
      )
      seq2seq_outputs = np.zeros(
          (num_seq2seq_rows, projection_horizon, outputs.shape[-1])
      )
      seq2seq_active_entries = np.zeros(
          (num_seq2seq_rows, projection_horizon, active_entries.shape[-1])
      )
      seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
      seq2seq_stabilized_weights = (
          np.zeros((num_seq2seq_rows, projection_horizon + 1))
          if stabilized_weights is not None
          else None
      )

      total_seq2seq_rows = 0  # we use this to shorten any trajectories later

      for i in range(num_patients):
        sequence_length = int(sequence_lengths[i])

        for t in range(
            1, sequence_length - projection_horizon
        ):  # shift outputs back by 1
          seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[
              i, t - 1, :
          ]  # previous state output
          seq2seq_original_index[total_seq2seq_rows] = i
          seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

          max_projection = min(projection_horizon, sequence_length - t)

          seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = (
              active_entries[i, t : t + max_projection, :]
          )
          seq2seq_previous_treatments[
              total_seq2seq_rows, :max_projection, :
          ] = previous_treatments[i, t - 1 : t + max_projection - 1, :]
          seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = (
              current_treatments[i, t : t + max_projection, :]
          )
          seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[
              i, t : t + max_projection, :
          ]
          seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
          seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :] = (
              current_covariates[i, t : t + max_projection, :]
          )

          if (
              seq2seq_stabilized_weights is not None
          ):  # Also including SW of one-step-ahead prediction
            seq2seq_stabilized_weights[total_seq2seq_rows, :] = (
                stabilized_weights[i, t - 1 : t + max_projection]
            )

          total_seq2seq_rows += 1

      # Filter everything shorter
      seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
      seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]
      seq2seq_active_encoder_r = seq2seq_active_encoder_r[
          :total_seq2seq_rows, :
      ]
      seq2seq_previous_treatments = seq2seq_previous_treatments[
          :total_seq2seq_rows, :, :
      ]
      seq2seq_current_treatments = seq2seq_current_treatments[
          :total_seq2seq_rows, :, :
      ]
      seq2seq_current_covariates = seq2seq_current_covariates[
          :total_seq2seq_rows, :, :
      ]
      seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
      seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
      seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
      if seq2seq_stabilized_weights is not None:
        seq2seq_stabilized_weights = seq2seq_stabilized_weights[
            :total_seq2seq_rows
        ]

      # Package outputs
      seq2seq_data = {
          'init_state': seq2seq_state_inits,
          'original_index': seq2seq_original_index,
          'active_encoder_r': seq2seq_active_encoder_r,
          'prev_treatments': seq2seq_previous_treatments,
          'current_treatments': seq2seq_current_treatments,
          'current_covariates': seq2seq_current_covariates,
          'prev_outputs': seq2seq_current_covariates[:, :, :1],
          'static_features': seq2seq_current_covariates[:, 0, 1:],
          'outputs': seq2seq_outputs,
          'sequence_lengths': seq2seq_sequence_lengths,
          'active_entries': seq2seq_active_entries,
          'unscaled_outputs': (
              seq2seq_outputs * self.scaling_params['output_stds']
              + self.scaling_params['output_means']
          ),
      }
      if seq2seq_stabilized_weights is not None:
        seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

      self.data_original = deepcopy(self.data)
      self.data = seq2seq_data
      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      if save_encoder_r:
        self.encoder_r = encoder_r[:, :seq_length, :]

      self.processed_sequential = True
      self.exploded = True

    else:
      logger.info(
          '%s',
          f'{self.subset_name} Dataset already processed (multiple sequences)',
      )

    return self.data

  def process_sequential_test(
      self, projection_horizon, encoder_r=None, save_encoder_r=False
  ):
    """Pre-process test dataset for multiple-step-ahead prediction.

    takes the last n-steps according to the projection horizon

    Args:
        projection_horizon: Projection horizon
        encoder_r: Representations of encoder
        save_encoder_r: Save all encoder representations (for cross-attention of
          EDCT)

    Returns:
        processed sequential test data
    """

    assert self.processed

    if not self.processed_sequential:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before testing (multiple'
          ' sequences)',
      )

      sequence_lengths = self.data['sequence_lengths']
      outputs = self.data['outputs']
      current_treatments = self.data['current_treatments']
      previous_treatments = self.data['prev_treatments'][
          :, 1:, :
      ]  # Without zero_init_treatment
      current_covariates = self.data['current_covariates']

      num_patient_points, max_seq_length, _ = outputs.shape

      if encoder_r is not None:
        seq2seq_state_inits = np.zeros(
            (num_patient_points, encoder_r.shape[-1])
        )
      else:
        seq2seq_state_inits = None
      seq2seq_active_encoder_r = np.zeros(
          (num_patient_points, max_seq_length - projection_horizon)
      )
      seq2seq_previous_treatments = np.zeros((
          num_patient_points,
          projection_horizon,
          previous_treatments.shape[-1],
      ))
      seq2seq_current_treatments = np.zeros(
          (num_patient_points, projection_horizon, current_treatments.shape[-1])
      )
      seq2seq_current_covariates = np.zeros(
          (num_patient_points, projection_horizon, current_covariates.shape[-1])
      )
      seq2seq_outputs = np.zeros(
          (num_patient_points, projection_horizon, outputs.shape[-1])
      )
      seq2seq_active_entries = np.zeros(
          (num_patient_points, projection_horizon, 1)
      )
      seq2seq_sequence_lengths = np.zeros(num_patient_points)

      for i in range(num_patient_points):
        fact_length = int(sequence_lengths[i]) - projection_horizon
        if encoder_r is not None:
          seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
        seq2seq_active_encoder_r[i, :fact_length] = 1.0

        seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
        if fact_length >= 1:
          seq2seq_previous_treatments[i] = previous_treatments[
              i, fact_length - 1 : fact_length + projection_horizon - 1, :
          ]
        seq2seq_current_treatments[i] = current_treatments[
            i, fact_length : fact_length + projection_horizon, :
        ]
        seq2seq_outputs[i] = outputs[
            i, fact_length : fact_length + projection_horizon, :
        ]
        seq2seq_sequence_lengths[i] = projection_horizon
        # Disabled teacher forcing for test dataset
        if fact_length >= 1:
          seq2seq_current_covariates[i] = np.repeat(
              [current_covariates[i, fact_length - 1]],
              projection_horizon,
              axis=0,
          )

      # Package outputs
      seq2seq_data = {
          'active_encoder_r': seq2seq_active_encoder_r,
          'prev_treatments': seq2seq_previous_treatments,
          'current_treatments': seq2seq_current_treatments,
          'current_covariates': seq2seq_current_covariates,
          'prev_outputs': seq2seq_current_covariates[:, :, :1],
          'static_features': seq2seq_current_covariates[:, 0, 1:],
          'outputs': seq2seq_outputs,
          'sequence_lengths': seq2seq_sequence_lengths,
          'active_entries': seq2seq_active_entries,
          'unscaled_outputs': (
              seq2seq_outputs * self.scaling_params['output_stds']
              + self.scaling_params['output_means']
          ),
          'patient_types': self.data['patient_types'],
          'patient_ids_all_trajectories': (
              self.data['patient_ids_all_trajectories']
              if 'patient_ids_all_trajectories' in self.data
              else None
          ),
          'patient_current_t': (
              self.data['patient_current_t']
              if 'patient_curent_t' in self.data
              else None
          ),
      }
      if encoder_r is not None:
        seq2seq_data['init_state'] = seq2seq_state_inits

      self.data_original = deepcopy(self.data)
      self.data = seq2seq_data
      data_shapes = {k: v.shape for k, v in self.data.items() if v is not None}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      if save_encoder_r and encoder_r is not None:
        self.encoder_r = encoder_r[:, : max_seq_length - projection_horizon, :]

      self.processed_sequential = True

    else:
      logger.info(
          '%s',
          f'{self.subset_name} Dataset already processed (multiple sequences)',
      )

    return self.data

  def process_autoregressive_test(
      self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False
  ):
    """Pre-process test dataset for multiple-step-ahead prediction.

    axillary dataset placeholder for autoregressive prediction

    Args:
        encoder_r: Representations of encoder
        encoder_outputs: encoder outputs
        projection_horizon: Projection horizon
        save_encoder_r: Save all encoder representations (for cross-attention of
          EDCT)

    Returns:
        autoregressive test data
    """

    assert self.processed_sequential

    if not self.processed_autoregressive:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before testing'
          ' (autoregressive)',
      )

      current_treatments = self.data_original['current_treatments']
      prev_treatments = self.data_original['prev_treatments'][
          :, 1:, :
      ]  # Without zero_init_treatment

      sequence_lengths = self.data_original['sequence_lengths']
      num_patient_points, max_seq_length = current_treatments.shape[:2]

      current_dataset = dict()  # Same as original, but only with last n-steps
      current_dataset['current_covariates'] = np.zeros((
          num_patient_points,
          projection_horizon,
          self.data_original['current_covariates'].shape[-1],
      ))
      current_dataset['prev_treatments'] = np.zeros((
          num_patient_points,
          projection_horizon,
          self.data_original['prev_treatments'].shape[-1],
      ))
      current_dataset['current_treatments'] = np.zeros((
          num_patient_points,
          projection_horizon,
          self.data_original['current_treatments'].shape[-1],
      ))
      current_dataset['init_state'] = np.zeros(
          (num_patient_points, encoder_r.shape[-1])
      )
      current_dataset['active_encoder_r'] = np.zeros(
          (num_patient_points, max_seq_length - projection_horizon)
      )
      current_dataset['active_entries'] = np.ones(
          (num_patient_points, projection_horizon, 1)
      )

      for i in range(num_patient_points):
        fact_length = int(sequence_lengths[i]) - projection_horizon
        current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
        if encoder_outputs is not None:
          current_dataset['current_covariates'][i, 0, 0] = encoder_outputs[
              i, fact_length - 1
          ]
        current_dataset['active_encoder_r'][i, :fact_length] = 1.0
        current_dataset['prev_treatments'][i] = prev_treatments[
            i, fact_length - 1 : fact_length + projection_horizon - 1, :
        ]
        current_dataset['current_treatments'][i] = current_treatments[
            i, fact_length : fact_length + projection_horizon, :
        ]

      current_dataset['prev_outputs'] = current_dataset['current_covariates'][
          :, :, :1
      ]
      current_dataset['static_features'] = self.data_original['static_features']

      self.data_processed_seq = deepcopy(self.data)
      self.data = current_dataset
      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      if save_encoder_r:
        self.encoder_r = encoder_r[:, : max_seq_length - projection_horizon, :]

      self.processed_autoregressive = True

    else:
      logger.info(
          '%s', f'{self.subset_name} Dataset already processed (autoregressive)'
      )

    return self.data

  def process_sequential_multi(self, projection_horizon):
    """Pre-process test dataset for multiple-step-ahead prediction.

    for multi-input model: marking rolling origin with
        'future_past_split'

    Args:
        projection_horizon: Projection horizon

    Returns:
        processed sequential multi data
    """

    assert self.processed_sequential

    if not self.processed_autoregressive:
      self.data_processed_seq = self.data
      self.data = deepcopy(self.data_original)
      self.data['future_past_split'] = (
          self.data['sequence_lengths'] - projection_horizon
      )
      self.processed_autoregressive = True

    else:
      logger.info(
          '%s', f'{self.subset_name} Dataset already processed (autoregressive)'
      )

    return self.data

  def process_sequential_rep_est(self, projection_horizon):
    assert self.processed

    if not self.processed_autoregressive:
      self.data['future_past_split'] = (
          self.data['sequence_lengths'] - projection_horizon
      )
      self.process_autoregressive = True
    else:
      logger.info(
          '%s', f'{self.subset_name} Dataset already processed (autoregressive)'
      )

    return self.data

  def process_sequential_split(self):
    assert self.processed
    logger.info('%s', f'Augmenting {self.subset_name} dataset before training')

    self.data_before_aug = deepcopy(self.data)
    data_length = self.data_before_aug['prev_treatments'].shape[0]
    valid_keys = [
        k
        for k, v in self.data_before_aug.items()
        if hasattr(v, '__len__') and len(v) == data_length
    ]
    self.data = DefaultDict(list)
    for patient_id in range(data_length):
      sample_seq_len = int(self.data_before_aug['sequence_lengths'][patient_id])
      sample_future_past_split = np.arange(
          1, sample_seq_len + 1, dtype=np.int64
      )
      self.data['future_past_split'].append(sample_future_past_split)
      for vk in valid_keys:
        self.data[vk].append(
            np.repeat(
                self.data_before_aug[vk][patient_id][None, Ellipsis],
                sample_seq_len,
                axis=0,
            )
        )
    for k in self.data:
      self.data[k] = np.concatenate(self.data[k], axis=0)
    return self.data


class SyntheticCancerDatasetCollection(SyntheticDatasetCollection):
  """Dataset collection.

  (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
  """

  def __init__(
      self,
      chemo_coeff,
      radio_coeff,
      num_patients,
      seed=100,
      window_size=15,
      max_seq_length=60,
      projection_horizon=5,
      lag=0,
      cf_seq_mode='sliding_treatment',
      treatment_mode='multiclass',
      few_shot_sample_num=-1,
      same_subjs_train_test=False,
      **kwargs,
  ):
    """Initialization.

    Args:
      chemo_coeff: Confounding coefficient of chemotherapy
      radio_coeff: Confounding coefficient of radiotherapy
      num_patients: Number of patients in dataset
      seed: random seed
      window_size: Used for biased treatment assignment
      max_seq_length: Max length of time series
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      lag: Lag for treatment assignment window
      cf_seq_mode: sliding_treatment / random_trajectories
      treatment_mode: multiclass / multilabel
      few_shot_sample_num: if > 0, resampling training samples to make the
        dataset few-shot learning
      same_subjs_train_test: use same subjects in train and test
      **kwargs: other args

    Returns:
      dataset collection
    """
    super(SyntheticCancerDatasetCollection, self).__init__()
    self.seed = seed
    np.random.seed(seed)

    if same_subjs_train_test:
      self.train_f = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'train',
          lag=lag,
          treatment_mode=treatment_mode,
      )
      self.val_f = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'val',
          lag=lag,
          treatment_mode=treatment_mode,
          reuse_params=deepcopy(self.train_f.params),
      )
      self.test_cf_one_step = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'test',
          mode='counterfactual_one_step',
          lag=lag,
          treatment_mode=treatment_mode,
          reuse_params=deepcopy(self.train_f.params),
      )
      self.test_cf_treatment_seq = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'test',
          mode='counterfactual_treatment_seq',
          projection_horizon=projection_horizon,
          lag=lag,
          cf_seq_mode=cf_seq_mode,
          treatment_mode=treatment_mode,
          reuse_params=deepcopy(self.train_f.params),
      )
    else:
      self.train_f = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['train'],
          window_size,
          max_seq_length,
          'train',
          lag=lag,
          treatment_mode=treatment_mode,
      )
      self.val_f = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['val'],
          window_size,
          max_seq_length,
          'val',
          lag=lag,
          treatment_mode=treatment_mode,
      )
      self.test_cf_one_step = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'test',
          mode='counterfactual_one_step',
          lag=lag,
          treatment_mode=treatment_mode,
      )
      self.test_cf_treatment_seq = SyntheticCancerDataset(
          chemo_coeff,
          radio_coeff,
          num_patients['test'],
          window_size,
          max_seq_length,
          'test',
          mode='counterfactual_treatment_seq',
          projection_horizon=projection_horizon,
          lag=lag,
          cf_seq_mode=cf_seq_mode,
          treatment_mode=treatment_mode,
      )

    self.projection_horizon = projection_horizon
    self.autoregressive = True
    self.has_vitals = False
    self.train_scaling_params = self.train_f.get_scaling_params()

    self.max_seq_length = max_seq_length

    if few_shot_sample_num > 0:
      self.train_f.few_shot_sampling(few_shot_sample_num, self.seed)

  def save_data_in_crn_format(self, savepath):
    pickle_map = {
        'chemo_coeff': self.train_f.chemo_coeff,
        'radio_coeff': self.train_f.radio_coeff,
        'num_time_steps': self.max_seq_length,
        'training_data': self.train_f.data,
        'validation_data': self.val_f.data,
        'test_data': self.test_cf_one_step.data,
        #   'test_data_factuals': test_data_factuals,
        'test_data_seq': self.test_cf_treatment_seq.data,
        'scaling_data': self.train_scaling_params,
        'window_size': self.train_f.window_size,
    }
    logger.info('%s', 'Saving pickle map to {}'.format(savepath))
    if not os.path.exists(savepath):
      os.makedirs(savepath, exist_ok=True)
    filepath = os.path.join(
        savepath,
        'cancer_sim_cc-{}_rc-{}_seed-{}.pt'.format(
            self.train_f.chemo_coeff, self.train_f.radio_coeff, self.seed
        ),
    )
    with open(filepath, 'wb') as f:
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
