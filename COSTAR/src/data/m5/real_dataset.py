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

"""Generate real M5 dataset."""

import copy
import logging

import numpy as np
import sklearn.model_selection
from src import ROOT_PATH
from src.data.dataset_collection import RealDatasetCollection
from src.data.m5.load_data import load_m5_data_processed
import torch

deepcopy = copy.deepcopy
train_test_split = sklearn.model_selection.train_test_split
Dataset = torch.utils.data.Dataset

logger = logging.getLogger(__name__)


class M5RealDataset(Dataset):
  """Pytorch-style real-world M5 dataset."""

  def __init__(
      self,
      treatments,
      outcomes,
      vitals,
      static_features,
      outcomes_unscaled,
      scaling_params,
      sequence_lengths,
      subset_name,
  ):
    """M5 real dataset.

    Args:
      treatments: DataFrame with treatments; multiindex by (patient_id,
        timestep)
      outcomes: DataFrame with outcomes; multiindex by (patient_id, timestep)
      vitals: DataFrame with vitals (time-varying covariates); multiindex by
        (patient_id, timestep)
      static_features: DataFrame with static features
      outcomes_unscaled: DataFrame with unscaled outcomes; multiindex by
        (patient_id, timestep)
      scaling_params: Standard normalization scaling parameters
      sequence_lengths: sequence_lengths
      subset_name: train / val / test

    Returns:
      pytorch dataset
    """
    assert treatments.shape[0] == outcomes.shape[0]
    assert outcomes.shape[0] == vitals.shape[0]

    self.subset_name = subset_name
    user_sizes = sequence_lengths

    # Conversion to np.arrays
    active_entries = np.ones_like(outcomes).astype(float)
    active_entries[np.isnan(outcomes)] = 1.0
    treatments[np.isnan(treatments)] = 0.0
    outcomes[np.isnan(outcomes)] = 0.0
    vitals[np.isnan(vitals)] = 0.0
    outcomes_unscaled[np.isnan(outcomes_unscaled)] = 0.0

    self.data = {
        'sequence_lengths': user_sizes - 1,
        'prev_treatments': treatments[:, :-1, :],
        'vitals': vitals[:, 1:, :],
        'next_vitals': vitals[:, 2:, :],
        'current_treatments': treatments[:, 1:, :],
        'static_features': static_features,
        'active_entries': active_entries[:, 1:, :],
        'outputs': outcomes[:, 1:, :],
        'unscaled_outputs': outcomes_unscaled[:, 1:, :],
        'prev_outputs': outcomes[:, :-1, :],
    }

    self.scaling_params = scaling_params
    self.processed = True
    self.processed_sequential = False
    self.processed_autoregressive = False
    self.exploded = False

    data_shapes = {k: v.shape for k, v in self.data.items()}
    logger.info(
        '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
    )

    self.norm_const = 1.0

  def __getitem__(self, index):
    if 'id_t_in_unexploded' in self.data.keys():
      result = {}
      for k, v in self.data.items():
        if k not in [
            'unexploded_vitals',
            'unexploded_next_vitals',
            'unexploded_prev_treatments',
            'unexploded_current_treatments',
            'id_t_in_unexploded',
        ]:
          result[k] = torch.as_tensor(v[index]).to(torch.get_default_dtype())
      id_t_in_unexploded = self.data['id_t_in_unexploded'][index]
      id_in_unexploded = id_t_in_unexploded[0]
      start_t = id_t_in_unexploded[1]
      end_t = id_t_in_unexploded[2]
      if 'unexploded_vitals' in self.data:
        result['vitals'] = result['active_entries'] * torch.as_tensor(
            self.data['unexploded_vitals'][id_in_unexploded, start_t:end_t]
        ).to(torch.get_default_dtype())
        result['next_vitals'] = result['active_entries'][1:] * torch.as_tensor(
            self.data['unexploded_next_vitals'][id_in_unexploded, start_t:end_t]
        ).to(torch.get_default_dtype())
      if 'id_t_in_unexploded_prev_treatment' in self.data:
        start_t_pt, end_t_pt = self.data['id_t_in_unexploded_prev_treatment'][
            index
        ][1:3]
        result['prev_treatments'] = result['active_entries'] * torch.as_tensor(
            self.data['unexploded_prev_treatments'][
                id_in_unexploded, start_t_pt:end_t_pt
            ]
        ).to(torch.get_default_dtype())
      else:
        result['prev_treatments'] = result['active_entries'] * torch.as_tensor(
            self.data['unexploded_prev_treatments'][
                id_in_unexploded, start_t:end_t
            ]
        ).to(torch.get_default_dtype())
      result['current_treatments'] = result['active_entries'] * torch.as_tensor(
          self.data['unexploded_current_treatments'][
              id_in_unexploded, start_t:end_t
          ]
      ).to(torch.get_default_dtype())
    else:
      result = {
          k: torch.as_tensor(v[index]).to(torch.get_default_dtype())
          for k, v in self.data.items()
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
    return len(self.data['active_entries'])

  def explode_trajectories(self, projection_horizon):
    """Convert test dataset to a dataset with rolling origin.

    Args:
        projection_horizon: projection horizon

    Returns:
        data
    """
    assert self.processed

    logger.info(
        '%s',
        f'Exploding {self.subset_name} dataset before testing (multiple'
        ' sequences)',
    )

    outputs = self.data['outputs']
    prev_outputs = self.data['prev_outputs']
    sequence_lengths = self.data['sequence_lengths']
    vitals = self.data['vitals']
    next_vitals = self.data['next_vitals']
    active_entries = self.data['active_entries']
    unexploded_current_treatments = self.data['current_treatments']
    unexploded_previous_treatments = self.data['prev_treatments']
    static_features = self.data['static_features']
    if 'stabilized_weights' in self.data:
      stabilized_weights = self.data['stabilized_weights']
    else:
      stabilized_weights = None

    num_patients, max_seq_length, _ = outputs.shape
    num_seq2seq_rows = num_patients * max_seq_length

    seq2seq_id_t_in_unexploded = np.zeros((num_seq2seq_rows, 3), dtype=np.int64)
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
        seq2seq_outputs[total_seq2seq_rows, : (t + 1), :] = outputs[
            i, : (t + 1), :
        ]
        seq2seq_prev_outputs[total_seq2seq_rows, : (t + 1), :] = prev_outputs[
            i, : (t + 1), :
        ]
        seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
        seq2seq_id_t_in_unexploded[total_seq2seq_rows, 0] = i
        seq2seq_id_t_in_unexploded[total_seq2seq_rows, 1] = 0
        seq2seq_id_t_in_unexploded[total_seq2seq_rows, 2] = max_seq_length
        seq2seq_static_features[total_seq2seq_rows] = static_features[i]

        total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
    seq2seq_id_t_in_unexploded = seq2seq_id_t_in_unexploded[:total_seq2seq_rows]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    if 'stabilized_weights' in self.data:
      seq2seq_stabilized_weights = seq2seq_stabilized_weights[
          :total_seq2seq_rows
      ]

    new_data = {
        # 'prev_treatments': seq2seq_previous_treatments,
        # 'current_treatments': seq2seq_current_treatments,
        'unexploded_prev_treatments': unexploded_previous_treatments,
        'unexploded_current_treatments': unexploded_current_treatments,
        'static_features': seq2seq_static_features,
        'prev_outputs': seq2seq_prev_outputs,
        'outputs': seq2seq_outputs,
        # 'vitals': seq2seq_vitals,
        # 'next_vitals': seq2seq_next_vitals,
        'unexploded_vitals': vitals,
        'unexploded_next_vitals': next_vitals,
        'id_t_in_unexploded': seq2seq_id_t_in_unexploded,
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

    data_shapes = {k: v.shape for k, v in self.data.items()}
    logger.info(
        '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
    )

  def explode_trajectories_no_onfly(self, projection_horizon):
    """Convert test dataset to a dataset with rolling origin.

    Args:
        projection_horizon: projection horizon

    Returns:
        data
    """
    assert self.processed

    logger.info(
        '%s',
        f'Exploding {self.subset_name} dataset before testing (multiple'
        ' sequences)',
    )

    outputs = self.data['outputs']
    prev_outputs = self.data['prev_outputs']
    sequence_lengths = self.data['sequence_lengths']
    vitals = self.data['vitals']
    next_vitals = self.data['next_vitals']
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
    seq2seq_vitals = np.zeros(
        (num_seq2seq_rows, max_seq_length, vitals.shape[-1])
    )
    seq2seq_next_vitals = np.zeros(
        (num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1])
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
        seq2seq_vitals[total_seq2seq_rows, : (t + 1), :] = vitals[
            i, : (t + 1), :
        ]
        seq2seq_next_vitals[
            total_seq2seq_rows, : min(t + 1, sequence_length - 1), :
        ] = next_vitals[i, : min(t + 1, sequence_length - 1), :]
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
    seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
    seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
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
        'vitals': seq2seq_vitals,
        'next_vitals': seq2seq_next_vitals,
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

    data_shapes = {k: v.shape for k, v in self.data.items()}
    logger.info(
        '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
    )

  def process_sequential(
      self,
      encoder_r,
      projection_horizon,
      encoder_outputs=None,
      save_encoder_r=False,
  ):
    """Pre-process dataset for multiple-step-ahead prediction.

    explodes dataset to a larger one with rolling origin

    Args:
        encoder_r: Representations of encoder
        projection_horizon: Projection horizon
        encoder_outputs: One-step-ahead predcitions of encoder
        save_encoder_r: Save all encoder representations (for cross-attention of
          EDCT)

    Returns:
        data
    """

    assert self.processed

    if not self.processed_sequential:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before training (multiple'
          ' sequences)',
      )

      outputs = self.data['outputs']
      prev_outputs = self.data['prev_outputs']
      sequence_lengths = self.data['sequence_lengths']
      active_entries = self.data['active_entries']
      unexploded_current_treatments = self.data['current_treatments']
      unexploded_previous_treatments = self.data['prev_treatments']
      static_features = self.data['static_features']
      stabilized_weights = (
          self.data['stabilized_weights']
          if 'stabilized_weights' in self.data
          else None
      )

      num_patients, max_seq_length, _ = outputs.shape

      num_seq2seq_rows = num_patients * max_seq_length

      seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
      seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, max_seq_length))
      seq2seq_original_index = np.zeros((num_seq2seq_rows,))
      seq2seq_id_t_in_unexploded = np.zeros(
          (num_seq2seq_rows, 3), dtype=np.int64
      )
      seq2seq_static_features = np.zeros(
          (num_seq2seq_rows, static_features.shape[-1])
      )
      seq2seq_outputs = np.zeros(
          (num_seq2seq_rows, projection_horizon, outputs.shape[-1])
      )
      seq2seq_prev_outputs = np.zeros(
          (num_seq2seq_rows, projection_horizon, prev_outputs.shape[-1])
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
          seq2seq_id_t_in_unexploded[total_seq2seq_rows, 0] = i
          seq2seq_id_t_in_unexploded[total_seq2seq_rows, 1] = t
          seq2seq_id_t_in_unexploded[total_seq2seq_rows, 2] = t + max_projection
          seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[
              i, t : t + max_projection, :
          ]
          seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
          seq2seq_static_features[total_seq2seq_rows] = static_features[i]
          if encoder_outputs is not None:  # For auto-regressive evaluation
            seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = (
                encoder_outputs[i, t - 1 : t + max_projection - 1, :]
            )
          else:  # train / val of decoder
            seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = (
                prev_outputs[i, t : t + max_projection, :]
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
      seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
      seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
      seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
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
          # 'prev_treatments': seq2seq_previous_treatments,
          # 'current_treatments': seq2seq_current_treatments,
          'unexploded_prev_treatments': unexploded_previous_treatments,
          'unexploded_current_treatments': unexploded_current_treatments,
          'id_t_in_unexploded': seq2seq_id_t_in_unexploded,
          'static_features': seq2seq_static_features,
          'prev_outputs': seq2seq_prev_outputs,
          'outputs': seq2seq_outputs,
          'unscaled_outputs': (
              seq2seq_outputs * self.scaling_params['output_stds']
              + self.scaling_params['output_means']
          ),
          'sequence_lengths': seq2seq_sequence_lengths,
          'active_entries': seq2seq_active_entries,
      }
      if seq2seq_stabilized_weights is not None:
        seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

      self.data_original = deepcopy(self.data)
      self.data_processed_seq = deepcopy(
          seq2seq_data
      )  # For auto-regressive evaluation (self.data will be changed)
      self.data = seq2seq_data

      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      if save_encoder_r:
        self.encoder_r = encoder_r[:, :max_seq_length, :]

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
        data
    """

    assert self.processed

    if not self.processed_sequential:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before testing (multiple'
          ' sequences)',
      )

      outputs = self.data['outputs']
      prev_outputs = self.data['prev_outputs']
      sequence_lengths = self.data['sequence_lengths']
      unexploded_current_treatments = self.data['unexploded_current_treatments']
      unexploded_previous_treatments = self.data['unexploded_prev_treatments']
      id_t_in_unexploded = self.data['id_t_in_unexploded']
      # vitals = self.data['vitals']

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
      seq2seq_id_t_in_unexploded = np.zeros(
          (num_patient_points, 3), dtype=np.int64
      )
      seq2seq_outputs = np.zeros(
          (num_patient_points, projection_horizon, outputs.shape[-1])
      )
      seq2seq_prev_outputs = np.zeros(
          (num_patient_points, projection_horizon, outputs.shape[-1])
      )
      seq2seq_active_entries = np.zeros(
          (num_patient_points, projection_horizon, 1)
      )
      seq2seq_sequence_lengths = np.zeros(num_patient_points)
      seq2seq_original_index = np.zeros(num_patient_points)

      for i in range(num_patient_points):
        fact_length = int(sequence_lengths[i]) - projection_horizon
        if encoder_r is not None:
          seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
        seq2seq_active_encoder_r[i, :fact_length] = 1.0
        seq2seq_original_index[i] = i

        seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
        seq2seq_id_t_in_unexploded[i, 0] = id_t_in_unexploded[i, 0]
        seq2seq_id_t_in_unexploded[i, 1] = fact_length
        seq2seq_id_t_in_unexploded[i, 2] = fact_length + projection_horizon
        seq2seq_outputs[i] = outputs[
            i, fact_length : fact_length + projection_horizon, :
        ]
        seq2seq_prev_outputs[i] = prev_outputs[
            i, fact_length : fact_length + projection_horizon, :
        ]

        seq2seq_sequence_lengths[i] = projection_horizon

      # Package outputs
      seq2seq_data = {
          'original_index': seq2seq_original_index,
          'active_encoder_r': seq2seq_active_encoder_r,
          'unexploded_prev_treatments': unexploded_previous_treatments,
          'unexploded_current_treatments': unexploded_current_treatments,
          'static_features': self.data['static_features'],
          'prev_outputs': seq2seq_prev_outputs,
          'outputs': seq2seq_outputs,
          'unscaled_outputs': (
              seq2seq_outputs * self.scaling_params['output_stds']
              + self.scaling_params['output_means']
          ),
          'sequence_lengths': seq2seq_sequence_lengths,
          'active_entries': seq2seq_active_entries,
          'id_t_in_unexploded': seq2seq_id_t_in_unexploded,
      }
      if encoder_r is not None:
        seq2seq_data['init_state'] = seq2seq_state_inits

      self.data_original = deepcopy(self.data)
      self.data_processed_seq = deepcopy(
          seq2seq_data
      )  # For auto-regressive evaluation (self.data will be changed)
      self.data = seq2seq_data

      data_shapes = {k: v.shape for k, v in self.data.items()}
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
        data
    """

    assert self.processed_sequential

    if not self.processed_autoregressive:
      logger.info(
          '%s',
          f'Processing {self.subset_name} dataset before testing'
          ' (autoregressive)',
      )

      unexploded_current_treatments = self.data_original[
          'unexploded_current_treatments'
      ]
      unexploded_previous_treatments = self.data_original[
          'unexploded_prev_treatments'
      ]
      id_t_in_unexploded = self.data_original['id_t_in_unexploded']

      sequence_lengths = self.data_original['sequence_lengths']
      num_patient_points = sequence_lengths.shape[0]

      current_dataset = dict()  # Same as original, but only with last n-steps
      current_dataset['prev_outputs'] = np.zeros((
          num_patient_points,
          projection_horizon,
          self.data_original['outputs'].shape[-1],
      ))
      current_dataset['init_state'] = np.zeros(
          (num_patient_points, encoder_r.shape[-1])
      )
      current_dataset['active_encoder_r'] = np.zeros(
          (num_patient_points, int(sequence_lengths.max() - projection_horizon))
      )
      current_dataset['active_entries'] = np.ones(
          (num_patient_points, projection_horizon, 1)
      )
      current_dataset['id_t_in_unexploded'] = np.zeros(
          (num_patient_points, 3), dtype=np.int64
      )
      current_dataset['id_t_in_unexploded_prev_treatment'] = np.zeros(
          (num_patient_points, 3), dtype=np.int64
      )

      for i in range(num_patient_points):
        fact_length = int(sequence_lengths[i]) - projection_horizon
        current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
        current_dataset['prev_outputs'][i, 0, :] = encoder_outputs[
            i, fact_length - 1
        ]
        current_dataset['active_encoder_r'][i, :fact_length] = 1.0
        current_dataset['id_t_in_unexploded'][i, 0] = id_t_in_unexploded[i, 0]
        current_dataset['id_t_in_unexploded'][i, 1] = fact_length
        current_dataset['id_t_in_unexploded'][i, 2] = (
            fact_length + projection_horizon
        )
        current_dataset['id_t_in_unexploded_prev_treatment'][i, 0] = (
            id_t_in_unexploded[i, 0]
        )
        current_dataset['id_t_in_unexploded_prev_treatment'][i, 1] = (
            fact_length - 1
        )
        current_dataset['id_t_in_unexploded_prev_treatment'][i, 2] = (
            fact_length + projection_horizon - 1
        )

      current_dataset['static_features'] = self.data_original['static_features']
      current_dataset['unexploded_prev_treatments'] = (
          unexploded_previous_treatments
      )
      current_dataset['unexploded_current_treatments'] = (
          unexploded_current_treatments
      )

      self.data_processed_seq = deepcopy(self.data)
      self.data = current_dataset
      data_shapes = {k: v.shape for k, v in self.data.items()}
      logger.info(
          '%s', f'Shape of processed {self.subset_name} data: {data_shapes}'
      )

      if save_encoder_r:
        self.encoder_r = encoder_r[
            :, : int(max(sequence_lengths) - projection_horizon), :
        ]

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
        data
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


class M5RealDatasetCollection(RealDatasetCollection):
  """Dataset collection (train_f, val_f, test_f)."""

  def __init__(
      self,
      path,
      min_seq_length = 30,
      max_seq_length = 60,
      seed = 100,
      max_number = None,
      split=None,
      projection_horizon = 5,
      autoregressive=True,
      treatment_bucketize='uniform',
      **kwargs,
  ):
    """M5 real dataset collection.

    Args:
      path: Path with M5 dataset (HDFStore)
      min_seq_length: Min sequence lenght in cohort
      max_seq_length: Max sequence lenght in cohort
      seed: Seed for random cohort patient selection
      max_number: Maximum number of patients in cohort
      split: Ratio of train / val / test split
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      autoregressive: is autoregressive
      treatment_bucketize: uniform or quantile
      **kwargs: kwargs

    Returns:
      dataset collection
    """
    super(M5RealDatasetCollection, self).__init__()
    if not split:
      split = {'val': 0.2, 'test': 0.2}
    self.seed = seed
    (
        valid_seq_splits,
        treatments,
        outcomes,
        vitals,
        static_features,
        outcomes_unscaled,
        scaling_params,
        sequence_lengths,
    ) = load_m5_data_processed(
        ROOT_PATH + '/' + path,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        max_number=max_number,
        data_seed=seed,
        treatment_bucktize=treatment_bucketize,
        **kwargs,
    )

    # Train/val/test random_split
    # always split by item_id
    valid_seq_splits['i'] = valid_seq_splits.index
    item_ids, item_ids_test = train_test_split(
        valid_seq_splits['item_id'].unique(),
        test_size=split['test'],
        random_state=seed,
    )
    train_indices = (
        valid_seq_splits.set_index('item_id').loc[item_ids]['i'].tolist()
    )
    test_indices = (
        valid_seq_splits.set_index('item_id').loc[item_ids_test]['i'].tolist()
    )

    (
        treatments_train,
        outcomes_train,
        vitals_train,
        outcomes_unscaled_train,
        treatments_test,
        outcomes_test,
        vitals_test,
        outcomes_unscaled_test,
    ) = (
        treatments[train_indices],
        outcomes[train_indices],
        vitals[train_indices],
        outcomes_unscaled[train_indices],
        treatments[test_indices],
        outcomes[test_indices],
        vitals[test_indices],
        outcomes_unscaled[test_indices],
    )
    static_features_train, static_features_test = (
        static_features[train_indices],
        static_features[test_indices],
    )
    sequence_lengths_train, sequence_lengths_test = (
        sequence_lengths[train_indices],
        sequence_lengths[test_indices],
    )

    if split['val'] > 0.0:
      item_ids_train, item_ids_val = train_test_split(
          item_ids,
          test_size=split['val'] / (1 - split['test']),
          random_state=2 * seed,
      )
      train_indices = (
          valid_seq_splits.set_index('item_id')
          .loc[item_ids_train]['i']
          .tolist()
      )
      val_indices = (
          valid_seq_splits.set_index('item_id').loc[item_ids_val]['i'].tolist()
      )

      (
          treatments_train,
          outcomes_train,
          vitals_train,
          outcomes_unscaled_train,
          treatments_val,
          outcomes_val,
          vitals_val,
          outcomes_unscaled_val,
      ) = (
          treatments[train_indices],
          outcomes[train_indices],
          vitals[train_indices],
          outcomes_unscaled[train_indices],
          treatments[val_indices],
          outcomes[val_indices],
          vitals[val_indices],
          outcomes_unscaled[val_indices],
      )
      static_features_train, static_features_val = (
          static_features[train_indices],
          static_features[val_indices],
      )
      sequence_lengths_train, sequence_lengths_val = (
          sequence_lengths[train_indices],
          sequence_lengths[val_indices],
      )
    else:
      treatments_val = None
      outcomes_val = None
      vitals_val = None
      static_features_val = None
      outcomes_unscaled_val = None
      sequence_lengths_val = None

    self.train_f = M5RealDataset(
        treatments_train,
        outcomes_train,
        vitals_train,
        static_features_train,
        outcomes_unscaled_train,
        scaling_params,
        sequence_lengths_train,
        'train',
    )
    if split['val'] > 0.0:
      self.val_f = M5RealDataset(
          treatments_val,
          outcomes_val,
          vitals_val,
          static_features_val,
          outcomes_unscaled_val,
          scaling_params,
          sequence_lengths_val,
          'val',
      )
    self.test_f = M5RealDataset(
        treatments_test,
        outcomes_test,
        vitals_test,
        static_features_test,
        outcomes_unscaled_test,
        scaling_params,
        sequence_lengths_test,
        'test',
    )

    self.projection_horizon = projection_horizon
    self.has_vitals = True
    self.autoregressive = autoregressive
    self.processed_data_encoder = True


class M5RealDatasetCategoryDomainCollection(RealDatasetCollection):
  """Dataset collection (train_f, val_f, test_f)."""

  def __init__(
      self,
      path,
      min_seq_length = 30,
      max_seq_length = 60,
      seed = 100,
      max_number = None,
      split=None,
      projection_horizon = 5,
      autoregressive=True,
      treatment_bucketize='uniform',
      src_cat_domains=None,
      tgt_cat_domains=None,
      few_shot_sample_num=0.01,
      use_few_shot=False,
      use_src_test=False,
      **kwargs,
  ):
    """M5 real dataset domain.

    Args:
      path: Path with M5 dataset (HDFStore)
      min_seq_length: Min sequence lenght in cohort
      max_seq_length: Max sequence lenght in cohort
      seed: Seed for random cohort patient selection
      max_number: Maximum number of patients in cohort
      split: Ratio of train / val / test split
      projection_horizon: Range of tau-step-ahead prediction (tau =
        projection_horizon + 1)
      autoregressive: is autoregressive
      treatment_bucketize: uniform or quantile
      src_cat_domains: source categories
      tgt_cat_domains: target categories
      few_shot_sample_num: number of samples in few-shot
      use_few_shot: if use few-shot
      use_src_test: if use source domain test data
      **kwargs: kwargs

    Returns:
      dataset collection
    """
    super(M5RealDatasetCategoryDomainCollection, self).__init__()
    if not split:
      split = {'val': 0.2, 'test': 0.2}
    self.seed = seed

    src_split = {'val': split['tr_val'], 'test': split['tr_test']}
    src_train_f, src_val_f, src_test_f = self._load_domain_data(
        path,
        min_seq_length,
        max_seq_length,
        max_number,
        src_split,
        seed,
        treatment_bucketize,
        src_cat_domains,
        few_shot_sample_num=0,
        use_src_test=use_src_test,
        **kwargs,
    )
    self.src_train_f, self.src_val_f = src_train_f, src_val_f
    tgt_split = {'val': split['te_val'], 'test': split['te_test']}
    tgt_train_f, tgt_val_f, tgt_test_f = self._load_domain_data(
        path,
        min_seq_length,
        max_seq_length,
        max_number,
        tgt_split,
        seed,
        treatment_bucketize,
        tgt_cat_domains,
        few_shot_sample_num,
        use_src_test,
        **kwargs,
    )
    self.test_train_f, self.test_val_f = tgt_train_f, tgt_val_f
    if use_few_shot:
      self.train_f, self.val_f = self.test_train_f, self.test_val_f
    else:
      self.train_f, self.val_f = self.src_train_f, self.src_val_f

    if not use_src_test:
      self.test_f = tgt_test_f
    else:
      self.test_f = src_test_f

    self.projection_horizon = projection_horizon
    self.has_vitals = True
    self.autoregressive = autoregressive
    self.processed_data_encoder = True

  def _load_domain_data(
      self,
      path,
      min_seq_length,
      max_seq_length,
      max_number,
      split,
      seed,
      treatment_bucketize,
      cat_domains,
      few_shot_sample_num,
      use_src_test,
      **kwargs,
  ):
    (
        valid_seq_splits,
        treatments,
        outcomes,
        vitals,
        static_features,
        outcomes_unscaled,
        scaling_params,
        sequence_lengths,
    ) = load_m5_data_processed(
        ROOT_PATH + '/' + path,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        max_number=max_number,
        data_seed=seed,
        treatment_bucktize=treatment_bucketize,
        cat_domains=cat_domains,
        **kwargs,
    )

    # Train/val/test random_split
    # always split by item_id
    valid_seq_splits['i'] = valid_seq_splits.index
    item_ids, item_ids_test = train_test_split(
        valid_seq_splits['item_id'].unique(),
        test_size=split['test'],
        random_state=seed,
    )
    total_item_id_num = len(item_ids) + len(item_ids_test)
    train_indices = (
        valid_seq_splits.set_index('item_id').loc[item_ids]['i'].tolist()
    )
    test_indices = (
        valid_seq_splits.set_index('item_id').loc[item_ids_test]['i'].tolist()
    )

    (
        _,
        _,
        _,
        _,
        treatments_test,
        outcomes_test,
        vitals_test,
        outcomes_unscaled_test,
    ) = (
        treatments[train_indices],
        outcomes[train_indices],
        vitals[train_indices],
        outcomes_unscaled[train_indices],
        treatments[test_indices],
        outcomes[test_indices],
        vitals[test_indices],
        outcomes_unscaled[test_indices],
    )
    _, static_features_test = (
        static_features[train_indices],
        static_features[test_indices],
    )
    _, sequence_lengths_test = (
        sequence_lengths[train_indices],
        sequence_lengths[test_indices],
    )

    if split['val'] > 0.0:
      item_ids_train, item_ids_val = train_test_split(
          item_ids,
          test_size=split['val'] / (1 - split['test']),
          random_state=2 * seed,
      )
      train_indices = (
          valid_seq_splits.set_index('item_id')
          .loc[item_ids_train]['i']
          .tolist()
      )
      val_indices = (
          valid_seq_splits.set_index('item_id').loc[item_ids_val]['i'].tolist()
      )

      (
          treatments_train,
          outcomes_train,
          vitals_train,
          outcomes_unscaled_train,
          treatments_val,
          outcomes_val,
          vitals_val,
          outcomes_unscaled_val,
      ) = (
          treatments[train_indices],
          outcomes[train_indices],
          vitals[train_indices],
          outcomes_unscaled[train_indices],
          treatments[val_indices],
          outcomes[val_indices],
          vitals[val_indices],
          outcomes_unscaled[val_indices],
      )
      static_features_train, static_features_val = (
          static_features[train_indices],
          static_features[val_indices],
      )
      sequence_lengths_train, sequence_lengths_val = (
          sequence_lengths[train_indices],
          sequence_lengths[val_indices],
      )
    else:
      raise NotImplementedError()

    max_number = min(max_number, total_item_id_num)
    tr_item_num = 0
    if max_number is not None and max_number > 0:
      if isinstance(few_shot_sample_num, float):
        tr_item_num = int(max_number * few_shot_sample_num)
      elif isinstance(few_shot_sample_num, int):
        tr_item_num = few_shot_sample_num
    else:
      tr_item_num = len(item_ids_train)
    if tr_item_num > 0 and tr_item_num < len(item_ids_train):
      item_ids_train = np.random.choice(
          item_ids_train, size=tr_item_num, replace=False
      )
      train_indices = (
          valid_seq_splits.set_index('item_id')
          .loc[item_ids_train]['i']
          .tolist()
      )
      treatments_train = treatments[train_indices]
      outcomes_train = outcomes[train_indices]
      vitals_train = vitals[train_indices]
      outcomes_unscaled_train = outcomes_unscaled[train_indices]
      static_features_train = static_features[train_indices]
      sequence_lengths_train = sequence_lengths[train_indices]

    train_f = M5RealDataset(
        treatments_train,
        outcomes_train,
        vitals_train,
        static_features_train,
        outcomes_unscaled_train,
        scaling_params,
        sequence_lengths_train,
        'train',
    )
    if split['val'] > 0.0:
      val_f = M5RealDataset(
          treatments_val,
          outcomes_val,
          vitals_val,
          static_features_val,
          outcomes_unscaled_val,
          scaling_params,
          sequence_lengths_val,
          'val',
      )
    else:
      val_f = None

    if not use_src_test:
      test_f = M5RealDataset(
          treatments_test,
          outcomes_test,
          vitals_test,
          static_features_test,
          outcomes_unscaled_test,
          scaling_params,
          sequence_lengths_test,
          'test',
      )
    else:
      test_f = M5RealDataset(
          treatments_val,
          outcomes_val,
          vitals_val,
          static_features_val,
          outcomes_unscaled_val,
          scaling_params,
          sequence_lengths_val,
          'test',
      )
    return train_f, val_f, test_f
