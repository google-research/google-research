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

"""Base classes of synthetic and real datasets.

including train/val/test splits and preprocessing steps.
"""

import copy
import logging
import numpy as np
import sklearn.model_selection

deepcopy = copy.deepcopy
train_test_split = sklearn.model_selection.train_test_split

logger = logging.getLogger(__name__)


class SyntheticDatasetCollection:
  """Dataset collection (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)."""

  def __init__(self, **kwargs):
    _ = kwargs
    self.seed = None

    self.processed_data_encoder = False
    self.processed_data_decoder = False
    self.processed_data_multi = False
    self.processed_data_msm = False
    self.processed_data_rep = False
    self.processed_data_rep_head = False
    self.processed_data_rep_est = False

    self.train_f = None
    self.val_f = None
    self.test_cf_one_step = None
    self.test_cf_treatment_seq = None
    self.train_scaling_params = None
    self.projection_horizon = None

    self.autoregressive = None
    self.has_vitals = None

  def process_data_encoder(self):
    self.train_f.process_data(self.train_scaling_params)
    self.val_f.process_data(self.train_scaling_params)
    self.test_cf_one_step.process_data(self.train_scaling_params)
    self.processed_data_encoder = True

  def process_propensity_train_f(
      self, propensity_treatment, propensity_history
  ):
    """Generate stabilized weights for RMSN for the train subset.

    Args:
        propensity_treatment: Propensity treatment network
        propensity_history: Propensity history network
    """
    prop_treat_train_f = propensity_treatment.get_propensity_scores(
        self.train_f
    )
    prop_hist_train_f = propensity_history.get_propensity_scores(self.train_f)
    self.train_f.data['stabilized_weights'] = np.prod(
        prop_treat_train_f / prop_hist_train_f, axis=2
    )

  def process_data_decoder(self, encoder, save_encoder_r=False):
    """Used by CRN, RMSN, EDCT."""
    self.train_f.process_data(self.train_scaling_params)
    self.val_f.process_data(self.train_scaling_params)
    self.test_cf_treatment_seq.process_data(self.train_scaling_params)

    # Representation generation / One-step ahead prediction with encoder
    r_train_f = encoder.get_representations(self.train_f)
    r_val_f = encoder.get_representations(self.val_f)
    r_test_cf_treatment_seq = encoder.get_representations(
        self.test_cf_treatment_seq
    )
    try:
      outputs_test_cf_treatment_seq = encoder.get_predictions(
          self.test_cf_treatment_seq
      )
    except AttributeError:
      outputs_test_cf_treatment_seq = None

    # Splitting time series wrt specified projection horizon.
    # Preparing test sequences.
    self.train_f.process_sequential(
        r_train_f, self.projection_horizon, save_encoder_r=save_encoder_r
    )
    self.val_f.process_sequential(
        r_val_f, self.projection_horizon, save_encoder_r=save_encoder_r
    )
    # Considers only last timesteps according to projection horizon
    self.test_cf_treatment_seq.process_sequential_test(
        self.projection_horizon,
        r_test_cf_treatment_seq,
        save_encoder_r=save_encoder_r,
    )
    self.test_cf_treatment_seq.process_autoregressive_test(
        r_test_cf_treatment_seq,
        outputs_test_cf_treatment_seq,
        self.projection_horizon,
        save_encoder_r=save_encoder_r,
    )
    self.processed_data_decoder = True

  def process_data_multi(self, generative_style_predict=False):
    """Used by CT."""
    self.train_f.process_data(self.train_scaling_params)
    if hasattr(self, 'val_f') and self.val_f is not None:
      self.val_f.process_data(self.train_scaling_params)
      if generative_style_predict:
        self.val_f.process_sequential_split()

    self.test_cf_one_step.process_data(self.train_scaling_params)
    if generative_style_predict:
      self.test_cf_one_step.process_sequential_test(1)
      self.test_cf_one_step.process_sequential_multi(1 - 1)

    self.test_cf_treatment_seq.process_data(self.train_scaling_params)
    if generative_style_predict:
      self.test_cf_treatment_seq.process_sequential_test(
          self.projection_horizon + 1
      )
      self.test_cf_treatment_seq.process_sequential_multi(
          self.projection_horizon
      )
    else:
      self.test_cf_treatment_seq.process_sequential_test(
          self.projection_horizon
      )
      self.test_cf_treatment_seq.process_sequential_multi(
          self.projection_horizon
      )
    self.processed_data_multi = True

  def process_data_rep(self, max_train_length):
    """Used by self-supervised representation learning methods."""
    self.train_f.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=True
    )
    self.val_f.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=False
    )
    self.test_cf_one_step.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=False
    )
    self.processed_data_rep = True

  def process_data_rep_head(self, rep, max_train_length, save_rep_r=False):
    """Used by headers attached to self-supervised representation learning methods."""
    self.train_f.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=False
    )
    self.val_f.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=False
    )
    self.test_cf_treatment_seq.process_data_rep(
        self.train_scaling_params, max_train_length, is_training=False
    )

    # Representation generation / One-step ahead prediction with encoder
    r_train_f = rep.get_representations(self.train_f)
    r_val_f = rep.get_representations(self.val_f)
    r_test_cf_treatment_seq = rep.get_representations(
        self.test_cf_treatment_seq
    )
    outputs_test_cf_treatment_seq = None

    # Splitting time series wrt specified projection horizon.
    # Preparing test sequences.
    self.train_f.process_sequential_rep_head(
        r_train_f, self.projection_horizon, save_encoder_r=save_rep_r
    )
    self.val_f.process_sequential_rep_head(
        r_val_f, self.projection_horizon, save_encoder_r=save_rep_r
    )
    # Considers only last timesteps according to projection horizon
    self.test_cf_treatment_seq.process_sequential_test_rep_head(
        self.projection_horizon,
        r_test_cf_treatment_seq,
        save_encoder_r=save_rep_r,
    )
    self.test_cf_treatment_seq.process_autoregressive_test_rep_head(
        r_test_cf_treatment_seq,
        outputs_test_cf_treatment_seq,
        self.projection_horizon,
        save_encoder_r=save_rep_r,
    )
    self.processed_data_rep_head = True

  def process_data_rep_est(self):
    self.train_f.process_data(self.train_scaling_params)
    self.val_f.process_data(self.train_scaling_params)
    self.test_cf_one_step.process_data(self.train_scaling_params)
    self.test_cf_treatment_seq.process_data(self.train_scaling_params)
    # self.test_cf_treatment_seq.process_sequential_rep_est(self.projection_horizon)
    self.processed_data_rep_est = True

  def split_train_f_holdout(self, holdout_ratio=0.1):
    """Used by G-Net."""
    if not hasattr(self, 'train_f_holdout') and holdout_ratio > 0.0:
      self.train_f_holdout = deepcopy(self.train_f)
      for k, v in self.train_f.data.items():
        self.train_f.data[k], self.train_f_holdout.data[k] = train_test_split(
            v, test_size=holdout_ratio, random_state=self.seed
        )
      logger.info(
          '%s',
          f'Splited train_f on train_f: {len(self.train_f)} and'
          f' train_f_holdout: {len(self.train_f_holdout)}',
      )

  def explode_cf_treatment_seq(self, mc_samples=1):
    """Producing mc_samples copies of test_cf_treatment_seq subset.

    for further MC-Sampling (e.g. for G-Net).

    Args:
      mc_samples: Number of copies

    Returns:
      data
    """
    if not hasattr(self, 'test_cf_treatment_seq_mc'):
      logger.info('%s', f'Exploding test_cf_treatment_seq {mc_samples} times')

      self.test_cf_treatment_seq_mc = []
      for m in range(mc_samples):
        self.test_cf_treatment_seq_mc.append(self.test_cf_treatment_seq)
        self.test_cf_treatment_seq_mc[m].data = deepcopy(
            self.test_cf_treatment_seq.data
        )


class RealDatasetCollection:
  """Dataset collection (train_f, val_f, test_f)."""

  def __init__(self, **kwargs):
    _ = kwargs
    self.seed = None

    self.processed_data_encoder = False
    self.processed_data_decoder = False
    self.processed_data_propensity = False
    self.processed_data_msm = False

    self.train_f = None
    self.val_f = None
    self.test_f = None
    self.train_scaling_params = None
    self.projection_horizon = None

    self.autoregressive = None
    self.has_vitals = None

  def process_data_encoder(self):
    pass

  def process_propensity_train_f(
      self, propensity_treatment, propensity_history
  ):
    """Generate stabilized weights for RMSN for the train subset.

    Args:
        propensity_treatment: Propensity treatment network
        propensity_history: Propensity history network
    """
    prop_treat_train_f = propensity_treatment.get_propensity_scores(
        self.train_f
    )
    prop_hist_train_f = propensity_history.get_propensity_scores(self.train_f)
    self.train_f.data['stabilized_weights'] = np.prod(
        prop_treat_train_f / prop_hist_train_f, axis=2
    )

  def process_data_decoder(self, encoder, save_encoder_r=False):
    """Used by CRN, RMSN, EDCT."""
    # Multiplying test trajectories
    self.test_f.explode_trajectories(self.projection_horizon)

    # Representation generation / One-step ahead prediction with encoder
    r_train_f = encoder.get_representations(self.train_f)
    r_val_f = encoder.get_representations(self.val_f)
    r_test_f = encoder.get_representations(self.test_f)
    outputs_test_f = encoder.get_predictions(self.test_f)

    # Splitting time series wrt specified projection horizon.
    # Preparing test sequences.
    self.train_f.process_sequential(
        r_train_f, self.projection_horizon, save_encoder_r=save_encoder_r
    )
    self.val_f.process_sequential(
        r_val_f, self.projection_horizon, save_encoder_r=save_encoder_r
    )

    self.test_f.process_sequential_test(
        self.projection_horizon, r_test_f, save_encoder_r=save_encoder_r
    )
    self.test_f.process_autoregressive_test(
        r_test_f,
        outputs_test_f,
        self.projection_horizon,
        save_encoder_r=save_encoder_r,
    )

    self.processed_data_decoder = True

  def process_data_multi(self, generative_style_predict=False):
    """Used by CT."""
    _ = generative_style_predict
    self.test_f_multi = deepcopy(self.test_f)

    # Multiplying test trajectories
    self.test_f_multi.explode_trajectories(self.projection_horizon)

    self.test_f_multi.process_sequential_test(self.projection_horizon)
    self.test_f_multi.process_sequential_multi(self.projection_horizon)

    self.processed_data_multi = True

  def process_data_rep_est(self):
    pass

  def split_train_f_holdout(self, holdout_ratio=0.1):
    """Used by G-Net."""
    if not hasattr(self, 'train_f_holdout') and holdout_ratio > 0.0:
      self.train_f_holdout = deepcopy(self.train_f)
      for k, v in self.train_f.data.items():
        self.train_f.data[k], self.train_f_holdout.data[k] = train_test_split(
            v, test_size=holdout_ratio, random_state=self.seed
        )
      logger.info(
          '%s',
          f'Splited train_f on train_f: {len(self.train_f)} and'
          f' train_f_holdout: {len(self.train_f_holdout)}',
      )

  def explode_cf_treatment_seq(self, mc_samples=1):
    """Producing mc_samples copies of test_cf_treatment_seq subset.

    for further MC-Sampling (e.g. for G-Net).

    Args:
      mc_samples: Number of copies

    Returns:
      data
    """
    if not hasattr(self, 'test_f_mc'):
      self.test_f_mc = []
      for m in range(mc_samples):
        logger.info('%s', f'Exploding test_f {mc_samples} times')
        self.test_f_mc.append(self.test_f_multi)
        self.test_f_mc[m].data = deepcopy(self.test_f_multi.data)
