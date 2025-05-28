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

"""Configs for training the diffusion models."""

import ml_collections


def config(**kwargs):
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """default config for diffusion model training."""
  return config(
      ds=4000,  #  size of the dataset
      bs=500,  #  batch size
      dataset='LorenzDataset',  # for options see ode_datasets_test.py
      dataset_timesteps=60,
      seed=37,
      lr=1e-4,
      epochs=10000,
      channels=32,
      attention=False,  # Whether to use self attention in UNet
      # diffusion type e.g. (VariancePreserving, SubVariancePreserving)
      difftype='VarianceExploding',
      # whether to use correlated noise: e.g. BrownianCovariance, PinkCovariance
      noisetype='Identity',
      ic_conditioning=False,  # whether to condition on first 3 timesteps
  )
