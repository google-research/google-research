# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Utility functions for model setup and reproducibility."""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from united_earth_surface_fluxes import model_baselines
from united_earth_surface_fluxes import model_moe


def set_seed(seed):
  """Sets the random seed for reproducibility."""
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.config.experimental.enable_op_determinism()


def get_model_class(model_name):
  """Mapping function to resolve model classes by string name."""
  if model_name == 'FracMoE':
    return model_moe.FracMoE
  elif model_name == 'FTTransformer':
    return model_baselines.FTTransformer
  elif model_name == 'SoftMoE':
    return model_moe.SoftRoutingMoE
  elif model_name == 'UniTeD':
    return model_moe.UniTeD
  elif model_name == 'MLP':
    return model_baselines.MLP
  else:
    raise ValueError(f'Unknown model name: {model_name}')


def parse_period_list(period_str):
  """Parses a period list string into a list of daily date strings."""
  if not period_str:
    return []
  if ':' in period_str:
    start_str, end_str = period_str.split(':', 1)
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    date_range = pd.date_range(start_date, end_date, freq='D')
    return [d.strftime('%Y-%m-%d') for d in date_range]
  return [s.strip() for s in period_str.split(',') if s.strip()]
