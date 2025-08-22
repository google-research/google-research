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

"""Utility methods for Machine Learning."""
import logging
import os
import tempfile
import time
from typing import Dict, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.io import gfile


class StandardScaler:
  """A modification of sklearn's StandardScaler that allows to select axes."""

  def __init__(self, feature_axes = 0):
    """Initializes a StandardScaler with 0 mean and standard deviation of 1.

    Args:
      feature_axes: int or tuple of ints. The axes over which mean and std are
        to be standardized. In the original scikit implementation this is 0.
    """
    self.mean = 0
    self.std = 1
    self.feature_axes = feature_axes

  def fit(self, x):
    """Computes the mean and std to be used for later scaling."""
    if self.feature_axes is not None:
      self.mean = x.mean(axis=self.feature_axes, keepdims=True)
      self.std = x.std(axis=self.feature_axes, keepdims=True)

  def transform(self, x):
    """Performs standardization by centering and scaling."""
    return (x - self.mean) / self._handle_zeros_in_std()

  def fit_transform(self, x):
    """Performs both fit and transform on the input."""
    self.fit(x)
    return self.transform(x)

  def store(self, path):
    """Stores the scaler in a given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
      f.truncate(0)
      joblib.dump(self, f)

  def _handle_zeros_in_std(self):
    """Handles case of zero std.

    Returns:
      Array/scalar of the same shape as `self.std`, with zeros replaced by ones.
    """
    if np.isscalar(self.std):
      if self.std == 0:
        return 1
      return self.std
    elif isinstance(self.std, np.ndarray):
      std = self.std.copy()
      std[std == 0.0] = 1.0
      return std
    raise ValueError('Unknown std type.')


def read_scaler_files(scaler_dir):
  """Reads and loads StandardScaler from a folder."""
  files = os.listdir(scaler_dir)
  scalers = {}
  for f in files:
    with open(os.path.join(scaler_dir, f), 'rb') as scaler_file:
      scalers[f] = joblib.load(scaler_file)
  return scalers


def save_model_to_directory(model, dirpath):
  """Saves a model in TF format.

  Arguments:
    model: Keras model instance to be saved.
    dirpath: The path to where the model should be saved.
  """
  model.save(dirpath, save_format='tf')


def save_history_to_folder(
    history, directory
):
  """Saves a training history.

  Arguments:
    history: Training history instance to be saved.
    directory: The path to where the history should be saved.
  """
  df = pd.DataFrame(history)
  filename = 'history.csv'
  with open(os.path.join(directory, filename), mode='w') as f:
    df.to_csv(f)
