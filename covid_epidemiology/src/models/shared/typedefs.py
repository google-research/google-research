# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Python typing definitions.

Library of typing definitions used through the project
"""

import enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

FeatureScaler = Union[preprocessing.MinMaxScaler, preprocessing.StandardScaler]
GroundTruthTimeSeriesTuple = Tuple[tf.Tensor, Dict[str, tf.Tensor],
                                   Dict[str, tf.Tensor], List[str],
                                   Dict[str, Dict[str, np.ndarray]]]
TimeSeriesFeature = Dict[str, np.ndarray]  # location-->values map
StaticFeature = Dict[str, float]  # location-->values map
TimeSeriesFeaturesDict = Dict[str, Optional[TimeSeriesFeature]]
StaticFeaturesDict = Dict[str, Optional[StaticFeature]]
FeatureScalersDict = Dict[str, Optional[FeatureScaler]]


class FeatureType(enum.Enum):
  """Type of feature."""

  TIMESERIES = "timeseries"
  STATIC = "static"
