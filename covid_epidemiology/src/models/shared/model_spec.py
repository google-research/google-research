# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Defines structure for specifying model features and their properties.."""
import enum
from typing import Any, Dict, List, Optional

import dataclasses


class ForecastMethod(enum.Enum):
  NONE = "none"
  CONSTANT = "constant"
  XGBOOST = "xgboost"
  PERIODIC_WEEKLY = "weekly"


class EncoderWeightSignConstraint(enum.Enum):
  """Constraint on the sign of the encoder weights."""
  NONE = "none"
  POSITIVE = "positive"
  NEGATIVE = "negative"


@dataclasses.dataclass
class FeatureSpec:
  """Spec for defining a feature in an encoder."""
  name: str
  initializer: Optional[int] = 0
  forecast_method: "ForecastMethod" = ForecastMethod.XGBOOST
  apply_lasso: bool = False
  weight_sign_constraint: EncoderWeightSignConstraint = EncoderWeightSignConstraint.NONE


@dataclasses.dataclass
class EncoderSpec:
  """Spec for defining a variable encoder in a model."""
  encoder_name: str
  encoder_type: str
  vaccine_type: Optional[str] = None
  static_feature_specs: Optional[List[FeatureSpec]] = dataclasses.field(
      default_factory=list)
  covariate_feature_specs: Optional[List[FeatureSpec]] = dataclasses.field(
      default_factory=list)
  covariate_feature_time_offset: int = 0
  covariate_feature_window: int = 4
  encoder_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelSpec:
  """Spec for defining a model."""
  encoder_specs: List[EncoderSpec]
  hparams: Dict[str, Any]

  def _get_feature_specs(self,
                         feature_spec_type):
    """Get feature specs from the model.

    Args:
      feature_spec_type: The type of feature spec to get. Raises if not static
        or covariate.

    Returns:
      A map from the feature spec name to it's full feature spec.
    """
    if feature_spec_type not in {"static", "covariate"}:
      raise ValueError(
          f"The feature spec type {feature_spec_type} is not supported")

    feature_spec_from_name = {}
    for encoder in self.encoder_specs:
      cur_specs = getattr(encoder, f"{feature_spec_type}_feature_specs", None)
      if not cur_specs:
        continue

      for feature in cur_specs:
        feature_spec_from_name[feature.name] = feature

    return feature_spec_from_name

  @property
  def covariate_feature_specs(self):
    """Gets the feature specs of all the covariates in the model."""
    return self._get_feature_specs("covariate")

  @property
  def static_feature_specs(self):
    """Gets the feature specs of all the static features in the model."""
    return self._get_feature_specs("static")

  @property
  def covariate_names(self):
    """Gets the names of all the covariates in the model."""
    return list(self.covariate_feature_specs.keys())

  @property
  def static_names(self):
    """Gets the names of all the static features in the model."""
    return list(self.static_feature_specs.keys())
