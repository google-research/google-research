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

"""Configuration of the evaluation pipeline."""

import dataclasses

import fiddle as fdl
import optax

from imp.max.config import base
from imp.max.optimization import config as opt_config
from imp.max.optimization import schedules


# ----------------------------------------------------------------------
# ------------------------------ METRICS -------------------------------
# ----------------------------------------------------------------------


@dataclasses.dataclass
class BaseMetric(base.Config):
  """Base configuration for metric functions."""

  name: str | None = None


@dataclasses.dataclass
class Accuracy(BaseMetric):
  """Parameters for Accuracy metric."""

  name: str = 'accuracy'
  top: tuple[int, Ellipsis] = (1, 5)


@dataclasses.dataclass
class RetrievalRecall(BaseMetric):
  """Parameters for Recall at different levels."""

  name: str = 'retrieval_recall'
  at: tuple[int, Ellipsis] = (1, 5, 10)
  return_median_rank: bool = True
  instance_selection_method: str = 'boundary'


@dataclasses.dataclass
class MetricStack(base.Config):
  """Parameters for a stack of different metrics."""

  classification: tuple[BaseMetric, Ellipsis] = ()
  retrieval: tuple[BaseMetric, Ellipsis] = ()


# ----------------------------------------------------------------------
# ----------------------------- EXECUTION ------------------------------
# ----------------------------------------------------------------------


@dataclasses.dataclass
class Evaluation(base.Config):
  """Full configuration for the evaluation pipeline."""

  restore_path: str | None = None
  metric: MetricStack = dataclasses.field(default_factory=MetricStack)
  metrics_postprocess_fn: fdl.Partial | None = None


# ----------------------------------------------------------------------
# ------------------------------ SERVINGS ------------------------------
# ----------------------------------------------------------------------


@dataclasses.dataclass
class BaseServing(base.Config):
  """Base configuration for serving classes."""

  name: str | None = None


@dataclasses.dataclass
class BaseLinearClassification(BaseServing):
  """Parameters for linear classification serving."""

  batch_size: int = 128
  regularization: float = 0.
  input_noise_std: float = 0.
  seed: int = 0
  verbose: int = 0


@dataclasses.dataclass
class OnlineLinearClassification(BaseLinearClassification):
  """Parameters for online linear classification serving."""

  name: str = 'online_linear_classifier'
  total_steps: int = 50000
  learning_rate: optax.Schedule = dataclasses.field(
      default_factory=lambda: schedules.get_schedule(  # pylint: disable=g-long-lambda
          opt_config.WarmupCosineDecayLearningRate(
              init_value=0.0,
              peak_value=1e-4,
              warmup_steps=200,
              decay_steps=50000,
              end_value=0.0,
          )
      )
  )


@dataclasses.dataclass
class OfflineLinearClassification(BaseLinearClassification):
  """Parameters for offline linear classification serving."""

  name: str = 'offline_linear_classifier'
  total_epochs: int = 100
  learning_rate: float = 1e-4
