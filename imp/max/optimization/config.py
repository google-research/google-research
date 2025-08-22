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

"""Configuration of the optimization pipeline."""

import dataclasses

import fiddle as fdl
import jax
from jax import numpy as jnp
import optax

from imp.max.config import base
from imp.max.config import validators
from imp.max.core import constants
from imp.max.utils import typing

AggregationMethod = constants.ObjectiveAggregation
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName


@dataclasses.dataclass
class BaseObjective(base.Config):
  """Base configuration for objective functions."""

  name: str | None = None
  loss_weight: float = 1.0
  dtype: jax.typing.DTypeLike | None = None


@dataclasses.dataclass
class BaseSequenceObjective(BaseObjective):
  """The base objective for token sequences.

  Attributes:
    route_key: The route feature name in the data collection or its
      corresponding dataflow.
    predictions_key: The output feature name of the model predictions in the
      data collection or its corresponding dataflow.
    targets_key: The output feature name of the prediction targets in the
      data collection or its corresponding dataflow.
    left_shift_targets: Whether to shift the targets by one token. This is only
      used in the autoregressive decoder training. If set to True, targets are
      shifted one token to the left and predictions are truncated at end by one
      token accordingly (so that predictions predict the next target at each
      position).
  """
  # TODO(b/277977414): Encapsulate all keys in one object
  route_key: str = ''
  predictions_key: str = ''
  targets_key: str = ''
  modality_token_weights: tuple[typing.ObjectiveModalityTokenWeight, Ellipsis] = ()
  left_shift_targets: bool = False


@dataclasses.dataclass
class ObjectiveAggregator(BaseObjective):
  """Config for objective that aggregates multiple objectives in a dataset."""

  name: str | None = constants.Objective.OBJECTIVE_AGGREGATOR
  loss_weight: float = 1.0
  objectives: tuple[BaseObjective, Ellipsis] = ()
  aggregation_method: str = AggregationMethod.SUM
  dtype: jax.typing.DTypeLike | None = None


@dataclasses.dataclass
class CrossModalNCE(BaseObjective):
  """Parameters for cross-modal correspondence using NCE loss."""

  name: str = constants.Objective.CROSS_MODAL_NCE
  modality_pair_weights: tuple[typing.ObjectiveModalityPairWeight, Ellipsis] = ()
  hparams_route_key: str = DataFeatureRoute.ENCODER
  temperature: float = 0.07
  margin: float = 0.0


@dataclasses.dataclass
class SigmoidBinaryCrossEntropy(BaseSequenceObjective):
  """Sigmoid binary cross entropy objective for token sequences.

  Attributes:
    name: The name for this objective.
    route_key: The route feature name in the data collection or its
      corresponding dataflow.
    predictions_key: The output feature name of the model predictions in the
      data collection or its corresponding dataflow.
    targets_key: The output feature name of the prediction targets in the
      data collection or its corresponding dataflow.
  """
  name: str = constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY
  route_key: str = DataFeatureRoute.ENCODER
  predictions_key: str = DataFeatureName.LOGITS
  targets_key: str = DataFeatureName.LABEL


@dataclasses.dataclass
class SoftmaxCrossEntropy(SigmoidBinaryCrossEntropy):
  """Softmax cross entropy objective for token sequences.

  Attributes:
    name: The name for this objective.
    one_hot_targets: Whether the target (labels) are one-hot vectors.
  """
  name: str = constants.Objective.SOFTMAX_CROSS_ENTROPY
  one_hot_targets: bool = True


@dataclasses.dataclass
class MeanSquaredError(BaseSequenceObjective):
  """Mean squared error (MSE) loss objective.

  Attributes:
    name: The name for this objective.
    predictions_key: The feature name of the model predictions in the outputs
      dictionary.
    targets_key:  The feature name of the prediction targets in the inputs
      dictionary.
    z_score_predictions: If True, the predictions will be normalized using
      z-score (normalize to mean and std).
    z_score_targets: If True, the target (groundtruth) features will be
      normalized using z-score (normalize to mean and std).
  """
  name: str = constants.Objective.MEAN_SQUARED_ERROR
  route_key: str = DataFeatureRoute.ENCODER
  predictions_key: str = DataFeatureName.FEATURES
  targets_key: str = DataFeatureName.TOKEN_RAW
  z_score_predictions: bool = False
  z_score_targets: bool = False


@dataclasses.dataclass
class LearningRate(base.Config):
  """Base configuration for learning_rate schedulers."""

  name: str | None = None


@dataclasses.dataclass
class ConstantLearningRate(LearningRate):
  """Constant learning_rate."""

  name: str = constants.Schedule.CONSTANT_LR
  value: float = 0.001


@dataclasses.dataclass
class CosineDecayLearningRate(LearningRate):
  """Half-period cosine decay learning_rate schedule."""

  name: str = constants.Schedule.COSINE_DECAY_LR
  init_value: float = 0.001
  decay_steps: int = 100000
  alpha: float = 0.0


@dataclasses.dataclass
class WarmupCosineDecayLearningRate(LearningRate):
  """Half-period cosine decay learning_rate schedule with linear warmup."""

  name: str = constants.Schedule.WARMUP_COSINE_DECAY_LR
  init_value: float = 0.0
  peak_value: float = 0.001
  warmup_steps: int = 10000
  decay_steps: int = 100000
  end_value: float = 0.0


@dataclasses.dataclass
class PreWarmupCosineDecayLearningRate(WarmupCosineDecayLearningRate):
  """Cosine decay schedule with linear warmup and linear pre-warmup.

  Adds a pre-warmup linear LR if specified. A small LR value for <5k steps
  can reduce the chance of model divergence early in training.

  Attributes:
    name: the name of the config.
    pre_warmup_steps: if nonzero, apply a linear pre-warmup from
      `pre_warmup_init_value` to `init_value` for the given steps.
    pre_warmup_init_value: the starting value for pre-warmup.
  """

  name: str = constants.Schedule.PRE_WARMUP_COSINE_DECAY_LR
  pre_warmup_steps: int = 0
  pre_warmup_init_value: float = 0.


@dataclasses.dataclass
class Optimizer(base.Config):
  """Base configuration for optimizers."""

  name: str = ''
  learning_rate: LearningRate = dataclasses.field(
      default_factory=ConstantLearningRate
  )


@dataclasses.dataclass
class SgdOptimizer(Optimizer):
  """Configuration of standard Momentum optimizer (see optax.sgd for info)."""

  name: str = constants.Optimizer.SGD
  momentum: float = 0.9
  nesterov: bool = False


@dataclasses.dataclass
class AdamOptimizer(Optimizer):
  """Configuration of standard Adam optimizer  (see optax.adam for info)."""

  name: str = constants.Optimizer.ADAM
  b1: float = 0.9
  b2: float = 0.999
  eps: float = 1e-07
  eps_root: float = 0.0


@dataclasses.dataclass
class AdamWOptimizer(AdamOptimizer):
  """Configuration of AdamW optimizer (see optax.adamw for info)."""

  name: str = constants.Optimizer.ADAM_W
  weight_decay: float = 0.0


@dataclasses.dataclass
class GaLoreAdamWOptimizer(AdamWOptimizer):
  """Configuration of GaLoreAdamW optimizer (see arXiv:2403.03507 for info)."""

  name: str = constants.Optimizer.GALORE_ADAM_W
  rank: int = 128
  svd_frequency: int = 250


@dataclasses.dataclass
class AdafactorOptimizer(Optimizer):
  """Configuration of Adafactor optimizer (see optax.adafactor for info)."""

  name: str = constants.Optimizer.ADAFACTOR
  momentum: float | None = 0.9
  decay_rate: float = 0.999
  decay_offset: int = 0
  factored: bool = True
  min_dim_size_to_factor: int = 128
  multiply_by_parameter_scale: bool = False
  clipping_threshold: float | None = 1.0
  dtype_momentum: jax.typing.DTypeLike = jnp.float32
  weight_decay_rate: float | None = None
  eps: float = 1e-7
  weight_decay_mask: optax.MaskOrFn = None


@validators.lock
@dataclasses.dataclass
class Optimization(base.Config):
  """Full configuration for the entire optimization pipeline."""

  optimizer: Optimizer = dataclasses.field(default_factory=AdamOptimizer)
  loss: tuple[BaseObjective, Ellipsis] = ()
  restore_path: str | None = None
  max_checkpoints: int = 50
  total_steps: int = 500000
  save_checkpoint_freq: int = 10000
  gradient_clip_norm: float = 0.0
  metrics_postprocess_fn: fdl.Partial | None = None
