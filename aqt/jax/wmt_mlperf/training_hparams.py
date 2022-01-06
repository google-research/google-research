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

# pylint: disable=g-importing-member
"""Utilities for managing the hyperparameter configurations of experiments."""

import dataclasses
import typing
from typing import Optional, TypeVar

from absl import flags

from aqt.jax.flax import struct as flax_struct
from aqt.jax.wmt_mlperf import models
from aqt.utils import hparams_utils as os_hparams_utils

T = TypeVar('T')

FLAGS = flags.FLAGS

dataclass = flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass


# TODO(malmaud): Have these be subclasses of a new 'Train' class that implements
# the training loop, in analogy to how Transformer.HParams is a subclass of
# Transformer. That will be a non-trivial refactoring of train.py, so
# postpone for a follow-up CL.


@dataclass
class LearningRateSchedulerHParams:
  """The set of hyperparameters used to control the learning rate scheduler.

  See train.create_learning_rate_scheduler for the corresponding
  implementation.
  """

  # A string with factors separated by '*' that defines the schedule.
  # Consists of:
  # * constant: interpreted as the constant value,
  # * linear_warmup: interpreted as linear warmup until warmup_steps,
  # * rsqrt_decay: divide by square root of max(step, warmup_steps)
  # * decay_every: Every k steps decay the learning rate by decay_factor.
  # * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.
  # For example, "constant * linear_warmup * rsqrt_decay".
  factors: str

  # the starting constant for the lr schedule.
  base_learning_rate: float

  # how many steps to warm up for in the warmup schedule.
  warmup_steps: int

  # The amount to decay the learning rate by.
  decay_factor: Optional[float]

  # How often to decay the learning rate.
  steps_per_decay: Optional[int]

  # Steps per cycle when using cosine decay.
  steps_per_cycle: Optional[int]


# TODO(b/176825799): Create shared training hparams for different models i.e.
# wmt_mlperf and imagenet.
@dataclass
class TrainingHParams:
  """The set of hyperparameters used by the training loop."""

  metadata: os_hparams_utils.HParamsMetadata

  # Optimizer parameters
  learning_rate_schedule: LearningRateSchedulerHParams
  per_host_batch_size: int
  num_train_steps: int
  weight_decay: float  # Decay factor for AdamW style weight decay
  beta1: float
  beta2: float
  eps: float

  # RNG parameters
  random_seed: int  # Integer for PRNG random seed
  hardware_rng: bool  # Whether to use hardware rng for dropout

  # Auto clip activation quantization parameter
  activation_bound_update_freq: int  # Update frequency. If set to a value > 0,
  # will set update_bounds to True at step = 1 * freq, step = 2 * freq, etc..
  # If set to a value <= 0, update_bounds will always be False.
  # If GetBounds is used anywhere in the model, then this parameter should be
  # set to a value > 0.
  # TODO(wanglisa): Validate this before training starts (b/162350655)

  activation_bound_start_step: int

  prefer_int8_to_int32_dot: bool  # lax.dot inputs with an int8 dtype.

  # Weight outlier regularization parameters
  weight_outlier_regularization: float
  weight_outlier_regularization_regex: str

  model_hparams: models.Transformer.HParams

  @classmethod
  def from_config_dict(cls, config_dict):
    return os_hparams_utils.load_dataclass_from_config_dict(cls, config_dict)
