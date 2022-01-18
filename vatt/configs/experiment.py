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

# Lint as: python3
"""Config definitions for video model training/evaluation tasks."""

import dataclasses
from typing import Optional, Tuple, Union

from vatt.configs import audio as aud_config
from vatt.configs import base_config
from vatt.configs import dataset as ds_config
from vatt.configs import multimodal as mm_config
from vatt.configs import video as vid_config


@dataclasses.dataclass
class RuntimeConfig(base_config.Config):
  """High-level configurations for Runtime."""

  distribution_strategy: str = 'tpu'
  tpu: Optional[str] = None


@dataclasses.dataclass
class LearningRate(base_config.Config):
  learning_rate_base: float = 0.0001
  total_steps: int = 500000


@dataclasses.dataclass
class CosineDecayLearningRate(LearningRate):
  learning_rate_base: float = 0.0001
  warmup_learning_rate: float = 0.0
  warmup_steps: int = 5000
  total_steps: int = 500000


@dataclasses.dataclass
class StepwiseCosineDecayLearningRate(LearningRate):
  warmup_learning_rate: float = 0.0
  warmup_steps: int = 5000
  learning_rate_levels: Tuple[float, Ellipsis] = (0.0001, 0.00005)
  learning_rate_steps: Tuple[int, Ellipsis] = (5000, 500000)
  total_steps: int = 500000


@dataclasses.dataclass
class OptimizerConfig(base_config.Config):
  """Configuration for Optimizers.

  Attributes:
    name: The name of the optimizer. Defaults to None.
    nesterov: Whether or not to apply Nesterov momentum. Defaults to None.
  """
  name: str = ''
  learning_rate: LearningRate = LearningRate()


@dataclasses.dataclass
class MomentumOptimizer(OptimizerConfig):
  name: str = 'Momentum'
  momentum: float = 0.9
  nesterov: bool = False
  learning_rate: LearningRate = LearningRate()


@dataclasses.dataclass
class MomentumWOptimizer(OptimizerConfig):
  name: str = 'MomentumW'
  weight_decay: float = 5e-5
  momentum: float = 0.9
  nesterov: bool = False
  learning_rate: LearningRate = LearningRate()


@dataclasses.dataclass
class AdamOptimizer(OptimizerConfig):
  name: str = 'Adam'
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  learning_rate: LearningRate = LearningRate()


@dataclasses.dataclass
class AdamWOptimizer(OptimizerConfig):
  name: str = 'AdamW'
  weight_decay: float = 5e-5
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  learning_rate: LearningRate = LearningRate()


@dataclasses.dataclass
class TrainConfig(base_config.Config):
  """Configuration for training."""

  input: Optional[ds_config.Dataset] = None
  max_checkpoints: int = 50
  iterations_per_loop: int = 50
  save_checkpoint_freq: int = 10000
  optimizer: OptimizerConfig = OptimizerConfig()
  gradient_clip_norm: float = 0.0
  gradient_clip_norm_cls: Optional[float] = None


@dataclasses.dataclass
class EvalConfig(base_config.Config):
  """Configuration for evaluation."""

  input: Optional[ds_config.Dataset] = None


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Base configuration for an image classification experiment."""

  model_dir: str = ''
  mode: str = ''
  task: str = ''
  checkpoint_path: Optional[str] = None
  strategy_config: RuntimeConfig = RuntimeConfig()
  model_config: Optional[Union[
      vid_config.ModelConfig,
      aud_config.ModelConfig,
      mm_config.ModelConfig
      ]] = None
  train: Optional[TrainConfig] = None
  eval: Optional[EvalConfig] = None


@dataclasses.dataclass
class Pretrain(ExperimentConfig):
  """Configuration of the Self-Supervised Multimodal Pretrain experiment."""

  task: str = 'Pretrain'
  checkpoint_path: Optional[str] = None
  model_config: mm_config.TxFACModel = mm_config.TxFACModel()
  train: TrainConfig = TrainConfig(
      input=ds_config.Pretrain(),
      optimizer=AdamOptimizer(
          learning_rate=StepwiseCosineDecayLearningRate(
              warmup_learning_rate=0.0,
              warmup_steps=5000,
              learning_rate_levels=(0.0001, 0.00005),
              learning_rate_steps=(5000, 500000),
              total_steps=500000,)))
  eval: EvalConfig = EvalConfig(input=ds_config.Evaluation())


@dataclasses.dataclass
class Finetune(ExperimentConfig):
  """Configuration of the Fine-Tune experiment."""

  task: str = 'Finetune'
  checkpoint_path: Optional[str] = None
  model_config: vid_config.ModelConfig = vid_config.ViTBase()
  train: TrainConfig = TrainConfig(
      input=ds_config.Finetune(),
      max_checkpoints=10,
      save_checkpoint_freq=5000,
      gradient_clip_norm=0.75,
      gradient_clip_norm_cls=0.0,
      optimizer=MomentumOptimizer(
          learning_rate=CosineDecayLearningRate(
              warmup_steps=2500,
              learning_rate_base=0.005,
              total_steps=100000),
          ))
  eval: EvalConfig = EvalConfig(input=ds_config.Finetune())
