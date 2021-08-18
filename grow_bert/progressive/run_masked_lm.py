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

# pylint: disable=line-too-long
r"""Creating the task and start trainer.

Supported modes: train, eval, train_and_eval, continuous_eval
The ProgressiveMaskedLM class is a subclass of ProgressivePolicy. This means
that a progressive trainer instead of a base trainer.
"""
# pylint: enable=line-too-long
# Lint as: python3

import dataclasses
from absl import app
from absl import flags
import gin

from grow_bert.progressive import masked_lm
from grow_bert.progressive import utils
from official.common import flags as tfm_flags
from official.modeling import optimization
from official.modeling.fast_training.progressive import train_lib
from official.modeling.fast_training.progressive import trainer as prog_trainer_lib
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.data import pretrain_dataloader
from official.utils.misc import distribution_utils


FLAGS = flags.FLAGS


AdamWeightDecay = optimization.AdamWeightDecayConfig
PolynomialLr = optimization.PolynomialLrConfig
PolynomialWarmupConfig = optimization.PolynomialWarmupConfig


@dataclasses.dataclass
class BertOptimizationConfig(optimization.OptimizationConfig):
  """Bert optimization config."""
  optimizer: optimization.OptimizerConfig = optimization.OptimizerConfig(
      type='adamw',
      adamw=AdamWeightDecay(
          weight_decay_rate=0.01,
          exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']))
  learning_rate: optimization.LrConfig = optimization.LrConfig(
      type='polynomial',
      polynomial=PolynomialLr(
          initial_learning_rate=1e-4,
          decay_steps=1000000,
          end_learning_rate=0.0))
  warmup: optimization.WarmupConfig = optimization.WarmupConfig(
      type='polynomial', polynomial=PolynomialWarmupConfig(warmup_steps=10000))


def get_exp_config():
  """Get ExperimentConfig."""

  params = cfg.ExperimentConfig(
      task=masked_lm.MaskedLMConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          small_train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=prog_trainer_lib.ProgressiveTrainerConfig(
          progressive=masked_lm.ProgStackingConfig(),
          optimizer_config=BertOptimizationConfig(),
          train_steps=1000000),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return utils.config_override(params, FLAGS)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = get_exp_config()

  distribution_strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with distribution_strategy.scope():
    task = masked_lm.ProgressiveMaskedLM(
        strategy=distribution_strategy,
        progressive_config=params.trainer.progressive,
        optimizer_config=params.trainer.optimizer_config,
        train_data_config=params.task.train_data,
        small_train_data_config=params.task.small_train_data,
        task_config=params.task)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=FLAGS.model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
