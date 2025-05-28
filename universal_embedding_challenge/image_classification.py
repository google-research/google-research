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

"""Image classification configuration definition."""
import os

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.configs import backbones
from official.vision.configs import common
from official.vision.configs import image_classification


@exp_factory.register_config_factory('vit_with_bottleneck_imagenet_pretrain')
def image_classification_imagenet_vit_pretrain():
  """Image classification on imagenet with vision transformer."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = (
      image_classification.IMAGENET_TRAIN_EXAMPLES // train_batch_size)
  config = cfg.ExperimentConfig(
      task=image_classification.ImageClassificationTask(
          model=image_classification.ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              kernel_initializer='zeros',
              backbone=backbones.Backbone(
                  type='vit',
                  vit=backbones.VisionTransformer(
                      model_name='vit-b16',
                      representation_size=64,
                      init_stochastic_depth_rate=0.1,
                      original_init=False,
                      transformer=backbones.Transformer(
                          dropout_rate=0.0, attention_dropout_rate=0.0)))),
          losses=image_classification.Losses(
              l2_weight_decay=0.0,
              label_smoothing=0.1,
              one_hot=False,
              soft_labels=True),
          train_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              aug_type=common.Augmentation(
                  type='randaug',
                  randaug=common.RandAugment(
                      magnitude=9, exclude_ops=['Cutout'])),
              mixup_and_cutmix=common.MixupAndCutmix(label_smoothing=0.1)),
          validation_data=image_classification.DataConfig(
              input_path=os.path.join(
                  image_classification.IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=300 * steps_per_epoch,
          validation_steps=image_classification.IMAGENET_VAL_EXAMPLES //
          eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
                  'adamw': {
                      'weight_decay_rate': 0.05,
                      'include_in_weight_decay': r'.*(kernel|weight):0$',
                      'gradient_clip_norm': 0.0
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.0005 * train_batch_size / 512,
                      'decay_steps': 300 * steps_per_epoch,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
