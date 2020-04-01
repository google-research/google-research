# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def get_config():
  return tf.contrib.training.HParams(**{
      'total_bs': 64,
      'eval_total_bs': 16,
      'dataset_name': 'imagenet32',
      'dataset_config': tf.contrib.training.HParams(),
      'model_name': 'SlicedChannelModel',
      'model_config': tf.contrib.training.HParams(**{
          'optim': tf.contrib.training.HParams(**{
              'max_lr': 1e-4,
              'warmup': 5000,
              'grad_clip_norm': 1.0,
              'ema': 0.99995,
              'optimizer': 'adam',
              'adam_beta1': 0.9,
              'adam_beta2': 0.999,
          }),
          'dropout': 0.04,
          'img_size': 32,
          'ardec': tf.contrib.training.HParams(**{
              'emb_dim': 1536,
              'hdim_factor': 1,
              'emb_init_scale': 5.0,
              'num_heads': 16,
              'num_exterior_layers': 8,
              'num_outer_layers': 8,
              'num_inner_layers': 8,
              'res_init_scale': 1e-10,
          }),
      })
  })
