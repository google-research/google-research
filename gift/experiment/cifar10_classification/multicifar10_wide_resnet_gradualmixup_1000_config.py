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

"""Config for Cifar10 classification with end2end training."""

import ml_collections


def get_config():
  """Returns the experiment configuration for WideResnet on Cifar10."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'wide_resnet_cifar10_cls_gradual_mixup'

  # Train mode
  config.train_mode = 'self_adaptive_gradual_mixup'
  config.pretrained = ml_collections.ConfigDict()
  config.pretrained.only_backbone_pretrained = False
  config.pretrained.checkpoint_path = ''

  # Task
  config.task_name = 'multi_env_identity_dm_cls'
  config.dataset_name = 'multi_cifar10'
  config.data_augmentations = ['random_flip']
  config.train_environments = ['cifar', 'translated']
  config.eval_environments = ['cifar', 'translated']
  config.labeled_environments = ['cifar']
  config.unlabeled_environments = ['translated']

  # Model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'
  config.model_name = 'wide_resnet'
  config.blocks_per_group = 4
  config.channel_multiplier = 10
  config.num_outputs = 10
  config.dropout_rate = 0.0
  config.output_dim = 10

  # Training
  config.optimizer = 'adam'
  config.opt_hparams = {'weight_decay': 0.001}
  config.l2_decay_factor = .0
  config.max_grad_norm = 5.0
  config.label_smoothing = None
  config.num_training_steps = 1000
  config.num_training_epochs = None
  config.eval_frequency = 100
  config.batch_size = 512
  config.eval_batch_size = 512
  config.rng_seed = 0

  # Learning rate
  config.steps_per_epoch = 50000 // config.batch_size
  config.total_steps = config.num_training_steps
  config.base_lr = 0.000002 * (config.batch_size / 256)
  config.lr_hparams = {
      'learning_rate_schedule': 'compound',
      'factors': 'constant * decay_every',
      'initial_learning_rate': config.base_lr,
      'steps_per_decay': 100,
      'decay_factor': 0.99,
  }

  # Pipeline params
  config.confidence_quantile_threshold = 0.3
  config.self_supervised_label_transformation = 'sharp'
  config.label_temp = 0.5
  config.self_training_iterations = 5
  config.reinitialize_optimizer_at_each_step = False
  config.restart_learning_rate = False
  config.pseudo_labels_train_mode = False
  config.stop_gradient_for_interpolations = True
  config.ground_truth_factor_params = {'mode': 'constant', 'initial_value': 0.0}
  config.inter_env_interpolation = False
  config.intra_env_interpolation = False
  config.unlabeled_interpolation = True
  config.mixup_layer_set = [0, 1, 2, 3]
  config.interpolation_method = 'plain_convex_combination'
  config.intra_interpolation_method = 'plain_convex_combination'

  config.interpolation_mode = 'hard'
  config.ot_label_cost = 0.1
  config.ot_l2_cost = 0.0000000
  config.ot_noise_cost = 0.0

  config.intra_mixup_factor_params = {'mode': 'constant', 'initial_value': 0.0}
  config.beta_schedule_params = {'mode': 'constant', 'initial_value': 1.0}
  config.alpha_schedule_param = {'mode': 'constant', 'initial_value': 1.0}

  config.inter_mixup_factor_params = {'mode': 'constant', 'initial_value': 0.0}
  config.inter_beta_schedule_params = {'mode': 'constant', 'initial_value': 1.0}
  config.inter_alpha_schedule_param = {'mode': 'constant', 'initial_value': 1.0}

  config.unlabeled_mixup_factor_params = {
      'mode': 'constant',
      'initial_value': 1.0
  }
  config.unlabeled_beta_params = {
      'mode': 'linear_decay',
      'initial_value': 10,
      'min_value': 0,
      'total_steps': config.total_steps,
      'num_steps': config.self_training_iterations
  }
  config.unlabeled_alpha_params = {
      'mode': 'linear_grow',
      'initial_value': 1,
      'max_value': 10,
      'total_steps': config.total_steps,
      'num_steps': config.self_training_iterations
  }

  # IRM related
  config.penalty_weight = 0.0
  config.penalty_anneal_iters = 0

  # Continual learning related:
  config.gift_factor = 0.001

  # Domain Mapper related:
  config.aux_weight = 0
  config.aux_l2 = 0

  # logging
  config.write_summary = True  # write TB and XM summary
  config.checkpoint = True  # do checkpointing
  config.keep_ckpts = 3
  config.keep_env_ckpts = False
  config.write_xm_measurements = True
  config.trial = 0

  return config
