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

"""Main training script for Cascaded Nets."""
import collections
import os
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags
import numpy as np
import torch
from torch import optim
from cascaded_networks.datasets.dataset_handler import DataHandler
from cascaded_networks.models import densenet
from cascaded_networks.models import resnet
from cascaded_networks.modules import eval_handler
from cascaded_networks.modules import losses
from cascaded_networks.modules import train_handler
from cascaded_networks.modules import utils

# Setup Flags
FLAGS = flags.FLAGS
flags.DEFINE_string('gcs_path', None, 'gcs_path dir')
flags.DEFINE_bool('hyper_param_sweep', None, 'conducting hyperparam sweep')
flags.DEFINE_integer('n_gpus', None, 'Number of GPUs')

config_flags.DEFINE_config_file(
    name='config',
    default=None,
    help_string='Path to the Training configuration.')


def main(_):
  config = FLAGS.config

  if config.debug:
    config.epochs = 5

  # Make reproducible
  utils.make_reproducible(config.random_seed)

  # Parse GCS bucket path
  gcs_subpath = config.local_output_dir

  # Setup output directory
  out_basename = f'td({config.lambda_val})' if config.cascaded else 'std'
  out_basename += f',seed_{config.random_seed}'
  if FLAGS.hyper_param_sweep:
    out_basename += f',bs={config.batch_size}'
    out_basename += f',lr={config.learning_rate}'
    out_basename += f',wd={config.weight_decay}'

  save_root = os.path.join(gcs_subpath, config.experiment_name, out_basename)
  logging.info('Saving experiment to %s', save_root)

  # Flag check
  if config.tdl_mode == 'EWS':
    assert config.tdl_alpha is not None, 'tdl_alpha not set'
  elif config.tdl_mode == 'noise':
    assert config.noise_var is not None, 'noise_var not set'
  utils.save_flags(FLAGS, save_root, config)

  # Device
  device = torch.device('cuda'
                        if torch.cuda.is_available() and config.use_gpu
                        else 'cpu')

  # Set dataset root
  dataset_root = '/tmp/dataset'
  if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)

  # Data Handler
  data_dict = {
      'dataset_name': config.dataset_name,
      'data_root': dataset_root,
      'val_split': config.val_split,
      'split_idxs_root': 'split_idxs',
      'noise_type': config.augmentation_noise_type,
      'load_previous_splits': True,
  }
  data_handler = DataHandler(**data_dict)

  # Model
  model_dict = {
      'seed': config.random_seed,
      'num_classes': data_handler.num_classes,
      'pretrained': False,
      'cascaded': config.cascaded,
      'lambda_val': config.lambda_val,
      'tdl_alpha': config.tdl_alpha,
      'tdl_mode': config.tdl_mode,
      'noise_var': config.noise_var,
      'bn_opts': {
          'temporal_affine': config.bn_time_affine,
          'temporal_stats': config.bn_time_stats,
      },
      'imagenet': config.dataset_name == 'ImageNet2012',
  }

  # Model init op
  if config.model_key.startswith('resnet'):
    model_init_op = resnet
  elif config.model_key.startswith('densenet'):
    model_init_op = densenet

  # Initialize net
  net = model_init_op.__dict__[config.model_key](**model_dict).to(device)

  # Save model config
  model_dict['model_key'] = config.model_key
  utils.save_model_config(model_dict, save_root, config)

  # Optimizer
  optimizer = optim.SGD(net.parameters(),
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        nesterov=config.nesterov)

  # Scheduler
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=config.lr_milestones,
                                                gamma=config.lr_schedule_gamma)

  # Criterion
  criterion = losses.categorical_cross_entropy

  # Set Loaders
  train_loader = data_handler.build_loader('train', config)
  val_loader = data_handler.build_loader('val', config)
  test_loader = data_handler.build_loader('test', config)

  # train and eval functions
  train_fxn = train_handler.get_train_loop(net.timesteps,
                                           data_handler.num_classes,
                                           config)
  eval_fxn = eval_handler.get_eval_loop(net.timesteps,
                                        data_handler.num_classes,
                                        config)

  # Metrics container
  metrics = {
      'train': collections.defaultdict(list),
      'val': collections.defaultdict(list),
      'test': collections.defaultdict(float),
  }

  for epoch_i in range(config.epochs):
    # Train net
    train_loss, train_acc = train_fxn(net, train_loader, criterion,
                                      optimizer, device)

    # Log train metrics
    metrics['train']['loss'].append((epoch_i, train_loss))
    metrics['train']['acc'].append((epoch_i, train_acc))

    # Update lr scheduler
    lr_scheduler.step()

    if epoch_i % config.eval_freq == 0:
      # Evaluate net
      val_loss, val_acc = eval_fxn(net, val_loader, criterion, device)

      # Log eval metrics
      metrics['val']['loss'].append((epoch_i, val_loss))
      metrics['val']['acc'].append((epoch_i, val_acc))

    if config.cascaded:
      train_loss_val = np.mean(train_loss, axis=0)[-1]
      train_acc_val = np.mean(train_acc, axis=0)[-1] * 100
    else:
      train_loss_val = np.mean(train_loss, axis=0)
      train_acc_val = np.mean(train_acc, axis=0) * 100

    logging.info('Epoch %d/%d -- Acc: %0.2f -- Loss: %0.6f',
                 epoch_i+1, config.epochs, train_acc_val, train_loss_val)

    if epoch_i % config.upload_freq == 0:
      utils.save_model(net, optimizer, save_root, epoch_i, config)
      utils.save_metrics(metrics, save_root, config)

  # Evaluate test set
  test_loss, test_acc = eval_fxn(net, test_loader, criterion, device)
  metrics['test']['loss'] = test_loss
  metrics['test']['acc'] = test_acc

  # Save model and metrics
  utils.save_model(net, optimizer, save_root, epoch_i, config)
  utils.save_metrics(metrics, save_root, config)

if __name__ == '__main__':
  app.run(main)
