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

"""Evaluation pipeline for active selective prediction."""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict

from active_selective_prediction import methods
from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import general_util
import numpy as np
import tensorflow as tf


def eval_selective_prediction(
    source_dataset,
    target_dataset,
    method,
    method_config,
    model_arch_name,
    model_arch_kwargs,
    source_train_ds,
    target_test_ds,
    logger,
):
  """Evaluates specified error detection method."""
  t0 = time.time()
  if 'finetune_method' not in method_config:
    method_config['finetune_method'] = 'joint_train'
  if method == 'sr':
    sp = methods.SR(
        model_arch_name=model_arch_name,
        model_arch_kwargs=model_arch_kwargs,
        model_path=method_config['model_path'],
        source_train_ds=source_train_ds,
        label_budget=method_config['label_budget'],
        batch_size=method_config['batch_size'],
        sampling_rounds=method_config['sampling_rounds'],
        max_epochs=method_config['max_epochs'],
        patience_epochs=method_config['patience_epochs'],
        min_epochs=method_config['min_epochs'],
        optimizer_name=method_config['optimizer_name'],
        optimizer_kargs=method_config['optimizer_kargs'],
        sampling_method=method_config['sampling_method'],
        sampling_kwargs=method_config['sampling_kwargs'],
        finetune_method=method_config['finetune_method'],
        finetune_kwargs=method_config['finetune_kwargs'],
        debug_info=method_config['debug_info'],
        print_freq=method_config['print_freq'],
    )
  elif method == 'de':
    sp = methods.DE(
        model_arch_name=model_arch_name,
        model_arch_kwargs=model_arch_kwargs,
        source_train_ds=source_train_ds,
        model_path=method_config['model_path'],
        num_models=method_config['num_models'],
        label_budget=method_config['label_budget'],
        batch_size=method_config['batch_size'],
        sampling_rounds=method_config['sampling_rounds'],
        max_epochs=method_config['max_epochs'],
        patience_epochs=method_config['patience_epochs'],
        min_epochs=method_config['min_epochs'],
        optimizer_name=method_config['optimizer_name'],
        optimizer_kargs=method_config['optimizer_kargs'],
        sampling_method=method_config['sampling_method'],
        sampling_kwargs=method_config['sampling_kwargs'],
        finetune_method=method_config['finetune_method'],
        finetune_kwargs=method_config['finetune_kwargs'],
        debug_info=method_config['debug_info'],
        print_freq=method_config['print_freq'],
    )
  elif method == 'aspest':
    sp = methods.ASPEST(
        model_arch_name=model_arch_name,
        model_arch_kwargs=model_arch_kwargs,
        source_train_ds=source_train_ds,
        label_budget=method_config['label_budget'],
        batch_size=method_config['batch_size'],
        sampling_rounds=method_config['sampling_rounds'],
        max_epochs=method_config['max_epochs'],
        patience_epochs=method_config['patience_epochs'],
        min_epochs=method_config['min_epochs'],
        optimizer_name=method_config['optimizer_name'],
        optimizer_kargs=method_config['optimizer_kargs'],
        model_path=method_config['model_path'],
        num_models=method_config['num_models'],
        self_train_kwargs=method_config[
            'self_train_kwargs'
        ],
        finetune_method=method_config['finetune_method'],
        finetune_kwargs=method_config['finetune_kwargs'],
        debug_info=method_config['debug_info'],
        print_freq=method_config['print_freq'],
    )
  else:
    raise ValueError(f'Not supported error detection method {method}!')
  results = sp.get_results(target_test_ds)
  logger.info(
      'Evaluate %s -> %s, %s, time used:  %.2fs',
      source_dataset,
      target_dataset,
      method,
      time.time() - t0,
  )
  return results


def print_metrics(name, metrics, logger):
  """Prints metrics."""
  logger.info('Dataset %s', name)
  logger.info(
      'Adapted Model Test Accuracy: %.2f%%', 100 * metrics['adapted_model_acc']
  )
  if 'surrogate_model_acc' in metrics:
    logger.info(
        'Surrogate Model Accuracy: %.2f%%', 100 * metrics['surrogate_model_acc']
    )
  cov_acc_95 = metrics['coverage_set'][metrics['accuracy_set'] >= 0.95][-1]
  acc_cov_95 = metrics['accuracy_set'][metrics['coverage_set'] >= 0.95][0]
  auacc = np.trapz(y=metrics['accuracy_set'], x=metrics['coverage_set'])
  logger.info('Coverage at Accuracy >= 95%%: %.2f%%', cov_acc_95*100)
  logger.info('Accuracy at Coverage >= 95%%: %.2f%%', acc_cov_95*100)
  logger.info('AUACC %.4f\n', auacc)


def main():
  parser = argparse.ArgumentParser(
      description='pipeline for detecting dataset shift.'
  )
  parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
  parser.add_argument(
      '--seed', default=100, type=int, help='set a fixed random seed.'
  )
  parser.add_argument(
      '--repeat-times',
      default=1,
      type=int,
      help='repeat times to compute mean and variance.',
  )
  parser.add_argument(
      '--source-dataset',
      default='color_mnist',
      choices=[
          'color_mnist',
          'cifar10',
          'domainnet',
          'fmow',
          'amazon_review',
          'otto',
      ],
      type=str,
      help='specify source dataset.',
  )
  parser.add_argument(
      '--method',
      default='sr',
      choices=[
          'sr',
          'de',
          'aspest',
      ],
      type=str,
      help='specify the error detection method.',
  )
  parser.add_argument(
      '--method-config-file',
      default='./configs/sr.json',
      type=str,
      help='the path to the method config file.',
  )
  parser.add_argument(
      '--log-file',
      default='',
      type=str,
      help='the path to the file for logging.',
  )
  parser.add_argument(
      '--debug',
      action='store_true',
      help='Whether to set debug mode.'
  )
  args = parser.parse_args()
  handlers = [logging.StreamHandler()]
  if args.log_file:
    handlers.append(logging.FileHandler(args.log_file))
  logger = logging.getLogger(__name__)
  logging.basicConfig(
      format='%(message)s', level=logging.DEBUG, handlers=handlers
  )
  state = {k: v for k, v in args.__dict__.items()}
  logger.info(state)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  general_util.set_random_seed(args.seed)
  source_dataset = args.source_dataset
  method = args.method
  with open(args.method_config_file) as config_file:
    all_method_config = json.load(config_file)
    method_config = all_method_config[source_dataset]
  if args.debug:
    method_config['debug_info'] = True
  logger.info(method_config)
  if source_dataset == 'color_mnist':
    model_arch_name = 'simple_convnet'
    model_arch_kwargs = {
        'num_classes': 10,
    }
    batch_size = 128
    source_train_ds = data_util.get_color_mnist_dataset(
        split='train', batch_size=batch_size, shuffle=True, drop_remainder=False
    )
    target_datasets = {}
    for target_dataset in ['SVHN']:
      if target_dataset == 'SVHN':
        target_test_ds = data_util.get_svhn_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  elif source_dataset == 'cifar10':
    model_arch_name = 'cifar_resnet'
    model_arch_kwargs = {
        'num_classes': 10,
    }
    batch_size = 128
    source_train_ds = data_util.get_cifar10_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
    )
    target_datasets = {}
    for target_dataset in ['CINIC-10']:
      if target_dataset == 'CINIC-10':
        target_test_ds = data_util.get_cinic10_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
            max_size=30000,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  elif source_dataset == 'domainnet':
    model_arch_name = 'resnet50'
    model_arch_kwargs = {
        'num_classes': 345,
        'backbone_weights': 'imagenet',
    }
    batch_size = 128
    source_train_ds = data_util.get_domainnet_dataset(
        domain_name='real',
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
    )
    target_datasets = {}
    for target_dataset in [
        'DomainNet-painting',
        'DomainNet-clipart',
        'DomainNet-infograph',
        'DomainNet-sketch',
    ]:
      if target_dataset == 'DomainNet-painting':
        target_test_ds = data_util.get_domainnet_dataset(
            domain_name='painting',
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      elif target_dataset == 'DomainNet-clipart':
        target_test_ds = data_util.get_domainnet_dataset(
            domain_name='clipart',
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      elif target_dataset == 'DomainNet-infograph':
        target_test_ds = data_util.get_domainnet_dataset(
            domain_name='infograph',
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      elif target_dataset == 'DomainNet-sketch':
        target_test_ds = data_util.get_domainnet_dataset(
            domain_name='sketch',
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  elif source_dataset == 'fmow':
    model_arch_name = 'densenet121'
    model_arch_kwargs = {
        'num_classes': 62,
        'backbone_weights': 'imagenet',
    }
    batch_size = 128
    source_train_ds = data_util.get_fmow_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
        include_meta=False,
    )
    target_datasets = {}
    for target_dataset in ['FMoW-OOD']:
      if target_dataset == 'FMoW-OOD':
        target_test_ds = data_util.get_fmow_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
            include_meta=False,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  elif source_dataset == 'amazon_review':
    model_arch_name = 'roberta_mlp'
    model_arch_kwargs = {
        'num_classes': 5,
    }
    batch_size = 128
    source_train_ds = data_util.get_amazon_review_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=False,
        include_meta=False,
    )
    target_datasets = {}
    for target_dataset in ['Amazon-Review-OOD-subset-1']:
      if target_dataset == 'Amazon-Review-OOD':
        target_test_ds = data_util.get_amazon_review_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
            include_meta=False,
        )
      elif 'subset' in target_dataset:
        subset_index = int(target_dataset.split('-')[-1])
        target_test_ds = data_util.get_amazon_review_test_sub_dataset(
            subset_index=subset_index,
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  elif source_dataset == 'otto':
    model_arch_name = 'simple_mlp'
    model_arch_kwargs = {
        'num_classes': 9,
    }
    batch_size = 128
    source_train_ds = data_util.get_otto_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        drop_remainder=False,
    )
    target_datasets = {}
    for target_dataset in ['otto-test']:
      if target_dataset == 'otto-test':
        target_test_ds = data_util.get_otto_dataset(
            split='test',
            batch_size=batch_size,
            shuffle=False,
            drop_remainder=False,
        )
      else:
        raise ValueError(f'Unsupported target dataset {target_dataset}!')
      target_datasets[target_dataset] = target_test_ds
  else:
    raise ValueError(f'Unsupported source dataset {source_dataset}!')

  for name in target_datasets:
    target_test_ds = target_datasets[name]
    for k in range(args.repeat_times):
      general_util.set_random_seed(args.seed * (k + 1))
      metrics = eval_selective_prediction(
          source_dataset,
          name,
          method,
          method_config,
          model_arch_name,
          model_arch_kwargs,
          source_train_ds,
          target_test_ds,
          logger,
      )
      print_metrics(name, metrics, logger)


if __name__ == '__main__':
  main()
