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

"""Evaluates the accuracy of a source trained model."""

import argparse
import logging
import os
from typing import Any, Dict

from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import general_util
from active_selective_prediction.utils import model_util
from active_selective_prediction.utils import tf_util
import numpy as np
import tensorflow as tf


def load_pretrained_model(
    model_path,
    source_train_ds,
    model_arch_name,
    model_arch_kwargs,
):
  """Loads a pretrained model."""
  init_inputs, _ = next(iter(source_train_ds))
  if isinstance(init_inputs, dict):
    input_shape = tuple(init_inputs['input_ids'].shape[1:])
  else:
    input_shape = tuple(init_inputs.shape[1:])
  if model_arch_name == 'simple_convnet':
    model = model_util.get_simple_convnet(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
    )
  elif model_arch_name == 'cifar_resnet':
    model = model_util.get_cifar_resnet(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
    )
  elif model_arch_name == 'simple_mlp':
    model = model_util.get_simple_mlp(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
    )
  elif model_arch_name == 'densenet121':
    model = model_util.get_densenet121(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
        weights=model_arch_kwargs['backbone_weights'],
    )
  elif model_arch_name == 'resnet50':
    model = model_util.get_resnet50(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
        weights=model_arch_kwargs['backbone_weights'],
    )
  elif model_arch_name == 'distilbert':
    model = model_util.get_distilbert(
        num_classes=model_arch_kwargs['num_classes'],
    )
  elif model_arch_name == 'roberta_mlp':
    model = model_util.get_roberta_mlp(
        input_shape=input_shape,
        num_classes=model_arch_kwargs['num_classes'],
    )
  else:
    raise ValueError(
        f'Not supported model architecture: {model_arch_name}'
    )
  # Makes an initial forward pass to create model Variables.
  model(init_inputs, training=False)
  model.load_weights(os.path.join(model_path, 'checkpoint')).expect_partial()
  return model


def eval_model(model, test_ds):
  """Evaluates the accuracy of the model."""
  preds = []
  labels = []
  for batch_x, batch_y in test_ds:
    batch_pred = tf_util.get_model_prediction(model, batch_x)
    preds.extend(batch_pred.numpy())
    labels.extend(batch_y.numpy())
  preds = np.array(preds)
  labels = np.array(labels)
  return np.mean(preds == labels)


def main():
  parser = argparse.ArgumentParser(
      description='pipeline for detecting dataset shift.'
  )
  parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
  parser.add_argument(
      '--seed', default=100, type=int, help='set a fixed random seed.'
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
      '--model-path',
      required=True,
      type=str,
      help='the path to the pre-trained model.',
  )
  args = parser.parse_args()
  handlers = [logging.StreamHandler()]
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
  model_path = args.model_path
  if source_dataset == 'color_mnist':
    model_arch_name = 'simple_convnet'
    model_arch_kwargs = {
        'num_classes': 10,
    }
    batch_size = 128
    source_train_ds = data_util.get_color_mnist_dataset(
        split='train', batch_size=batch_size, shuffle=True, drop_remainder=False
    )
    source_val_ds = data_util.get_color_mnist_dataset(
        split='test', batch_size=batch_size, shuffle=False, drop_remainder=False
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
    source_val_ds = data_util.get_cifar10_dataset(
        split='test', batch_size=batch_size, shuffle=False, drop_remainder=False
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
    source_val_ds = data_util.get_domainnet_dataset(
        domain_name='real',
        split='test',
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False,
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
    source_val_ds = data_util.get_fmow_dataset(
        split='id_val',
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False,
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
    source_val_ds = data_util.get_amazon_review_dataset(
        split='id_val',
        batch_size=batch_size,
        shuffle=False,
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
    source_val_ds = data_util.get_otto_dataset(
        split='val',
        batch_size=batch_size,
        shuffle=False,
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

  model = load_pretrained_model(
      model_path=model_path,
      source_train_ds=source_train_ds,
      model_arch_name=model_arch_name,
      model_arch_kwargs=model_arch_kwargs,
  )
  source_acc = eval_model(model, source_val_ds)
  print(f'Source accuracy on {source_dataset} is: {source_acc:.2%}')
  for name in target_datasets:
    target_test_ds = target_datasets[name]
    target_acc = eval_model(model, target_test_ds)
    print(f'Target accuracy on {name} is: {target_acc:.2%}')


if __name__ == '__main__':
  main()
