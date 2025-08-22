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

"""Trains a model on a source dataset."""

import argparse
import os

from active_selective_prediction.utils import data_util
from active_selective_prediction.utils import general_util
from active_selective_prediction.utils import model_util
import tensorflow as tf


def main():
  parser = argparse.ArgumentParser(
      description='pipeline for detecting dataset shift'
  )
  parser.add_argument('--gpu', default='0', type=str, help='which gpu to use.')
  parser.add_argument(
      '--seed', default=100, type=int, help='set a fixed random seed.'
  )
  parser.add_argument(
      '--dataset',
      default='color_mnist',
      choices=[
          'cifar10',
          'domainnet',
          'color_mnist',
          'fmow',
          'amazon_review',
          'otto',
      ],
      type=str,
      help='which dataset to train a model',
  )
  parser.add_argument(
      '--save-dir',
      default='./checkpoints/standard_supervised/',
      type=str,
      help='the dir to save trained model',
  )
  args = parser.parse_args()
  state = {k: v for k, v in args.__dict__.items()}
  print(state)
  seed = args.seed
  dataset = args.dataset
  save_dir = args.save_dir
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  general_util.set_random_seed(seed)
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  if dataset == 'color_mnist':
    train_ds = data_util.get_color_mnist_dataset(
        split='train', batch_size=128, shuffle=True, drop_remainder=False
    )
    val_ds = data_util.get_color_mnist_dataset(
        split='test', batch_size=200, shuffle=False, drop_remainder=False
    )
    epochs = 20
    learning_rate = 1e-3
    num_classes = 10
    init_inputs, _ = next(iter(train_ds))
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_simple_convnet(
        input_shape=input_shape, num_classes=num_classes
    )
  elif dataset == 'cifar10':
    train_ds = data_util.get_cifar10_dataset(
        split='train',
        batch_size=128,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
    )
    val_ds = data_util.get_cifar10_dataset(
        split='test', batch_size=200, shuffle=False, drop_remainder=False
    )
    epochs = 200
    learning_rate = 1e-1
    num_classes = 10
    init_inputs, _ = next(iter(train_ds))
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_cifar_resnet(
        input_shape=input_shape, num_classes=num_classes
    )
  elif dataset == 'domainnet':
    train_ds = data_util.get_domainnet_dataset(
        domain_name='real',
        split='train',
        batch_size=128,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
    )
    val_ds = data_util.get_domainnet_dataset(
        domain_name='real',
        split='test',
        batch_size=128,
        shuffle=False,
        drop_remainder=False,
    )
    epochs = 50
    learning_rate = 1e-4
    num_classes = 345
    init_inputs, _ = next(iter(train_ds))
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_resnet50(
        input_shape=input_shape,
        num_classes=num_classes,
        weights='imagenet',
    )
  elif dataset == 'fmow':
    train_ds = data_util.get_fmow_dataset(
        split='train',
        batch_size=128,
        shuffle=True,
        drop_remainder=False,
        data_augment=True,
        include_meta=False,
    )
    val_ds = data_util.get_fmow_dataset(
        split='id_val',
        batch_size=128,
        shuffle=False,
        drop_remainder=False,
        include_meta=False,
    )
    epochs = 50
    learning_rate = 1e-4
    num_classes = 62
    init_inputs, _ = next(iter(train_ds))
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_densenet121(
        input_shape=input_shape,
        num_classes=num_classes,
        weights='imagenet',
    )
  elif dataset == 'amazon_review':
    train_ds = data_util.get_amazon_review_dataset(
        split='train',
        batch_size=128,
        shuffle=True,
        drop_remainder=False,
        include_meta=False,
    )
    val_ds = data_util.get_amazon_review_dataset(
        split='id_val',
        batch_size=128,
        shuffle=False,
        drop_remainder=False,
        include_meta=False,
    )
    epochs = 200
    learning_rate = 1e-3
    num_classes = 5
    train_ds_iter = iter(train_ds)
    init_inputs, _ = next(train_ds_iter)
    for _ in train_ds_iter:
      pass
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_roberta_mlp(
        input_shape=input_shape,
        num_classes=num_classes,
    )
  elif dataset == 'otto':
    train_ds = data_util.get_otto_dataset(
        split='train',
        batch_size=128,
        shuffle=True,
        drop_remainder=False,
    )
    val_ds = data_util.get_otto_dataset(
        split='val',
        batch_size=128,
        shuffle=False,
        drop_remainder=False,
    )
    epochs = 200
    learning_rate = 1e-3
    num_classes = 9
    train_ds_iter = iter(train_ds)
    init_inputs, _ = next(train_ds_iter)
    for _ in train_ds_iter:
      pass
    input_shape = tuple(init_inputs.shape[1:])
    model = model_util.get_simple_mlp(
        input_shape=input_shape,
        num_classes=num_classes,
    )
  else:
    raise ValueError(f'Unsupported dataset {dataset}!')
  # Builds model
  model(init_inputs)
  model.summary()
  if dataset == 'color_mnist':
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
  elif dataset == 'cifar10':
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    def scheduler_func(epoch, lr):
      if epoch == 80:
        lr *= 0.1
      elif epoch == 120:
        lr *= 0.1
      elif epoch == 160:
        lr *= 0.1
      elif epoch == 180:
        lr *= 0.5
      return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        scheduler_func
    )
    callbacks = [lr_scheduler]
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )
  elif dataset in ['domainnet', 'fmow', 'amazon_review', 'otto']:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(
        train_ds, epochs=epochs, validation_data=val_ds
    )
  model.save_weights(
      os.path.join(save_dir, f'{dataset}', 'checkpoint')
  )


if __name__ == '__main__':
  main()
