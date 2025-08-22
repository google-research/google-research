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

"""Builds custom datasets."""

import argparse
import os

import active_selective_prediction.tfds_generators.tfds_amazon_review  # pylint: disable=unused-import
import active_selective_prediction.tfds_generators.tfds_cinic10  # pylint: disable=unused-import
import active_selective_prediction.tfds_generators.tfds_fmow  # pylint: disable=unused-import
import active_selective_prediction.tfds_generators.tfds_otto  # pylint: disable=unused-import
import tensorflow as tf
import tensorflow_datasets as tfds


def build_amazon_review_dataset(
    data_dir,
    raw_dataset_dir,
):
  """Builds Amazon Review dataset.

  Args:
    data_dir: data directory to save the dataset.
    raw_dataset_dir: raw dataset directory.
  """
  builder = tfds.builder(
      name='amazon_review',
      data_dir=data_dir,
  )
  download_config = tfds.download.DownloadConfig(
      manual_dir=raw_dataset_dir+'/wilds_data/amazon_v2.1/'
  )

  builder.download_and_prepare(
      download_dir=data_dir,
      download_config=download_config,
  )


def build_mnist_related_datasets(
    data_dir,
):
  """Builds MNIST, MNIST Corrupted, SVHN and MNIST-M datasets.

  Args:
    data_dir: data directory to save the dataset.
  """
  builder = tfds.builder(name='mnist', data_dir=data_dir)
  builder.download_and_prepare(download_dir=data_dir)

  builder = tfds.builder(name='svhn_cropped', data_dir=data_dir)
  builder.download_and_prepare(download_dir=data_dir)


def build_cifar10_related_datasets(
    data_dir,
    raw_dataset_dir,
):
  """Builds CIFAR-10 and CINIC-10 datasets.

  Args:
    data_dir: data directory to save the dataset.
    raw_dataset_dir: raw dataset directory.
  """
  builder = tfds.builder(name='cifar10', data_dir=data_dir)
  builder.download_and_prepare(download_dir=data_dir)
  builder = tfds.builder(name='cinic10', data_dir=data_dir)
  download_config = tfds.download.DownloadConfig(
      manual_dir=raw_dataset_dir + '/cinic10'
  )
  builder.download_and_prepare(
      download_dir=data_dir, download_config=download_config
  )


def build_domainnet_dataset(
    data_dir,
):
  """Builds DomainNet dataset.

  Args:
    data_dir: data directory to save the dataset.
  """
  domains = ['real', 'painting', 'clipart', 'quickdraw', 'infograph', 'sketch']
  for domain in domains:
    builder = tfds.builder(name=f'domainnet/{domain}', data_dir=data_dir)
    builder.download_and_prepare(
        download_dir=data_dir
    )


def build_fmow_dataset(
    data_dir,
    raw_dataset_dir,
):
  """Builds FMoW dataset.

  Args:
    data_dir: data directory to save the dataset.
    raw_dataset_dir: raw dataset directory.
  """
  builder = tfds.builder(name='fmow', data_dir=data_dir)
  download_config = tfds.download.DownloadConfig(
      manual_dir=raw_dataset_dir+'/wilds_data/fmow_v1.1'
  )
  builder.download_and_prepare(
      download_dir=data_dir, download_config=download_config
  )


def build_otto_dataset(
    data_dir,
    raw_dataset_dir,
):
  """Builds Otto dataset.

  Args:
    data_dir: data directory to save the dataset.
    raw_dataset_dir: raw dataset directory.
  """
  builder = tfds.builder(
      name='otto',
      data_dir=data_dir,
  )
  download_config = tfds.download.DownloadConfig(
      manual_dir=raw_dataset_dir+'/otto-group-product-classification/'
  )
  builder.download_and_prepare(
      download_dir=data_dir, download_config=download_config
  )


def main():
  parser = argparse.ArgumentParser(
      description='Builds custom tensorflow datasets.'
  )
  parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
  parser.add_argument(
      '--dataset',
      default='mnist',
      choices=[
          'mnist',
          'cifar10',
          'domainnet',
          'fmow',
          'amazon_review',
          'otto',
      ],
      type=str,
      help='which dataset to build.',
  )
  parser.add_argument(
      '--data-dir',
      default='./tensorflow_datasets/',
      type=str,
      help='data directory to save the dataset.',
  )
  parser.add_argument(
      '--raw-dataset-dir',
      default='./raw_datasets/',
      type=str,
      help='raw dataset directory.',
  )
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  if args.dataset == 'amazon_review':
    build_amazon_review_dataset(args.data_dir, args.raw_dataset_dir)
  elif args.dataset == 'mnist':
    build_mnist_related_datasets(args.data_dir)
  elif args.dataset == 'cifar10':
    build_cifar10_related_datasets(args.data_dir, args.raw_dataset_dir)
  elif args.dataset == 'domainnet':
    build_domainnet_dataset(args.data_dir)
  elif args.dataset == 'fmow':
    build_fmow_dataset(args.data_dir, args.raw_dataset_dir)
  elif args.dataset == 'otto':
    build_otto_dataset(args.data_dir, args.raw_dataset_dir)
  else:
    raise ValueError(
        f'The dataset {args.dataset} is not implemented in this repo!'
    )


if __name__ == '__main__':
  main()
