# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Dataset loading and file I/O utilities for data attribution pipelines."""

from collections.abc import Sequence
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
import tqdm

from reversible_data_attribution import utils

if 'gfile' not in globals():
  gfile = None

FLAGS = flags.FLAGS

if not flags.FLAGS.is_parsed():
  flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')
  flags.DEFINE_integer('num_epochs', 5, 'Number of training epochs.')
  flags.DEFINE_integer('batch_size', 10, 'Batch size for training.')
  flags.DEFINE_string('device', 'cpu', 'Device to run computation on.')
  flags.DEFINE_enum(
      'dataset',
      'mnist',
      [
          'mnist',
          'cifar',
          'cifar10',
          'cifar100',
          'fashion_mnist',
          'imdb',
          'mnist_corrupted',
          'synthetic',
      ],
      'Dataset to use: "mnist", "cifar" (or "cifar10"/"cifar100"),'
      ' "fashion_mnist", "imdb", "mnist_corrupted", or "synthetic".',
  )
  flags.DEFINE_string(
      'data_path',
      None,
      'Path to local dataset npz file. If None, uses default path for chosen'
      ' dataset.',
  )
  flags.DEFINE_integer(
      'num_train', 200, 'Number of MNIST training samples to load.'
  )
  flags.DEFINE_integer('num_test', 50, 'Number of MNIST test samples to load.')
  flags.DEFINE_string(
      'model_type',
      'small_dnn',
      'Model architecture: "linear", "small_dnn", "dnn", "cnn", "vit", or'
      ' "resnet".',
  )
  flags.DEFINE_list(
      'inds_to_remove',
      ['0'],
      'List of indices to remove for counterfactual training.',
  )
  """To prevent our cloudtop from going out of memory, we could also decide NOT to save the artifacts."""
  flags.DEFINE_string(
      'workdir',
      None,
      'Directory to save the training artifacts and CSV.',
  )
  flags.DEFINE_boolean(
      'save_checkpoints',
      True,
      'Whether to save the checkpoints.',
  )
  flags.DEFINE_enum(
      'noise_type',
      'none',
      [
          'none',
          'label_flip',
          'gaussian',
          'both',
          'lowpass',
          'highpass',
          'low_pass',
          'high_pass',
      ],
      'Type of noise to apply to the training dataset.',
  )
  flags.DEFINE_float(
      'noise_rate',
      0.1,
      'Fraction of training dataset to corrupt with noise (between 0.0 and'
      ' 1.0).',
  )
  flags.DEFINE_float(
      'gaussian_std',
      0.1,
      'Standard deviation of Gaussian noise added to inputs.',
  )
  flags.DEFINE_float(
      'cutoff_freq',
      0.5,
      'Cutoff frequency for lowpass / highpass filtering.',
  )
  flags.DEFINE_integer(
      'filter_order',
      2,
      'Filter order for lowpass / highpass filtering.',
  )
  flags.DEFINE_float(
      'sampling_rate',
      60.0,
      'Sampling rate for lowpass / highpass filtering.',
  )
  flags.DEFINE_integer(
      'noise_seed',
      42,
      'Random seed for noise generation.',
  )
  flags.DEFINE_float(
      'random_masking',
      None,
      'Proportion of gradient coordinates to keep track via Bernoulli(p) random'
      ' masking for forward Adam update.',
  )
  flags.DEFINE_bool(
      'use_reversible_transform',
      False,
      'Whether to use the reversible transform for backward Adam update.',
  )


def _make_dirs(dir_path):
  """Creates directory path supporting local, CNS, and GCS paths."""
  if not dir_path:
    return
  if gfile is not None:
    try:
      if hasattr(gfile, 'MakeDirs'):
        gfile.MakeDirs(dir_path)
        return
      elif hasattr(gfile, 'makedirs'):
        gfile.makedirs(dir_path)
        return
    except Exception:
      pass
  os.makedirs(dir_path, exist_ok=True)


def _glob(pattern):
  """Globs files supporting local, CNS, and GCS paths."""
  if gfile is not None:
    try:
      if hasattr(gfile, 'Glob'):
        return list(gfile.Glob(pattern))
      elif hasattr(gfile, 'glob'):
        return list(gfile.glob(pattern))
    except Exception:
      pass
  import glob

  return glob.glob(pattern)


def _list_dir(dir_path):
  """Lists directory contents supporting local, CNS, and GCS paths."""
  if gfile is not None:
    try:
      if hasattr(gfile, 'ListDir'):
        return list(gfile.ListDir(dir_path))
      elif hasattr(gfile, 'listdir'):
        return list(gfile.listdir(dir_path))
    except Exception:
      pass
  return os.listdir(dir_path)


def _is_dir(dir_path):
  """Checks if path is a directory supporting local, CNS, and GCS paths."""
  if gfile is not None:
    try:
      if hasattr(gfile, 'IsDirectory'):
        return bool(gfile.IsDirectory(dir_path))
      elif hasattr(gfile, 'isdir'):
        return bool(gfile.isdir(dir_path))
    except Exception:
      pass
  return os.path.isdir(dir_path)


def _exists(file_path):
  """Checks if path exists supporting local, CNS, and GCS paths."""
  if gfile is not None:
    try:
      if hasattr(gfile, 'Exists'):
        return bool(gfile.Exists(file_path))
      elif hasattr(gfile, 'exists'):
        return bool(gfile.exists(file_path))
    except Exception:
      pass
  return os.path.exists(file_path)


def _open_file(file_path, mode = 'r'):
  """Opens file supporting local, CNS, and GCS paths."""
  if gfile is not None:
    try:
      if hasattr(gfile, 'GFile'):
        return gfile.GFile(file_path, mode)
      elif hasattr(gfile, 'Open'):
        return gfile.Open(file_path, mode)
    except Exception:
      pass

  return open(file_path, mode)


def load_mnist_data(
    filepath = None,
    num_train = 200,
    num_test = 50,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
):
  """Loads and preprocesses a subset of the MNIST dataset from an npz file."""
  if not filepath:
    filepath = 'third_party/google_research/google_research/reversible_data_attribution/mnist/mnist_local.npz'

  candidates = [
      filepath,
      os.path.join(os.getcwd(), filepath),
      os.path.join(os.path.dirname(__file__), 'mnist', 'mnist_local.npz'),
  ]

  resolved_path = None
  for p in candidates:
    if os.path.exists(p):
      resolved_path = p
      break

  if resolved_path is None:
    raise FileNotFoundError(
        f'Could not locate MNIST dataset at {filepath}. Tried: {candidates}'
    )

  logging.info('Loading MNIST dataset from %s', resolved_path)
  with np.load(resolved_path) as data:
    x_train = data['x_train'][:num_train].astype(np.float32) / 255.0
    y_train = data['y_train'][:num_train].astype(np.int64)
    x_test = data['x_test'][:num_test].astype(np.float32) / 255.0
    y_test = data['y_test'][:num_test].astype(np.int64)

  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)

  if noise_type.lower() != 'none' and noise_rate > 0.0:
    logging.info(
        'Applying %s noise to MNIST train set (rate=%.2f)...',
        noise_type,
        noise_rate,
    )
    x_train, y_train, _ = utils.apply_dataset_noise(
        x_train=x_train,
        y_train=y_train,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=10,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  return x_train, y_train, x_test, y_test


def load_mnist_corrupted_data(
    filepath,
    num_train = 200,
    num_test = 50,
):
  """Loads and preprocesses a subset of the corrupted MNIST dataset from an npz file."""
  return load_mnist_data(filepath, num_train=num_train, num_test=num_test)


def load_fashion_mnist_data(
    filepath = None,
    num_train = 200,
    num_test = 50,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
):
  """Loads and preprocesses a subset of the Fashion-MNIST dataset from an npz file."""
  if not filepath:
    filepath = 'third_party/google_research/google_research/reversible_data_attribution/fashion_mnist/fashion_mnist.npz'

  candidates = [
      filepath,
      os.path.join(os.getcwd(), filepath),
      os.path.join(
          os.path.dirname(__file__), 'fashion_mnist', 'fashion_mnist.npz'
      ),
      os.path.join(
          os.path.dirname(__file__), 'fashion_mnist', 'fashion_mnist_local.npz'
      ),
  ]

  resolved_path = None
  for p in candidates:
    if os.path.exists(p):
      resolved_path = p
      break

  if resolved_path is None:
    raise FileNotFoundError(
        f'Could not locate Fashion-MNIST dataset at {filepath}. Tried:'
        f' {candidates}'
    )

  logging.info('Loading Fashion-MNIST dataset from %s', resolved_path)
  with np.load(resolved_path) as data:
    x_train = data['x_train'][:num_train].astype(np.float32) / 255.0
    y_train = data['y_train'][:num_train].astype(np.int64)
    x_test = data['x_test'][:num_test].astype(np.float32) / 255.0
    y_test = data['y_test'][:num_test].astype(np.int64)

  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)

  if noise_type.lower() != 'none' and noise_rate > 0.0:
    logging.info(
        'Applying %s noise to Fashion-MNIST train set (rate=%.2f)...',
        noise_type,
        noise_rate,
    )
    x_train, y_train, _ = utils.apply_dataset_noise(
        x_train=x_train,
        y_train=y_train,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=10,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  return x_train, y_train, x_test, y_test


def _vectorize_sequences(
    sequences, dimension = 1000
):
  """Converts variable-length sequence of word indices to multi-hot binary vectors."""
  results = np.zeros((len(sequences), dimension), dtype=np.float32)
  for i, sequence in enumerate(sequences):
    if isinstance(sequence, (list, np.ndarray)):
      valid_indices = [int(idx) for idx in sequence if int(idx) < dimension]
      results[i, valid_indices] = 1.0
  return results


def load_imdb_data(
    filepath = None,
    num_train = 200,
    num_test = 50,
    num_words = 1000,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
):
  """Loads and vectorizes a subset of the IMDB dataset from an npz file."""
  if not filepath:
    filepath = 'third_party/google_research/google_research/reversible_data_attribution/imdb/imdb_local.npz'

  candidates = [
      filepath,
      os.path.join(os.getcwd(), filepath),
      os.path.join(os.path.dirname(__file__), 'imdb', 'imdb_local.npz'),
      os.path.join(os.path.dirname(__file__), 'imdb', 'imdb.npz'),
  ]

  resolved_path = None
  for p in candidates:
    if os.path.exists(p):
      resolved_path = p
      break

  if resolved_path is None:
    raise FileNotFoundError(
        f'Could not locate IMDB dataset at {filepath}. Tried: {candidates}'
    )

  logging.info('Loading IMDB dataset from %s', resolved_path)
  with np.load(resolved_path, allow_pickle=True) as data:
    x_train_raw = data['x_train'][:num_train]
    y_train = data['y_train'][:num_train].astype(np.int64)
    x_test_raw = data['x_test'][:num_test]
    y_test = data['y_test'][:num_test].astype(np.int64)

  x_train = _vectorize_sequences(x_train_raw, dimension=num_words)
  x_test = _vectorize_sequences(x_test_raw, dimension=num_words)

  if noise_type.lower() != 'none' and noise_rate > 0.0:
    logging.info(
        'Applying %s noise to IMDB train set (rate=%.2f)...',
        noise_type,
        noise_rate,
    )
    x_train, y_train, _ = utils.apply_dataset_noise(
        x_train=x_train,
        y_train=y_train,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=2,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  return x_train, y_train, x_test, y_test


def load_cifar_data(
    filepath,
    num_train = 200,
    num_test = 50,
    flatten = True,
    noise_type = 'none',
    noise_rate = 0.0,
    gaussian_std = 0.1,
    cutoff_freq = 0.5,
    filter_order = 2,
    sampling_rate = 60.0,
    seed = 42,
    num_classes = 10,
):
  """Loads and preprocesses a subset of the CIFAR dataset from an npz file."""
  candidates = [
      filepath,
      os.path.join(os.getcwd(), filepath),
      os.path.join(os.path.dirname(__file__), 'cifar', 'cifar_local.npz'),
  ]

  resolved_path = None
  for p in candidates:
    if os.path.exists(p):
      resolved_path = p
      break

  if resolved_path is None:
    raise FileNotFoundError(
        f'Could not locate CIFAR dataset at {filepath}. Tried: {candidates}'
    )

  logging.info('Loading CIFAR dataset from %s', resolved_path)
  with tqdm.tqdm(total=4, desc='Loading CIFAR data') as pbar:
    with np.load(resolved_path) as data:
      pbar.set_description('Reading NPZ archive')
      x_train_raw = data['x_train'][:num_train]
      y_train_raw = data['y_train'][:num_train]
      x_test_raw = data['x_test'][:num_test]
      y_test_raw = data['y_test'][:num_test]
      pbar.update(1)

      pbar.set_description('Normalizing train split')
      x_train = x_train_raw.astype(np.float32) / 255.0
      y_train = y_train_raw.astype(np.int64)
      pbar.update(1)

      pbar.set_description('Normalizing test split')
      x_test = x_test_raw.astype(np.float32) / 255.0
      y_test = y_test_raw.astype(np.int64)
      pbar.update(1)

    if flatten:
      pbar.set_description('Flattening features')
      x_train = x_train.reshape(x_train.shape[0], -1)
      x_test = x_test.reshape(x_test.shape[0], -1)
    pbar.update(1)

  if noise_type.lower() != 'none' and noise_rate > 0.0:
    logging.info(
        'Applying %s noise to CIFAR train set (rate=%.2f)...',
        noise_type,
        noise_rate,
    )
    x_train, y_train, _ = utils.apply_dataset_noise(
        x_train=x_train,
        y_train=y_train,
        noise_type=noise_type,
        noise_rate=noise_rate,
        num_classes=num_classes,
        gaussian_std=gaussian_std,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        sampling_rate=sampling_rate,
        seed=seed,
    )

  return x_train, y_train, x_test, y_test


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info(
      'pipeline.py direct execution is deprecated. Please run'
      ' run_data_cleansing_eval.py instead.'
  )


if __name__ == '__main__':
  app.run(main)
