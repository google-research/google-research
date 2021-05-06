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
"""Data loaders."""

from typing import Generator, Tuple

import jax
import numpy as onp
from jax import numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from enum import Enum
import re
import os
import warnings
from enum import Enum

imdb = tf.keras.datasets.imdb
sequence = tf.keras.preprocessing.sequence


class Task(Enum):
  REGRESSION = "regression"
  CLASSIFICATION = "classification"


class ImgDatasets(Enum):
  CIFAR10 = "cifar10"
  CIFAR100 = "cifar100"
  MNIST = "mnist"


class UCIRegressionDatasets(Enum):
  BOSTON = "boston"
  ENERGY = "energy"
  YACHT = "yacht"
  CONCRETE = "concrete"
  NAVAL = "naval"
  ELEVATORS = "elevators"
  KEGGU = "keggu"
  KEGGD = "keggd"
  PROTEIN = "protein"
  POL = "pol"
  SKILLCRAFT = "skillcraft"


_UCI_REGRESSION_FILENAMES = {
    UCIRegressionDatasets.BOSTON: "boston.npz",
    UCIRegressionDatasets.ENERGY: "energy.npz",
    UCIRegressionDatasets.YACHT: "yacht.npz",
    UCIRegressionDatasets.CONCRETE: "concrete.npz",
    UCIRegressionDatasets.NAVAL: "naval.npz",
    UCIRegressionDatasets.ELEVATORS: "wilson_elevators.npz",
    UCIRegressionDatasets.KEGGU: "wilson_keggundirected.npz",
    UCIRegressionDatasets.KEGGD: "wilson_keggdirected.npz",
    UCIRegressionDatasets.PROTEIN: "wilson_protein.npz",
    UCIRegressionDatasets.POL: "wilson_pol.npz",
    UCIRegressionDatasets.SKILLCRAFT: "wilson_skillcraft.npz"
}

# Format: (img_mean, img_std)
_ALL_IMG_DS_STATS = {
    ImgDatasets.CIFAR10: ((0.49, 0.48, 0.44), (0.2, 0.2, 0.2)),
    ImgDatasets.CIFAR100: ((0.49, 0.48, 0.44), (0.2, 0.2, 0.2)),
    ImgDatasets.MNIST: ((0.1307,), (0.3081,))
}

_IMDB_CONFIG = {"max_features": 20000, "max_len": 100, "num_train": 20000}


def load_imdb_dataset():
  """Load the IMDB reviews dataset.

  Code adapted from the code for
  _How Good is the Bayes Posterior in Deep Neural Networks Really?_:
  https://github.com/google-research/google-research/blob/master/cold_posterior_bnn/imdb/imdb_data.py
  """
  (x_train, y_train), (x_test, y_test) = imdb.load_data(
      path="./datasets", num_words=_IMDB_CONFIG["max_features"])
  num_train = _IMDB_CONFIG["num_train"]
  x_train, x_val = x_train[:num_train], x_train[num_train:]
  y_train, y_val = y_train[:num_train], y_train[num_train:]

  def preprocess(x, y, max_length):
    x = sequence.pad_sequences(x, maxlen=max_length)
    y = onp.array(y)
    x = onp.array(x)
    return x, y

  max_length = _IMDB_CONFIG["max_len"]
  x_train, y_train = preprocess(x_train, y_train, max_length=max_length)
  x_val, y_val = preprocess(x_val, y_val, max_length=max_length)
  x_test, y_test = preprocess(x_test, y_test, max_length=max_length)
  data_info = {"num_classes": 2}
  return (x_train, y_train), (x_test, y_test), (x_val, y_val), data_info


def load_image_dataset(split,
                       batch_size,
                       name="cifar10",
                       repeat=False,
                       shuffle=False,
                       shuffle_seed=None):
  """Loads the dataset as a generator of batches."""
  # Do no data augmentation.
  ds, dataset_info = tfds.load(
      name, split=split, as_supervised=True, with_info=True)
  num_classes = dataset_info.features["label"].num_classes
  num_examples = dataset_info.splits[split].num_examples
  num_channels = dataset_info.features["image"].shape[-1]

  def img_to_float32(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label

  ds = ds.map(img_to_float32).cache()
  ds_stats = _ALL_IMG_DS_STATS[ImgDatasets(name)]

  def img_normalize(image, label):
    """Normalize the image to zero mean and unit variance."""
    mean, std = ds_stats
    image -= tf.constant(mean, shape=[1, 1, num_channels], dtype=image.dtype)
    image /= tf.constant(std, shape=[1, 1, num_channels], dtype=image.dtype)
    return image, label

  ds = ds.map(img_normalize)
  if batch_size == -1:
    batch_size = num_examples
  if repeat:
    ds = ds.repeat()
  if shuffle:
    ds = ds.shuffle(buffer_size=10 * batch_size, seed=shuffle_seed)
  ds = ds.batch(batch_size)
  return tfds.as_numpy(ds), num_classes, num_examples


def get_image_dataset(name):
  train_set, n_classes, _ = load_image_dataset("train", -1, name)
  train_set = next(iter(train_set))

  test_set, _, _ = load_image_dataset("test", -1, name)
  test_set = next(iter(test_set))

  data_info = {"num_classes": n_classes}

  return train_set, test_set, data_info


def load_uci_regression_dataset(name,
                                split_seed,
                                train_fraction=0.9,
                                data_dir="uci_datasets"):
  """Load a UCI dataset from an npz file.

  Ported from
  https://github.com/wjmaddox/drbayes/blob/master/experiments/uci_exps/bayesian_benchmarks/data.py.
  """
  path = os.path.join(data_dir,
                      _UCI_REGRESSION_FILENAMES[UCIRegressionDatasets(name)])
  data_arr = onp.load(path)
  x, y = data_arr["x"], data_arr["y"]

  indices = jax.random.permutation(jax.random.PRNGKey(split_seed), len(x))
  indices = onp.asarray(indices)
  x, y = x[indices], y[indices]

  n_train = int(train_fraction * len(x))
  x_train, y_train = x[:n_train], y[:n_train]
  x_test, y_test = x[n_train:], y[n_train:]

  def normalize_with_stats(arr, arr_mean=None, arr_std=None):
    return (arr - arr_mean) / arr_std

  def normalize(arr):
    eps = 1e-6
    arr_mean = arr.mean(axis=0, keepdims=True)
    arr_std = arr.std(axis=0, keepdims=True) + eps
    return normalize_with_stats(arr, arr_mean, arr_std), arr_mean, arr_std

  x_train, x_mean, x_std = normalize(x_train)
  y_train, y_mean, y_std = normalize(y_train)
  x_test = normalize_with_stats(x_test, x_mean, x_std)
  y_test = normalize_with_stats(y_test, y_mean, y_std)

  data_info = {"y_scale": float(y_std)}

  return (x_train, y_train), (x_test, y_test), data_info


def _parse_uci_regression_dataset(name_str):
  """Parse name and seed for uci regression data.

  E.g. yacht_2 is the yacht dataset with seed 2.
  """
  pattern_string = "(?P<name>[a-z]+)_(?P<seed>[0-9]+)"
  pattern = re.compile(pattern_string)
  matched = pattern.match(name_str)
  if matched:
    name = matched.group("name")
    seed = matched.group("seed")
    return name, seed
  return None, None


def load_npz_array(filename):
  arr = onp.load(filename, allow_pickle=True)
  return ((arr["x_train"], arr["y_train"]), (arr["x_test"], arr["y_test"]),
          arr["data_info"].item())


def batch_split_axis(batch, n_split):
  """Reshapes batch to have first axes size equal n_split."""
  x, y = batch
  n = x.shape[0]
  n_new = n / n_split
  assert n_new == int(n_new), (
      "First axis cannot be split: batch dimension was {} when "
      "n_split was {}.".format(x.shape[0], n_split))
  n_new = int(n_new)
  return tuple(arr.reshape([n_split, n_new, *arr.shape[1:]]) for arr in (x, y))


def pmap_dataset(ds, n_devices):
  """Shard the dataset to devices."""
  n_data = len(ds[0])
  if n_data % n_devices:
    new_len = n_devices * (n_data // n_devices)
    warning_str = ("Dataset of length {} can not be split onto {} devices."
                   "Truncating to {} data points.".format(
                       n_data, n_devices, new_len))
    warnings.warn(warning_str, UserWarning)
    ds = (arr[:new_len] for arr in ds)
  return jax.pmap(lambda x: x)(batch_split_axis(ds, n_devices))


def make_ds_pmap_fullbatch(name, dtype, n_devices=None, truncate_to=None):
  """Make train and test sets sharded over batch dim."""
  name = name.lower()
  n_devices = n_devices or len(jax.local_devices())
  if name in ImgDatasets._value2member_map_:
    train_set, test_set, data_info = get_image_dataset(name)
    loaded = True
    task = Task.CLASSIFICATION
  elif name == "imdb":
    train_set, test_set, _, data_info = load_imdb_dataset()
    dtype = jnp.int32
    loaded = True
    task = Task.CLASSIFICATION
  elif name[-4:] == ".npz":
    train_set, test_set, data_info = load_npz_array(name)
    loaded = True
    task = Task.CLASSIFICATION
  else:
    name, seed = _parse_uci_regression_dataset(name)
    loaded = name is not None
    if name is not None:
      train_set, test_set, data_info = load_uci_regression_dataset(
          name, int(seed))
      loaded = True
      task = Task.REGRESSION

  if not loaded:
    raise ValueError("Unknown dataset name: {}".format(name))

  if truncate_to:
    assert truncate_to % n_devices == 0, (
        "truncate_to should be devisible by n_devices, but got values "
        "truncate_to={}, n_devices={}".format(truncate_to, n_devices))
    train_set = tuple(arr[:truncate_to] for arr in train_set)

  train_set, test_set = tuple(
      pmap_dataset(ds, n_devices) for ds in (train_set, test_set))

  train_set, test_set = map(lambda ds: (ds[0].astype(dtype), ds[1]),
                            (train_set, test_set))

  return train_set, test_set, task, data_info
