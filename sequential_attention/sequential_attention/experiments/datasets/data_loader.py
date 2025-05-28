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

"""Data loader functions to read various tabular datasets."""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def load_mice():
  """Loads the Mice dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  filling_value = -100000
  cache_filepath = os.path.join(DATA_DIR, "mice/Data_Cortex_Nuclear.csv")

  with open(cache_filepath, "r") as fp:
    x = np.genfromtxt(
        fp.readlines(),
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
  with open(cache_filepath, "r") as fp:
    classes = np.genfromtxt(
        fp.readlines(),
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )

  for i, row in enumerate(x):
    for j, val in enumerate(row):
      if val == filling_value:
        x[i, j] = np.mean(
            [
                x[k, j]
                for k in range(classes.shape[0])
                if np.all(classes[i] == classes[k])
            ]
        )

  y = np.zeros((classes.shape[0]), dtype=np.uint8)
  for i, row in enumerate(classes):
    for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
      y[i] += (2**j) * (val == label)

  x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)

  indices = np.arange(x.shape[0])
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]
  classes = classes[indices]

  print("Data loaded...")
  print("Data shapes:")
  print("x shape: {}, y shape: {}".format(x.shape, y.shape))

  is_classification = True
  num_classes = 8

  x_train = pd.DataFrame(x[: x.shape[0] * 4 // 5])
  x_test = pd.DataFrame(x[x.shape[0] * 4 // 5 :])
  y_train = pd.DataFrame(y[: x.shape[0] * 4 // 5], dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(y[x.shape[0] * 4 // 5 :], dtype=np.int32).iloc[:, 0]

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)


def load_isolet():
  """Loads the Isolet dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  cache_filepath_train = os.path.join(DATA_DIR, "isolet/isolet1+2+3+4.data")
  cache_filepath_test = os.path.join(DATA_DIR, "isolet/isolet5.data")

  # use gfile to read from cns, otherwise can just use np.genfromtxt
  with open(cache_filepath_train, "r") as fp:
    x_train = np.genfromtxt(
        fp.readlines(), delimiter=",", usecols=range(0, 617), encoding="UTF-8"
    )
  with open(cache_filepath_train, "r") as fp:
    y_train = np.genfromtxt(
        fp.readlines(), delimiter=",", usecols=[617], encoding="UTF-8"
    )
  with open(cache_filepath_test, "r") as fp:
    x_test = np.genfromtxt(
        fp.readlines(), delimiter=",", usecols=range(0, 617), encoding="UTF-8"
    )
  with open(cache_filepath_test, "r") as fp:
    y_test = np.genfromtxt(
        fp.readlines(), delimiter=",", usecols=[617], encoding="UTF-8"
    )

  x = MinMaxScaler(feature_range=(0, 1)).fit_transform(
      np.concatenate((x_train, x_test))
  )
  x_train = x[: len(y_train)]
  x_test = x[len(y_train) :]

  print("Data loaded...")
  print("Data shapes:")
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  is_classification = True
  num_classes = 26

  x_train = pd.DataFrame(x_train)
  x_test = pd.DataFrame(x_test)
  y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)


def load_activity():
  """Loads the Activity dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  cache_filepath_train_x = os.path.join(DATA_DIR, "activity/X_train.txt")
  cache_filepath_train_y = os.path.join(DATA_DIR, "activity/y_train.txt")
  cache_filepath_test_x = os.path.join(DATA_DIR, "activity/X_test.txt")
  cache_filepath_test_y = os.path.join(DATA_DIR, "activity/y_test.txt")
  with open(cache_filepath_train_x, "r") as fp:
    x_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_test_x, "r") as fp:
    x_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_train_y, "r") as fp:
    y_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_test_y, "r") as fp:
    y_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")

  x = MinMaxScaler(feature_range=(0, 1)).fit_transform(
      np.concatenate((x_train, x_test))
  )
  x_train = x[: len(y_train)]
  x_test = x[len(y_train) :]

  print("Data loaded...")
  print("Data shapes:")
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  is_classification = True
  num_classes = 6

  x_train = pd.DataFrame(x_train)
  x_test = pd.DataFrame(x_test)
  y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)


def load_coil():
  """Loads the Coil dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  samples = []
  for i in range(1, 21):  # classes
    for image_index in range(72):  # examples
      image_filename = "obj%d__%d.png" % (i, image_index)
      image_filename = os.path.join(
          DATA_DIR, f"coil/coil-20-proc/{image_filename}"
      )
      with Image.open(image_filename) as obj_img:
        rescaled = obj_img.resize((20, 20))
        pixels_values = [float(x) for x in list(rescaled.getdata())]
      sample = np.array(pixels_values + [i])
      samples.append(sample)
  samples = np.array(samples)
  np.random.shuffle(samples)
  data = samples[:, :-1]
  targets = (samples[:, -1] + 0.5).astype(np.int64)
  data = (data - data.min()) / (data.max() - data.min())

  l = data.shape[0] * 4 // 5
  x_train = data[:l]
  y_train = targets[:l] - 1
  x_test = data[l:]
  y_test = targets[l:] - 1

  is_classification = True
  num_classes = 20

  x_train = pd.DataFrame(x_train)
  x_test = pd.DataFrame(x_test)
  y_train = pd.DataFrame(y_train, dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(y_test, dtype=np.int32).iloc[:, 0]

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)


def load_data(fashion=False, digit=None, normalize=False):
  """Loads the data for image datasets."""

  if fashion:
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  if digit is not None and 0 <= digit and digit <= 9:
    train = test = {y: [] for y in range(10)}
    for x, y in zip(x_train, y_train):
      train[y].append(x)
    for x, y in zip(x_test, y_test):
      test[y].append(x)

    for y in range(10):
      train[y] = np.asarray(train[y])
      test[y] = np.asarray(test[y])

    x_train = train[digit]
    x_test = test[digit]

  x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(
      np.float32
  )
  x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(
      np.float32
  )

  if normalize:
    x = np.concatenate((x_train, x_test))
    x = (x - x.min()) / (x.max() - x.min())
    x_train = x[: len(y_train)]
    x_test = x[len(y_train) :]

  return (x_train, y_train), (x_test, y_test)


def load_mnist():
  """Loads the MNIST dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  train, test = load_data(fashion=False, normalize=True)

  is_classification = True
  num_classes = 10

  x_train = pd.DataFrame(train[0])
  x_test = pd.DataFrame(test[0])
  y_train = pd.DataFrame(train[1], dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(test[1], dtype=np.int32).iloc[:, 0]

  print("Data loaded...")
  print("Data shapes:")
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)


def load_fashion():
  """Loads the Fashion dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  train, test = load_data(fashion=True, normalize=True)

  is_classification = True
  num_classes = 10

  x_train = pd.DataFrame(train[0])
  x_test = pd.DataFrame(test[0])
  y_train = pd.DataFrame(train[1], dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(test[1], dtype=np.int32).iloc[:, 0]

  print("Data loaded...")
  print("Data shapes:")
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  return (x_train, x_test, y_train, y_test, is_classification, num_classes)
