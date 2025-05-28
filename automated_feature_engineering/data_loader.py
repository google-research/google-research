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
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import datasets as sk_datasets
from sklearn import model_selection
from sklearn import preprocessing
import tensorflow as tf


DATA_DIR = "gs://data-imputation/"
LOCAL_CACHE_DIR = "./"
DataReturnType = Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Any, Any
]
_EMPTY_NP_ARRAY = np.empty(0)
_IMPUTE_VALUE = "0"


def load_ames_housing(
    seed = 44, test_size_ratio = 0.2
):
  """Load Ames housing dataset. Regression. (1460, 80)."""
  housing = sk_datasets.fetch_openml(name="house_prices", as_frame=True)

  x = housing.data
  y = housing.target

  x_train, x_test, y_train, y_test = (
      model_selection.model_selection.train_test_split(
          x, y, test_size=test_size_ratio, random_state=seed
      )
  )

  is_classification = False

  print("Data loaded...")
  print("Column names:")
  print(x_train.columns)

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      None,
      _EMPTY_NP_ARRAY,
  )


def generate_feature_engineering_synthetic_data(
    num_samples = 10000,
    num_features = 5,
    seed = None,
    test_size_ratio = 0.2,
    zipf_distribution = 10.0,
    triangle_left = 0,
    triangle_center = 5,
    triangle_right = 6,
):
  """Generates synthetic dataset for regression task.

  Args:
    num_samples: number of samples to generate.
    num_features: feature dimension in generated dataset.
    seed: random number generator seed.
    test_size_ratio: proportion of data used for test set.
    zipf_distribution: distribution parameter for the zipf distribution.
    triangle_left: left bound for the triangular distribution..
    triangle_center: center for the triangular distribution..
    triangle_right: right bound for the triangular distribution.

  Returns:
    Generated synthetic dataset and pertinent metadata.
  """

  x = np.random.uniform(low=-1, high=1, size=(num_samples, num_features - 2))
  x1 = np.random.zipf(a=zipf_distribution, size=(num_samples, 1))
  x2 = np.reshape(
      np.random.triangular(
          triangle_left, triangle_center, triangle_right, num_samples
      ),
      (-1, 1),
  )

  features = np.concatenate([x, x1, x2], axis=-1)
  labels = np.exp(x2[:, -1])

  features = pd.DataFrame(features)
  labels = pd.Series(labels)

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      features, labels, test_size=test_size_ratio, random_state=seed
  )

  print("Synthetic data loaded...")

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      False,
      None,
      _EMPTY_NP_ARRAY,
  )


def generate_feature_engineering_synthetic_data_temporal(
    num_samples = 100,
    target_column = "label",
    temporal_lag = 7,
    context_window_len = 10,
    seed = None,
    test_size_ratio = 0.2,
):
  """Generates temporal synthetic dataset for regression task.

  Args:
    num_samples: number of samples to generate.
    target_column: name of the target column.
    temporal_lag: time lag along the temporal dimension used for label creation.
    context_window_len: size of the time series prediction context window.
    seed: random number generator seed.
    test_size_ratio: proportion of data used for test set.

  Returns:
    Generated temporal synthetic dataset and pertinent metadata.
  """

  temporal_lag = min(temporal_lag, context_window_len)
  category = ["cat"] * num_samples
  event = ["event"] * num_samples
  num_random_samples = num_samples + context_window_len
  all_price = np.random.randint(low=0, high=100, size=(num_random_samples,))
  prices = []
  labels = []
  for i in range(num_samples):
    prices.append(str(all_price[i : i + context_window_len].tolist()))
    labels.append(str(all_price[i + context_window_len - temporal_lag]))

  features = {
      "category": category,
      "price": prices,
      "event": event,
      target_column: labels,
  }

  features = pd.DataFrame(features)
  labels = pd.Series(labels)

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      features, labels, test_size=test_size_ratio, random_state=seed
  )

  print("Temporal synthetic data loaded...")

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      False,
      None,
      _EMPTY_NP_ARRAY,
  )


def load_m5(
    target_col,
    seed = 4,
    test_size_ratio = 0.2,
    use_rows = 2000,
):
  """Loads the m5 sample dataset generated by the FTE.

  This dataset presents sequential features as strings of arrays.

  Args:
    target_col: name of the target column.
    seed: seed for random number generator.
    test_size_ratio: ratio used to allocate test set.
    use_rows: number of rows to use for training and eval.

  Returns:
    Train and test data, with associated properties.
  """
  cache_filepath = "m5_vertex_ai_fte_split_output_train_staging_100k"
  data_path = os.path.join(LOCAL_CACHE_DIR, cache_filepath)

  if not os.path.isfile(data_path):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR, cache_filepath), data_path, overwrite=True
    )

  # Use a test set to ensure numerical columns are loaded as float32.
  df_test = pd.read_csv(data_path, nrows=200)
  float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
  float32_cols = {c: np.float32 for c in float_cols}

  dataset = pd.read_csv(
      data_path, engine="c", dtype=float32_cols, nrows=use_rows
  )
  dataset = dataset.replace("__MISSING__", _IMPUTE_VALUE)

  # Note this x includes all feature columns, as the historical "sales"
  # data are used as a feature as well.
  x = dataset
  y = dataset[target_col].to_frame()

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=test_size_ratio, random_state=seed
  )

  is_classification = False
  num_classes = None

  print("Data loaded...")
  print("Column names:")
  print(x_train.columns)

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_fraud(
    seed = 4, test_size_ratio = 0.2, use_rows = 600000
):
  """Loads the IEEE fraud detection dataset of size (600000, 871).

  Data is downloaded from:
  https://www.kaggle.com/competitions/ieee-fraud-detection/data.

  Args:
    seed: seed for random number generator.
    test_size_ratio: ratio used to allocate test set.
    use_rows: number of rows to use for training and eval.

  Returns:
    Train and test data, with associated properties.
  """

  cache_filepath = "train_transaction_identity.csv"
  if not os.path.isfile(cache_filepath):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR, cache_filepath),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath),
        overwrite=True,
    )

  data_path = os.path.join(LOCAL_CACHE_DIR, cache_filepath)

  # Use a test set to ensure numerical columns are loaded as float32.
  df_test = pd.read_csv(data_path, nrows=200)
  float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
  float32_cols = {c: np.float32 for c in float_cols}

  dataset = pd.read_csv(
      data_path, engine="c", dtype=float32_cols, nrows=use_rows
  )

  x = dataset.iloc[:, 2:]
  y = dataset.iloc[:, 1]
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=test_size_ratio, random_state=seed
  )

  is_classification = True
  num_classes = 2

  print("Data loaded...")
  print("Column names:")
  print(x_train.columns)

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_synthetic(
    num_samples = 100000,
    num_salient_features = 100,
    num_features = 10000,
    seed = 4,
    test_size_ratio = 0.2,
    noise_magnitude = 0.2,
):
  """Generate synthetic data with desired attributes.

  Args:
    num_samples: number of samples to generate.
    num_salient_features: number of non-noise features.
    num_features: total number of features.
    seed: seed for random number generator.
    test_size_ratio: ratio of test set out of all samples generated.
    noise_magnitude: magnitude of noise to add.

  Returns:
    Generated train and test datasets, and their attributes.
  """
  cache_filepath = (
      "synthetic_data_"
      + str(num_samples)
      + "_"
      + str(num_features)
      + "_"
      + str(num_salient_features)
      + ".npz"
  )
  try:
    loaded_data = np.load(cache_filepath)
    input_features = loaded_data["input_features"]
    salient_features = loaded_data["salient_features"]
    output_labels = loaded_data["output_labels"]
  except OSError:
    # cache_filepath does not exist.
    input_features = np.random.uniform(
        low=-1, high=1, size=(num_samples, num_features)
    )

    salient_features = np.random.choice(num_features, num_salient_features)
    np.random.shuffle(salient_features)

    subset1 = salient_features[: num_salient_features // 5]
    subset2 = salient_features[
        num_salient_features // 5 : 2 * num_salient_features // 5
    ]
    subset3 = salient_features[
        2 * num_salient_features // 5 : 3 * num_salient_features // 5
    ]
    subset4 = salient_features[
        3 * num_salient_features // 5 : 4 * num_salient_features // 5
    ]
    subset5 = salient_features[4 * num_salient_features // 5 :]

    # Various nonlinear functions that depend on different salient features.
    term1 = np.mean(np.exp(input_features[:, subset1] - 1), axis=1)

    term2 = np.exp(
        np.mean(np.abs(np.sin(2 * np.pi * input_features[:, subset2])), axis=1)
    )

    term3 = np.mean(-1.0 * np.log(1.1 + input_features[:, subset3]), axis=1)

    term4 = np.mean(input_features[:, subset4], axis=1)

    term5 = 1.0 / (
        1.0 + np.mean(np.abs(np.tanh(input_features[:, subset5])), axis=1)
    )

    # Construct the logit
    aggregate_term = 1.0 * ((term1 + term2 + term3 + term4 + term5) - 3)

    # Add noise
    noise = np.random.randn(num_samples)
    aggregate_term += noise_magnitude * noise
    output_labels = (aggregate_term > 0).astype(int)

    np.savez(
        cache_filepath,
        input_features=input_features,
        salient_features=salient_features,
        output_labels=output_labels,
    )

  x = pd.DataFrame(input_features)
  y = pd.DataFrame(output_labels).iloc[:, 0]

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=test_size_ratio, random_state=seed
  )

  # TODO(yihed): Generalize to regression as well.
  is_classification = True
  num_classes = 2

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      salient_features,
  )


def load_safe(seed = 4, test_size_ratio = 0.2):
  """Loads the safe dataset."""

  # Data is downloaded from here:
  # https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data.

  cache_filepath = "safe_train.csv"
  if not os.path.isfile(cache_filepath):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR, cache_filepath),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath),
        overwrite=True,
    )

  dataset = pd.read_csv(os.path.join(LOCAL_CACHE_DIR, cache_filepath))
  x = dataset.iloc[:, 2:]
  y = dataset.iloc[:, 1]

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=test_size_ratio, random_state=seed
  )

  is_classification = True
  num_classes = 2

  print("Data loaded...")
  print("Column names:")
  print(x_train.columns)

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_year(seed = 4, test_size_ratio = 0.2):
  """Loads the year dataset."""

  # Data is downloaded from here:
  # https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd.

  cache_filepath = "YearPredictionMSD.txt"
  if not os.path.isfile(cache_filepath):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR, cache_filepath),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath),
        overwrite=True,
    )

  dataset = pd.read_csv(
      os.path.join(LOCAL_CACHE_DIR, cache_filepath), header=None
  )
  x = dataset.iloc[:, 1:]
  y = dataset.iloc[:, 0]

  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=test_size_ratio, random_state=seed
  )

  is_classification = False

  print("Data loaded...")
  print("Column names:")
  print(x_train.columns)

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      None,
      _EMPTY_NP_ARRAY,
  )


def load_mice():
  """Loads the Mice dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  filling_value = -100000
  cache_filepath = "Data_Cortex_Nuclear.csv"
  if not os.path.isfile(cache_filepath):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "MICE/", cache_filepath),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath),
        overwrite=True,
    )

  x = np.genfromtxt(
      cache_filepath,
      delimiter=",",
      skip_header=1,
      usecols=range(1, 78),
      filling_values=filling_value,
      encoding="UTF-8",
  )
  classes = np.genfromtxt(
      cache_filepath,
      delimiter=",",
      skip_header=1,
      usecols=range(78, 81),
      dtype=None,
      encoding="UTF-8",
  )

  for i, row in enumerate(x):
    for j, val in enumerate(row):
      if val == filling_value:
        x[i, j] = np.mean([
            x[k, j]
            for k in range(classes.shape[0])
            if np.all(classes[i] == classes[k])
        ])

  y = np.zeros((classes.shape[0]), dtype=np.uint8)
  for i, row in enumerate(classes):
    for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
      y[i] += (2**j) * (val == label)

  x = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(x)

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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_isolet():
  """Loads the Isolet dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  cache_filepath_train = "isolet1+2+3+4.data"
  cache_filepath_test = "isolet5.data"
  if not os.path.isfile(cache_filepath_train):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "isolet/", cache_filepath_train),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_train),
        overwrite=True,
    )
  if not os.path.isfile(cache_filepath_test):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "isolet/", cache_filepath_test),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_test),
        overwrite=True,
    )

  x_train = np.genfromtxt(
      cache_filepath_train,
      delimiter=",",
      usecols=range(0, 617),
      encoding="UTF-8",
  )
  y_train = np.genfromtxt(
      cache_filepath_train, delimiter=",", usecols=[617], encoding="UTF-8"
  )
  x_test = np.genfromtxt(
      cache_filepath_test,
      delimiter=",",
      usecols=range(0, 617),
      encoding="UTF-8",
  )
  y_test = np.genfromtxt(
      cache_filepath_test, delimiter=",", usecols=[617], encoding="UTF-8"
  )

  x = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_activity():
  """Loads the Activity dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  cache_filepath_train_x = "final_X_train.txt"
  cache_filepath_train_y = "final_y_train.txt"
  cache_filepath_test_x = "final_X_test.txt"
  cache_filepath_test_y = "final_y_test.txt"
  if not os.path.isfile(cache_filepath_train_x):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "dataset_uci/", cache_filepath_train_x),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_train_x),
        overwrite=True,
    )
  if not os.path.isfile(cache_filepath_train_y):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "dataset_uci/", cache_filepath_train_y),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_train_y),
        overwrite=True,
    )
  if not os.path.isfile(cache_filepath_test_x):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "dataset_uci/", cache_filepath_test_x),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_test_x),
        overwrite=True,
    )
  if not os.path.isfile(cache_filepath_test_y):
    tf.io.gfile.copy(
        os.path.join(DATA_DIR + "dataset_uci/", cache_filepath_test_y),
        os.path.join(LOCAL_CACHE_DIR, cache_filepath_test_y),
        overwrite=True,
    )
  x_train = np.genfromtxt(cache_filepath_train_x, encoding="UTF-8")
  x_test = np.genfromtxt(cache_filepath_test_x, encoding="UTF-8")
  y_train = np.genfromtxt(cache_filepath_train_y, encoding="UTF-8")
  y_test = np.genfromtxt(cache_filepath_test_y, encoding="UTF-8")

  x = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_coil():
  """Loads the Coil dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""

  processed_data_filename = "coil_data.npz"
  # Instead of processing the image data again, save the processed data for
  # efficiency of subsequent runs.
  try:
    loaded_data = np.load(LOCAL_CACHE_DIR + processed_data_filename)
    data = np.float32(loaded_data["data"])
    targets = np.float32(loaded_data["targets"])
  except IOError:
    samples = []
    os.makedirs(LOCAL_CACHE_DIR + "coil-20-proc/", exist_ok=True)
    for i in range(1, 21):
      for image_index in range(72):
        image_filename = "obj%d__%d.png" % (i, image_index)
        tf.io.gfile.copy(
            os.path.join(DATA_DIR + "coil-20-proc/", image_filename),
            os.path.join(LOCAL_CACHE_DIR + "coil-20-proc/", image_filename),
            overwrite=True,
        )
        obj_img = Image.open(
            os.path.join(LOCAL_CACHE_DIR + "coil-20-proc/", image_filename)
        )
        rescaled = obj_img.resize((20, 20))
        pixels_values = [float(x) for x in list(rescaled.getdata())]
        sample = np.array(pixels_values + [i])
        samples.append(sample)
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())
    np.savez(
        LOCAL_CACHE_DIR + processed_data_filename, data=data, targets=targets
    )

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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def load_data(
    fashion = False, digit = None, normalize = False
):
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

    x_train = np.asarray(train[digit])
    x_test = np.asarray(test[digit])

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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


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

  return (
      x_train,
      x_test,
      y_train,
      y_test,
      is_classification,
      num_classes,
      _EMPTY_NP_ARRAY,
  )


def get_mtime(path):
  """Gets file modification time for the path.

  Args:
    path: whose mtime is needed

  Returns:
    mtime in nanseconds (type int)
  """
  stats = tf.io.gfile.stat(path)
  return int(stats.mtime_nsec)


def is_more_recent(path_of_interest, path_to_compare_with):
  """Determines if the path of interest is more recent than path to compare.

  Args:
    path_of_interest: path to test
    path_to_compare_with: path to compare with

  Returns:
    True if path of interest is more recent
  """
  return get_mtime(path_of_interest) > get_mtime(path_to_compare_with)


def ensure_source_file_exists(path):
  """Ensures path exists.

  Args:
    path: path to test

  Raises:
    FileNotFoundError:
  """
  if not os.path.isfile(path) and not tf.io.gfile.exists(path):
    raise FileNotFoundError(path + " does not exist")
  if tf.io.gfile.isdir(path):
    raise FileNotFoundError(path + " is not a file")


def create_parent_dirs(path, mode = 0o777):
  """Creates parent directories for a given path.

  Args:
    path: path whose parents need to be created if not existent
    mode: mode with which the parents should be created

  Raises:
    FileExistsError: if parent exists and is not a directory
  """
  d = os.path.dirname(path)
  if os.path.isfile(d):
    raise FileExistsError(d + "is a not a dir and exists")
  if os.path.isdir(d):
    return
  os.makedirs(d, mode=mode, exist_ok=True)


def update_cached_file(data_file, cache_file):
  """Updates cache file if does not exist or source is more recent.

  Args:
    data_file: original file that needs to be copied into cache
    cache_file: cache file that needs to be as recent as data file
  """
  ensure_source_file_exists(data_file)
  create_parent_dirs(cache_file, 0o777)
  if not os.path.isfile(cache_file) or not is_more_recent(
      cache_file, data_file
  ):
    if tf.io.gfile.exists(cache_file):
      tf.io.gfile.remove(cache_file)
    tf.io.gfile.copy(data_file, cache_file)


def update_cache(data_dir, cache_dir, files):
  """Updates multiple cache files that are listed in arguments.

  Args:
    data_dir: source data dir from where the file need to be copied
    cache_dir: directory into which the file need to be copied
    files: list of files that need to be copied
  """
  if len(files) == 1 and isinstance(files, str):
    update_cached_file(
        os.path.join(data_dir, files[0]), os.path.join(cache_dir, files[0])
    )
    return
  for file in files:
    update_cached_file(
        os.path.join(data_dir, file), os.path.join(cache_dir, file)
    )


def separate_data_and_labels(
    df, label
):
  """splits the dataframe into data and label dataframes.

  Args:
    df: to split
    label:  name of label column

  Returns:
    data and label dataframes
  """
  label_df = df[label]
  data_df = df.drop(label, axis=1)
  return (data_df, label_df)


def date_to_year_month_day(
    df,
    date_col,
    year_col = "Year",
    month_col = "Month",
    day_col = "Day",
    drop_date_col = True,
):
  """Extracts year, month and day from date column and add to frame.

  Args:
    df: data frame to be modified
    date_col: name of the column that is date column
    year_col: name of the new year column
    month_col: name of the new month column
    day_col: name of hte new day column
    drop_date_col: delete date column or not

  Returns:
    data frame with columns for year, month and day
  """
  year = df[date_col].dt.year
  month = df[date_col].dt.month
  day = df[date_col].dt.day
  df[year_col] = year
  df[month_col] = month
  df[day_col] = day
  if drop_date_col:
    df = df.drop(date_col, axis=1)
  return df


def create_mask_columns(
    df,
    source_col,
    target_col_names,
    based_on_values,
    drop_source_col = True,
):
  """spread a column into number of columns based on contents.

  Args:
    df: dataframe containing the source column
    source_col: of the source column
    target_col_names: list of new target columns to create
    based_on_values: list of matching values for each of target columns
    drop_source_col: whether to drop source column or not

  Returns:
    dataframe with mask columns

  Raises:
    ValueError: if months and based on values are not of same length
  """
  if len(target_col_names) != len(based_on_values):
    raise ValueError(
        "target_col_names and based_on_values must have the same length"
    )
  for c, v in zip(target_col_names, based_on_values):

    def has(x):
      if isinstance(x, float):
        return 0
      # pylint: disable=cell-var-from-loop
      return 1 if v in x else 0

    df[c] = df[source_col].apply(has)
  if drop_source_col:
    df = df.drop(source_col, axis=1)
  return df


def create_month_list(use_full_names = False):
  """Returns: list of months."""
  if not use_full_names:
    return [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
  return [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December",
  ]


def load_diabetes(
    seed = 44,
    test_size_ratio = 0.2,
    nrows = None,
    data_dir = os.path.join(DATA_DIR, "diabetes"),
    cache_dir = os.path.join(LOCAL_CACHE_DIR, "diabetes"),
    discard_cols = None,
):
  """Loads kaggle diabetes data.

  Args:
    seed: random state for train test split. use same value across multiple runs
      to consistently generate same shuffle order while splitting data
    test_size_ratio: ratio of test data set to total data set lengths
    nrows: number of rows to use, if not provided, all rows are used data_dir :
      source directory containing diabetes data
    data_dir: source directory for data
    cache_dir: cache directory to copy data into. Default value is curre nt dir
    discard_cols: list of column names to discard from data

  Returns:
    train, test data and labels
    whether data is for classification problems or not
    number of classes if this is for classification
    an empty numpy array

  Raises:
    KeyError: if label column is not found
  """
  label = "Outcome"
  file = "diabetes.csv"
  dest_file = os.path.join(cache_dir, file)
  update_cache(data_dir, cache_dir, [file])
  df = pd.read_csv(dest_file, nrows=nrows)
  if discard_cols:
    df = df.drop(columns=discard_cols)
  if label not in df.columns:
    raise KeyError(label + " not found")
  data_df, label_df = separate_data_and_labels(df, label)
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      data_df, label_df, random_state=seed, test_size=test_size_ratio
  )
  return (x_train, x_test, y_train, y_test, True, 2, _EMPTY_NP_ARRAY)


def build_criteo_display_ads_columns(
    num_numeric_cols = 13, num_categorical_cols = 26
):
  """Builds columns for criteo data as it does not have labels.

  Args:
    num_numeric_cols: number of integer columns, default is 13
    num_categorical_cols: number of categorical columns, default is 26

  Returns:
    a list of column names including a label column named as Label
  """
  cols = ["Label"]
  cols.extend([f"numeric_{i}" for i in range(num_numeric_cols)])
  cols.extend([f"categorical_{i}" for i in range(num_categorical_cols)])
  return cols


def load_criteo_ads_display(
    seed = 44,
    test_size_ratio = 0.2,
    nrows = None,
    data_dir = os.path.join(DATA_DIR, "criteo_display_ads"),
    cache_dir = os.path.join(LOCAL_CACHE_DIR, "criteo_display_ads"),
    discard_cols = None,
):
  """Loads criteo display ads data.

  Criteo data is tab separate and there are no headers in the datafile.

  Args:
    seed: random state for train test split. use same value across multiple runs
      to consistently generate same shuffle order while splitting data
    test_size_ratio: ratio of test data set to total data set lengths
    nrows: number of rows to use, if not provided, all rows are used
    data_dir: source of test data. default value is gs://data-imputation
    cache_dir: cache directory to copy data into. Default value is current dir
    discard_cols: list of column indices to discard from data

  Returns:
    train, test data and labels
    whether data is for classification problems or not
    number of classes if this is for classification
    an empty numpy array

  Raises:
    KeyError: if label column is not found
  """
  file = "train.txt"
  dest_file = os.path.join(cache_dir, file)
  update_cache(data_dir, cache_dir, [file])
  columns = build_criteo_display_ads_columns(13, 26)
  df = pd.read_csv(dest_file, nrows=nrows, sep="\t", names=columns)
  label = df.columns[0]
  for i in range(1, 14):
    df[columns[i]] = df[columns[i]].astype("Int64")
  if discard_cols:
    df = df.drop(df.columns[discard_cols], axis=1)
  if label not in df.columns:
    raise KeyError(label + " not found")
  df = df.dropna()
  data_df, label_df = separate_data_and_labels(df, label)
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      data_df, label_df, random_state=seed, test_size=test_size_ratio
  )
  return (x_train, x_test, y_train, y_test, True, 2, _EMPTY_NP_ARRAY)


def load_rossmann_sales_data(
    seed = 44,
    test_size_ratio = 0.2,
    nrows = None,
    data_dir = os.path.join(DATA_DIR, "rossmann"),
    cache_dir = os.path.join(LOCAL_CACHE_DIR, "rossmann"),
    discard_cols = None,
    discard_store_cols = None,
    label = "Sales",
):
  """Loads Rossmann's sales data.

  Rossman's data is made of 2 files, train.csv and store.csv. This
  loader joins the 2 files as store.csv provides information per store
  and train.csv provides temporal store data.

  Args:
    seed: random state for train test split. use same value across multiple runs
      to consistently generate same shuffle order while splitting data
    test_size_ratio: ratio of test data set to total data set lengths
    nrows: number of rows to use, if not provided, all rows are used
    data_dir: source of test data. default value is gs://data-imputation
    cache_dir: cache directory to copy data into. Default value is current dir
    discard_cols: list of column names to discard from data
    discard_store_cols: list of store file column names to discard from data
    label: name of the label column

  Returns:
    train, test data and labels
    whether data is for classification problems or not
    number of classes if this is for classification
    an empty numpy array

  Raises:
    KeyError: if label column is not found
  """
  files = ["train.csv", "store.csv"]
  dest_file = os.path.join(cache_dir, files[0])
  update_cache(data_dir, cache_dir, files)
  df = pd.read_csv(dest_file, nrows=nrows)
  if discard_cols:
    df = df.drop(columns=discard_cols)
  if label not in df.columns:
    raise KeyError(label + " not found")
  store_df = pd.read_csv(os.path.join(cache_dir, files[1]))
  if discard_store_cols:
    store_df = store_df.drop(columns=discard_store_cols)
  df = df.merge(store_df, on="Store", how="left")
  df["StateHoliday"] = df["StateHoliday"].astype("str")
  df["SchoolHoliday"] = df["SchoolHoliday"].astype("str")
  df["Date"] = pd.to_datetime(df["Date"])
  df = create_mask_columns(
      df, "PromoInterval", create_month_list(), create_month_list(), True
  )
  df = date_to_year_month_day(df, "Date")
  df = df.dropna()
  data_df, label_df = separate_data_and_labels(df, label)
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      data_df, label_df, random_state=seed, test_size=test_size_ratio
  )
  return (x_train, x_test, y_train, y_test, False, None, _EMPTY_NP_ARRAY)


def load_kaggle_heart_disease_prediction(
    seed = 44,
    test_size_ratio = 0.2,
    nrows = None,
    data_dir = os.path.join(DATA_DIR, "kaggle_heart_disease_prediction"),
    cache_dir = os.path.join(
        LOCAL_CACHE_DIR, "kaggle_heart_disease_prediction"
    ),
    discard_cols = None,
):
  """Load kaggle heart disease data.

  Args:
    seed: random state for train test split. use same value across multiple runs
      to consistently generate same shuffle order while splitting data
    test_size_ratio: ratio of test data set to total data set lengths
    nrows: number of rows to use, if not provided, all rows are used
    data_dir: source of test data. default value is gs://data-imputation
    cache_dir: cache directory to copy data into. Default value is current dir
    discard_cols: list of column names to discard from data

  Returns:
    train, test data and labels
    whether data is for classification problems or not
    number of classes if this is for classification
    an empty numpy array

  Raises:
    KeyError: if label column is not found
  """
  label = "Heart Disease"
  file = "Heart_Disease_Prediction.csv"
  dest_file = os.path.join(cache_dir, file)
  update_cache(data_dir, cache_dir, [file])
  df = pd.read_csv(dest_file, nrows=nrows)
  df["Heart Disease"].replace({"Presence": 1, "Absence": 0}, inplace=True)
  if discard_cols:
    df = df.drop(columns=discard_cols)
  if label not in df.columns:
    raise KeyError(label + " not found")
  data_df, label_df = separate_data_and_labels(df, label)
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      data_df, label_df, random_state=seed, test_size=test_size_ratio
  )
  return (x_train, x_test, y_train, y_test, True, 2, _EMPTY_NP_ARRAY)
