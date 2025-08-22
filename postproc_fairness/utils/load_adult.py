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

"""Util methods for UCI data operations."""

import pandas as pd
import tensorflow as tf
from tensorflow_model_remediation import min_diff

_UCI_DATA_URL_TEMPLATE = 'postproc_fairness/data/adult/adult.{}'

# Column Names corresponding to the UCI data.
_UCI_COLUMN_NAMES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]

# Cache of the full dataset so we don't have to download it every time.
_uci_full_dataframes = {}


def _get_full_uci_dataframe(url):
  """Retrieves the full DataFrame from the given UCI url using pd.read_csv."""
  if url in _uci_full_dataframes:
    return _uci_full_dataframes[url]

  # Download and cache results.
  data = pd.read_csv(url, names=_UCI_COLUMN_NAMES, header=None)
  _uci_full_dataframes[url] = data
  return data


def _get_uci_data_from_url(url, sample=None):
  """Retrieves an optionally sampled DataFrame from the UCI url."""
  data = _get_full_uci_dataframe(url)
  if sample is not None:
    data = data.sample(frac=sample, replace=False, random_state=1)
  # Filter out rows that don't have 'education' feature set.
  data = data[data['education'].notnull()]
  # Convert 'age' values to int.
  data['age'] = data['age'].astype(int)
  # Create 'target' column corresponding to income.
  data['target'] = data['income'].str.contains('>50K').astype(int)
  data['sex'] = data['sex'].str.contains('Female').astype(int)
  # Remove 'income' feature since it's used as the target.
  data.pop('income')
  # Remove 'fnlwgt' and 'rage' features since we don't use them in the model.
  data.pop('fnlwgt')
  # data.pop('race')

  return data


def get_uci_data(split='train', sample=None):
  """Retrieves optionally sampled UCI income data.

  Retrieves the UCI income dataset. If sampled, only a fraction of the examples
  will be returned.
  The following processing is applied:
    - Any example without 'education' data is removed.
    - 'age' values are converted to int types.
    - 'income' is replaced by a column called 'target' which contains a binary
      int value: 1 if 'income' is '>50k' and 0 otherwise.
    - 'fnlwgt' and 'race' features are removed.

  Args:
    split: Default: 'train'. Split for the data. Can be either 'train' or
      'test'.
    sample: Default: `None`. Number between `0` and `1` representing the
      fraction of the data that will be used. If `None`, the entire dataset will
      be used.

  Returns:
    A DataFrame containing UCI income dataset examples.
  """
  if split not in set(['train', 'test']):
    raise ValueError(
        "split must be one of 'train' or 'test', given: {}".format(split)
    )
  suffix = 'data' if split == 'train' else 'test'
  uci_train_url = _UCI_DATA_URL_TEMPLATE.format(suffix)
  return _get_uci_data_from_url(uci_train_url, sample=sample)


def convert_to_dataset(dataframe, shuffle=False, batch_size=None):
  """Converts DataFrame into a tf.data.Dataset.

  The DataFrame must have the label in the feature named 'target'

  Args:
    dataframe: The DataFrame to be converted.
    shuffle: boolean. Default: False. If true, the dataset will be shuffled.
    batch_size: Optional. If set, the dataset will be batched with this batch
      size. Otherwise the dataset is returned unbatched.

  Returns:
    A tf.data.Dataset corresponding to the passed in DataFrame.
  """
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=5000)  # Reasonable but arbitrary buffer_size.
  if batch_size:
    ds = ds.batch(batch_size)
  return ds


def get_uci_min_diff_datasets(
    split='train', sample=None, original_batch_size=128, min_diff_batch_size=32
):
  """Creates 3 UCI datasets for MinDiff training (original, male, female).

  Creates the 3 datasets that need to be packed together to use MinDiff on the
  UCI dataset when targeting a FNR gap between male and female slices. The
  datasets are:
    - original: The original dataset used for training. This will be UCI
      dataset sampled according to the `sample` parameter.
    - MinDiff male: A dataset containing only positively labeled male examples.
      This dataset will be a subset of the original dataset (i.e. will change in
      size according to the `sample` parameter).
    - MinDiff female: A dataset containing only positively labeled female
      examples. This dataset will be a subset of the full data, regardless of
      the value of the `sample` parameter.

  Args:
    split: Default: 'train'. Split for the data. Can be either 'train' or
      'test'.
    sample: Default: `None`. Number between `0` and `1` representing the
      fraction of the data that will be used. If `None`, the entire dataset will
      be used.
    original_batch_size: Default: 128. Batch size for the original dataset.
    min_diff_batch_size: Default: 32. Batch size for the min_diff datasets (male
      and female).

  Returns:
    A tuple of datasets: (original, min_diff_male, min_diff_female).
  """
  sampled = get_uci_data(split=split, sample=sample)
  male_pos = sampled[(sampled['sex'] == ' Male') & (sampled['target'] == 1)]

  # Use full dataset to get extra Female examples.
  full = get_uci_data(split=split)
  female_pos = full[(full['sex'] == ' Female') & (full['target'] == 1)]

  # Convert to tf.data.Dataset
  original_ds = convert_to_dataset(
      sampled, shuffle=True, batch_size=original_batch_size
  )
  min_diff_male_ds = convert_to_dataset(male_pos, shuffle=True).batch(
      min_diff_batch_size, drop_remainder=True
  )
  min_diff_female_ds = convert_to_dataset(female_pos, shuffle=True).batch(
      min_diff_batch_size, drop_remainder=True
  )

  return original_ds, min_diff_male_ds, min_diff_female_ds


def get_uci_with_min_diff_dataset(split='train', sample=None):
  """Creates a single dataset containing MinDiff data.

  Creates a dataset that can be used directly as an input to a `MinDiffModel`.
  This is done by taking the datasets returned by
  `get_uci_min_diff_datasets` and calling
  `min_diff.keras.utils.pack_min_diff_data` to pack them together.

  Args:
    split: Default: 'train'. Split for the data. Can be either 'train' or
      'test'.
    sample: Default: `None`. Number between `0` and `1` representing the
      fraction of the data that will be used. If `None`, the entire dataset will
      be used.

  Returns:
    A single `tf.data.Dataset` that contains MinDiff data.
  """
  original_ds, min_diff_male_ds, min_diff_female_ds = get_uci_min_diff_datasets(
      split=split, sample=sample
  )
  return min_diff.keras.utils.pack_min_diff_data(
      original_dataset=original_ds,
      sensitive_group_dataset=min_diff_female_ds,
      nonsensitive_group_dataset=min_diff_male_ds,
  )


def get_uci_model(model_class=tf.keras.Model):
  """Create the model to be trained on UCI income data.

  The model created uses the keras Functional API. It expects the following
  inputs from the UCI dataset: ['age', 'workclass', 'education', 'sex',
    'education-num', 'marital-status', 'occupation', 'relationship',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
  These features should be provided as a dictionary. Any additional features
  provided will be ignored with a warning.
  Users can pass in a custom class to be used instead of `tf.keras.Model` as
  long as the class supports the Functional API.

  Args:
    model_class: Default: `tf.keras.Model`. Optional custom model class that can
      be used to create the model. The class must be a subclass of
      `tf.keras.Model` and support the Functional API. Note that
      `tf.keras.Sequential` or a subclass of it does not meet this second
      criteria.

  Returns:
    A model to be used on UCI income data.

  Raises:
    TypeError: If `model_class` is not a subclass of `tf.keras.Model`.
    TypeError: If `model_class` is a subclass of `tf.keras.Sequential`.
  """
  if not isinstance(model_class, type):
    raise TypeError(
        '`model_class` must be a class, given: {}'.format(model_class)
    )

  if not issubclass(model_class, tf.keras.Model):
    raise TypeError(
        '`model_class` must be a subclass of `tf.keras.Model`, '
        'given: {}'.format(model_class)
    )
  if issubclass(model_class, tf.keras.Sequential):
    raise TypeError(
        '`model_class` must support the Functional API and '
        'therefore cannot be a subclass of `tf.keras.Sequential`, '
        'given: {}'.format(model_class)
    )
  inputs = {}  # Dictionary of input layers.
  # List of either input layers or preprocessing layers built on top of inputs.
  features = []

  def _add_input_feature(input_layer, feature=None):
    feature = feature if feature is not None else input_layer
    inputs[input_layer.name] = input_layer
    features.append(feature)

  # Numeric inputs.
  numeric_column_names = [
      'education-num',
      'capital-gain',
      'capital-loss',
      'hours-per-week',
  ]
  for col_name in numeric_column_names:
    numeric_input = tf.keras.Input(shape=(1,), name=col_name)
    _add_input_feature(numeric_input)

  # Bucketized age feature.
  age_input = tf.keras.Input(shape=(1,), name='age')
  bucketized_age_feature = tf.keras.layers.Discretization(
      bins=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
  )(age_input)
  encoded_feature = tf.keras.layers.CategoryEncoding(max_tokens=12)(
      bucketized_age_feature
  )
  _add_input_feature(age_input, encoded_feature)

  # Categorical inputs.
  uci_df = get_uci_data()  # UCI data is used to index categorical features.
  categorical_column_names = [
      'sex',
      'native-country',
      'workclass',
      'occupation',
      'marital-status',
      'relationship',
      'education',
  ]
  for col_name in categorical_column_names:
    categorical_input = tf.keras.Input(
        shape=(1,), name=col_name, dtype=tf.string
    )
    vocabulary = uci_df[col_name].unique()
    feature_index = tf.keras.layers.StringLookup(
        vocabulary=vocabulary, mask_token=None
    )(categorical_input)
    # Note that we need to add 1 to max_tokens to account for the 'UNK' token
    # that StringLookup adds.
    encoded_feature = tf.keras.layers.CategoryEncoding(
        max_tokens=len(vocabulary) + 1
    )(feature_index)
    _add_input_feature(categorical_input, encoded_feature)

  # Crossed columns
  # cross: Education x Occupation
  num_bins = 1000
  encoded_feature = tf.keras.layers.experimental.preprocessing.HashedCrossing(
      num_bins=num_bins, output_mode='one_hot'
  )([inputs['education'], inputs['occupation']])
  features.append(encoded_feature)

  # cross: Education x Occupation x Age
  num_bins = 50000
  encoded_feature = tf.keras.layers.experimental.preprocessing.HashedCrossing(
      num_bins=num_bins, output_mode='one_hot'
  )([inputs['education'], inputs['occupation'], bucketized_age_feature])
  features.append(encoded_feature)

  # Build model from inputs.
  concatenated_features = tf.keras.layers.concatenate(features)
  x = tf.keras.layers.Dense(64, activation='relu')(concatenated_features)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
  return model_class(inputs, outputs)
