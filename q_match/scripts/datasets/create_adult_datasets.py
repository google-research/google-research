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

"""Creates the Adult 1% Dataset."""
import os
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import tensorflow as tf


_READ_PATH = '/tmp/data/'
_WRITE_PATH = '/tmp/data/'

_READ_PATH = flags.DEFINE_string('read_path',
                                 _READ_PATH, 'Path to find data files.')
_WRITE_PATH = flags.DEFINE_string('write_path',
                                  _WRITE_PATH, 'Path to write TF Records.')


SCHEMA = """age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands."""


FLAGS = flags.FLAGS


def _parse_schema(schema):
  """Returns the numerical columns, categorical columns, and target column."""
  variables = dict()
  columns = []
  for row in schema.split('\n'):
    name, value = row.split(': ')
    columns.append(name)
    if value != 'continuous.':
      categoricals = value.split(', ')
      value = {cat.strip().strip('.'): i+1
               for i, cat in enumerate(categoricals)}
      value.update({'?': 0})
    variables[name] = value
  columns.append('target')
  variables.update({'target': {'<=50K': 1, '>50K': 0}})

  numerical_features = [
      name for name, value in variables.items() if value == 'continuous.'
  ]
  categorical_features = [
      name
      for name, value in variables.items()
      if (value != 'continuous.' and name != 'target')
  ]
  target = 'target'
  return numerical_features, categorical_features, target, variables, columns


def _make_tf_record(
    dataframe,
    name,
    numerical_features,
    categorical_features,
    target,
    variables,
    write_path,
    dataset_name='adult',
):
  """Write the TF record of the dataframe."""
  tf.io.gfile.makedirs(os.path.join(write_path, dataset_name))
  with tf.io.TFRecordWriter(
      os.path.join(write_path, dataset_name, name + '.tfrecord')
  ) as file_writer:
    for i in range(len(dataframe)):
      row = dataframe.iloc[i]
      features = dict()
      for num_feature in numerical_features:
        features[num_feature] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[row[num_feature]]))
      for cat_feature in categorical_features:
        value = row[cat_feature].strip()
        for feature_value in list(variables[cat_feature].keys()):
          features[cat_feature + '_' + feature_value] = tf.train.Feature(
              float_list=tf.train.FloatList(value=[0.0])
          )
        features[cat_feature + '_' + value] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[1.0])
        )

      features[target] = tf.train.Feature(
          int64_list=tf.train.Int64List(
              value=[int(variables[target][row[target].strip().strip('.')])]
          )
      )
      record_bytes = tf.train.Example(
          features=tf.train.Features(feature=features)
      ).SerializeToString()
      file_writer.write(record_bytes)


def make_datasets(read_path, write_path):
  """Reads the orignal dataset, applies transformations, and saves splits."""
  numerical_features, categorical_features, target, variables, columns = (
      _parse_schema(SCHEMA)
  )
  with tf.io.gfile.GFile(os.path.join(read_path, 'adult.data')) as f:
    df = pd.read_csv(f, header=None, names=columns)
  with tf.io.gfile.GFile(os.path.join(read_path, 'adult.test')) as f:
    test = pd.read_csv(f, header=None, names=columns, skiprows=[0])

  # transform numerical features
  qt = QuantileTransformer(n_quantiles=128, random_state=0)
  transformed_array = qt.fit_transform(df[numerical_features])
  df_transformed_numerical = pd.DataFrame(
      data=transformed_array, columns=numerical_features
  )
  df = pd.concat(
      [df_transformed_numerical, df[categorical_features + [target]]], axis=1
  )

  ## also transform the test data
  transformed_array = qt.transform(test[numerical_features])
  df_transformed_numerical = pd.DataFrame(
      data=transformed_array, columns=numerical_features
  )
  test = pd.concat(
      [df_transformed_numerical, test[categorical_features + [target]]], axis=1
  )

  one_hot_cat_features = []
  for cat_feature in categorical_features:
    for feature_value in list(variables[cat_feature].keys()):
      one_hot_cat_features.append(cat_feature+'_'+feature_value)

  # tf record code formerly defined here.

  # pretextval = len(df)  # pretext + val is the original dataset
  pretext = int(len(df)*.95)
  pretext_val = int(len(df)*.97)
  train = int(len(df)*.01)

  np.random.seed(1)
  pretextval_idx = np.random.choice(len(df), len(df), replace=False)
  pretext_idx = pretextval_idx[:pretext]  # pretext training
  pretext_val_idx = pretextval_idx[
      pretext:pretext_val
  ]  # validation for pretext task
  val_idx = pretextval_idx[pretext_val:]  # take the tail of this set as the val
  train_idx = pretextval_idx[:train]  # take the head of this as the train

  dataset_name = 'adult1p'
  _make_tf_record(
      df.iloc[pretext_idx],
      'pretext',
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      target=target,
      variables=variables,
      write_path=write_path,
      dataset_name=dataset_name,
  )
  _make_tf_record(
      df.iloc[pretext_val_idx],
      'pretext_val',
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      target=target,
      variables=variables,
      write_path=write_path,
      dataset_name=dataset_name,
  )
  _make_tf_record(
      df.iloc[val_idx],
      'val',
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      target=target,
      variables=variables,
      write_path=write_path,
      dataset_name=dataset_name,
  )
  _make_tf_record(
      df.iloc[train_idx],
      'train',
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      target=target,
      variables=variables,
      write_path=write_path,
      dataset_name=dataset_name,
  )
  _make_tf_record(
      test,
      'test',
      numerical_features=numerical_features,
      categorical_features=categorical_features,
      target=target,
      variables=variables,
      write_path=write_path,
      dataset_name=dataset_name,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  read_path = _READ_PATH.value
  write_path = _WRITE_PATH.value

  make_datasets(read_path, write_path)


if __name__ == '__main__':
  app.run(main)

