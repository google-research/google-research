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

"""File for preprocessing the rossmann dataset."""
import calendar
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

train = pd.read_csv('./train.csv')
store = pd.read_csv('./store.csv')

month_abbrs = calendar.month_abbr[1:]
month_abbrs[8] = 'Sept'


def preprocess(df, stores):
  """Function preprocess the dataset.

  1) make integer Year,Month,Day columns instead of Date
  2) join data from store table

  Args:
      df: pandas dataframe containing training data
      stores: panda datafrome with store data

  Returns:
       df: joint dataframe data with the store table
  """

  date = (df['Date'].values)

  df = df.drop(['Date'], axis=1)
  date_ = np.zeros((date.shape[0], 3))
  for i in range(date.shape[0]):
    split = date[i].split('-')
    date_[i, 0] = split[0]
    date_[i, 1] = split[1]
    date_[i, 2] = split[2]
  df['Year'] = date_[:, 0]
  df['Month'] = date_[:, 1]
  df['Day'] = date_[:, 2]
  df = df.join(stores, on='Store', rsuffix='_right')
  df = df.drop(['Store_right'], axis=1)
  promo2_start_months = [
      (s.split(',') if not pd.isnull(s) else []) for s in df['PromoInterval']
  ]

  for month_abbr in month_abbrs:
    df['Promo2Start_' + month_abbr] = np.array([
        (1 if month_abbr in s else 0) for s in promo2_start_months
    ])
  df = df.drop(['PromoInterval'], axis=1)
  df.StateHoliday.replace({0: '0'}, inplace=True)

  df = df.fillna(0)
  df.StoreType.replace({0: '0'}, inplace=True)
  df.Assortment.replace({0: '0'}, inplace=True)

  return df


def get_str_column_names(df):
  """Returns string columns names in a dataframe.

  Args:
      df: pandas dataframe containing training data

  Returns:
       str_name: string column names
  """

  str_names = []
  for col in df.columns:
    for x in df[col]:
      if isinstance(x, str):
        str_names.append(col)
        break

  return str_names


def fix_strs(df, cat_names, test_df=None):
  """Transform categorical columns with strings using LabelEncoder.

  Args:
      df: pandas dataframe containing training data
      cat_names: string column names
      test_df: pandas dataframe containing testing data

  Returns:
       df: transformed train dataframe
       test_df: transformed test dataframe
  """

  df[cat_names] = df[cat_names].fillna(0)
  if test_df is not None:
    test_df[cat_names] = test_df[cat_names].fillna(0)
  for col in cat_names:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    if test_df is not None:
      test_df[col] = enc.transform(test_df[col])
  return df, test_df


train_prepared_fixed_date = preprocess(train, store)

# Set training dataset to be samples in year 2014 and testing to 2015
train_inds = train_prepared_fixed_date[train_prepared_fixed_date['Year'] ==
                                       2014].index
test_inds = train_prepared_fixed_date[train_prepared_fixed_date['Year'] ==
                                      2015].index
train = train_prepared_fixed_date.iloc[train_inds]
test = train_prepared_fixed_date.iloc[test_inds]


# Adjusting string column attribuites
str_cat_columns = get_str_column_names(train_prepared_fixed_date)
train, test = fix_strs(train, str_cat_columns, test)

all_cat_names = ([
    'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'Promo2'
] + ['Promo2Start_' + month_abbr for month_abbr in month_abbrs])
# Sort data based on store, year, month and day
train.sort_values(by=['Store', 'Year', 'Month', 'Day'], inplace=True)
test.sort_values(by=['Store', 'Year', 'Month', 'Day'], inplace=True)

# Save data
preprocessed_dataset_path = './'
train.to_csv(
    os.path.join(preprocessed_dataset_path, 'train_processed.csv'),
    sep='\t',
    header=False,
    index=False)
test.to_csv(
    os.path.join(preprocessed_dataset_path, 'test_processed.csv'),
    sep='\t',
    header=False,
    index=False)

with open(os.path.join(preprocessed_dataset_path, 'cd'), 'w') as cd:
  for idx, name in enumerate(train.columns):
    cd.write('{}\t{}\n'.format(
        idx, 'Label' if name == 'Sales' else
        ('Categ\t' + name if name in all_cat_names else 'Num\t' + name)))
