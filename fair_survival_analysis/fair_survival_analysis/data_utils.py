# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Utilitites to preprocess SEER and SUPPORT datasets.

This module has utility functions to load and preprocess the SEER and SUPPORT
datasets for learning and evaluation of Survival Analysis methods. Preprocessing
involves standard scaling of the features, excluding the protected attribute,
imputing the missing values (for SEER) and removing outliers.

Not designed to be called directly, would be called when running a function from
fair_survival_analysis.fair_survival_analysis

"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_support(data, prot='sex', fair_strategy=None):
  """Function to preprocess a loaded SUPPORT dataset.

  Imputation and standard rescaling is performed.

  Args:
    data:
      pandas dataframe of the loaded SUPPORT csv.
    prot:
      a string to idetify the name of the column containing the protected
      attributes.
    fair_strategy:
     Fairness strategy to be incorporated while preprocessing. Can be one of
     None, "unawaereness" or "coupled".

  Returns:
    a tuple of the shape of (x, t, e, a). Where 'x' are the features, t is the
    event (or censoring) times, e is the censoring indicator and a is a vector
    of the protected attributes.

  """

  x1 = data[[
      'age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp', 'pafi', 'alb',
      'bili', 'crea', 'sod', 'ph', 'glucose', 'bun', 'urine', 'adlp', 'adls'
  ]]

  catfeats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']

  if fair_strategy in ['unawareness', 'coupled']:

    print('Note: For ' + fair_strategy, ' the protected feature is removed...')

    catfeats = set(catfeats)
    catfeats = list(catfeats - set([prot]))

  x2 = pd.get_dummies(data[catfeats])

  x = np.concatenate([x1, x2], axis=1)

  t = data['d.time'].values

  e = data['death'].values

  # Imputation

  x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)

  # Standard scaling

  x = StandardScaler().fit_transform(x)

  a = data[prot].values

  remove = ~np.isnan(t)

  return x[remove], t[remove], e[remove], a[remove]


def preprocess_seer(data, prot='RACE1V', fair_strategy=None):
  """Function to preprocess a loaded SEER dataset.

  Imputation and standard rescaling is performed.

  Args:
    data:
      pandas dataframe of the loaded SEER csv.
    prot:
      a string to idetify the name of the column containing the protected
      attributes.
    fair_strategy:
     Fairness strategy to be incorporated while preprocessing. Can be one of
     None, "unawaereness" or "coupled".

  Returns:
    a tuple of the shape of (x, t, e, a). Where 'x' are the features, t is the
    event (or censoring) times, e is the censoring indicator and a is a vector
    of the protected attributes.

  """

  keep = (data['X']['RACE1V'] == '01') | (data['X']['RACE1V'] == '02')

  x = data['X'][keep]
  t = data['T'][keep]
  e = data['E'][keep]
  a = data['A'][keep]

  catfeats = data['X'].columns.tolist()

  if fair_strategy in ['unawareness', 'coupled']:

    print('Note: For ' + fair_strategy, ' the protected feature is removed...')

    catfeats = set(catfeats)
    catfeats = list(catfeats - set([prot]))

  x = x[catfeats]

  x = x.values
  t = t.astype(float)[:, 0]
  e = e.astype(float)[:, 0]
  a = a.values

  x = StandardScaler().fit_transform(x)

  remove = t > 300

  return x[~remove], t[~remove], e[~remove], a[~remove]
