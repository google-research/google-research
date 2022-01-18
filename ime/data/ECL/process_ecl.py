# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""File for preprocessing the electricity dataset."""
import io
import urllib.request
import zipfile

import pandas as pd

# Download dataset from website
zipurl = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
with urllib.request.urlopen(zipurl) as zipresp:
  with zipfile.ZipFile(io.BytesIO(zipresp.read())) as zfile:
    zfile.extractall('.')
# Read dataset
data_ecl = pd.read_csv(
    'LD2011_2014.txt', parse_dates=True, sep=';', decimal=',', index_col=0)
# Resample dataset so it has 1h interval
data_ecl = data_ecl.resample('1h', closed='right').sum()
# Filter out instances with missing values
data_ecl = data_ecl.loc[:, data_ecl.cumsum(axis=0).iloc[8920] != 0]
data_ecl.index = data_ecl.index.rename('date')
data_ecl = data_ecl['2012':]
data_ecl.to_csv('ECL.csv')
