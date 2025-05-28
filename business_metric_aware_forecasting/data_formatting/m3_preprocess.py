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

"""Script to pre-proccess M3 dataset."""

import pandas as pd
from sklearn.utils import shuffle


def main():
  df = pd.read_excel('../data/m3/M3C.xls', 'M3Month')
  df['Category'] = df['Category'].str.strip()
  df = shuffle(df[df['Category'] == 'INDUSTRY'])
  df.to_csv('../data/m3/m3_industry_monthly_shuffled.csv', index=False)


if __name__ == '__main__':
  main()
