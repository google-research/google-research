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

"""Downloads and prepares the Poker dataset."""

import os
import wget

URL_TRAIN = 'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data'
URL_TEST = 'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data'


def main():

  if not os.path.exists('./data'):
    os.makedirs('./data')

  # We only demonstrate training for optimized hyperparameters here,
  # without validation.

  wget.download(URL_TRAIN, 'data/train_poker.csv')
  wget.download(URL_TEST, 'data/test_poker.csv')

if __name__ == '__main__':
  main()
