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

"""Preprocess Penn-Treebank dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np


def main():
  with open('train.txt') as finp:
    lines = finp.read().strip().replace('\n', '<eos>')
    words = lines.split(' ')

  vocab, index = {}, {}
  for word in sorted(words):
    if word not in vocab:
      index[len(vocab)] = word
      vocab[word] = len(vocab)
  print('vocab size: {}'.format(len(vocab)))

  x_train = [vocab[word] for word in words] + [vocab['<eos>']]
  x_train = np.array(x_train, dtype=np.int32)

  with open('valid.txt') as finp:
    lines = finp.read().strip().replace('\n', '<eos>')
    words = lines.split(' ')

  x_valid = [vocab[word] for word in words] + [vocab['<eos>']]
  x_valid = np.array(x_valid, dtype=np.int32)

  with open('test.txt') as finp:
    lines = finp.read().strip().replace('\n', '<eos>')
    words = lines.split(' ')

  x_test = [vocab[word] for word in words] + [vocab['<eos>']]
  x_test = np.array(x_test, dtype=np.int32)

  print('train size: {}'.format(np.size(x_train)))
  print('valid size: {}'.format(np.size(x_valid)))
  print('test size: {}'.format(np.size(x_test)))

  with open('ptb.pkl', 'wb') as fout:
    pickle.dump((x_train, x_valid, x_test, vocab, index), fout, protocol=2)


if __name__ == '__main__':
  main()
