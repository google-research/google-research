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

"""Word-level language model corpus."""

import os
import pickle
import wget

urls = {
    'train':
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
    'valid':
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
    'test':
        'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
}


class Dictionary(object):
  """Dictionary for language model."""

  def __init__(self):
    self.word2idx = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
    self.idx2word = ['<unk>', '<pad>', '<mask>']
    self.word2frq = {}

  def add_word(self, word):
    if word not in self.word2idx:
      self.idx2word.append(word)
      self.word2idx[word] = len(self.idx2word) - 1
    if word not in self.word2frq:
      self.word2frq[word] = 1
    else:
      self.word2frq[word] += 1
    return self.word2idx[word]

  def __len__(self):
    return len(self.idx2word)

  def __getitem__(self, item):
    if item in self.word2idx:
      return self.word2idx[item]
    else:
      return self.word2idx['<unk>']

  def rebuild_by_freq(self, thd=3):
    """Prune low frequency words."""
    self.word2idx = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
    self.idx2word = ['<unk>', '<pad>', '<mask>']

    for k, v in self.word2frq.items():
      if v >= thd and (k not in self.idx2word):
        self.idx2word.append(k)
        self.word2idx[k] = len(self.idx2word) - 1

    print('Number of words:', len(self.idx2word))
    return len(self.idx2word)


class Corpus(object):
  """Word-level language model corpus."""

  def __init__(self, path, thd=0):
    """Initialization.

    Args:
      path: path to corpus location, the folder should include 'train.txt',
        'valid.txt' and 'test.txt'
      thd: tokens that appears less then thd times in train.txt will be replaced
        by <unk>
    """

    if not os.path.exists(path):
      os.mkdir(path)

    dict_file_name = os.path.join(path, 'dict.pkl')
    if os.path.exists(dict_file_name):
      print('Loading dictionary...')
      self.dictionary = pickle.load(open(dict_file_name, 'rb'))
      build_dict = False
    else:
      self.dictionary = Dictionary()
      build_dict = True

    train_path = os.path.join(path, 'train.txt')
    if not os.path.exists(train_path):
      wget.download(urls['train'], train_path)
    self.train = self.tokenize(
        train_path, build_dict=build_dict, thd=thd)

    valid_path = os.path.join(path, 'valid.txt')
    if not os.path.exists(valid_path):
      wget.download(urls['valid'], valid_path)
    self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    test_path = os.path.join(path, 'test.txt')
    if not os.path.exists(test_path):
      wget.download(urls['test'], test_path)
    self.test = self.tokenize(os.path.join(path, 'test.txt'))

    if build_dict:
      print('Saving dictionary...')
      dict_file_name = os.path.join(path, 'dict.pkl')
      pickle.dump(self.dictionary, open(dict_file_name, 'wb'))

  def tokenize(self, path, build_dict=False, thd=0):
    """Tokenizes a text file."""

    assert os.path.exists(path)

    if build_dict:
      # Add words to the dictionary
      with open(path, 'r') as f:
        for line in f:
          words = line.split()
          for word in words:
            self.dictionary.add_word(word)
      if thd > 1:
        self.dictionary.rebuild_by_freq(thd)

    # Tokenize file content
    ids_list = []
    with open(path, 'r') as f:
      for line in f:
        words = line.split()
        ids = []
        for word in words:
          ids.append(self.dictionary[word])
        ids_list.append(ids)

    return ids_list
