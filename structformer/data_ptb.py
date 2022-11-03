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

"""PTB style parsed corpus."""

import os
import pickle
import re

import nltk
from nltk.corpus import ptb

WORD_TAGS = [
    'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
    'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
    'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
    'WP$', 'WRB'
]
CURRENCY_TAGS_WORDS = ['#', '$', 'C$', 'A$']
ELLIPSIS = [
    '*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*'
]
PUNCTUATION_TAGS = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
PUNCTUATION_WORDS = [
    '.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!',
    '...', '-LCB-', '-RCB-'
]


class Corpus(object):
  """PTB style parsed corpus."""

  def __init__(self, dict_path):
    """Initialization.

    Args:
      dict_path: path to dictionary folder
    Raises:
      Exception: missing dictionary
    """

    dict_file_name = os.path.join(dict_path, 'dict.pkl')
    if os.path.exists(dict_file_name):
      self.dictionary = pickle.load(open(dict_file_name, 'rb'))
    else:
      raise Exception

    all_file_ids = ptb.fileids()
    train_file_ids = []
    valid_file_ids = []
    test_file_ids = []
    rest_file_ids = []
    for file_id in all_file_ids:
      if 'WSJ/00/WSJ_0200.MRG' <= file_id <= 'WSJ/21/WSJ_2199.MRG':
        train_file_ids.append(file_id)
      if 'WSJ/22/WSJ_2200.MRG' <= file_id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(file_id)
      if 'WSJ/23/WSJ_2300.MRG' <= file_id <= 'WSJ/23/WSJ_2399.MRG':
        test_file_ids.append(file_id)
      elif ('WSJ/00/WSJ_0000.MRG' <= file_id <= 'WSJ/01/WSJ_0199.MRG') or \
          ('WSJ/24/WSJ_2400.MRG' <= file_id <= 'WSJ/24/WSJ_2499.MRG'):
        rest_file_ids.append(file_id)

    self.train, self.train_sens, self.train_trees, self.train_nltktrees \
        = self.tokenize(train_file_ids)
    self.valid, self.valid_sens, self.valid_trees, self.valid_nltktress \
        = self.tokenize(valid_file_ids)
    self.test, self.test_sens, self.test_trees, self.test_nltktrees \
        = self.tokenize(test_file_ids)
    self.rest, self.rest_sens, self.rest_trees, self.rest_nltktrees \
        = self.tokenize(rest_file_ids)

  def filter_words(self, tree):
    words = []
    for w, tag in tree.pos():
      if tag in WORD_TAGS:
        w = w.lower()
        w = re.sub('[0-9]+', 'N', w)
        words.append(w)
    return words

  def add_words(self, file_ids):
    for file_id_i in file_ids:
      sentences = ptb.parsed_sents(file_id_i)
      for sen_tree in sentences:
        words = self.filter_words(sen_tree)
        for word in words:
          self.dictionary.add_word(word)

  def tokenize(self, file_ids):
    """Tokenizes a mrg file."""

    def tree2list(tree):
      if isinstance(tree, nltk.Tree):
        if tree.label() in WORD_TAGS:
          w = tree.leaves()[0].lower()
          w = re.sub('[0-9]+', 'N', w)
          return w
        else:
          root = []
          for child in tree:
            c = tree2list(child)
            if c:
              root.append(c)
          if len(root) > 1:
            return root
          elif len(root) == 1:
            return root[0]
      return []

    sens_idx = []
    sens = []
    trees = []
    nltk_trees = []
    for file_id_i in file_ids:
      sentences = ptb.parsed_sents(file_id_i)
      for sen_tree in sentences:
        words = self.filter_words(sen_tree)
        sens.append(words)
        idx = []
        for word in words:
          idx.append(self.dictionary[word])
        sens_idx.append(idx)
        trees.append(tree2list(sen_tree))
        nltk_trees.append(sen_tree)

    return sens_idx, sens, trees, nltk_trees
