# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Functions for preprocessing text.

Functions are copied and modified based on
https://raw.githubusercontent.com/hendrycks/error-detection/master/NLP/Categorization/20%20Newsgroups.ipynb
by Dan Hendrycks

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import numpy as np
import tensorflow as tf

# disable pylint for keeping the original code from Hendrycks.
# pylint: disable=bare-except
# pylint: disable=invalid-name
# pylint: disable=g-explicit-length-test
# pylint: disable=invalid-unary-operand-type
# pylint: disable=dangerous-default-value


def load_data(filename, stop_words=[]):
  """Load the raw dataset."""

  # Differently from Hendrycks's code, we don't throw away stop words,

  x, y = [], []
  with tf.gfile.Open(filename, 'r') as f:
    for line in f:
      line = re.sub(r'\W+', ' ', line).strip()
      if line[1] == ' ':
        x.append(line[1:])
        y.append(line[0])
      else:
        x.append(line[2:])
        y.append(line[:2])
      x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
  return x, np.array(y, dtype=int)


def get_vocab(dataset):
  """Count words and build vocab."""
  vocab = {}

  # create a counter for each word
  for example in dataset:
    example_as_list = example.split()
    for word in example_as_list:
      vocab[word] = 0

  for example in dataset:
    example_as_list = example.split()
    for word in example_as_list:
      vocab[word] += 1

  # sort from greatest to least by count
  return collections.OrderedDict(
      sorted(vocab.items(), key=lambda x: x[1], reverse=True))


def text_to_rank(dataset, _vocab, desired_vocab_size=15000):
  """Encode words to ids.

  Args:
    dataset: the text from load_data
    _vocab: a _ordered_ dictionary of vocab words and counts from get_vocab
    desired_vocab_size: the desired vocabulary size. words no longer in vocab
      become unk

  Returns:
    the text corpus with words mapped to their vocab rank,
    with all sufficiently infrequent words mapped to unk;
    unk has rank desired_vocab_size.
    (the infrequent word cutoff is determined by desired_vocab size)
  """
  # pylint: disable=invalid-name
  _dataset = dataset[:]  # aliasing safeguard
  vocab_ordered = list(_vocab)
  count_cutoff = _vocab[vocab_ordered[
      desired_vocab_size - 1]]  # get word by its rank and map to its count

  word_to_rank = {}
  for i in range(len(vocab_ordered)):
    # we add one to make room for any future padding symbol with value 0
    word_to_rank[vocab_ordered[i]] = i + 1

  for i in range(len(_dataset)):
    example = _dataset[i]
    example_as_list = example.split()
    for j in range(len(example_as_list)):
      try:
        if _vocab[example_as_list[j]] >= count_cutoff and word_to_rank[
            example_as_list[j]] < desired_vocab_size:
          # we need to ensure that other words below the word
          # on the edge of our desired_vocab size
          # are not also on the count cutoff
          example_as_list[j] = word_to_rank[example_as_list[j]]
        else:
          example_as_list[j] = desired_vocab_size  # UUUNNNKKK
      # pylint: disable=bare-except
      except:
        example_as_list[j] = desired_vocab_size  # UUUNNNKKK
    _dataset[i] = example_as_list

  return _dataset


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
  """Pads each sequence to the same length.

  If maxlen is provided, any sequence longer
  than maxlen is truncated to maxlen.
  Truncation happens off either the beginning (default) or
  the end of the sequence.
  Supports post-padding and pre-padding (default).

  Args:
    sequences: list of lists where each element is a sequence
    maxlen: int, maximum length
    dtype: type to cast the resulting sequence.
    padding: 'pre' or 'post', pad either before or after each sequence.
    truncating: 'pre' or 'post', remove values from sequences larger than maxlen
      either in the beginning or in the end of the sequence
    value: float, value to pad the sequences to the desired value.

  Returns:
    numpy array with dimensions (number_of_sequences, maxlen).
  """
  lengths = [len(s) for s in sequences]

  nb_samples = len(sequences)
  if maxlen is None:
    maxlen = np.max(lengths)

  # take the sample shape from the first non empty sequence
  # checking for consistency in the main loop below.
  sample_shape = tuple()
  for s in sequences:
    # pylint: disable=g-explicit-length-test
    if len(s) > 0:
      sample_shape = np.asarray(s).shape[1:]
      break

  x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
  for idx, s in enumerate(sequences):
    # pylint: disable=g-explicit-length-test
    if len(s) == 0:
      continue  # empty list was found
    if truncating == 'pre':
      # pylint: disable=invalid-unary-operand-type
      trunc = s[-maxlen:]
    elif truncating == 'post':
      trunc = s[:maxlen]
    else:
      raise ValueError('Truncating type "%s" not understood' % truncating)

    # check `trunc` has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
      raise ValueError(
          'Shape of sample %s of sequence at position %s is different from expected shape %s'
          % (trunc.shape[1:], idx, sample_shape))

    if padding == 'post':
      x[idx, :len(trunc)] = trunc
    elif padding == 'pre':
      x[idx, -len(trunc):] = trunc
    else:
      raise ValueError('Padding type "%s" not understood' % padding)
  return x


def partion_data_in_two(dataset, dataset_labels, in_sample_labels, oos_labels):
  """Partition dataset into in-distribution and OODs by labels.

  Args:
    dataset: the text from text_to_rank
    dataset_labels: dataset labels
    in_sample_labels: a list of newsgroups which the network will/did train on
    oos_labels: the complement of in_sample_labels; these newsgroups the network
      has never seen

  Returns:
    the dataset partitioned into in_sample_examples, in_sample_labels,
    oos_examples, and oos_labels in that order
  """
  _dataset = dataset[:]  # aliasing safeguard
  _dataset_labels = dataset_labels

  in_sample_idxs = np.zeros(np.shape(_dataset_labels), dtype=bool)
  ones_vec = np.ones(np.shape(_dataset_labels), dtype=int)
  for label in in_sample_labels:
    in_sample_idxs = np.logical_or(in_sample_idxs,
                                   _dataset_labels == label * ones_vec)

  oos_sample_idxs = np.zeros(np.shape(_dataset_labels), dtype=bool)
  for label in oos_labels:
    oos_sample_idxs = np.logical_or(oos_sample_idxs,
                                    _dataset_labels == label * ones_vec)

  return _dataset[in_sample_idxs], _dataset_labels[in_sample_idxs], _dataset[
      oos_sample_idxs], _dataset_labels[oos_sample_idxs]


# our network trains only on a subset of classes, say 6,
# but class number 7 might still
# be an in-sample label: we need to squish the labels to be in {0,...,5}
def relabel_in_sample_labels(labels):
  """Relabel in-distribution labels from 1,3,5 to 0,1,2 for training."""
  labels_as_list = labels.tolist()

  set_of_labels = []
  for label in labels_as_list:
    set_of_labels.append(label)
  labels_ordered = sorted(list(set(set_of_labels)))

  relabeled = np.zeros(labels.shape, dtype=int)
  for i in range(len(labels_as_list)):
    relabeled[i] = labels_ordered.index(labels_as_list[i])

  return relabeled
