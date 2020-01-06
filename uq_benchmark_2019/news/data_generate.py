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

r"""Generate encoded np arrays of 20 news groups for in-distribution and OODs.

The topics of the 20 news groups
    # 1. alt.atheism
    # 2. talk.politics.guns
    # 3. talk.politics.mideast
    # 4. talk.politics.misc
    # 5. talk.religion.misc
    # 6. soc.religion.christian

    # 7. comp.sys.ibm.pc.hardware
    # 8. comp.graphics
    # 9. comp.os.ms-windows.misc
    # 10. comp.sys.mac.hardware
    # 11. comp.windows.x

    # 12. rec.autos
    # 13. rec.motorcycles
    # 14. rec.sport.baseball
    # 15. rec.sport.hockey

    # 16. sci.crypt
    # 17. sci.electronics
    # 18. sci.space
    # 19. sci.med

    # 20. misc.forsale

The original raw dataset is downloaded from Dan Hendrycks's github page
https://github.com/hendrycks/error-detection/tree/master/NLP/Categorization/data

The program does the following steps for generating the dataset:
  1. Encode words into integers.
    Total vocabulary size is ~70,000, we truncated it into 30,000
    0 (padding), 1, 2, ..., 30,000 (unknown)
  2. Truncate text into 250 words
    mean=205.799433277, std=413.26162286
    text > 250, truncate from the beginning
    text < 250, padding from the beginning
  3. Split the dataset into in- and out-of-distribution
    in-distribution: even number classes,
    OOD: odd number classes
  4. relabel in-distribution from 0,2,4... to 0,1,2, for the ease of
  encoding labels while training classifier

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import random
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from uq_benchmark_2019.news import data_utils
from uq_benchmark_2019.news import data_utils_from_hendrycks as data_utilsh

# parameters
FLAGS = flags.FLAGS

flags.DEFINE_float('tr_frac', 0.9, 'The fraction of data used for training')
flags.DEFINE_integer('random_seed', 1234, 'The random seed')
flags.DEFINE_string('tr_data_file', '20ng-train.txt', 'text for training')
flags.DEFINE_string('test_data_file', '20ng-train.txt', 'text for testing')
flags.DEFINE_string('out_dir', '20news_data', 'directory to output dataset')
flags.DEFINE_integer('fix_len', 250, 'sequence length')
flags.DEFINE_integer('vocab_size', 30000, 'vocab size')
flags.DEFINE_integer('n_class', 20, 'number of classes in total')
flags.DEFINE_boolean('shortfrag', False, 'if fragment text into short pieces')
flags.DEFINE_float(
    'sample_rate', 0.02,
    'sampling rate for short fragments, only used when shortfrag=True')
flags.DEFINE_integer(
    'filter_label', -1,
    ('If only sequences from the class=filter_label are used for training.'
     '-1 means no filter.'))

FLAGS = flags.FLAGS


def make_dataset(params):
  """Make np arrays for 20 news groups."""

  # odd number classes are in-distribution, even number classes are OODs.
  if params['filter_label'] == -1:
    to_include = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
  else:
    to_include = [params['filter_label']]
  to_exclude = list(set(range(params['n_class'])).difference(to_include))
  logging.info('exclude classes=%s', to_exclude)

  logging.info('Loading raw data')
  x_train, y_train = data_utilsh.load_data(params['tr_data_file'])
  x_test, y_test = data_utilsh.load_data(params['test_data_file'])

  logging.info('Get vocab and encode words to ints')
  # vocab is a dict ordered by word freqs
  vocab = data_utilsh.get_vocab(x_train)
  # words with top vocab_size-1 freqs are encoded as 1 to vocab_size-1,
  # words with less freqs are encoded as vocab_size for unknown token
  # sentences > max_len is truncated from the beginning,
  # sentences < max_len is padded from the beginning with 0.
  # so the total vocab = vocab_size-1 (specific words) + 1 (unk) + 1 (padding)
  x_train = data_utilsh.text_to_rank(x_train, vocab, params['vocab_size'])
  x_test = data_utilsh.text_to_rank(x_test, vocab, params['vocab_size'])

  # shuffle
  np.random.seed(params['random_seed'])
  indices = np.arange(len(x_train))
  np.random.shuffle(indices)
  x_train = [x_train[i] for i in indices]
  y_train = [y_train[i] for i in indices]

  indices = np.arange(len(x_test))
  np.random.shuffle(indices)
  x_test = [x_test[i] for i in indices]
  y_test = [y_test[i] for i in indices]

  # split into train/dev
  n_dev = int(len(x_train) * (1 - params['tr_frac']))
  x_dev = x_train[-n_dev:]
  y_dev = y_train[-n_dev:]
  x_train = x_train[:-n_dev]
  y_train = y_train[:-n_dev]

  # if fragment text into short pieces
  if params['shortfrag']:
    logging.info('sampling sub-text.')
    x_train, y_train = data_utils.fragment_into_short_sentence(
        x_train, y_train, params['fix_len'], params['sample_rate'])
    logging.info('x_train_frag=%s', x_train[0])

    x_dev, y_dev = data_utils.fragment_into_short_sentence(
        x_dev, y_dev, params['fix_len'], params['sample_rate'])
    logging.info('x_dev_frag=%s', x_dev[0])

    x_test, y_test = data_utils.fragment_into_short_sentence(
        x_test, y_test, params['fix_len'], params['sample_rate'])
    logging.info('x_test_frag=%s', x_test[0])

  else:
    logging.info('pad original text with 0s.')
    # pad text to achieve the same length
    x_train = data_utilsh.pad_sequences(x_train, maxlen=params['fix_len'])
    x_dev = data_utilsh.pad_sequences(x_dev, maxlen=params['fix_len'])
    x_test = data_utilsh.pad_sequences(x_test, maxlen=params['fix_len'])

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)

  # partition data into in-distribution and OODs by their labels
  in_sample_examples, in_sample_labels, oos_examples, oos_labels =\
  data_utilsh.partion_data_in_two(x_train, np.array(y_train), to_include, to_exclude)
  dev_in_sample_examples, dev_in_sample_labels, dev_oos_examples, dev_oos_labels =\
  data_utilsh.partion_data_in_two(x_dev, np.array(y_dev), to_include, to_exclude)
  test_in_sample_examples, test_in_sample_labels, test_oos_examples, test_oos_labels =\
  data_utilsh.partion_data_in_two(x_test, np.array(y_test), to_include, to_exclude)

  class_freq = np.bincount(in_sample_labels)
  logging.info('in_sample_labels_freq=%s', class_freq)

  class_freq = np.bincount(dev_in_sample_labels)
  logging.info('dev_in_sample_labels_freq=%s', class_freq)

  class_freq = np.bincount(dev_oos_labels)
  logging.info('dev_oos_labels_freq=%s', class_freq)

  # relabel in-distribution from 0,2,4... to 0,1,2, for encoding labels
  # when training classifier
  # safely assumes there is an example for each in_sample class i
  # n both the training and dev class
  in_sample_labels = data_utilsh.relabel_in_sample_labels(in_sample_labels)
  dev_in_sample_labels = data_utilsh.relabel_in_sample_labels(
      dev_in_sample_labels)
  test_in_sample_labels = data_utilsh.relabel_in_sample_labels(
      test_in_sample_labels)

  logging.info('# word id>15000=%s', np.sum(in_sample_labels > 15000))
  logging.info(
      'n_tr_in=%s, n_val_in=%s, n_val_ood=%s, n_test_in=%s, n_test_ood=%s',
      in_sample_labels.shape[0], dev_in_sample_labels.shape[0],
      dev_oos_labels.shape[0], test_in_sample_labels.shape[0],
      test_oos_labels.shape[0])
  logging.info('example in_sample_examples1=%s, \n in_sample_examples2=%s',
               in_sample_examples[0], in_sample_examples[1])

  ## save to disk
  if params['shortfrag']:
    # if fragment text into fix-length short pieces,
    # we subsample short pieces with sample_rate
    # so the data file name has the this parameter
    out_file_name = '20news_encode_maxlen{}_vs{}_rate{}_in{}_trfrac{}.pkl'.format(
        params['fix_len'], params['vocab_size'], params['sample_rate'],
        '-'.join([str(x) for x in to_include]), params['tr_frac'])
  else:
    # if we do not fragment text, we use all text examples
    # Given a fixed length, pad text if it is shorter than the fixed length,
    # truncate text if it is longer than the fixed length.
    out_file_name = '20news_encode_maxlen{}_vs{}_in{}_trfrac{}.pkl'.format(
        params['fix_len'], params['vocab_size'],
        '-'.join([str(x) for x in to_include]), params['tr_frac'])
  with tf.gfile.Open(os.path.join(params['out_dir'], out_file_name), 'wb') as f:
    pickle.dump(in_sample_examples, f)
    pickle.dump(in_sample_labels, f)
    pickle.dump(oos_examples, f)
    pickle.dump(oos_labels, f)

    pickle.dump(dev_in_sample_examples, f)
    pickle.dump(dev_in_sample_labels, f)
    pickle.dump(dev_oos_examples, f)
    pickle.dump(dev_oos_labels, f)

    pickle.dump(test_in_sample_examples, f)
    pickle.dump(test_in_sample_labels, f)
    pickle.dump(test_oos_examples, f)
    pickle.dump(test_oos_labels, f)
    pickle.dump(vocab, f)


def main(_):

  random.seed(FLAGS.random_seed)

  params = {
      'tr_data_file': FLAGS.tr_data_file,
      'test_data_file': FLAGS.test_data_file,
      'filter_label': FLAGS.filter_label,
      'fix_len': FLAGS.fix_len,
      'n_class': FLAGS.n_class,
      'vocab_size': FLAGS.vocab_size,
      'out_dir': FLAGS.out_dir,
      'tr_frac': FLAGS.tr_frac,
      'random_seed': FLAGS.random_seed,
      'shortfrag': FLAGS.shortfrag,
      'sample_rate': FLAGS.sample_rate,
  }

  if not tf.gfile.Exists(params['out_dir']):
    tf.gfile.MkDir(params['out_dir'])

  logging.info('params=%s', params)
  make_dataset(params)


if __name__ == '__main__':
  app.run(main)
