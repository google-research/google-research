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

r"""Process ROCStories data into tfrecords.

  process_rocstories -- --raw_dir=$ROC_FILES \
      --output_base=/tmp/rocstories_spring_winter_train --vocab_file=$VOCAB

We produce 20 shards of data, and will use shards 0-17 for train, 18 for valid,
19 for test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf  # tf

from summae import data_util
from summae import util

FLAGS = flags.FLAGS

flags.DEFINE_string('raw_dir', '',
                    'paths to rocstories raw data.')
flags.DEFINE_integer('num_shards', 20, 'Number of shards')
flags.DEFINE_string('output_base', '',
                    ('Will output output_base.x.tfrecord for'
                     'x=0, ... num_shards-1.'))
flags.DEFINE_string('vocab_file', '',
                    'If specified, encode using t2t SubwordTextEncoder.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  paths = [os.path.join(FLAGS.raw_dir, x) for x in [
      'ROCStories__spring2016 - ROCStories_spring2016.csv',
      'ROCStories_winter2017 - ROCStories_winter2017.csv']]

  assert paths, FLAGS.raw_dir
  logging.info('Reading from: %s', paths)
  logging.info('Loading vocabulary file from %s', FLAGS.vocab_file)
  tk = util.get_tokenizer(FLAGS.vocab_file)
  assert tk

  writers = data_util.get_filewriters(FLAGS.output_base, 'all',
                                      FLAGS.num_shards)
  sharder = data_util.get_text_sharder(FLAGS.num_shards)
  count = 0

  for p in paths:
    logging.info('Opening %s', p)
    with tf.gfile.Open(p) as f:
      reader = csv.reader(f)
      next(reader)  # Header
      for r in reader:
        assert len(r) == 7
        storyid = r[0]
        storytitle = r[1]
        sentences = r[2:7]
        context_features = tf.train.Features(feature={
            'storyid': data_util.tf_bytes_feature(storyid),
            'storytitle': data_util.tf_bytes_feature(storytitle),
        })
        seq_ex = data_util.sents2seqex(sentences,
                                       tk,
                                       context_features=context_features,
                                       add_eos=False,
                                       add_untokenized=True)
        writers[sharder(storyid)].write(seq_ex.SerializeToString())
        count += 1
  data_util.close_writers(writers)
  logging.info('Wrote %d records to %d shards.', count, FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
