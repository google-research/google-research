# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Run Classifier to score all data.

Data scorer wtih classifier.

This file is intended for a dataset that is split into 14 chunks.
"""

import csv
import os
import pickle
from typing import Sequence

from absl import app
from absl import flags
from scipy.special import softmax
import tensorflow as tf
import transformers

tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'save_dir', default=None,
    help='Directory to store scores data.')
flags.DEFINE_integer(
    'slice', default=0,
    help='Which slice of data to process.')
flags.DEFINE_string(
    'bert_base_dir', default=None,
    help='Directory of German BERT.')
flags.DEFINE_string(
    'bert_clf_dir', default=None,
    help='Directory of German BERT domain classifier.')
flags.DEFINE_string(
    'target_text', default=None,
    help='Filename with target text. This data will be labeled by model.')


PROC_SIZE = 300000


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Grab pretrain text data
  targets_decoded_pt = []
  for i in range(1, 9):
    with tf.io.gfile.GFile(FLAGS.target_text % i, 'rb') as f:
      pt_targs_tmp = pickle.load(f)
    targets_decoded_pt.extend(pt_targs_tmp)

  # Load model
  cache_dir = '/tmp/'  # model weights get temporarily written to this directory
  path = FLAGS.bert_base_dir
  trained_path = FLAGS.bert_clf_dir
  config = transformers.BertConfig.from_pretrained(
      os.path.join(trained_path, 'config.json'), num_labels=2,
      cache_dir=cache_dir)
  tokenizer = transformers.BertTokenizer.from_pretrained(
      path, cache_dir=cache_dir)
  model = transformers.TFBertForSequenceClassification.from_pretrained(
      os.path.join(trained_path, 'tf_model.h5'), config=config,
      cache_dir=cache_dir)

  start = PROC_SIZE * FLAGS.slice
  end = PROC_SIZE * (FLAGS.slice + 1)
  if FLAGS.slice == 14:
    end = 9999999
  encoding = tokenizer(
      targets_decoded_pt[start:end],
      return_tensors='tf',
      padding=True,
      truncation=True,
      max_length=512)

  train_dataset = tf.data.Dataset.from_tensor_slices((
      dict(encoding),
  ))
  train_dataset = train_dataset.batch(256)
  logits = model.predict(train_dataset)

  probs = softmax(logits.logits, axis=1)

  clf_score_name = FLAGS.save_dir + '/CLR_scores_' + str(FLAGS.slice) + '.csv'
  with tf.io.gfile.GFile(clf_score_name, 'w') as f:
    writer = csv.writer(f)
    for p in probs:
      writer.writerow([p[1]])


if __name__ == '__main__':
  app.run(main)
