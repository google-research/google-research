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
import jax
import numpy as np
from scipy.special import softmax
import tensorflow as tf
import transformers

from data_selection.wmt import decode
from data_selection.wmt import input_pipeline


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
flags.DEFINE_string(
    'dataset_name', default=None,
    help='Name of dataset if targets not provided.')
flags.DEFINE_string(
    'data_dir', default=None,
    help='Dataset dir if targets not provided.')
flags.DEFINE_string(
    'vocab_path', default=None,
    help='Vocab file if targets not provided.')
flags.DEFINE_bool(
    'split_tokenizer', default=False,
    help='Use 1 or 2 tokenizers if targets not provided.')
flags.DEFINE_bool(
    'clf_inputs', default=False,
    help='Classify the input language.')
flags.DEFINE_bool(
    'clf_targets', default=True,
    help='Classify the target language.')
flags.DEFINE_integer(
    'paracrawl_size', default=0,
    help='Number of examples to sample from paracrawl.')

PROC_SIZE = 300000


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Grab pretrain text data
  if FLAGS.target_text:
    targets_decoded_pt = []
    for i in range(1, 9):
      with tf.io.gfile.GFile(FLAGS.target_text % i, 'rb') as f:
        pt_targs_tmp = pickle.load(f)
      targets_decoded_pt.extend(pt_targs_tmp)
  else:
    train_ds, (encoder_in, encoder_tgt) = input_pipeline.get_wmt_is_datasets(
        n_devices=jax.local_device_count(),
        dataset_name=FLAGS.dataset_name,
        shard_idx=jax.process_index(),
        shard_count=jax.process_count(),
        data_dir=FLAGS.data_dir,
        vocab_path=FLAGS.vocab_path,
        target_vocab_size=32000,
        batch_size=1024,
        max_length=256,
        paracrawl_size=FLAGS.paracrawl_size,
        split_tokenizer=FLAGS.split_tokenizer)

    train_data = iter(train_ds)
    eos_id = decode.EOS_ID
    def decode_tokens(encoder, toks):
      valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
      return encoder.detokenize(valid_toks).numpy().decode('utf-8')
    targets = []
    inputs = []
    for x in train_data:
      trg = x['targets']._numpy()  # pylint:disable=protected-access
      ins = x['inputs']._numpy()  # pylint:disable=protected-access
      targets.append(trg)
      inputs.append(ins)

    # flatten targets_decoded_pt
    # pylint:disable=g-complex-comprehension
    targets_flat = [t for batch_t in targets for t in batch_t]
    inputs_flat = [t for batch_t in inputs for t in batch_t]
    # pylint:enable=g-complex-comprehension

    # decode only the slice for this one
    targets_decoded_pt = []
    start = PROC_SIZE * FLAGS.slice
    end = PROC_SIZE * (FLAGS.slice + 1)
    if FLAGS.slice == 14:
      end = 9999999
    for i, x in enumerate(targets_flat[start:end]):
      if FLAGS.clf_inputs:
        input_decode = decode_tokens(encoder_in, inputs_flat[i + start])
      if FLAGS.clf_targets:
        target_decode = decode_tokens(encoder_tgt, x)
      if FLAGS.clf_inputs and FLAGS.clf_targets:
        decode_tok = input_decode + ' [SEP] ' + target_decode
      else:
        decode_tok = target_decode if FLAGS.clf_targets else input_decode
      targets_decoded_pt.append(decode_tok)

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

  if FLAGS.target_text:
    # If we read the entire dataset from text, select the slice to encode
    start = PROC_SIZE * FLAGS.slice
    end = PROC_SIZE * (FLAGS.slice + 1)
    if FLAGS.slice == 14:
      end = 9999999
    input_targets = targets_decoded_pt[start:end]
  else:
    # the targets were decoded above so just use the ones that were decoded
    input_targets = targets_decoded_pt
  encoding = tokenizer(
      input_targets,
      return_tensors='tf',
      padding=True,
      truncation=True,
      max_length=512)

  train_dataset = tf.data.Dataset.from_tensor_slices((
      dict(encoding),
  ))
  batch_size = 256
  if FLAGS.clf_inputs and FLAGS.clf_targets:
    # multiling model is larger
    batch_size = 128
  train_dataset = train_dataset.batch(batch_size)
  logits = model.predict(train_dataset)

  probs = softmax(logits.logits, axis=1)

  clf_score_name = FLAGS.save_dir + '/CLR_scores_' + str(FLAGS.slice) + '.csv'
  with tf.io.gfile.GFile(clf_score_name, 'w') as f:
    writer = csv.writer(f)
    for p in probs:
      writer.writerow([p[1]])


if __name__ == '__main__':
  app.run(main)
