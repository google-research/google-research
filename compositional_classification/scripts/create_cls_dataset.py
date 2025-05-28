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

# coding=utf-8
"""Functions to create the classification dataset."""

import collections
import json
import os
import random

from absl import app
from absl import flags
from absl import logging
import cfq_parse_util
import ijson
import tensorflow as tf

FLAGS = flags.FLAGS
Node = cfq_parse_util.Node

flags.DEFINE_string('cfq_root', None, 'Path to the CFQ dataset')
flags.DEFINE_string('split_file', 'random_split', 'Name of the dataset split '
                    'in CFQ')
flags.DEFINE_string('output_dir', None, 'Path to save dataset')
flags.DEFINE_enum('negative_example', 'random', ['random', 'model_output'],
                  'Method to generate negative examples')
flags.DEFINE_string('model_output_dir', None, 'Path to baseline output dir')
flags.DEFINE_integer('max_neg', 3, 'Number of neg examples per CFQ pair')
flags.DEFINE_boolean('sort_by_score', True,
                     'Whether to sort outputs of different models by scores')
flags.DEFINE_boolean(
    'train_hold_out', True,
    'Whether to set the 5 percent of train set aside as holdout')

flags.mark_flag_as_required('cfq_root')
flags.mark_flag_as_required('output_dir')
flags.register_validator('cfq_root', os.path.exists, 'CFQ dataset not found')


def parse_cfq_json(cfq_dataset_fpath):
  """Parses CFQ dataset file to produce list of (question, query) pairs."""
  cfq_data, question, query = [], '', ''
  with open(cfq_dataset_fpath) as fd:
    # Iteratively parses json file to save memory.
    parser = ijson.parse(fd)
    for prefix, _, value in parser:
      if prefix == 'item.questionPatternModEntities':
        question = value
      elif prefix == 'item.sparqlPatternModEntities':
        query = value
      if question and query:
        cfq_data.append(cfq_parse_util.preprocess(question, query))
        question, query = '', ''
  return cfq_data


def neg_random_sample(pos, sample_pool):
  """Randomly samples an element, not same as pos."""
  # Simple safety check. (It's not perfect. e.g. sample_pool == [pos] * 2)
  assert sample_pool and not (len(sample_pool) == 1 and sample_pool[0] == pos)

  while True:
    neg = random.choice(sample_pool)
    if pos != neg:
      return neg


def generate_dataset_random(
    pos_data):
  """Generates a balanced cls dataset with random negative examples."""
  all_queries = list(zip(*pos_data))[1]

  dataset = []
  for question, query in pos_data:
    dataset.append((question, query, 1))
    selected_queries = [query]
    for _ in range(FLAGS.max_neg):
      while True:
        neg_query = neg_random_sample(query, all_queries)
        if neg_query not in selected_queries:
          dataset.append((question, neg_query, 0))
          selected_queries.append(neg_query)
          break
  return dataset


def neg_model_output(
    question, pos_query,
    model_output_dict):
  """Returns negative examples from model output, or None if not available."""
  if question not in model_output_dict:
    return None

  def check_qr_duplicate(query, qr_tree,
                         ref_queries):
    """Returns True if the input query does not overlap with the ref list."""
    for ref_query, ref_qr_tree, _ in ref_queries:
      if cfq_parse_util.sparql_query_synonym(ref_query, query, ref_qr_tree,
                                             qr_tree):
        return False
    return True

  # Model outputs are a dictionary of model name and (query, score) pairs.
  model_outputs = model_output_dict[question]

  # Model negatives can be chosen either by sorting scores or by rotation
  pos_qr_tree = cfq_parse_util.parse_sparql_query(pos_query)
  if FLAGS.sort_by_score:
    # Sort all outputs by their scores.
    model_outputs_all = []
    for model_name, pairs in model_outputs.items():
      for pair in pairs:
        model_outputs_all.append(pair + (model_name,))
    model_outputs_all.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    neg_outputs = [(query, model_name)
                   for (query, score, model_name) in model_outputs_all
                   if query != pos_query]
    # Select up to max_neg queries, filtering invalid and duplicates
    ref_neg_outputs, neg_idx = [(pos_query, pos_qr_tree, 'positive')], 0
    while len(ref_neg_outputs) < FLAGS.max_neg + 1:
      neg_query, neg_model_name = neg_outputs[neg_idx]
      try:
        neg_qr_tree = cfq_parse_util.parse_sparql_query(neg_query)
        if check_qr_duplicate(neg_query, neg_qr_tree, ref_neg_outputs):
          ref_neg_outputs.append((neg_query, neg_qr_tree, neg_model_name))
      except (AssertionError, ValueError):
        # Failed to parse query. Move on to the next query.
        pass
    # Remove the first query (positive). Remove query parse tree.
    valid_neg_outputs = [(x[0], x[2]) for x in ref_neg_outputs[1:]]
  else:
    # Rotating models, choose the best remaining output of the model
    model_names, model_idx = cfq_parse_util.CFQ_BASELINE_ORDER, 0
    model_neg_idxs = {name: 0 for name in model_names
                     }  # Output index in each model
    total_model_negs = sum([len(model_outputs[model]) for model in model_names])
    ref_neg_outputs = [(pos_query, pos_qr_tree, 'positive')]
    # Select up to max_neg queries, filtering invalid and overlapping queries
    while (len(ref_neg_outputs) < FLAGS.max_neg + 1 and
           sum(model_neg_idxs.values()) < total_model_negs):
      model_name = model_names[model_idx]
      while model_neg_idxs[model_name] < len(model_outputs[model_name]):
        neg_query, _ = model_outputs[model_name][model_neg_idxs[model_name]]
        model_neg_idxs[
            model_name] += 1  # Increase output idx of the model before parsing
        try:
          neg_qr_tree = cfq_parse_util.parse_sparql_query(neg_query)
          if check_qr_duplicate(neg_query, neg_qr_tree, ref_neg_outputs):
            ref_neg_outputs.append((neg_query, neg_qr_tree, model_name))
            break
        except (AssertionError, ValueError):
          # Failed to parse query. Move on to the next query.
          pass
      model_idx = (model_idx + 1) % len(model_names)  # Move to the next model
    # Remove the first query (positive). Remove query parse tree.
    valid_neg_outputs = [(x[0], x[2]) for x in ref_neg_outputs[1:]]

  return valid_neg_outputs


def generate_dataset_model(
    pos_data):
  """Generates a cls dataset with CFQ baseline outputs."""
  model_output_dict = cfq_parse_util.load_model_output(FLAGS.model_output_dir)
  dataset = []
  model_counter = collections.Counter()
  for question, pos_query in pos_data:
    neg_queries = neg_model_output(question, pos_query, model_output_dict)
    dataset.append((question, pos_query, 1))
    for neg_query, model_name in neg_queries:
      dataset.append((question, neg_query, 0))
      model_counter[model_name] += 1
  logging.info('Model negative statistics: %s.', model_counter.most_common())
  return dataset


def create_example(
    question, query, label,
    text_encoder):
  """Creates a tf.train.Example after converting sentences into ids."""
  example = collections.OrderedDict()
  example['question'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=text_encoder.encode(question)))
  example['query'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=text_encoder.encode(query)))
  example['label'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[label]))
  tf_example = tf.train.Example(features=tf.train.Features(feature=example))
  return tf_example


def write_dataset_split(cfq_data, idxs,
                        fpath,
                        text_encoder):
  """Creates dataset split from give indices and writes to file."""
  pos_data = [cfq_data[i] for i in idxs]
  logging.info('Use %s CFQ data', len(pos_data))
  if FLAGS.negative_example == 'random':
    cls_data = generate_dataset_random(pos_data)
  elif FLAGS.negative_example == 'model_output':
    cls_data = generate_dataset_model(pos_data)
  else:
    raise ValueError(f'Wrong negative sample method: {FLAGS.negative_example}')

  logging.info('Write %s examples to %s', len(cls_data), fpath)
  writer = tf.io.TFRecordWriter(fpath)
  for question, query, label in cls_data:
    tf_example = create_example(question, query, label, text_encoder)
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(argv):
  del argv  # Unused

  # Loads CFQ dataset and dataset split.
  cfq_dataset_fpath = os.path.join(FLAGS.cfq_root, 'dataset.json')
  cfq_split_fpath = os.path.join(FLAGS.cfq_root, 'splits', FLAGS.split_file)

  logging.info('Load CFQ dataset from %s. '
               'This takes a long time.', cfq_dataset_fpath)
  cfq_data = parse_cfq_json(cfq_dataset_fpath)

  logging.info('Load dataset split from %s', cfq_split_fpath)
  with open(cfq_split_fpath, 'r') as fd:
    data = json.load(fd)
    train_idxs = data['trainIdxs']
    dev_idxs = data['devIdxs']
    test_idxs = data['testIdxs']

  # Output paths
  train_fpath = os.path.join(FLAGS.output_dir, 'train.tfrecord')
  dev_fpath = os.path.join(FLAGS.output_dir, 'dev.tfrecord')
  test_fpath = os.path.join(FLAGS.output_dir, 'test.tfrecord')
  vocab_fpath = os.path.join(FLAGS.output_dir, 'vocab.txt')
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Creates vocaburary and writes to file.
  logging.info('Create vocabulary')
  word_counter = collections.Counter()
  for question, query in cfq_data:
    for w in question.split(' ') + query.split(' '):
      word_counter[w] += 1
  logging.info('%s words in total', len(word_counter))

  vocab = ['<pad>', '<EOS>', '<OOV>', '<SEP>', '<CLS>'] + list(
      word_counter.keys())
  logging.info('Write %s words to %s', len(vocab), vocab_fpath)
  with open(vocab_fpath, 'w') as fd:
    fd.write('\n'.join(vocab))

  # Creates tokenizer from the vocabulary.
  text_encoder = cfq_parse_util.load_text_encoder(vocab_fpath)

  # Writes dataset into tfrecord files. Negative samples are generated.
  random.seed(123)  # Random seed for reproducibility
  if not FLAGS.train_hold_out:
    # Use all train dataset
    write_dataset_split(cfq_data, train_idxs, train_fpath, text_encoder)
  else:
    # 5% of the train dataset is kept as train_holdout set
    num_hold_out = len(train_idxs) // 20
    random.shuffle(train_idxs)
    train_train_idxs = train_idxs[:-num_hold_out]
    train_hold_idxs = train_idxs[-num_hold_out:]
    train_hold_fpath = os.path.join(FLAGS.output_dir, 'train_holdout.tfrecord')
    write_dataset_split(cfq_data, train_train_idxs, train_fpath, text_encoder)
    write_dataset_split(cfq_data, train_hold_idxs, train_hold_fpath,
                        text_encoder)
  write_dataset_split(cfq_data, dev_idxs, dev_fpath, text_encoder)
  write_dataset_split(cfq_data, test_idxs, test_fpath, text_encoder)

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
