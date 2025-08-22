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
"""Function to create the tree-annotated classification dataset."""

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


def parse_cfq_json_tree(
    cfq_dataset_fpath):
  """Parses CFQ file to produce list of question, query, and parse trees."""
  cfq_data, question, query = [], '', ''
  with open(cfq_dataset_fpath) as fd, open(cfq_dataset_fpath) as fd2:
    # Iteratively parses json file to save memory.
    parser = ijson.parse(fd)
    rule_tree_iters = iter(ijson.items(fd2, 'item.ruleTree'))
    for prefix, _, value in parser:
      if prefix == 'item.questionPatternModEntities':
        question = value
      elif prefix == 'item.sparqlPatternModEntities':
        query = value

      if question and query:
        # Make question parse tree from ruleTree
        rule_tree = next(rule_tree_iters)
        qs_tree, qs_struct_tokens = cfq_parse_util.parse_question_rule_tree(
            rule_tree, question)
        cfq_parse_util.preprocess_question_tree(qs_tree, question)

        # Preprocess question and query as the CFQ original code
        question, query = cfq_parse_util.preprocess(question, query)

        # Make query parse tree from the preprocessed query
        qr_tree = cfq_parse_util.parse_sparql_query(query)

        cfq_data.append((question, qs_struct_tokens, qs_tree, query, qr_tree))
        question, query = '', ''
  return cfq_data


def neg_random_sample(
    pos_qr, sample_pool):
  """Randomly samples query with parse tree, not same as pos_qr."""
  # Simple safety check. (It's not perfect. e.g. sample_pool == [pos] * 2)
  assert sample_pool and not (len(sample_pool) == 1 and
                              sample_pool[0][0] == pos_qr)

  while True:
    neg_qr, neg_qr_tree = random.choice(sample_pool)
    if pos_qr != neg_qr:
      return neg_qr, neg_qr_tree


def generate_dataset_random(
    pos_data
):
  """Generates a balanced cls dataset with random negative examples."""
  all_queries = [(example[3], example[4]) for example in pos_data]

  dataset = []
  for qs, qs_st_tokens, qs_tree, pos_qr, pos_qr_tree in pos_data:
    dataset.append((qs, qs_st_tokens, qs_tree, pos_qr, pos_qr_tree, 1))
    selected_queries = [pos_qr]
    for _ in range(FLAGS.max_neg):
      while True:
        neg_qr, neg_qr_tree = neg_random_sample(pos_qr, all_queries)
        if neg_qr not in selected_queries:
          dataset.append((qs, qs_st_tokens, qs_tree, neg_qr, neg_qr_tree, 0))
          selected_queries.append(neg_qr)
          break
  return dataset


def neg_model_output(
    question, pos_query, pos_qr_tree,
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
  if FLAGS.sort_by_score:
    # Sort all outputs by their scores.
    model_outputs_all = []
    for model_name, pairs in model_outputs.items():
      model_outputs_all.extend([pair + (model_name,) for pair in pairs])

    model_outputs_all.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    neg_outputs = [(query, model_name)
                   for (query, _, model_name) in model_outputs_all
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
    valid_neg_outputs = ref_neg_outputs[1:]  # Remove the first query (positive)
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
    valid_neg_outputs = ref_neg_outputs[1:]  # Remove the first query (positive)

  return valid_neg_outputs


def generate_dataset_model(
    pos_data
):
  """Generates a cls dataset with CFQ baseline outputs."""
  model_output_dict = cfq_parse_util.load_model_output(FLAGS.model_output_dir)
  dataset = []
  model_counter = collections.Counter()
  for qs, qs_st_tokens, qs_tree, pos_qr, pos_qr_tree in pos_data:
    neg_queries = neg_model_output(qs, pos_qr, pos_qr_tree, model_output_dict)
    dataset.append((qs, qs_st_tokens, qs_tree, pos_qr, pos_qr_tree, 1))
    for neg_qr, neg_qr_tree, model_name in neg_queries:
      dataset.append((qs, qs_st_tokens, qs_tree, neg_qr, neg_qr_tree, 0))
      model_counter[model_name] += 1
  logging.info('Model negative statistics: %s.', model_counter.most_common())
  return dataset


def create_example(
    question, qs_struct_tokens, qs_tree, query,
    qr_tree, label,
    text_encoder,
    xlink_mapping):
  """Creates a tf.train.Example after converting sentences into ids."""
  question_ids = text_encoder.encode(question)
  query_ids = text_encoder.encode(query)
  question_group, query_group = cfq_parse_util.convert_to_xlink_group_ids(
      question_ids, query_ids, xlink_mapping)

  example = collections.OrderedDict()
  example['question'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=question_ids))
  example['question_structure_tokens'] = tf.train.Feature(
      int64_list=tf.train.Int64List(
          value=text_encoder.encode(' '.join(qs_struct_tokens))))
  example['question_tree'] = tf.train.Feature(
      int64_list=tf.train.Int64List(
          value=cfq_parse_util.node_to_edge_list(qs_tree)))
  example['question_group'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=question_group))
  example['query'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_ids))
  example['query_tree'] = tf.train.Feature(
      int64_list=tf.train.Int64List(
          value=cfq_parse_util.node_to_edge_list(qr_tree)))
  example['query_group'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_group))
  example['label'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[label]))
  tf_example = tf.train.Example(features=tf.train.Features(feature=example))
  return tf_example


def write_dataset_split(
    cfq_data, idxs, fpath,
    text_encoder,
    xlink_mapping):
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
  for example in cls_data:
    tf_example = create_example(*example, text_encoder, xlink_mapping)
    writer.write(tf_example.SerializeToString())
  writer.close()


def main(argv):
  del argv  # Unused

  # Loads CFQ dataset and dataset split.
  cfq_dataset_fpath = os.path.join(FLAGS.cfq_root, 'dataset.json')
  cfq_split_fpath = os.path.join(FLAGS.cfq_root, 'splits', FLAGS.split_file)

  logging.info('Load CFQ dataset from %s. '
               'This takes a long time.', cfq_dataset_fpath)
  cfq_data = parse_cfq_json_tree(cfq_dataset_fpath)

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
  xlink_map_fpath = os.path.join(FLAGS.output_dir, 'xlink_mapping.pkl')
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Creates vocaburary and writes to file.
  logging.info('Create vocabulary')
  word_counter = collections.Counter()
  for question, qs_struct_tokens, _, query, _ in cfq_data:
    for w in question.split(' ') + qs_struct_tokens + query.split(' '):
      word_counter[w] += 1
  logging.info('%s words in total', len(word_counter))

  vocab = ['<pad>', '<EOS>', '<OOV>', '<SEP>', '<CLS>'] + list(
      word_counter.keys())
  logging.info('Write %s words to %s', len(vocab), vocab_fpath)
  with open(vocab_fpath, 'w') as fd:
    fd.write('\n'.join(vocab))

  # Creates tokenizer from the vocabulary.
  text_encoder = cfq_parse_util.load_text_encoder(vocab_fpath)

  # Load cross link mappings
  xlink_mapping = cfq_parse_util.load_xlink_mapping(xlink_map_fpath)

  # Writes dataset into tfrecord files. Negative samples are generated.
  random.seed(123)  # Random seed for reproducibility
  if not FLAGS.train_hold_out:
    # Use all train dataset
    write_dataset_split(cfq_data, train_idxs, train_fpath, text_encoder,
                        xlink_mapping)
  else:
    # 5% of the train dataset is kept as train_holdout set
    num_hold_out = len(train_idxs) // 20
    random.shuffle(train_idxs)
    train_train_idxs = train_idxs[:-num_hold_out]
    train_hold_idxs = train_idxs[-num_hold_out:]
    train_hold_fpath = os.path.join(FLAGS.output_dir, 'train_holdout.tfrecord')
    write_dataset_split(cfq_data, train_train_idxs, train_fpath, text_encoder,
                        xlink_mapping)
    write_dataset_split(cfq_data, train_hold_idxs, train_hold_fpath,
                        text_encoder, xlink_mapping)
  write_dataset_split(cfq_data, dev_idxs, dev_fpath, text_encoder,
                      xlink_mapping)
  write_dataset_split(cfq_data, test_idxs, test_fpath, text_encoder,
                      xlink_mapping)

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
