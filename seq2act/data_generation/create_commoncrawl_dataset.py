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

"""Creates common crawl dateset in TFRecord format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import operator
import os
from absl import app
from absl import flags

import attr
import numpy as np
import tensorflow.compat.v1 as tf  # tf

from seq2act.data_generation import common
from seq2act.data_generation import config
from seq2act.data_generation import proto_utils
from seq2act.data_generation import string_utils

_NUM_SHARDS_DEFAULT = 10
GOLD_NUM_IN_SHARD0 = 700

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_shards', _NUM_SHARDS_DEFAULT,
    'The number of sharded files to save the created dataset.')
flags.DEFINE_enum(
    'sharding', '700-golden-in-shard0', ['hash', '700-golden-in-shard0'],
    'The way to split data to sharding')
flags.DEFINE_string(
    'input_csv_file', None, 'Input CSV file of labeled data.')
flags.DEFINE_string(
    'input_instruction_json_file', None,
    'Input Json file of downloaded instructions.')
flags.DEFINE_string(
    'output_dir', None,
    'Output directory for generated tf example proto data.')


# Debug purpose only
counters = collections.Counter()
id_batch_map = {}  # dict of {Task_id: Batch num where the task comes from}
one_agreement_ids = []
# Pick out some examples out for manual check, {"reason": list_of_examples}
chosen_examples = collections.defaultdict(list)
# Stats on some features, {"interesting_feature_name": {"value": count}}
distributions = collections.defaultdict(collections.Counter)


@attr.s
@attr.s
class Action(object):
  verb_type = attr.ib()
  verb_start_pos = attr.ib()
  verb_end_pos = attr.ib()
  object_desc_start_pos = attr.ib()
  object_desc_end_pos = attr.ib()
  input_content_start_pos = attr.ib()
  input_content_end_pos = attr.ib()


def _annotation_to_actions(annotation_str):
  """Splits the annotated actions to list of Actions for easy to use."""
  # Example: [CLICK-28:31-42:81-0:0]->[CLICK-248:251-254:337-0:0]
  actions = []
  action_str_list = annotation_str.split('->')
  for action_str in action_str_list:
    action = Action()
    action_str = action_str[1:-1]
    parts = action_str.split('-')
    action.verb_type = parts[0]
    (action.verb_start_pos,
     action.verb_end_pos) = [int(x) for x in parts[1].split(':')]
    (action.object_desc_start_pos,
     action.object_desc_end_pos) = [int(x) for x in parts[2].split(':')]
    (action.input_content_start_pos,
     action.input_content_end_pos) = [int(x) for x in parts[3].split(':')]
    actions.append(action)
  return actions


def _task_to_features_dict(task, do_stats):
  """Converts one task to features dict.

  Args:
    task: Task instance, the task to be converted.
    do_stats: whether do stats on this task. Set it to False for debug or test.
  Returns:
    features: dict of (string, np.array) which contains columns and values:
    features = {
      'instruction_str': string of answer, np string array, shape = (1,)
      'instruction_word_id_seq': word id sequence of question, np string array,
          shape = (word_num,)

      'verb_id_seq': word id of verb, np int array, shape = (action_num,)
      'obj_desc_position_seq': index of word id of object in answer, np int
          array, shape = (actions * 2,)
      'input_str_position_seq': additional info of the action, shape =
          (action_num * 2,)

      'raters_count_per_task': shape = (1,)
      'agreement_count': shape = (1,)
  Raises:
    ValueError: raise error when fail to parse actions of tasks.
  """
  answer = task['instruction'].lower()
  features = {}
  features['instruction_str'] = np.array([answer], dtype=np.string_)
  tokens, _ = string_utils.tokenize_to_ids(answer)
  features['instruction_word_id_seq'] = np.array(tokens, dtype=np.int64)

  verb_id_seq = []
  verb_str_position_seq = []
  obj_desc_position_seq = []
  input_str_position_seq = []

  for action in task['actions']:
    try:
      verb_id = common.ActionTypes[action.verb_type.upper().strip()]
    except KeyError:
      raise ValueError('Verb "%s" cannot be recognized.' % action.verb_type)
    if verb_id == common.ActionTypes.OTHERS:
      verb = answer[action.verb_start_pos: action.verb_end_pos].strip().lower()
      verb_id = common.VERB_ID_MAP.get(verb, common.ActionTypes.OTHERS)
    verb_id_seq.append(verb_id.value)

    verb_str_position_seq.extend(string_utils.get_token_pos_from_char_pos(
        answer, action.verb_start_pos, action.verb_end_pos))
    if do_stats and task['agreement-count'] >= 2:
      distributions['longest_verb_str'][
          verb_str_position_seq[-1] - verb_str_position_seq[-2]] += 1

    obj_desc_position_seq.extend(string_utils.get_token_pos_from_char_pos(
        answer, action.object_desc_start_pos, action.object_desc_end_pos))
    if do_stats and task['agreement-count'] >= 2:
      distributions['longest_obj_desc'][
          obj_desc_position_seq[-1] - obj_desc_position_seq[-2]] += 1

    if not (action.input_content_start_pos == 0 and
            action.input_content_end_pos == 0):
      input_str_position_seq.extend(string_utils.get_token_pos_from_char_pos(
          answer, action.input_content_start_pos, action.input_content_end_pos))
      if do_stats and task['agreement-count'] >= 2:
        distributions['longest_input_str'][
            input_str_position_seq[-1] - input_str_position_seq[-2]] += 1
    else:
      input_str_position_seq.extend([config.LABEL_DEFAULT_VALUE_INT] * 2)

  features['verb_id_seq'] = np.array(verb_id_seq, dtype=np.int64)
  features['verb_str_position_seq'] = np.array(verb_str_position_seq,
                                               dtype=np.int64)
  features['obj_desc_position_seq'] = np.array(obj_desc_position_seq,
                                               dtype=np.int64)
  features['input_str_position_seq'] = np.array(input_str_position_seq,
                                                dtype=np.int64)

  features['agreement_count'] = np.array([task['agreement-count']],
                                         dtype=np.int64)

  if do_stats:
    distributions['step_num'][len(task['actions'])] += 1
    distributions['longest_instruction'][len(tokens)] += 1
    counters['total_verb_refs'] += len(verb_id_seq)
    counters['total_obj_refs'] += len(obj_desc_position_seq) / 2
    counters['total_input_refs'] += (
        (len(input_str_position_seq) -
         input_str_position_seq.count(config.LABEL_DEFAULT_VALUE_INT)) / 2)
    for verb in common.ActionTypes:
      if verb.value in verb_id_seq:
        counters['Instructions contain %s in verbs' % verb.name] += 1
    if input_str_position_seq.count(config.LABEL_DEFAULT_VALUE_INT) != len(
        input_str_position_seq):
      counters['Instructions contain INPUT Content'] += 1
    if ' and then ' in answer:
      chosen_examples['instruction_contains_and-then'].append(answer)
    if ' after ' in answer:
      chosen_examples['instruction_contains_after'].append(answer)
    if '. ' in answer:
      counters['instruction_contains_dot'] += 1
    if ', ' in answer:
      counters['instruction_contains_comma'] += 1

  return features


def _write_tasks_to_tf_example(id_tasks_dict, output_dir, num_shards, sharding):
  """Writes tasks as tf.Example.

  Args:
    id_tasks_dict: dict of task_id and list of Tasks.
    output_dir: string, the full path of outupt folder.
    num_shards: int, number of shards of output.
    sharding: from flag sharding enum, how to sharding the data.
  """
  tfrecord_writers = []
  for shard in range(num_shards):
    tfrecord_writers.append(tf.python_io.TFRecordWriter(
        os.path.join(output_dir, 'commoncrawl_%d.tfrecord' % shard)))

  def write_task(task, shard_id):
    try:
      features = _task_to_features_dict(task, do_stats=True)
    except ValueError:
      counters['ValueError'] += 1
    else:
      tfproto = proto_utils.features_to_tf_example(features)
      tfrecord_writers[shard_id].write(tfproto.SerializeToString())
      counters['examples_count_in_dataset'] += 1

  # Sharing mode
  if sharding == 'hash':
    for task_id, tasks in id_tasks_dict.items():
      shard_id = hash(task_id) % num_shards
      for task in tasks:
        write_task(task, shard_id)

  else:  # when sharding == '700-golden-in-shard0'
    # For testing purpose, put 700 100% agreement tasks to shard_0,
    # and then put the rest tasks to shard 1~9
    testing_count = 0
    for task_id, tasks in id_tasks_dict.items():
      if (testing_count < GOLD_NUM_IN_SHARD0 and
          tasks[0]['agreement-count'] == len(tasks) and len(tasks) >= 3):
        for task in tasks:
          write_task(tasks[0], shard_id=0)
        testing_count += 1
      else:
        shard_id = hash(task_id) % (num_shards -1) + 1
        for task in tasks:
          write_task(task, shard_id)


def _read_tasks(input_csv_file, input_instruction_json_file):
  """Reads rows from CSV file containing the annotations."""

  # We use `index+url` as the ID of an instruction
  def get_task_id(index, url):
    return '%s+%s' % (str(index), url)

  id_instruction_dict = {}
  with open(input_instruction_json_file, 'r') as f:
    for line in f:
      if line.strip():
        json_dict = json.loads(line)
        id_instruction_dict[get_task_id(
            json_dict['index'], json_dict['url'])] = json_dict['instructions']

  instruction_found = 0
  instruction_not_found = 0
  id_tasks_dict = collections.defaultdict(list)
  with open(input_csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      task_id = get_task_id(row['index'], row['url'])
      row['actions'] = _annotation_to_actions(row['annotation'])
      row['agreement-count'] = int(row['agreement-count'])
      if task_id in id_instruction_dict:
        row['instruction'] = id_instruction_dict[task_id]
        id_tasks_dict[task_id].append(row)
        instruction_found += 1
      else:
        instruction_not_found += 1

  if instruction_not_found == 0:
    print('All %s instructions match with annotations successfully.' %
          instruction_found)
  else:
    print('%s instructions match with annotations successfully.' %
          instruction_found)
    print('Warning: can not find instructions for %s annotations, probably you '
          'have not downloaded all the WARC files.' % instruction_not_found)
  return id_tasks_dict


def _generate_commoncrawl_dataset():
  """Generates commoncrawl dataset with the annotations."""
  assert FLAGS.input_csv_file.endswith('.csv')
  id_tasks_dict = _read_tasks(FLAGS.input_csv_file,
                              FLAGS.input_instruction_json_file)

  _write_tasks_to_tf_example(id_tasks_dict, FLAGS.output_dir,
                             FLAGS.num_shards, FLAGS.sharding)

  def sort_dict_by_key(the_dict):
    return sorted(the_dict.items(), key=operator.itemgetter(0))

  with open(os.path.join(FLAGS.output_dir, 'stats.txt'), 'w+') as stat_file:
    stat_file.write('stat_fix_dict: %s\n' % string_utils.stat_fix_dict)
    for key, count in sort_dict_by_key(counters):
      stat_file.write('%s: %s\n' % (key, count))
    for key, examples in sort_dict_by_key(chosen_examples):
      stat_file.write('%s: %s\n' % (key, len(examples)))
    for key, distribution in distributions.items():
      stat_file.write('%s: %s\n' % (key, sort_dict_by_key(distribution)))

  for key, examples in chosen_examples.items():
    with open(os.path.join(FLAGS.output_dir, key), 'w+') as writer:
      writer.write('\n'.join(examples))


def main(_):
  _generate_commoncrawl_dataset()


if __name__ == '__main__':
  FLAGS.set_default('logtostderr', True)
  app.run(main)
