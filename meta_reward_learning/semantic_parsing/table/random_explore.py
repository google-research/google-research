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

"""Script for random exploration."""
import json
import multiprocessing
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
from meta_reward_learning.semantic_parsing.nsm \
import computer_factory
from meta_reward_learning.semantic_parsing.nsm \
import data_utils
from meta_reward_learning.semantic_parsing.nsm \
import env_factory
from meta_reward_learning.semantic_parsing.nsm \
import executor_factory
from meta_reward_learning.semantic_parsing.table \
import utils

# FLAGS
FLAGS = tf.app.flags.FLAGS

# Experiment name
tf.flags.DEFINE_string('output_dir', '', 'output directory')
tf.flags.DEFINE_string(
    'experiment_name', 'experiment', 'All outputs of this experiment is'
    ' saved under a folder with the same name.')
tf.app.flags.DEFINE_integer('n_epoch', 1000,
                            'Max number of valid tokens during decoding.')

# Data
tf.app.flags.DEFINE_string('table_file', '', '.')
tf.app.flags.DEFINE_string(
    'train_file_tmpl', '',
    'Path to the file of training examples, a jsonl file.')

# Model
## Computer
tf.app.flags.DEFINE_integer('max_n_mem', 100,
                            'Max number of memory slots in the "computer".')
tf.app.flags.DEFINE_integer('max_n_exp', 3,
                            'Max number of expressions allowed in a program.')
tf.app.flags.DEFINE_integer('max_n_valid_indices', 100,
                            'Max number of valid tokens during decoding.')
tf.app.flags.DEFINE_string('executor', 'wtq',
                           'Which executor to use, wtq or wikisql.')

# Exploration
tf.app.flags.DEFINE_integer('n_explore_samples', 50, '.')
tf.app.flags.DEFINE_integer('save_every_n', 10, '.')
tf.app.flags.DEFINE_integer('id_start', 0, '.')
tf.app.flags.DEFINE_integer('id_end', 0, '.')

tf.app.flags.DEFINE_string('trigger_word_file', '', '.')

tf.app.flags.DEFINE_integer('n_process', -1, '.')

tf.logging.set_verbosity(tf.logging.INFO)


def get_experiment_dir():
  experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name)
  return experiment_dir


def random_explore(env, use_cache=True, trigger_dict=None):
  env = env.clone()
  env.use_cache = use_cache
  question_tokens = env.question_annotation['tokens']
  if 'pos_tag' in env.question_annotation:
    question_tokens += env.question_annotation['pos_tags']
  invalid_functions = []
  if trigger_dict is not None:
    for function, trigger_words in trigger_dict.iteritems():
      for w in trigger_words:
        if w in question_tokens:
          break
      else:
        invalid_functions.append(function)
  ob = env.start_ob
  while not env.done:
    invalid_actions = env.de_vocab.lookup(invalid_functions)
    valid_actions = ob[0].valid_indices
    new_valid_actions = list(set(valid_actions) - set(invalid_actions))
    # No action available anymore.
    if len(new_valid_actions) <= 0:
      return None
    new_action = np.random.randint(0, len(new_valid_actions))
    action = valid_actions.index(new_valid_actions[new_action])
    ob, _, _, _ = env.step(action)

  if sum(env.rewards) >= 1.0:
    return env.de_vocab.lookup(env.mapped_actions, reverse=True)
  else:
    return None


def run_random_exploration(shard_id):
  experiment_dir = get_experiment_dir()
  if not tf.gfile.Exists(experiment_dir):
    tf.gfile.MkDir(experiment_dir)

  if FLAGS.trigger_word_file:
    with gfile.Open(FLAGS.trigger_word_file, 'r') as f:
      trigger_dict = json.load(f)
      print 'use trigger words in {}'.format(FLAGS.trigger_word_file)
  else:
    trigger_dict = None

  # Load dataset.
  train_set = []
  with gfile.Open(FLAGS.train_file_tmpl.format(shard_id), 'r') as f:
    for line in f:
      example = json.loads(line)
      train_set.append(example)
  tf.logging.info('{} examples in training set.'.format(len(train_set)))

  table_dict = {}
  with gfile.Open(FLAGS.table_file) as f:
    for line in f:
      table = json.loads(line)
      table_dict[table['name']] = table
  tf.logging.info('{} tables.'.format(len(table_dict)))

  if FLAGS.executor == 'wtq':
    score_fn = utils.wtq_score
    process_answer_fn = lambda x: x
    executor_fn = executor_factory.WikiTableExecutor
  elif FLAGS.executor == 'wikisql':
    score_fn = utils.wikisql_score
    process_answer_fn = utils.wikisql_process_answer
    executor_fn = executor_factory.WikiSQLExecutor
  else:
    raise ValueError('Unknown executor {}'.format(FLAGS.executor))

  all_envs = []
  t1 = time.time()
  for i, example in enumerate(train_set):
    if i % 100 == 0:
      tf.logging.info('creating environment #{}'.format(i))
    kg_info = table_dict[example['context']]
    executor = executor_fn(kg_info)
    api = executor.get_api()
    type_hierarchy = api['type_hierarchy']
    func_dict = api['func_dict']
    constant_dict = api['constant_dict']
    interpreter = computer_factory.LispInterpreter(
        type_hierarchy=type_hierarchy,
        max_mem=FLAGS.max_n_mem,
        max_n_exp=FLAGS.max_n_exp,
        assisted=True)
    for v in func_dict.values():
      interpreter.add_function(**v)

    interpreter.add_constant(
        value=kg_info['row_ents'], type='entity_list', name='all_rows')

    de_vocab = interpreter.get_vocab()
    env = env_factory.QAProgrammingEnv(
        data_utils.Vocab([]),
        de_vocab,
        question_annotation=example,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        constant_value_embedding_fn=lambda x: None,
        score_fn=score_fn,
        max_cache_size=FLAGS.n_explore_samples * FLAGS.n_epoch * 10,
        name=example['id'])
    all_envs.append(env)

  program_dict = dict([(env.name, []) for env in all_envs])
  for i in xrange(1, FLAGS.n_epoch + 1):
    tf.logging.info('iteration {}'.format(i))
    t1 = time.time()
    for env in all_envs:
      for _ in xrange(FLAGS.n_explore_samples):
        program = random_explore(env, trigger_dict=trigger_dict)
        if program is not None:
          program_dict[env.name].append(program)
    t2 = time.time()
    tf.logging.info('{} sec used in iteration {}'.format(t2 - t1, i))

    if i % FLAGS.save_every_n == 0:
      tf.logging.info('saving programs and cache in iteration {}'.format(i))
      t1 = time.time()
      with gfile.Open(
          os.path.join(get_experiment_dir(), 'program_shard_{}-{}.json'.format(
              shard_id, i)), 'w') as f:
        program_str_dict = dict([
            (k, [' '.join(p) for p in v]) for k, v in program_dict.iteritems()
        ])
        json.dump(program_str_dict, f, sort_keys=True, indent=2)

      # cache_dict = dict([(env.name, list(env.cache._set)) for env in all_envs])
      t2 = time.time()
      tf.logging.info(
          '{} sec used saving programs and cache in iteration {}'.format(
              t2 - t1, i))

    n = len(all_envs)
    solution_ratio = len([env for env in all_envs if program_dict[env.name]
                         ]) * 1.0 / n
    tf.logging.info(
        'At least one solution found ratio: {}'.format(solution_ratio))
    n_programs_per_env = np.array(
        [len(program_dict[env.name]) for env in all_envs])
    tf.logging.info(
        'number of solutions found per example: max {}, min {}, avg {}, std {}'
        .format(n_programs_per_env.max(), n_programs_per_env.min(),
                n_programs_per_env.mean(), n_programs_per_env.std()))

    # Macro average length.
    mean_length = np.mean([
        np.mean([len(p)
                 for p in program_dict[env.name]])
        for env in all_envs
        if program_dict[env.name]
    ])
    tf.logging.info('macro average program length: {}'.format(mean_length))
    # avg_cache_size = sum([len(env.cache._set) for env in all_envs]) * 1.0 / len(all_envs)
    # tf.logging.info('average cache size: {}'.format(
    #  avg_cache_size))


def collect_programs():
  saved_programs = {}
  for i in xrange(FLAGS.id_start, FLAGS.id_end):
    with gfile.Open(
        os.path.join(get_experiment_dir(), 'program_shard_{}-{}.json'.format(
            i, FLAGS.n_epoch)), 'r') as f:
      program_shard = json.load(f)
      saved_programs.update(program_shard)
  saved_program_path = os.path.join(get_experiment_dir(), 'saved_programs.json')
  with gfile.Open(saved_program_path, 'w') as f:
    json.dump(saved_programs, f)
  print 'saved programs are aggregated in {}'.format(saved_program_path)


def main(unused_argv):
  ps = []
  for idx in xrange(FLAGS.id_start, FLAGS.id_end):
    p = multiprocessing.Process(target=run_random_exploration, args=(idx,))
    p.start()
    ps.append(p)
  for p in ps:
    p.join()
  collect_programs()


if __name__ == '__main__':
  tf.app.run()
