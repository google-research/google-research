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

# -*- coding: utf-8 -*-
"""Experiment to run supervised learning for Program Synthesis."""

from __future__ import division
from __future__ import print_function
import codecs
import json
import os
import time

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile


from meta_reward_learning.semantic_parsing import common_flags
from meta_reward_learning.semantic_parsing.nsm import agent_factory
from meta_reward_learning.semantic_parsing.nsm import computer_factory
from meta_reward_learning.semantic_parsing.nsm import data_utils
from meta_reward_learning.semantic_parsing.nsm import env_factory
from meta_reward_learning.semantic_parsing.nsm import executor_factory
from meta_reward_learning.semantic_parsing.nsm import graph_factory
from meta_reward_learning.semantic_parsing.nsm import model_factory
from meta_reward_learning.semantic_parsing.nsm import score_utils
from meta_reward_learning.semantic_parsing.nsm import word_embeddings
from meta_reward_learning.semantic_parsing.table import utils


# FLAGS
FLAGS = flags.FLAGS
flags.adopt_module_key_flags(common_flags)


def get_experiment_dir():
  tf.logging.info('out_dir: {}'.format(FLAGS.output_dir))
  experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_name)
  if not gfile.IsDirectory(FLAGS.output_dir):
    gfile.MakeDirs(FLAGS.output_dir)
  if not gfile.IsDirectory(experiment_dir):
    gfile.MakeDirs(experiment_dir)
  return experiment_dir


def get_init_model_path():
  """Helper function to get the initial model path."""
  # This code restores the last saved checkpoint in case of premption
  if not FLAGS.eval_only:
    ckpt_dir = os.path.join(get_experiment_dir(), FLAGS.saved_model_dir)
    current_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if current_ckpt is not None:
      tf.logging.info('Loading initial model from {}'.format(current_ckpt))
      return current_ckpt
    elif FLAGS.experiment_to_load:
      with gfile.Open(
          os.path.join(FLAGS.output_dir, FLAGS.experiment_to_load,
                       'best_model_info.json'), 'r') as f:
        best_model_info = json.load(f)
        best_model_path = best_model_info['best_model_path']
        tf.logging.info('Loading initial model from {}'.format(best_model_path))
        return best_model_path
    else:
      return ''
  else:
    if FLAGS.experiment_to_eval:
      with gfile.Open(
          os.path.join(FLAGS.output_dir, FLAGS.experiment_to_eval,
                       'best_model_info.json'), 'r') as f:
        best_model_info = json.load(f)
        best_model_path = best_model_info['best_model_path']
        return best_model_path
    else:
      raise ValueError('Please supply correct value for experiment_to_eval')


def get_init_score_path():
  """Helper function to get model for only initializing score function."""
  if FLAGS.init_score_path is not None:
    score_experiment_dir = os.path.join(FLAGS.output_dir, FLAGS.init_score_path)
    tf.logging.info(
        'Using {} for initializing score function'.format(score_experiment_dir))
    best_model_path = os.path.join(score_experiment_dir, 'best_model_info.json')
    if gfile.Exists(best_model_path):
      with gfile.Open(best_model_path, 'r') as f:
        best_model_info = json.load(f)
        score_model_path = best_model_info['best_model_path']
    else:
      ckpt_dir = os.path.join(score_experiment_dir, FLAGS.saved_model_dir)
      score_model_path = tf.train.latest_checkpoint(ckpt_dir)
  return score_model_path


def restore_prempted():
  """Helper function for restoring agent model.

  This function checks whether we are starting training from a pretrained
  model or restoring a premempted model.
  Returns:
    A boolean value.
  """
  ckpt_dir = os.path.join(get_experiment_dir(), FLAGS.saved_model_dir)
  current_ckpt = tf.train.latest_checkpoint(ckpt_dir)
  if current_ckpt is not None:
    return True
  else:
    return False


def get_saved_graph_config():
  if FLAGS.experiment_to_eval:
    with gfile.Open(
        os.path.join(FLAGS.output_dir, FLAGS.experiment_to_eval,
                     'graph_config.json'), 'r') as f:
      graph_config = json.load(f)
      return graph_config
  else:
    return None


def get_saved_experiment_config():
  if FLAGS.experiment_to_eval:
    with gfile.Open(
        os.path.join(FLAGS.output_dir, FLAGS.experiment_to_eval,
                     'experiment_config.json'), 'r') as f:
      experiment_config = json.load(f)
      return experiment_config
  else:
    return None


def show_samples(samples, de_vocab, env_dict=None):
  """Given a set of sample programs, generates a string showing programs."""
  string = ''
  for sample in samples:
    traj = sample.traj
    actions = traj.actions
    obs = traj.obs
    pred_answer = traj.answer
    string += u'\n'
    env_name = traj.env_name
    string += u'env {}\n'.format(env_name)
    if env_dict is not None:
      string += u'question: {}\n'.format(
          env_dict[env_name].question_annotation['question'])
      string += u'answer: {}\n'.format(
          env_dict[env_name].question_annotation['answer'])
    program = []
    for (a, ob) in zip(actions, obs):
      ob = ob[0]
      valid_tokens = de_vocab.lookup(ob.valid_indices, reverse=True)
      token = valid_tokens[a]
      program.append(token)
    program_str = ' '.join(program)
    if env_dict:
      program_str = unpack_program(program_str, env_dict[env_name])
    string += u'program: {}\n'.format(program_str)
    string += u'prediction: {}\n'.format(pred_answer)
    string += u'return: {}\n'.format(sum(traj.rewards))
    string += u'prob is {}\n'.format(sample.prob)
  return string


def collect_traj_for_program(env, program):
  """Converts a string program into the namedtuple Traj."""
  env_original = env
  env = env.clone()
  env.use_cache = False
  ob = env.start_ob

  for tk in program:
    valid_actions = list(ob[0].valid_indices)
    mapped_action = env.de_vocab.lookup(tk)
    try:
      action = valid_actions.index(mapped_action)
    except:  # pylint: disable=bare-except
      return None
    ob, _, _, _ = env.step(action)
  sim_features = env_factory.create_features(
      env_original, program, FLAGS.max_n_exp)
  traj = agent_factory.Traj(
      obs=env.obs,
      actions=env.actions,
      rewards=env.rewards,
      context=env.get_context(),
      env_name=env.name,
      answer=env.interpreter.result,
      sim_features=sim_features)
  return traj


def unpack_program(program_str, env):
  ns = env.interpreter.namespace
  processed_program = []
  for tk in program_str.split():
    if tk[:1] == 'v' and tk in ns:
      processed_program.append(unicode(ns[tk]['value']))
    else:
      processed_program.append(tk)
  return ' '.join(processed_program)


def load_programs(envs,
                  replay_buffer,
                  fn,
                  max_programs=None,
                  use_top_programs_only=False):
  """Loads the programs from a json file into a replay buffer."""
  if not gfile.Exists(fn):
    return
  with gfile.Open(fn, 'r') as f:
    program_dict = json.load(f)
  trajs = []
  n = 0
  total_env = 0
  n_found = 0
  traj_buffer = replay_buffer.traj_buffer
  for env in envs:
    total_env += 1
    found = False
    if env.name in program_dict:
      cur_trajs = []
      if env.name in traj_buffer:
        num_progs_present = len(traj_buffer[env.name])
      else:
        num_progs_present = 0
      program_str_list = program_dict[env.name]
      n += len(program_str_list)
      env.cache.cache_set = set(program_str_list)
      for program_str in program_str_list:
        program = program_str.split()
        # Change for having a weight for each component
        traj = collect_traj_for_program(env, program)
        if traj is not None:
          cur_trajs.append(traj)
          if not found:
            found = True
            n_found += 1
      if cur_trajs:
        if (max_programs is not None) or use_top_programs_only:
          cur_trajs = sorted(
              cur_trajs,
              key=lambda k: (k.sim_features[-1], k.sim_features[-2]),
              reverse=True)
          if use_top_programs_only:
            # Calculate the number of programs with the max score.
            max_feature_val = cur_trajs[0].sim_features[-1]
            num_programs = 0
            for traj in cur_trajs:
              if np.isclose(max_feature_val, traj.sim_features[-1]):
                num_programs += 1
              else:
                # Since the list is sorted, we can stop at first mismatch.
                break
          if max_programs is not None:
            if use_top_programs_only:
              # This is done for the meta training set programs usually.
              num_programs = max(num_programs, max_programs)
            else:
              # Select the  user specified number of programs.
              num_programs = max_programs
          cur_trajs = cur_trajs[:num_programs]
      cur_trajs = [
          traj._replace(idx=i + num_progs_present)
          for i, traj in enumerate(cur_trajs)
      ]
      trajs.extend(cur_trajs)
  tf.logging.info('@' * 100)
  tf.logging.info('loading programs from file {}'.format(fn))
  tf.logging.info('at least 1 solution found fraction: {}'.format(
      float(n_found) / total_env))
  replay_buffer.save_trajs(trajs)
  n_trajs_buffer = 0
  for v in replay_buffer.traj_buffer.values():
    n_trajs_buffer += len(v)
  tf.logging.info('{} programs in the file'.format(n))
  tf.logging.info('{} programs extracted'.format(len(trajs)))
  tf.logging.info('{} programs in the buffer'.format(n_trajs_buffer))
  tf.logging.info('@' * 100)


def get_program_shard_path(i):
  return os.path.join(FLAGS.saved_programs_dir,
                      FLAGS.program_shard_prefix + str(i) + '.json')


def get_train_shard_path(i):
  return os.path.join(FLAGS.train_shard_dir,
                      FLAGS.train_shard_prefix + str(i) + '.jsonl')


def load_jsonl(fn):
  result = []
  with gfile.Open(fn, 'r') as f:
    for line in f:
      data = json.loads(line)
      result.append(data)
  return result


def create_envs(table_dict, data_set, en_vocab, embedding_model):
  """Create the contextual envs representing queries and tables."""
  all_envs = []
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
  if FLAGS.trigger_words_file:
    with gfile.Open(FLAGS.trigger_words_file, 'r') as f:
      trigger_words_dict = json.load(f)
    trigger_words_dict = {k: set(v) for k, v in trigger_words_dict.iteritems()}
    tf.logging.info('Using trigger words in {}'.format(
        FLAGS.trigger_words_file))
  else:
    trigger_words_dict = None

  for i, example in enumerate(data_set):
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
    # pylint: disable=g-long-lambda
    constant_value_embedding_fn = lambda x: utils.get_embedding_for_constant(
        x, embedding_model, embedding_size=FLAGS.pretrained_embedding_size)
    # pylint: enable=g-long-lambda
    env = env_factory.QAProgrammingEnv(
        en_vocab,
        de_vocab,
        question_annotation=example,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        constant_value_embedding_fn=constant_value_embedding_fn,
        score_fn=score_fn,
        trigger_words_dict=trigger_words_dict,
        name=example['id'])
    all_envs.append(env)
  return all_envs


def create_agent(graph_config,
                 init_model_path,
                 pretrained_embeddings=None,
                 extra_monitors=False):
  tf.logging.info('Start creating and initializing graph')
  t1 = time.time()
  graph = graph_factory.MemorySeq2seqGraph(graph_config)
  # restore all the variables if explicitly passed in the FLAGS or
  # running the evaluation pipeline. Also, if restoring a preempted trainer
  # (i.e. already existing model), restore all the global variables
  # irrespective of the FLAGS passed.
  trainable_only = FLAGS.trainable_only  # and (not FLAGS.eval_only)
  trainable_only = trainable_only and (not restore_prempted())
  eval_graph = (FLAGS.eval_dev in ['meta', 'validation']) or FLAGS.eval_only
  graph.launch(
      init_model_path=init_model_path,
      trainable_only=trainable_only,
      init_score_path=None if eval_graph else get_init_score_path(),
      ckpt_from_another=FLAGS.ckpt_from_another)
  t2 = time.time()
  tf.logging.info('{} sec used to create and initialize graph'.format(t2 - t1))

  tf.logging.info('Start creating model and agent')
  t1 = time.time()
  model = model_factory.MemorySeq2seqModel(
      graph,
      batch_size=FLAGS.batch_size,
      maxlen=graph_config['core_config']['maxlen'],
      en_maxlen=graph_config['core_config']['en_maxlen'])

  if pretrained_embeddings is not None:
    model.init_pretrained_embeddings(pretrained_embeddings)
  agent = agent_factory.PGAgent(
      model,
      extra_monitors=extra_monitors,
      score_model=FLAGS.score_model,
      score_norm_fn=FLAGS.score_norm_fn,
      use_baseline=FLAGS.use_baseline)
  t2 = time.time()
  tf.logging.info('{} sec used to create model and agent'.format(t2 - t1))
  return agent


def init_experiment(fns,
                    use_gpu=False,
                    gpu_id='0',
                    extra_monitors=False,
                    fns_val=None,
                    eval_fn=False):
  dataset = []
  for fn in fns:
    dataset += load_jsonl(fn)
  dataset = utils.remove_long_contexts(dataset)
  if FLAGS.unittest:
    FLAGS.en_maxlen = 40
    FLAGS.val_batch_size = FLAGS.batch_size = 10
    num_envs = 100
    dataset = dataset[:num_envs]
  tf.logging.info('{} examples in dataset.'.format(len(dataset)))
  tables = load_jsonl(FLAGS.table_file)
  table_dict = dict([(table['name'], table) for table in tables])
  tf.logging.info('{} tables.'.format(len(table_dict)))

  # Load pretrained embeddings.
  embedding_model = word_embeddings.EmbeddingModel(FLAGS.vocab_file,
                                                   FLAGS.embedding_file)

  with gfile.Open(FLAGS.en_vocab_file, 'r') as f:
    vocab = json.load(f)
  en_vocab = data_utils.Vocab([])
  en_vocab.load_vocab(vocab)
  tf.logging.info('{} unique tokens in encoder vocab'.format(
      len(en_vocab.vocab)))
  tf.logging.info('{} examples in the dataset'.format(len(dataset)))

  # Create environments.
  envs = create_envs(table_dict, dataset, en_vocab, embedding_model)

  # if FLAGS.unittest:
  #   envs = envs[:25]
  tf.logging.info('{} environments in total'.format(len(envs)))

  graph_config = get_saved_graph_config()
  if graph_config is None:
    graph_config = {}

  graph_config['use_gpu'] = use_gpu
  graph_config['gpu_id'] = gpu_id
  graph_config['meta_learn'] = FLAGS.meta_learn
  graph_config['eval_batch_size'] = FLAGS.eval_batch_size

  if graph_config and FLAGS.eval_only:
    # If evaluating an saved model, just load its graph config.
    graph_config['score_fn_config'] = dict(use_reward_weights=False)
    graph_config['train_config'] = {}
    agent = create_agent(graph_config, get_init_model_path())
  else:
    if FLAGS.use_pretrained_embeddings:
      tf.logging.info('Using pretrained embeddings!')
      pretrained_embeddings = []
      for i in xrange(len(en_vocab.special_tks), en_vocab.size):
        pretrained_embeddings.append(
            utils.average_token_embedding(
                utils.find_tk_in_model(
                    en_vocab.lookup(i, reverse=True), embedding_model),
                embedding_model,
                embedding_size=FLAGS.pretrained_embedding_size))
      pretrained_embeddings = np.vstack(pretrained_embeddings)
    else:
      pretrained_embeddings = None

    # Model configuration and initialization.
    de_vocab = envs[0].de_vocab
    n_mem = FLAGS.max_n_mem
    n_builtin = de_vocab.size - n_mem
    en_pretrained_vocab_size = en_vocab.size - len(en_vocab.special_tks)

    graph_config['core_config'] = dict(
        maxlen=FLAGS.maxlen,
        en_maxlen=FLAGS.en_maxlen,
        max_n_valid_indices=FLAGS.max_n_valid_indices,
        n_mem=n_mem,
        n_builtin=n_builtin,
        use_attn=True,
        attn_size=FLAGS.attn_size,
        attn_vec_size=FLAGS.attn_vec_size,
        input_vocab_size=de_vocab.size,
        en_input_vocab_size=en_vocab.size,
        hidden_size=FLAGS.hidden_size,
        n_layers=FLAGS.n_layers,
        en_hidden_size=FLAGS.hidden_size,
        en_n_layers=FLAGS.en_n_layers,
        en_use_embeddings=True,
        en_embedding_size=FLAGS.en_embedding_size,
        value_embedding_size=FLAGS.value_embedding_size,
        en_pretrained_vocab_size=en_pretrained_vocab_size,
        en_pretrained_embedding_size=FLAGS.pretrained_embedding_size,
        add_lm_loss=FLAGS.lm_loss_coeff > 0.0,
        en_bidirectional=FLAGS.en_bidirectional,
        en_attn_on_constants=FLAGS.en_attn_on_constants)

    graph_config['output_type'] = 'softmax'
    if FLAGS.sampling_strategy not in [
        'probs', 'reward', 'probs_and_reward', 'st_estimator', 'urex'
    ]:
      raise ValueError('{} is not a valid sampling strategy'.format(
          FLAGS.sampling_strategy))
    graph_config['output_config'] = dict(
        output_vocab_size=de_vocab.size,
        sampling_strategy=FLAGS.sampling_strategy,
        use_logits=True)
    aux_loss_list = [('ent_reg', FLAGS.entropy_reg_coeff)]

    if FLAGS.lm_loss_coeff > 0.0:
      aux_loss_list.append(('en_lm_loss', FLAGS.lm_loss_coeff))
    if eval_fn:
      graph_config['train_config'] = {}
    else:
      graph_config['train_config'] = dict(
          aux_loss_list=aux_loss_list,
          learning_rate=FLAGS.learning_rate,
          max_grad_norm=FLAGS.max_grad_norm,
          adam_beta1=FLAGS.adam_beta1,
          l2_coeff=FLAGS.l2_coeff,
          meta_lr=FLAGS.meta_lr,
          plot_summaries=FLAGS.plot_summaries,
          debug=FLAGS.debug_model,
          optimizer=FLAGS.optimizer,
          momentum=FLAGS.momentum)

    num_features = 12 if envs[0].trigger_words_dict else 11
    graph_config['score_fn_config'] = dict(
        score_model=FLAGS.score_model,
        num_envs=len(envs),
        num_features=num_features,
        score_norm_fn=FLAGS.score_norm_fn,
        score_temperature=FLAGS.score_temperature,
        max_programs=FLAGS.max_programs)
    agent = create_agent(
        graph_config,
        get_init_model_path(),
        pretrained_embeddings=pretrained_embeddings,
        extra_monitors=extra_monitors)

  with gfile.Open(os.path.join(get_experiment_dir(), 'graph_config.json'),
                  'w') as f:
    json.dump(graph_config, f, sort_keys=True, indent=2)

  if FLAGS.meta_learn:
    if not eval_fn:
      assert fns_val is not None, "Can't use Meta learning without val set"
      # Remove the really long questions, ideally shouldn't be done but to
      # make the static graph much more smaller, dropping a tiny number of
      # really long contexts doesn't really make much of a difference.
      # Reason: MAML like updates with LSTMs in tensorflow only possible with
      # static graph.
      dataset_val = []
      for fn in fns_val:
        dataset_val += load_jsonl(fn)
      dataset_val = utils.remove_long_contexts(dataset_val)
      envs_val = create_envs(table_dict, dataset_val, en_vocab, embedding_model)
      if FLAGS.unittest:
        envs_val = envs_val[:10]
      envs = (envs, envs_val)

  return agent, envs


def fns_meta_learn():
  if FLAGS.use_validation_for_meta_train:
    assert FLAGS.num_val_shards == 0
    fns_val = [FLAGS.dev_file]
  else:
    fns_val = [
        get_train_shard_path(i)
        for i in xrange(FLAGS.shard_end - FLAGS.num_val_shards, FLAGS.shard_end)
    ]
  return fns_val


def run_experiment():
  print('=' * 100)
  if FLAGS.show_log:
    tf.logging.set_verbosity(tf.logging.INFO)

  is_eval_dev = FLAGS.eval_dev in ['meta', 'validation']
  if FLAGS.eval_only and is_eval_dev:
    raise ValueError(
        '{} evaluation can not be done if eval_only is true'.format(
            FLAGS.eval_dev))
  is_eval = FLAGS.eval_only or is_eval_dev
  experiment_dir = get_experiment_dir()
  experiment_config = get_saved_experiment_config()
  if experiment_config:
    FLAGS.embedding_file = experiment_config['embedding_file']
    FLAGS.vocab_file = experiment_config['vocab_file']
    FLAGS.en_vocab_file = experiment_config['en_vocab_file']
    FLAGS.table_file = experiment_config['table_file']

  if not is_eval:
    # if gfile.Exists(experiment_dir):
    #   gfile.DeleteRecursively(experiment_dir)
    # gfile.MakeDirs(experiment_dir)

    experiment_config = {
        'embedding_file': FLAGS.embedding_file,
        'vocab_file': FLAGS.vocab_file,
        'en_vocab_file': FLAGS.en_vocab_file,
        'table_file': FLAGS.table_file,
        'saved_program_file': FLAGS.saved_program_file,
        'experiment_to_load': FLAGS.experiment_to_load
    }

    with gfile.Open(
        os.path.join(get_experiment_dir(), 'experiment_config.json'), 'w') as f:
      json.dump(experiment_config, f)

  run_type = 'evaluation' if is_eval else 'experiment'
  print('Start {} {}.'.format(run_type, FLAGS.experiment_name))
  print('The data of this {} is saved in {}.'.format(run_type, experiment_dir))

  fns_to_eval = None
  if FLAGS.eval_only:
    print('Start evaluating the best model {}.'.format(get_init_model_path()))
    fns_to_eval = [FLAGS.eval_file]
  elif FLAGS.eval_dev == 'meta':
    print('Start the evaluator on meta training dataset')
    # Pick the last num_val_shards in training dataset for validation
    fns_to_eval = fns_meta_learn()
  elif FLAGS.eval_dev == 'validation':
    print('Start the evaluator on the validation dataset')
    fns_to_eval = [FLAGS.dev_file]
  else:
    print('Start training...')

  if is_eval:
    evaluator = Evaluator('Evaluator', fns_to_eval)
    evaluator.run()
    print('Evaluator finished')
    print('=' * 100)
  else:
    print('Start learner.')
    fns = [
        get_train_shard_path(i)
        for i in xrange(FLAGS.shard_start, FLAGS.shard_end -
                        FLAGS.num_val_shards)
    ]
    fns_val = fns_meta_learn() if FLAGS.meta_learn else None
    learner = Learner('Learner', fns, fns_val=fns_val)
    learner.run()
    print('Use tensorboard to monitor the training progress (see README).')
    print('Learner finished')


class SampleGenerator():

  def __init__(self, envs, agent=None, is_val=False, is_fixed=True):
    self.agent = agent
    self.is_val = is_val
    self.is_fixed = is_fixed
    self.replay_buffer = agent_factory.AllGoodReplayBuffer(
        agent, envs[0].de_vocab)

    # Load saved programs to warm start the replay buffer.
    if FLAGS.load_saved_programs:
      tf.logging.info('Loading the saved programs into the replay buffer.')
      if is_val:
        load_programs(
            envs,
            self.replay_buffer,
            FLAGS.saved_val_program_file,
            max_programs=FLAGS.max_val_programs,
            use_top_programs_only=True)
      else:
        load_programs(
            envs,
            self.replay_buffer,
            FLAGS.saved_program_file,
            max_programs=FLAGS.max_programs)
        # Assumes that the extra replay files add a disjoint set.
        if FLAGS.saved_replay_program_files:
          program_files = FLAGS.saved_replay_program_files.split(',')
          for filename in program_files:
            load_programs(
                envs,
                self.replay_buffer,
                filename,
                max_programs=FLAGS.max_programs)
      tf.logging.info('Loaded saved programs into the replay buffer.')
    if is_fixed:
      env_dict = self.replay_buffer.traj_buffer
      self.envs = [env for env in envs if env.name in env_dict]
      self.all_envs = envs
    else:
      self.envs = envs

    if is_val:
      self.env_batch_size = FLAGS.val_batch_size
    else:
      n_train_samples = 0
      # Replay samples are used by all algorithms currently
      if FLAGS.use_replay_samples_in_train:
        n_train_samples += FLAGS.n_replay_samples
      if FLAGS.use_policy_samples_in_train and \
          FLAGS.use_nonreplay_samples_in_train:
        raise ValueError(
            'Cannot use both on-policy samples and nonreplay samples '
            'for training!')
      if FLAGS.use_policy_samples_in_train or \
          FLAGS.use_nonreplay_samples_in_train:
        n_train_samples += FLAGS.n_policy_samples
      # Make sure that all the samples from the env batch
      # fits into one batch for training.
      if FLAGS.batch_size < n_train_samples:
        raise ValueError(
            'One batch have to at least contain samples from one environment.')
      self.env_batch_size = FLAGS.batch_size // n_train_samples
      self.n_train_samples = n_train_samples

  def update_buffer(self, fn, use_top_programs_only=False):
    try:
      if self.is_fixed:
        load_programs(
            self.all_envs,
            self.replay_buffer,
            fn,
            use_top_programs_only=use_top_programs_only)
        env_dict = self.replay_buffer.traj_buffer
        self.envs = [env for env in self.all_envs if env.name in env_dict]
      else:
        load_programs(
            self.envs,
            self.replay_buffer,
            fn,
            use_top_programs_only=use_top_programs_only)
    except ValueError:
      tf.logging.info('Not updating the replay buffer.')

  def generate_policy_samples(self, envs=None, n_samples=None):
    t1 = time.time()
    if envs is None:
      envs = self.envs
    if n_samples is None:
      n_samples = FLAGS.n_policy_samples
    if FLAGS.use_top_k_policy_samples:
      if n_samples == 1:
        policy_samples = self.agent.generate_samples(
            envs, n_samples=n_samples, greedy=True)
      else:
        policy_samples = self.agent.beam_search(envs, beam_size=n_samples)
    else:
      policy_samples = self.agent.generate_samples(
          envs, n_samples=n_samples, greedy=False)
    t2 = time.time()
    tf.logging.info('{} sec used generating {} policy samples'.format(
        t2 - t1, len(policy_samples)))

    return policy_samples

  def reweight_samples(self, samples, use_clipping=False, policy_samples=False):
    """Reweights the prob. of each sample according to the MAPO objective."""
    new_samples = []
    if use_clipping:
      min_replay_weight = FLAGS.min_replay_weight
    else:
      min_replay_weight = None
    for sample in samples:
      name = sample.traj.env_name
      if name in self.replay_buffer.prob_sum_dict:
        replay_prob = self.replay_buffer.prob_sum_dict[name]
        if min_replay_weight:
          replay_prob = max(replay_prob, min_replay_weight)
      else:
        replay_prob = 0.0
      if policy_samples:
        replay_prob = 1.0 - replay_prob
      new_samples.append(
          agent_factory.Sample(
              traj=sample.traj, prob=sample.prob * replay_prob))
    return new_samples

  def calc_clip_frac(self, batch_envs):
    """Calculates the clip fraction for MAPO."""
    if FLAGS.use_replay_prob_as_weight:
      n_clip = 0
      for env in batch_envs:
        name = env.name
        if name in self.replay_buffer.prob_sum_dict and \
          self.replay_buffer.prob_sum_dict[name] < FLAGS.min_replay_weight:
          n_clip += 1
      clip_frac = float(n_clip) / len(batch_envs)
    else:
      clip_frac = 0.0
    return clip_frac

  def generate_replay_samples(self, drop_last=True):
    """Generate samples from the replay buffer."""
    replay_buffer = self.replay_buffer
    agent = self.agent

    i = 0
    while True:
      # Split the env list into a iterator which returns
      # a list of envs in chunks of `env_batch_size`
      envs = self.envs
      env_iterator = data_utils.BatchIterator(
          dict(envs=envs), shuffle=True, batch_size=self.env_batch_size)

      for j, batch_dict in enumerate(env_iterator):
        batch_envs = batch_dict['envs']
        if drop_last and (len(batch_envs) < self.env_batch_size):
          # Don't use the last batch as it can be much smaller than other
          # batches, leading to high variance in gradients and pathological
          # behaviour. Note that since shuffling is turned on,
          # all the examples would be covered
          continue
        if self.is_val:
          val_sampling_strategy = 'probs'  # Since reward shouldn't show up
          if FLAGS.val_objective == 'iml':
            val_agent = None
          else:
            val_agent = agent
          replay_samples = replay_buffer.replay(
              batch_envs,
              n_samples=1,
              use_top_k=False,
              agent=val_agent,
              sampling_strategy=val_sampling_strategy,
              is_val=False)
          if FLAGS.val_objective == 'mapo':
            replay_samples = self.reweight_samples(
                replay_samples, use_clipping=True)
          yield replay_samples
        else:
          add_features_to_samples = FLAGS.meta_learn
          tf.logging.info('=' * 50)
          tf.logging.info('train iteration {}, batch {}: {} envs'.format(
              i, j, len(batch_envs)))
          t3 = time.time()
          n_explore = 0
          for _ in xrange(FLAGS.n_explore_samples):
            # Sampling (might be greedy) from the current policy
            # at a temparture of 1.
            explore_samples = agent.generate_samples(
                batch_envs,
                n_samples=1,
                use_cache=FLAGS.use_cache,
                greedy=FLAGS.greedy_exploration)
            # replay buffer only saves the trajectories with high rewards
            if add_features_to_samples:
              explore_samples = agent_factory.samples_with_features(
                  batch_envs, explore_samples, FLAGS.max_n_exp)
            replay_buffer.save(explore_samples)
            n_explore += len(explore_samples)

          # Extra exploration not done for wikitable currently
          if FLAGS.n_extra_explore_for_hard > 0:
            hard_envs = [
                env for env in batch_envs
                if not replay_buffer.has_found_solution(env.name)
            ]
            if hard_envs:
              for _ in xrange(FLAGS.n_extra_explore_for_hard):
                explore_samples = agent.generate_samples(
                    hard_envs,
                    n_samples=1,
                    use_cache=FLAGS.use_cache,
                    greedy=FLAGS.greedy_exploration)
                if add_features_to_samples:
                  explore_samples = agent_factory.samples_with_features(
                      batch_envs, explore_samples, FLAGS.max_n_exp)
                replay_buffer.save(explore_samples)
                n_explore += len(explore_samples)

            t4 = time.time()
            tf.logging.info(
                '{} sec used generating {} exploration samples.'.format(
                    t4 - t3, n_explore))

            tf.logging.info('{} samples saved in the replay buffer.'.format(
                replay_buffer.size))

          # Each algorithm is uses a different type of sampling from the
          # replay buffer. For a given env (i.e. query):
          # IML selects samples uniformly at random
          # MAPO, MML selects the samples from the normalized distribution of
          # samples according to the current policy
          # HARD_EM picks the top_k replay samples
          # The replay buffer is never truncated for wikitable
          replay_args = dict(
              use_top_k=FLAGS.use_top_k_replay_samples,
              agent=None if FLAGS.random_replay_samples else agent,
              truncate_at_n=FLAGS.truncate_replay_buffer_at_n)
          if FLAGS.sampling_strategy == 'urex':
            sampling_strategies = ['reward', 'probs']
            n_samples = FLAGS.n_replay_samples // 2
          else:
            sampling_strategies = [FLAGS.sampling_strategy]
            n_samples = FLAGS.n_replay_samples
          replay_samples = []
          for strategy in sampling_strategies:
            samples = replay_buffer.replay(
                batch_envs,
                n_samples=n_samples,
                sampling_strategy=strategy,
                **replay_args)
            # MAPO uses replay probability as weight while MML doesn't
            if FLAGS.use_replay_prob_as_weight:
              samples = self.reweight_samples(
                  samples, use_clipping=FLAGS.use_memory_weight_clipping)
            else:
              # Fixed replay weight is equal to 1.0, used for MML, IML
              samples = agent_factory.scale_probs(samples,
                                                  FLAGS.fixed_replay_weight)
            samples = sorted(samples, key=lambda x: x.traj.env_name)
            replay_samples += samples

          train_samples = []
          if FLAGS.use_replay_samples_in_train:
            if FLAGS.use_trainer_prob:
              replay_samples = [
                  sample._replace(prob=None) for sample in replay_samples
              ]
          train_samples += replay_samples

          if FLAGS.use_policy_samples_in_train or \
                        FLAGS.use_nonreplay_samples_in_train:
            policy_samples = self.generate_policy_samples(envs=batch_envs)
            samples = policy_samples

            if FLAGS.use_nonreplay_samples_in_train:
              nonreplay_samples = []
              for sample in policy_samples:
                if not replay_buffer.contain(sample.traj):
                  nonreplay_samples.append(sample)
              samples = nonreplay_samples
              tf.logging.info('{} non replay samples'.format(len(samples)))

            samples = self.reweight_samples(
                samples,
                use_clipping=FLAGS.use_memory_weight_clipping,
                policy_samples=True)

            train_samples += samples
            # Add sim_features to the samples to be added to the replay buffer
            if add_features_to_samples:
              samples = agent_factory.samples_with_features(
                  batch_envs, samples, FLAGS.max_n_exp)

            replay_buffer.save(samples)

          tf.logging.info('{} train samples'.format(len(train_samples)))
          yield train_samples
      i += 1


class Evaluator():

  def __init__(self, name, fns):
    self.name = name
    self.fns = fns
    self.eval_name = 'test' if FLAGS.eval_only else FLAGS.eval_dev

  def select_top(self, samples):
    top_dict = {}
    for sample in samples:
      name = sample.traj.env_name
      prob = sample.prob
      if name not in top_dict or prob > top_dict[name].prob:
        top_dict[name] = sample
    return top_dict.values()

  def run(self):
    agent, envs = init_experiment(
        self.fns,
        use_gpu=FLAGS.eval_use_gpu,
        gpu_id=str(FLAGS.eval_gpu_id),
        eval_fn=True)

    experiment_dir = get_experiment_dir()
    for env in envs:
      env.punish_extra_work = False
    graph = agent.model.graph
    dev_writer = tf.summary.FileWriter(
        os.path.join(experiment_dir, FLAGS.tb_log_dir, self.eval_name))
    best_dev_avg_return = 0.0
    best_model_path = ''
    best_model_dir = os.path.join(experiment_dir, FLAGS.best_model_dir)
    if not gfile.Exists(best_model_dir):
      gfile.MakeDirs(best_model_dir)
    i = 0
    current_ckpt = get_init_model_path()
    ckpt_dir = os.path.join(experiment_dir, FLAGS.saved_model_dir)
    env_dict = dict([(env.name, env) for env in envs])
    is_meta_eval = (FLAGS.eval_dev == 'meta') and FLAGS.meta_learn
    if is_meta_eval:
      sample_generator = SampleGenerator(envs, agent, is_val=True)
      rep_buffer = sample_generator.replay_buffer
      sample_generator = sample_generator.generate_replay_samples()
      beam_buffer = agent_factory.AllGoodReplayBuffer(de_vocab=envs[0].de_vocab)
      buf_writer = tf.summary.FileWriter(
          os.path.join(experiment_dir, FLAGS.tb_log_dir, 'buffer'))
      beam_writer = tf.summary.FileWriter(
          os.path.join(experiment_dir, FLAGS.tb_log_dir, 'beam'))
      samples_file = os.path.join(experiment_dir, 'val_beam_samples.json')
      all_samples_file = os.path.join(experiment_dir,
                                      'all_val_beam_samples.json')
    while True:
      t1 = time.time()
      true_n = len(envs)
      tf.logging.info('dev: iteration {}, evaluating {}.'.format(
          i, current_ckpt))
      env_batch_size = FLAGS.eval_batch_size
      env_iterator = data_utils.BatchIterator(
          dict(envs=envs), shuffle=False, batch_size=env_batch_size)
      dev_samples = []
      dev_samples_in_beam = []
      for j, batch_dict in enumerate(env_iterator):
        t3 = time.time()
        batch_envs = batch_dict['envs']
        tf.logging.info('=' * 50)
        tf.logging.info('{} iteration {}, batch {}: {} envs'.format(
            self.name, i, j, len(batch_envs)))
        new_samples_in_beam = agent.beam_search(
            batch_envs, beam_size=FLAGS.eval_beam_size, renorm=False)
        dev_samples_in_beam += new_samples_in_beam
        tf.logging.info('{} samples in beam, batch {}.'.format(
            len(new_samples_in_beam), j))
        t4 = time.time()
        tf.logging.info('{} sec used in evaluator batch {}.'.format(t4 - t3, j))

      # Account for beam search where the beam doesn't
      # contain any examples without error, which will make
      # len(dev_samples) smaller than len(envs).
      prob_dict = {}
      for sample in dev_samples_in_beam:
        env_name = sample.traj.env_name
        prob = sample.prob * sum(sample.traj.rewards)
        if env_name not in prob_dict:
          prob_dict[env_name] = [prob]
        else:
          prob_dict[env_name].append(prob)
      avg_total_prob = sum([np.mean(v) for v in prob_dict.values()]) / true_n
      dev_samples = self.select_top(dev_samples_in_beam)

      if is_meta_eval:
        old_dev_samples, new_dev_samples = [], []
        for s in dev_samples:
          if rep_buffer.check_not_in_buffer(s):
            new_dev_samples.append(s)
          else:
            old_dev_samples.append(s)
        beam_buffer.save(new_dev_samples)
        # Update the rep buffer too in order to only save the new dev samples
        rep_buffer.save(new_dev_samples)
        tf.logging.info('Saving new beam samples...')
        with gfile.Open(samples_file, 'w') as f:
          json.dump(beam_buffer.get_all_progs(), f, sort_keys=True, indent=2)
        tf.logging.info('{} saved.'.format(samples_file))
        # Recreate the beam_buffer
        beam_buffer = agent_factory.AllGoodReplayBuffer(
            de_vocab=envs[0].de_vocab)

        new_dev_prob = sum(
            [s.prob * sum(s.traj.rewards) for s in new_dev_samples]) / true_n
        old_dev_prob = sum(
            [s.prob * sum(s.traj.rewards) for s in old_dev_samples]) / true_n
        new_dev_samples = agent_factory.normalize_probs(new_dev_samples)
        old_dev_samples = agent_factory.normalize_probs(old_dev_samples)
        dev_samples = new_dev_samples + old_dev_samples
        # total_prob = new_dev_prob + old_dev_prob

        dev_avg_return, _ = agent.evaluate(
            new_dev_samples,
            writer=beam_writer,
            true_n=true_n,
            extra_monitors=dict(avg_prob=new_dev_prob))
        tf.logging.info('{} only beam avg return.'.format(dev_avg_return))

        # val_samples = sample_generator.next()
        # probs = agent.compute_probs([s.traj for s in val_samples])
        extra_buf_monitors = dict(avg_prob=old_dev_prob)
        dev_avg_return, _ = agent.evaluate(
            old_dev_samples,
            writer=buf_writer,
            true_n=true_n,
            extra_monitors=extra_buf_monitors)
        tf.logging.info('{} only buffer avg return.'.format(dev_avg_return))
      else:
        dev_samples = agent_factory.normalize_probs(dev_samples)

      dev_avg_return, dev_avg_len = agent.evaluate(
          dev_samples,
          writer=dev_writer,
          true_n=true_n,
          extra_monitors=dict(avg_prob=avg_total_prob))
      tf.logging.info('{} samples in non-empty beam.'.format(len(dev_samples)))
      tf.logging.info('true n is {}'.format(len(envs)))
      tf.logging.info('{} questions in dev set.'.format(len(envs)))
      tf.logging.info('{} {} avg return.'.format(self.eval_name,
                                                 dev_avg_return))
      global_step = agent.model.get_global_step()

      tf.logging.info('dev: avg return: {}, avg length: {}.'.format(
          dev_avg_return, dev_avg_len))
      if FLAGS.eval_only or (FLAGS.eval_dev == 'validation'):
        if dev_avg_return > best_dev_avg_return:
          best_model_path = graph.save(
              os.path.join(best_model_dir, 'model'), global_step)
          best_dev_avg_return = dev_avg_return
          tf.logging.info('New best {} avg returns is {}'.format(
              self.eval_name, best_dev_avg_return))
          tf.logging.info(
              'New best model is saved in {}'.format(best_model_path))
          with gfile.Open(
              os.path.join(experiment_dir, 'best_model_info.json'), 'w') as f:
            result = {'best_model_path': best_model_path}
            if FLAGS.eval_only:
              result['best_eval_avg_return'] = best_dev_avg_return
            else:
              result['best_dev_avg_return'] = best_dev_avg_return
            json.dump(result, f)

      if FLAGS.eval_only:
        # Save the decoding results for further.
        dev_programs_in_beam_dict = {}
        for sample in dev_samples_in_beam:
          name = sample.traj.env_name
          program = agent_factory.traj_to_program(sample.traj, envs[0].de_vocab)
          answer = sample.traj.answer
          reward = np.sum(sample.traj.rewards)
          if name in dev_programs_in_beam_dict:
            dev_programs_in_beam_dict[name].append((program, answer,
                                                    sample.prob, reward))
          else:
            dev_programs_in_beam_dict[name] = [(program, answer, sample.prob,
                                                reward)]

        t3 = time.time()
        with gfile.Open(
            os.path.join(experiment_dir,
                         'dev_programs_in_beam_{}.json'.format(i)), 'w') as f:
          json.dump(dev_programs_in_beam_dict, f)
        t4 = time.time()
        tf.logging.info(
            '{} sec used dumping programs in beam in eval iteration {}.'.format(
                t4 - t3, i))

        t3 = time.time()
        dev_sample_file = os.path.join(experiment_dir,
                                       'dev_samples_{}.txt'.format(i))
        with codecs.getwriter('utf-8')(gfile.GFile(dev_sample_file, 'w')) as f:
          for sample in dev_samples:
            f.write(show_samples([sample], envs[0].de_vocab, env_dict))
        t4 = time.time()
        tf.logging.info(
            '{} sec used logging dev samples in eval iteration {}.'.format(
                t4 - t3, i))

      t2 = time.time()
      tf.logging.info('{} sec used in eval iteration {}.'.format(t2 - t1, i))

      if FLAGS.eval_only or agent.model.get_global_step() >= FLAGS.n_steps:
        tf.logging.info('{} finished'.format(self.name))
        if FLAGS.eval_only:
          print('Eval average return (accuracy) of the best model is {}'.format(
              best_dev_avg_return))
        else:
          if is_meta_eval:
            tf.logging.info('Saving all collected beam samples...')
            with gfile.Open(all_samples_file, 'w') as f:
              json.dump(rep_buffer.get_all_progs(), f, sort_keys=True, indent=2)
          print('Best {} average return (accuracy) is {}'.format(
              self.eval_name, best_dev_avg_return))
          print('Best model is saved in {}'.format(best_model_path))
        return

      # Reload on the latest model.
      new_ckpt = None
      t1 = time.time()
      while new_ckpt is None or new_ckpt == current_ckpt:
        time.sleep(5)
        new_ckpt = tf.train.latest_checkpoint(ckpt_dir)
      t2 = time.time()
      tf.logging.info(
          '{} sec used waiting for new checkpoint in evaluator.'.format(t2 -
                                                                        t1))

      tf.logging.info('latest ckpt to evaluate is {}.'.format(new_ckpt))
      tf.logging.info('{} loading ckpt {}'.format(self.name, new_ckpt))
      t1 = time.time()
      graph.restore(new_ckpt)
      t2 = time.time()
      tf.logging.info('{} sec used in {} loading ckpt {}'.format(
          t2 - t1, self.name, new_ckpt))
      current_ckpt = new_ckpt


class Learner(object):

  def __init__(self, name, fns, fns_val=None):
    self.name = name
    self.save_every_n = FLAGS.save_every_n
    self.fns = fns
    self.fns_val = fns_val

  def run(self):
    """Run the learner."""
    # Writers to record training and replay information.
    experiment_dir = get_experiment_dir()
    train_writer = tf.summary.FileWriterCache.get(
        os.path.join(experiment_dir, FLAGS.tb_log_dir))
    # replay_writer = tf.summary.FileWriter(os.path.join(
    #  get_experiment_dir(), FLAGS.tb_log_dir, 'replay'))
    saved_model_dir = os.path.join(experiment_dir, FLAGS.saved_model_dir)
    if not gfile.Exists(saved_model_dir):
      gfile.MakeDirs(saved_model_dir)
    agent, envs = init_experiment(
        self.fns,
        use_gpu=FLAGS.train_use_gpu,
        gpu_id=str(FLAGS.train_gpu_id),
        extra_monitors=True,
        fns_val=self.fns_val)
    if isinstance(envs, tuple):
      envs, envs_val = envs
      for env in envs_val:
        env.punish_extra_work = False
    agent.train_writer = train_writer
    graph = agent.model.graph
    agent.train_writer.add_graph(graph.graph)
    current_ckpt = get_init_model_path()
    sample_generator = SampleGenerator(envs, agent)
    if FLAGS.meta_learn:
      dev_generator = SampleGenerator(envs_val, agent, is_val=True)
      val_sample_generator = dev_generator.generate_replay_samples()
      samples_file = os.path.join(experiment_dir, 'val_beam_samples.json')
    replay_sample_generator = sample_generator.generate_replay_samples()
    val_samples = None

    # Initialization related to score model
    score_model = FLAGS.score_model
    if (score_model in ['tabular', 'local_linear'
                       ]) or FLAGS.score_norm_fn == 'softmax':
      replay_buffer = sample_generator.replay_buffer
      agent.env_to_index = score_utils.create_env_index(replay_buffer)
      if score_model == 'tabular':
        if FLAGS.sampling_strategy == 'reward' and FLAGS.use_model_weight_init:
          weight_init_agent = agent
        else:
          weight_init_agent = None
        init_weights = score_utils.create_init_weights(
            score_norm_fn=FLAGS.score_norm_fn,
            replay_buffer=replay_buffer,
            num_envs=len(envs),
            max_programs=FLAGS.max_programs,
            agent=weight_init_agent)
        if not restore_prempted():
          agent.model.init_score_fn(init_weights)
      elif score_model == 'linear':  # Linear Softmax
        features = score_utils.get_features(
            replay_buffer, num_envs=len(envs), max_programs=FLAGS.max_programs)
        agent.model.init_score_fn(features)
      if FLAGS.score_norm_fn == 'softmax':
        num_trajs = score_utils.get_num_trajs(replay_buffer, len(envs))
        agent.model.init_num_trajs(num_trajs)

    n_train_samples = sample_generator.n_train_samples
    i = 0
    while True:
      tf.logging.info('Start train step {}'.format(i))
      t1 = time.time()
      replay_samples = replay_sample_generator.next()
      if FLAGS.meta_learn:
        if gfile.Exists(samples_file):
          tf.logging.info('Reading the val samples file..')
          dev_generator.update_buffer(samples_file, use_top_programs_only=True)
          gfile.Remove(samples_file)
          val_sample_generator = dev_generator.generate_replay_samples()
        val_samples = val_sample_generator.next()

      train_samples = replay_samples
      t2 = time.time()
      tf.logging.info('{} secs used waiting in train step {}.'.format(
          t2 - t1, i))

      if i % self.save_every_n == 0:
        eval_samples = sample_generator.generate_policy_samples(n_samples=1)
        eval_true_n = len(envs)
        extra_buf_monitors = {}
        # if FLAGS.meta_learn:
        #   probs = agent.compute_probs([s.traj for s in val_samples])
        #   extra_buf_monitors.update(val_loss=-np.mean(probs))
        avg_return, avg_len = agent.evaluate(
            eval_samples,
            writer=train_writer,
            true_n=eval_true_n,
            extra_monitors=extra_buf_monitors)
        tf.logging.info('train: avg return: {}, avg length: {}.'.format(
            avg_return, avg_len))

      t1 = time.time()
      if train_samples:
        if FLAGS.use_trainer_prob:
          train_samples = agent.update_replay_prob(
              train_samples, min_replay_weight=FLAGS.min_replay_weight)
        for _ in xrange(FLAGS.n_opt_step):
          agent.train(
              train_samples,
              val_samples=val_samples,
              parameters=dict(
                  en_rnn_dropout=FLAGS.dropout, rnn_dropout=FLAGS.dropout),
              min_prob=FLAGS.min_prob,
              scale=n_train_samples,
              de_vocab=envs[0].de_vocab,
              debug=FLAGS.debug_model)

      t2 = time.time()
      tf.logging.info(
          '{} sec used in training train iteration {}, {} samples.'.format(
              t2 - t1, i, len(train_samples)))
      i += 1
      if i % self.save_every_n == 0:
        t1 = time.time()
        current_ckpt = graph.save(
            os.path.join(saved_model_dir, 'model'),
            agent.model.get_global_step())
        t2 = time.time()
        tf.logging.info(
            '{} sec used saving model to {}, train iteration {}.'.format(
                t2 - t1, current_ckpt, i))
      if agent.model.get_global_step() >= FLAGS.n_steps:
        # Code for using Hparams plugin
        return




def main(_):
  run_experiment()


if __name__ == '__main__':
  app.run(main)
