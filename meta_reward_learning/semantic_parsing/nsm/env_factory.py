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

"""A collections of environments of sequence generations tasks."""

from __future__ import division
from __future__ import print_function

import collections
import pprint

from bloom_filter import BloomFilter
import numpy as np
import tensorflow.compat.v1 as tf
from meta_reward_learning.semantic_parsing.nsm import computer_factory
from meta_reward_learning.semantic_parsing.nsm import data_utils
from meta_reward_learning.semantic_parsing.nsm import tf_utils

# EPSILON
EPS = 1e-8


def create_features(env, program_tks, max_n_exp):
  """Create features to be used in the score function."""
  # Unique decoder variables used in the program
  variables = set(
      [tk for tk in program_tks if tk[0] == 'v' and tk[1:].isdigit()])
  ns = env.interpreter.namespace
  # Variables already present in the namespace i.e. entities and column names
  ns_vars = [tk for tk in variables if tk in ns]
  str_vals = [unicode(ns[tk]['value']) for tk in ns_vars]
  vars_to_val = dict(zip(ns_vars, str_vals))

  # Tokens which represent entities
  val_ent = env.ent_props['val_ent']
  ent_vars = [tk for tk, val in vars_to_val.iteritems() if val in val_ent]
  ents_index = [val_ent[vars_to_val[tk]][0] for tk in ent_vars]
  entities = [env.entities[i] for i in ents_index]
  sum_entity_length = {
      'datetime_list': 0.0,
      'string_list': 0.0,
      'num_list': 0.0
  }
  ent_lengths = [e['length'] for e in entities]
  for ent, w in zip(entities, ent_lengths):
    sum_entity_length[ent['type']] += w

  # Feature vector
  feature_size = 12 if env.trigger_words_dict else 11
  f = np.zeros(feature_size)
  # Entity features
  ent_sum = sum(sum_entity_length.values())
  ques_ent_sum = sum(env.ent_props['sum_entity_length'].values())
  # No of fractional entities where each entity is weighted by its length.
  f[0] = ent_sum / (ques_ent_sum + EPS)
  # No of fractional entities for each entity type weighted by its length
  f[1:4] = [
      v / (env.ent_props['sum_entity_length'][k] + EPS)
      for k, v in sum_entity_length.iteritems()
  ]
  # No of fractional entities
  f[4] = len(ent_vars) / (len(env.entities) + EPS)

  # Feature that the longest entity is present in the program or not, if there
  # is than one entity that has the maximum length, this feature
  # represents the fraction of entities present in the program that has the
  # max length.
  max_len = max(ent_lengths) if ent_lengths else 0.0
  if max_len == env.ent_props['max_length']:
    max_ent_sum = data_utils.max_sum(ent_lengths)
    f[5] = max_ent_sum / (env.ent_props['max_sum'] + EPS)

  # Column features
  # Only consider columns represnting differing things, for example,
  # year-date and year-string are considered the same column
  column_vars, column_vals = [], set()
  for v in ns_vars:
    if v not in ent_vars:
      str_val = vars_to_val[v].split('-')[0]
      if str_val not in column_vals:
        column_vars.append(v)
        column_vals.add(str_val)
  cols = env.de_vocab.lookup(column_vars)
  col_features = [env.id_feature_dict[col][0] for col in cols]
  col_sum, ques_col_sum = sum(col_features), env.col_props['sum']
  f[6] = col_sum / (ques_col_sum + EPS)
  f[7] = sum([(i > 0) for i in col_features]) / (env.col_props['num'] + EPS)
  max_w = max(col_features) if col_features else 0.0
  if max_w == env.col_props['max']:
    f[8] = data_utils.max_sum(col_features) / (env.col_props['max_sum'] + EPS)

  # (1 - n) where n = fractional number of expressions in the program
  num_exp = program_tks.count('(')
  f[9] = 1.0 - (num_exp / max_n_exp)

  # Function tokens
  if env.trigger_words_dict:
    fn_score = 0.0
    fn_tks = [tk for tk in program_tks if tk in env.trigger_words_dict]
    if fn_tks:
      for tk in fn_tks:
        if env.ques_tokens & env.trigger_words_dict[tk]:
          fn_score += 1.0
      fn_score /= len(fn_tks)
    f[10] = fn_score

  # Approximate set interection similarity feature
  denominator = ((2 * ques_ent_sum + ques_col_sum) * num_exp)
  f[-1] = (2 * ent_sum + col_sum) / (denominator + EPS)
  return f


def value_to_index(entities):
  """Maps entity values to their indexes.

  Args:
    entities: A list of entities.

  Returns:
    Mapping from entity values to their indexes.
  """
  val_to_ent = {}
  for index, e in enumerate(entities):
    val = unicode(e['value'])
    if val in val_to_ent:
      val_to_ent[val].append(index)
    else:
      val_to_ent[val] = [index]
  return val_to_ent


class Environment(object):
  """Environment with OpenAI Gym like interface."""

  def step(self, action):
    """Step function.

    Args:
      action: an action to execute against the environment.

    Returns:
      observation:
      reward:
      done:
      info:
    """
    raise NotImplementedError


# Use last action and the new variable's memory location as input.
ProgrammingObservation = collections.namedtuple(
    'ProgramObservation', ['last_actions', 'output', 'valid_actions'])


class QAProgrammingEnv(Environment):
  """An RL environment wrapper around an interpreter to

  learn to write programs based on question.
  """

  def __init__(self,
               en_vocab,
               de_vocab,
               question_annotation,
               answer,
               constant_value_embedding_fn,
               score_fn,
               interpreter,
               constants=None,
               punish_extra_work=True,
               init_interp=True,
               trigger_words_dict=None,
               max_cache_size=1e4,
               name='qa_programming'):
    self.name = name
    self.en_vocab = en_vocab
    self.de_vocab = de_vocab
    self.end_action = self.de_vocab.end_id
    self.score_fn = score_fn
    self.interpreter = interpreter
    self.answer = answer
    self.question_annotation = question_annotation
    self.constant_value_embedding_fn = constant_value_embedding_fn
    self.constants = constants
    self.punish_extra_work = punish_extra_work
    self.error = False
    self.trigger_words_dict = trigger_words_dict
    tokens = question_annotation['tokens']
    if 'pos_tags' in question_annotation:
      self.ques_tokens = set(tokens + question_annotation['pos_tags'])
    else:
      self.ques_tokens = set(tokens)

    en_inputs = en_vocab.lookup(tokens)
    self.n_builtin = len(de_vocab.vocab) - interpreter.max_mem
    self.n_mem = interpreter.max_mem
    self.n_exp = interpreter.max_n_exp
    max_n_constants = self.n_mem - self.n_exp

    constant_spans = []
    constant_values = []
    if constants is None:
      constants = []
    for c in constants:
      constant_spans.append([-1, -1])
      constant_values.append(c['value'])
      if init_interp:
        self.interpreter.add_constant(value=c['value'], type=c['type'])

    for entity in question_annotation['entities']:
      # Use encoder output at start and end (inclusive) step
      # to create span embedding.
      constant_spans.append([entity['token_start'], entity['token_end'] - 1])
      constant_values.append(entity['value'])
      if init_interp:
        self.interpreter.add_constant(
            value=entity['value'], type=entity['type'])

    constant_value_embeddings = [
        constant_value_embedding_fn(value) for value in constant_values
    ]

    if len(constant_values) > (self.n_mem - self.n_exp):
      tf.logging.info(
          'Not enough memory slots for example {}, which has {} constants.'
          .format(self.name, len(constant_values)))

    constant_spans = constant_spans[:max_n_constants]
    constant_value_embeddings = constant_value_embeddings[:max_n_constants]
    self.context = (en_inputs, constant_spans, constant_value_embeddings,
                    question_annotation['features'],
                    question_annotation['tokens'])

    # Create output features.
    prop_features = question_annotation['prop_features']
    self.id_feature_dict = {}
    for name, id in de_vocab.vocab.iteritems():
      self.id_feature_dict[id] = [0]
      if name in self.interpreter.namespace:
        val = self.interpreter.namespace[name]['value']
        if ((isinstance(val, str) or isinstance(val, unicode)) and
            val in prop_features):
          self.id_feature_dict[id] = prop_features[val]

    # Create features to make calculation of score function easy
    entities = question_annotation['entities']
    for e in entities:
      if e['type'] != 'datetime_list':
        e['length'] = e['token_end'] - e['token_start']
      else:
        # For datetime entities, either token_end or token_start is incorrect,
        # so need to look at the entity itself for calculating the length
        # Also, we shouldn't consider 'xxxx' or 'xx' while calculating the
        # entity length
        e['length'] = len(
            [x for x in e['value'][0].replace('x', '').split('-') if x])
    entity_lengths = [e['length'] for e in entities]
    max_entity_length = max(entity_lengths) if entity_lengths else 0.0
    max_entity_sum = data_utils.max_sum(entity_lengths)
    sum_entity_length = {'datetime_list': 0, 'string_list': 0, 'num_list': 0}
    for e, w in zip(entities, entity_lengths):
      sum_entity_length[e['type']] += w

    self.entities = entities
    self.ent_props = dict(
        max_sum=max_entity_sum,
        max_length=max_entity_length,
        val_ent=value_to_index(entities),
        sum_entity_length=sum_entity_length)
    col_features = [v[0] for v in self.id_feature_dict.values()]
    self.col_props = dict(
        sum=sum(col_features),
        max=max(col_features) if col_features else 0.0,
        max_sum=data_utils.max_sum(col_features),
        num=sum([i > 0 for i in col_features]))

    self.cache = SearchCache(name=name, max_elements=max_cache_size)
    self.use_cache = False
    self.reset()

  def get_context(self):
    return self.context

  def step(self, action, debug=False):
    self.actions.append(action)
    if debug:
      print('-' * 50)
      print(self.de_vocab.lookup(self.valid_actions, reverse=True))
      print('pick #{} valid action'.format(action))
      print('history:')
      print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
      # print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
      print('obs')
      pprint.pprint(self.obs)

    if action < len(self.valid_actions) and action >= 0:
      mapped_action = self.valid_actions[action]
    else:
      print('-' * 50)
      # print('env: {}, cache size: {}'.format(self.name, len(self.cache._set)))
      print('action out of range.')
      print('action:')
      print(action)
      print('valid actions:')
      print(self.de_vocab.lookup(self.valid_actions, reverse=True))
      print('pick #{} valid action'.format(action))
      print('history:')
      print(self.de_vocab.lookup(self.mapped_actions, reverse=True))
      print('obs')
      pprint.pprint(self.obs)
      print('-' * 50)
      mapped_action = self.valid_actions[action]

    self.mapped_actions.append(mapped_action)

    result = self.interpreter.read_token(
        self.de_vocab.lookup(mapped_action, reverse=True))

    self.done = self.interpreter.done
    # Only when the proram is finished and it doesn't have
    # extra work or we don't care, its result will be
    # scored, and the score will be used as reward.
    if self.done and not (self.punish_extra_work and
                          self.interpreter.has_extra_work()):
      reward = self.score_fn(self.interpreter.result, self.answer)
    else:
      reward = 0.0

    if self.done and self.interpreter.result == [computer_factory.ERROR_TK]:
      self.error = True

    if result is None or self.done:
      new_var_id = -1
    else:
      new_var_id = self.de_vocab.lookup(self.interpreter.namespace.last_var)
    valid_tokens = self.interpreter.valid_tokens()
    valid_actions = self.de_vocab.lookup(valid_tokens)

    # For each action, check the cache for the program, if
    # already tried, then not valid anymore.
    if self.use_cache:
      new_valid_actions = []
      cached_actions = []
      partial_program = self.de_vocab.lookup(self.mapped_actions, reverse=True)
      for ma in valid_actions:
        new_program = partial_program + [self.de_vocab.lookup(ma, reverse=True)]
        if not self.cache.check(new_program):
          new_valid_actions.append(ma)
        else:
          cached_actions.append(ma)
      valid_actions = new_valid_actions

    self.valid_actions = valid_actions
    self.rewards.append(reward)
    ob = (tf_utils.MemoryInputTuple(
        read_ind=mapped_action,
        write_ind=new_var_id,
        valid_indices=self.valid_actions),
          [self.id_feature_dict[a] for a in valid_actions])

    # If no valid actions are available, then stop.
    if not self.valid_actions:
      self.done = True
      self.error = True

    # If the program is not finished yet, collect the
    # observation.
    if not self.done:
      # Add the actions that are filtered by cache into the
      # training example because at test time, they will be
      # there (no cache is available).
      if self.use_cache:
        valid_actions = self.valid_actions + cached_actions
        true_ob = (tf_utils.MemoryInputTuple(
            read_ind=mapped_action,
            write_ind=new_var_id,
            valid_indices=valid_actions),
                   [self.id_feature_dict[a] for a in valid_actions])
        self.obs.append(true_ob)
      else:
        self.obs.append(ob)
    elif self.use_cache:
      # If already finished, save it in the cache.
      self.cache.save(self.de_vocab.lookup(self.mapped_actions, reverse=True))

    return ob, reward, self.done, {}
    # 'valid_actions': valid_actions, 'new_var_id': new_var_id}

  def reset(self):
    self.actions = []
    self.mapped_actions = []
    self.rewards = []
    self.done = False
    valid_actions = self.de_vocab.lookup(self.interpreter.valid_tokens())
    if self.use_cache:
      new_valid_actions = []
      for ma in valid_actions:
        partial_program = self.de_vocab.lookup(
            self.mapped_actions + [ma], reverse=True)
        if not self.cache.check(partial_program):
          new_valid_actions.append(ma)
      valid_actions = new_valid_actions
    self.valid_actions = valid_actions
    self.start_ob = (tf_utils.MemoryInputTuple(self.de_vocab.decode_id, -1,
                                               valid_actions),
                     [self.id_feature_dict[a] for a in valid_actions])
    self.obs = [self.start_ob]

  def interactive(self):
    self.interpreter.interactive()
    print('reward is: %s' % self.score_fn(self.interpreter))

  def clone(self):
    new_interpreter = self.interpreter.clone()
    new = QAProgrammingEnv(
        self.en_vocab,
        self.de_vocab,
        score_fn=self.score_fn,
        question_annotation=self.question_annotation,
        constant_value_embedding_fn=self.constant_value_embedding_fn,
        constants=self.constants,
        answer=self.answer,
        interpreter=new_interpreter,
        init_interp=False)
    new.actions = self.actions[:]
    new.mapped_actions = self.mapped_actions[:]
    new.rewards = self.rewards[:]
    new.obs = self.obs[:]
    new.done = self.done
    new.name = self.name
    # Cache is shared among all copies of this environment.
    new.cache = self.cache
    new.use_cache = self.use_cache
    new.valid_actions = self.valid_actions
    new.error = self.error
    new.id_feature_dict = self.id_feature_dict
    new.punish_extra_work = self.punish_extra_work
    new.trigger_words_dict = self.trigger_words_dict
    new.ques_tokens = self.ques_tokens
    new.ent_props = self.ent_props
    new.col_props = self.col_props
    new.entities = self.entities
    return new

  def show(self):
    program = ' '.join(
        self.de_vocab.lookup([o.read_ind for o in self.obs], reverse=True))
    valid_tokens = ' '.join(
        self.de_vocab.lookup(self.valid_actions, reverse=True))
    return 'program: {}\nvalid tokens: {}'.format(program, valid_tokens)


class SearchCache(object):

  def __init__(self, name, size=None, max_elements=1e4, error_rate=1e-8):
    self.name = name
    self.size = size
    self.max_elements = max_elements
    self.error_rate = error_rate
    self._set = BloomFilter(max_elements=max_elements, error_rate=error_rate)

  def check(self, tokens):
    return ' '.join(tokens) in self._set

  @property
  def cache_set(self):
    return self._set

  @cache_set.setter
  def cache_set(self, val):
    self._set = val

  def save(self, tokens):
    string = ' '.join(tokens)
    self._set.add(string)

  def is_full(self):
    return '(' in self._set

  def reset(self):
    self._set = BloomFilter(
        max_elements=self.max_elements, error_rate=self.error_rate)
