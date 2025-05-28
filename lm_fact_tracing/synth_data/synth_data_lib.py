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

"""Utilities for generating synthetic data for influence diagnostics."""
import abc
import dataclasses
import itertools
import os
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from absl import logging
import numpy as np
import tensorflow as tf


def convert_number_to_roman_literals(number):
  """Converts intege numbers to roman literals."""
  num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
  sym = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC', 'C', 'CD', 'D', 'CM', 'M']
  i = 12
  output = ''
  while number:
    div = number // num[i]
    number %= num[i]

    while div:
      output += sym[i]
      div -= 1
    i -= 1
  return output


class Fact(NamedTuple):
  """Represents a 'fact' about the world."""
  relation: int
  entities: Tuple[int, Ellipsis]

  @property
  def guid(self):
    """Two facts have the same GUID iff they have the same attribute values."""
    entity_str = '-'.join(f'e{ent}' for ent in self.entities)
    return f'r{self.relation}-{entity_str}'

  def __repr__(self):
    return f'{self.relation}: {self.entities}'


class Utterance(metaclass=abc.ABCMeta):
  """Represents a natural language expression of a Facts."""

  @property
  @abc.abstractmethod
  def guid(self):
    """Unique id of utterance."""
    pass

  @property
  @abc.abstractmethod
  def text(self):
    """textual form of utterance."""
    pass

  @abc.abstractmethod
  def to_t5_examples(self, mask_token = '<extra_id_0>'):
    """Converts the Utterance into multiple T5 span corruption examples."""
    pass


@dataclasses.dataclass(eq=True, frozen=True)
class SingleFactUtterance(Utterance):
  """Represents a natural language expression of a single Fact."""
  relation_expression: str
  entity_expressions: Tuple[str, Ellipsis]
  source_fact: Fact
  relation_expression_id: int
  entity_expression_ids: Tuple[int, Ellipsis]

  @property
  def guid(self):
    """Two utterances have same GUID iff they have the same attribute values."""
    entity_str = '-'.join(f'ex{exid}' for exid in self.entity_expression_ids)
    return (
        f'{self.source_fact.guid}-rx{self.relation_expression_id}-{entity_str}')

  def __repr__(self):
    return f'[{self.source_fact}] "{self.text}"'

  @property
  def text(self):
    return self.relation_expression.format(*self.entity_expressions)

  def to_t5_examples(self, mask_token = '<extra_id_0>'):
    """Converts the Utterance into multiple T5 span corruption examples."""
    examples = []
    for idx, ent in enumerate(self.entity_expressions):
      input_args = list(self.entity_expressions)
      input_args[idx] = mask_token
      input_str = self.relation_expression.format(*input_args)
      output_str = mask_token + ' ' + ent
      examples.append((input_str, output_str, f's{idx}'))
    return examples

  def __iter__(self):
    return iter(dataclasses.astuple(self))


@dataclasses.dataclass
class MultiFactUtterance(Utterance):
  """Represents a natural language expression of multiple Facts together."""
  utterances: Sequence[SingleFactUtterance]

  @property
  def guid(self):
    return ','.join([utterance.guid for utterance in self.utterances])

  @property
  def text(self):
    return ','.join([utterance.text for utterance in self.utterances])

  def to_t5_examples(self, mask_token = '<extra_id_0>'):
    """Converts the Utterance into multiple T5 span corruption examples."""
    examples = []

    for (i, utterance) in enumerate(self.utterances):
      examples.append(utterance.to_t5_examples(mask_token=f'<extra_id_{i}>'))

    multi_examples = []
    for utt_ios in itertools.product(*examples):
      inputs, outputs, mask_idxs = zip(*utt_ios)
      multi_examples.append(
          (','.join(inputs), ' '.join(outputs), ','.join(mask_idxs)))

    return multi_examples


class FactGenerationConfig(NamedTuple):
  num_relations: int  # Total number of unique relations.
  num_entities: int  # Total number of unique entities.
  num_facts: int  # Total number of unique facts.


def generate_facts(config,
                   rng):
  """Generates a random list of unique facts.

  Args:
    config: specifies how to randomly generate facts.
    rng: a random number generator to control randomness.

  Returns:
    facts: a list of unique Facts.
  """
  facts = set()
  partial_facts = set()
  while len(facts) < config.num_facts:
    relation = rng.randint(0, config.num_relations)
    entities = tuple(rng.randint(0, config.num_entities) for _ in range(2))

    partial_fact_already_seen = False
    for arg_idx, _ in enumerate(entities):
      partial_entities = tuple(
          e if i != arg_idx else None for i, e in enumerate(entities))
      partial_fact = (relation, partial_entities)
      if partial_fact not in partial_facts:
        partial_facts.add(partial_fact)
      else:
        partial_fact_already_seen = True

    if not partial_fact_already_seen:
      facts.add(Fact(relation, entities))
  return list(facts)


class ExpresserConfig(NamedTuple):
  num_entity_words: int  # Number of words in an entity expression.
  num_expressions_per_entity: int  # Different expressions for each entity.
  relation_vocab: Sequence[str]  # Words used to express relations.
  entity_vocab: Sequence[str]  # Words used to express entities.
  num_facts_per_utterance: int  # Number of words in an entity expression.


class FactExpresser:
  """Expresses a Fact as a natural language Utterance."""

  def __init__(self, fact_config,
               express_config, rng):
    self.rng = rng

    # Randomly generate different ways to express each entity.
    assert express_config.num_expressions_per_entity == 4
    num_entities = fact_config.num_entities

    self.entity_to_expressions = []
    for entity_idx in range(num_entities):
      exp1 = 'entity-' + str(entity_idx)
      exp2 = str(entity_idx) + '-entity'
      exp3 = 'entity-' + convert_number_to_roman_literals(entity_idx)
      exp4 = convert_number_to_roman_literals(entity_idx) + '-entity'
      self.entity_to_expressions.append([exp1, exp2, exp3, exp4])

    # Randomly generate different ways to express each relation.
    # NOTE: there is a chance that we will generate the same expression for two
    # different relations.
    self.relation_to_expressions = []
    for r_idx in range(fact_config.num_relations):
      expressions = []
      # Randomly decide how many relation words will be in the expression.
      # num_relation_words = express_config.num_relation_words

      # Total expression length is relation words + entity placeholders.
      # expression_length = fact_config.relation_arity + num_relation_words

      # Randomly sample words for the relation. Same for each expression of
      # the same relation
      r_rng = np.random.RandomState(seed=r_idx)
      expression_words = r_rng.choice(express_config.relation_vocab, size=1)

      expressions = expression_words[0].split(' | ')
      self.relation_to_expressions.append(expressions)

  def express(self, fact):
    """Expresses a Fact as a natural language Utterance."""
    # Randomly choose a relation expression.
    rx_choices = self.relation_to_expressions[fact.relation]
    rx_id = self.rng.choice(len(rx_choices))
    rx = rx_choices[rx_id]

    exs = []
    ex_ids = []
    for ent in fact.entities:
      ex_choices = self.entity_to_expressions[ent]
      ex_id = self.rng.choice(len(ex_choices))
      ex = ex_choices[ex_id]
      exs.append(ex)
      ex_ids.append(ex_id)

    return SingleFactUtterance(
        relation_expression=rx,
        entity_expressions=tuple(exs),
        source_fact=fact,
        relation_expression_id=rx_id,
        entity_expression_ids=tuple(ex_ids))

  def express_all(self, fact):
    """Expresses a Fact's all possible expressions ."""
    rx_choices = self.relation_to_expressions[fact.relation]
    utterances = []
    for (rx_id, rx) in enumerate(rx_choices):
      ent_exps = [
          enumerate(self.entity_to_expressions[ent]) for ent in fact.entities
      ]
      for exp in itertools.product(*ent_exps):
        exs = []
        ex_ids = []
        for (ex_id, ex) in exp:
          exs.append(ex)
          ex_ids.append(ex_id)
        utterance = SingleFactUtterance(
            relation_expression=rx,
            entity_expressions=tuple(exs),
            source_fact=fact,
            relation_expression_id=rx_id,
            entity_expression_ids=tuple(ex_ids))
        utterances.append(utterance)
    return utterances


class EvalExample(NamedTuple):
  """A test Utterance with all the train Utterances that support it."""
  test_utterance: SingleFactUtterance
  proponents: Tuple[SingleFactUtterance, Ellipsis]

  def __repr__(self):
    proponents_str = '\n'.join('  ' + repr(x) for x in self.proponents)
    return f'TEST: {self.test_utterance}\nPROPONENTS:\n{proponents_str}'


class DatasetConfig(NamedTuple):
  num_test_utterances: int  # Number of utterances to hold out for test.


class Dataset(NamedTuple):
  facts: List[Fact]
  expresser: FactExpresser
  train_utterances: List[Utterance]
  eval_examples: List[EvalExample]


def generate_diagnostic_dataset(fact_config,
                                express_config,
                                data_config):
  """Generates a dataset for influence diagnostics."""
  rng = np.random.RandomState(0)
  logging.info('generating facts...')
  facts = generate_facts(fact_config, rng)

  # Generate multiple utterances for each fact.
  logging.info('express facts...')
  expresser = FactExpresser(fact_config, express_config, rng)
  all_utterances_grouped_by_fact = []
  for fact in facts:
    utterances = set(expresser.express_all(fact))
    all_utterances_grouped_by_fact.append(utterances)

  # For some subset of facts, hold out one utterance as test.
  logging.info('test fact indices...')
  test_fact_indices = rng.choice(
      len(facts), size=data_config.num_test_utterances, replace=False)

  logging.info('eval examples...')
  eval_examples = []
  for i in test_fact_indices:
    utt_group = list(all_utterances_grouped_by_fact[i])
    rng.shuffle(utt_group)
    test_utt = utt_group.pop()
    all_utterances_grouped_by_fact[i] = set(utt_group)
    train_utts = tuple(utt_group)
    eval_examples.append(EvalExample(test_utt, train_utts))

  logging.info('training utterances...')
  # Remaining utterances are all training utterances.
  train_utterances = []
  for utt_group in all_utterances_grouped_by_fact:
    train_utterances.extend(utt_group)
  rng.shuffle(train_utterances)

  num_facts_per_utt = express_config.num_facts_per_utterance

  if num_facts_per_utt > 1:
    train_utterances = []
    for i in range(0, len(train_utterances), num_facts_per_utt):
      train_utterances.append(
          MultiFactUtterance(utterances=train_utterances[i:i +
                                                         num_facts_per_utt]))

  logging.info('generating dataset...')
  return Dataset(
      facts=facts,
      expresser=expresser,
      train_utterances=train_utterances,
      eval_examples=eval_examples)


def make_dataset(
    entity_vocabulary,
    relation_vocabulary,
    num_relations=10,
    num_entities=10_000,
    num_facts=50_000,
    num_entity_words=2,
    num_expressions_per_entity=1,
    num_test_utterances=5_000,
    num_facts_per_utterance=1,
):
  """Generates dataset from keyword arguments."""
  with tf.io.gfile.GFile(entity_vocabulary) as f:
    entity_vocab = [line.strip() for line in f]

  with tf.io.gfile.GFile(relation_vocabulary) as f:
    relation_vocab = [line.strip() for line in f]

  assert num_relations <= len(relation_vocab)

  fact_config = FactGenerationConfig(
      num_relations=num_relations,
      num_entities=num_entities,
      num_facts=num_facts,
  )

  # On average, the number of facts each entity will participate in is:
  # num_facts * relation_arity / num_entities = 10

  express_config = ExpresserConfig(
      num_entity_words=num_entity_words,
      num_expressions_per_entity=num_expressions_per_entity,
      relation_vocab=relation_vocab,
      entity_vocab=entity_vocab,
      num_facts_per_utterance=num_facts_per_utterance)

  data_config = DatasetConfig(num_test_utterances=num_test_utterances)

  return generate_diagnostic_dataset(fact_config, express_config, data_config)


def make_bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def make_tf_example(input_str,
                    output_str,
                    guid,
                    proponent_guids = None):
  """Creates a T5 training example."""
  if not proponent_guids:
    proponent_guids = []
  feature = {
      'inputs':
          make_bytes_feature([input_str.encode()]),
      'targets':
          make_bytes_feature([output_str.encode()]),
      'guid':
          make_bytes_feature([guid.encode()]),
      'proponent_guids':
          make_bytes_feature([x.encode() for x in proponent_guids]),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_sharded_filenames(shard_pattern):
  """Converts a sharded file path into a list of actual file paths."""
  if '@' not in shard_pattern:
    return [shard_pattern]

  components = shard_pattern.split('@')
  assert len(components) == 2
  path_prefix, num_shards_str = components

  num_shards = int(num_shards_str)
  if num_shards > 99999:
    raise ValueError('num_shards must not exceed 99999.')

  sharded_paths = []
  for i in range(num_shards):
    sharded_path = f'{path_prefix}-{i:05d}-of-{num_shards:05d}'
    sharded_paths.append(sharded_path)
  return sharded_paths


def write_tf_examples(shard_pattern, examples):
  """Writes an SSTable of TF Examples.

  Args:
    shard_pattern: a shard pattern, e.g. 'path/to/file@200'
    examples: a list of examples to write.
  """
  shard_paths = generate_sharded_filenames(shard_pattern)
  per_shard_length = len(examples) // int(shard_pattern.split('@')[-1])
  example_iter = iter(examples)
  for (s, path) in enumerate(shard_paths):
    with tf.io.TFRecordWriter(path) as tfrecord_writer:
      for _ in range(s * per_shard_length, (s + 1) * per_shard_length):
        try:
          example = next(example_iter)
          tfrecord_writer.write(example.SerializeToString())
        except StopIteration:
          break


def write_dataset(output_dir, dataset):
  """Writes a Dataset to disk."""
  tf.io.gfile.makedirs(output_dir)

  logging.info('writing train dataset')
  train_examples = []
  for utt in dataset.train_utterances:
    for i, (input_str, output_str, idxs) in enumerate(utt.to_t5_examples()):
      if isinstance(utt, MultiFactUtterance) and len(utt.utterances) > 1:
        guids = utt.guid.split(',')
        idxs = idxs.split(',')
        guids = [f'{guid}-{idxs[i]}' for (i, guid) in enumerate(guids)]
        guid = ','.join(guids)
      else:
        guid = f'{utt.guid}-{idxs}'
      train_examples.append(make_tf_example(input_str, output_str, guid))

  write_tf_examples(
      os.path.join(output_dir, 'train.tfrecord@1'), train_examples)

  logging.info('writing test dataset')
  test_examples = []
  for eval_ex in dataset.eval_examples:
    test_utt = eval_ex.test_utterance

    # Track proponent GUIDs.
    proponent_guids = []
    for proponent_utt in eval_ex.proponents:
      # We assume that every possible masking of a proponent is influential.
      # TODO(kguu): is this true?
      for i, _ in enumerate(test_utt.to_t5_examples()):
        proponent_guids.append(f'{proponent_utt.guid}-s{i}')

    for i, (input_str, output_str,
            idxs) in enumerate(test_utt.to_t5_examples()):
      guid = f'{test_utt.guid}-{idxs}'
      test_examples.append(
          make_tf_example(input_str, output_str, guid, proponent_guids))

  write_tf_examples(os.path.join(output_dir, 'test.tfrecord@1'), test_examples)
