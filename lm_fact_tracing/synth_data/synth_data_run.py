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

"""Synthesize example data with synth_data_lib."""
import os
from typing import Sequence

from absl import app
from absl import flags
from lm_fact_tracing.synth_data import synth_data_lib

_FILE_DIR = os.path.dirname(__file__)
_DEFAULT_ENTITY_VOCABULARY = os.path.join(_FILE_DIR, 'names.txt')
_DEFAULT_RELATION_VOCABULARY = os.path.join(_FILE_DIR, 'relations.txt')

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Output data path.', required=True)

_NUM_ENTITIES = flags.DEFINE_integer('num_entities', 5_000,
                                     'Number of entities.')

_NUM_RELATIONS = flags.DEFINE_integer('num_relations', 37,
                                      'Number of relations.')

_NUM_FACTS = flags.DEFINE_integer('num_facts', 50_000, 'Number of facts.')

_NUM_TEST_UTTERANCES = flags.DEFINE_integer('num_test_utterances', 5_000,
                                            'Number of test utterances.')

_NUM_ENTITY_WORDS = flags.DEFINE_integer('num_entity_words', 1,
                                         'Number of entity words.')

_NUM_EXPRESSIONS_PER_ENTITY = flags.DEFINE_integer(
    'num_expressions_per_entity', 4, 'Number of experessions per entity.')

_NUM_FACTS_PER_UTTERANCE = flags.DEFINE_integer(
    'num_facts_per_utterance', 2,
    'How many utterance to concatenate in train time.')

_ENTITY_VOCABULARY = flags.DEFINE_string(
    'entity_vocabulary', _DEFAULT_ENTITY_VOCABULARY,
    'Path to file containing entity strings.')

_RELATION_VOCABULARY = flags.DEFINE_string(
    'relation_vocabulary', _DEFAULT_RELATION_VOCABULARY,
    'Path to file containing relation strings.')


def main(unused_args):
  dataset = synth_data_lib.make_dataset(
      relation_vocabulary=_RELATION_VOCABULARY.value,
      entity_vocabulary=_ENTITY_VOCABULARY.value,
      num_relations=_NUM_RELATIONS.value,
      num_entities=_NUM_ENTITIES.value,
      num_facts=_NUM_FACTS.value,
      num_entity_words=_NUM_ENTITY_WORDS.value,
      num_expressions_per_entity=_NUM_EXPRESSIONS_PER_ENTITY.value,
      num_test_utterances=_NUM_TEST_UTTERANCES.value,
      num_facts_per_utterance=_NUM_FACTS_PER_UTTERANCE.value)

  synth_data_lib.write_dataset(_OUTPUT_PATH.value, dataset)


if __name__ == '__main__':
  app.run(main)
