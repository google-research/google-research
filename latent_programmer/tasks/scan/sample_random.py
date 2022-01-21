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

"""Sample a random SCAN task for some experiment."""

import enum
import random
from typing import Callable, Dict, List, Optional

import tensorflow as tf

from latent_programmer.tasks.scan import scan_vocab
from latent_programmer.tasks.scan import translate_scan


@enum.unique
class ScanExperiment(enum.Enum):
  """Kinds of decomposition-based generalization for SCAN tasks."""
  # No generalization.
  NONE = 0

  # For train, tasks have 1-4 parts. For test, tasks have 5-6 parts.
  LENGTH_1_4_TO_5_6 = 1

  # For train, tasks have exactly 4 parts. For test, tasks have 1-6 parts (but
  # not 4 parts).
  LENGTH_4_TO_1_6 = 2

  # For train, tasks have exactly 1 part. For test, tasks have 2-4 parts.
  LENGTH_1_TO_2_4 = 3

  # For train, half of the tasks have "left" as the only direction, and the
  # other half have "right" as the only direction. For test, all tasks have both
  # "left" and "right". All tasks have 2-4 parts, and each part must have a
  # direction component.
  COMPOSE_DIFFERENT_CONCEPTS = 4

  # For train, the tasks have only "left" in the first half of parts and only
  # "right" in the second half of parts. For test, the ordering is reversed. All
  # tasks have 2-4 parts, and each part must have a direction component.
  SWITCH_CONCEPT_ORDER = 5

  # For train, 10% of tasks are just "jump", and the other 90% of tasks do not
  # contain "jump". For test, all tasks contain "jump" (but aren't just "jump"
  # itself). This is SCAN's "addprim_jump" task, generalized to tasks with more
  # parts (all tasks have 1-4 parts).
  COMPOSE_NEW_OP = 6

  # For train, the tasks do not contain "around right". For test, the tasks all
  # contain "around right". This is SCAN's "template_around_right" task,
  # generalized to tasks with more parts (all tasks have 1-4 parts).
  EXTEND_OP_FUNCTIONALITY = 7


TEMPLATES = [
    ['<PRIM_NO_TURN>'],
    ['<PRIM>', '<DIR>'],
    ['<PRIM>', 'opposite', '<DIR>'],
    ['<PRIM>', 'around', '<DIR>'],
]
DIRECTIONS = ['left', 'right']
PRIMITIVES_NO_TURN = ['walk', 'look', 'run', 'jump']
PRIMITIVES = PRIMITIVES_NO_TURN + ['turn']
REPEATS = ['twice', 'thrice']
CONJUNCTIONS = ['and', 'after']

# Used for COMPOSE_DIFFERENT_CONCEPTS and SWITCH_CONCEPT_ORDER, where the
# concepts are either "left" or "right".
TEMPLATES_REQUIRE_DIRECTION = TEMPLATES[1:]


def generate_task(
    choose_num_parts,
    templates = None,
    directions = None,
    keep_fn = None):
  """Returns a random task with a good number of parts."""
  if not templates:
    templates = TEMPLATES
  if not directions:
    directions = DIRECTIONS
  if not keep_fn:
    keep_fn = lambda _: True

  num_parts = choose_num_parts()

  while True:
    parts = []
    for _ in range(num_parts):
      filled = []
      for token in random.choice(templates):
        if token == '<PRIM_NO_TURN>':
          filled.append(random.choice(PRIMITIVES_NO_TURN))
        elif token == '<PRIM>':
          filled.append(random.choice(PRIMITIVES))
        elif token == '<DIR>':
          filled.append(random.choice(directions))
        else:
          filled.append(token)
      parts.append(filled)
      if random.random() < 0.5:
        filled.append(random.choice(REPEATS))

    tokens = []
    needs_conjunction = False
    for part in parts:
      if needs_conjunction:
        tokens.append(random.choice(CONJUNCTIONS))
      tokens.extend(part)
      needs_conjunction = True

    if keep_fn(tokens):
      return tokens


def generate_task_switch_concept_order(is_train):
  """Returns a random task for SWITCH_CONCEPT_ORDER."""
  num_parts = random.randint(2, 4)

  first_half_num_parts = num_parts // 2
  second_half_num_parts = num_parts - first_half_num_parts
  first_half_direction = ['left'] if is_train else ['right']
  second_half_direction = ['right'] if is_train else ['left']

  first_half_tokens = generate_task(
      choose_num_parts=lambda: first_half_num_parts,
      templates=TEMPLATES_REQUIRE_DIRECTION,
      directions=first_half_direction)
  second_half_tokens = generate_task(
      choose_num_parts=lambda: second_half_num_parts,
      templates=TEMPLATES_REQUIRE_DIRECTION,
      directions=second_half_direction)
  conjunction = random.choice(CONJUNCTIONS)

  return first_half_tokens + [conjunction] + second_half_tokens


def tokens_contains(tokens, must_contain_str):
  return must_contain_str in ' '.join(tokens)


def sample_task(experiment, is_train):
  """Samples a random task."""
  if experiment == ScanExperiment.SWITCH_CONCEPT_ORDER.name:
    # Handle this case separately because it's the most different from the rest.
    return generate_task_switch_concept_order(is_train)

  choose_num_parts = None
  templates = None
  directions = None
  keep_fn = None

  if experiment == ScanExperiment.NONE.name:
    choose_num_parts = lambda: random.randint(1, 6)

  elif experiment == ScanExperiment.LENGTH_1_4_TO_5_6.name:
    choose_num_parts = ((lambda: random.randint(1, 4)) if is_train else
                        (lambda: random.randint(5, 6)))

  elif experiment == ScanExperiment.LENGTH_4_TO_1_6.name:
    choose_num_parts = ((lambda: 4) if is_train else
                        (lambda: random.choice([1, 2, 3, 5, 6])))

  elif experiment == ScanExperiment.LENGTH_1_TO_2_4.name:
    choose_num_parts = ((lambda: 1) if is_train else
                        (lambda: random.randint(2, 4)))

  elif experiment == ScanExperiment.COMPOSE_DIFFERENT_CONCEPTS.name:
    choose_num_parts = lambda: random.randint(2, 4)
    templates = TEMPLATES_REQUIRE_DIRECTION
    if is_train:
      directions = [random.choice(DIRECTIONS)]
    else:
      keep_fn = lambda tokens: (tokens_contains(tokens, 'left') and  # pylint: disable=g-long-lambda
                                tokens_contains(tokens, 'right'))

  elif experiment == ScanExperiment.COMPOSE_NEW_OP.name:
    choose_num_parts = lambda: random.randint(1, 4)
    if is_train:
      if random.random() < 0.1:
        return ['jump']
      else:
        keep_fn = lambda tokens: not tokens_contains(tokens, 'jump')
    else:
      keep_fn = lambda tokens: (tokens_contains(tokens, 'jump') and  # pylint: disable=g-long-lambda
                                len(tokens) != 1)

  elif experiment == ScanExperiment.EXTEND_OP_FUNCTIONALITY.name:
    choose_num_parts = lambda: random.randint(1, 4)
    if is_train:
      keep_fn = lambda tokens: not tokens_contains(tokens, 'around right')
    else:
      keep_fn = lambda tokens: tokens_contains(tokens, 'around right')

  else:
    raise ValueError('Unhandled experiment: {}'.format(experiment))

  assert choose_num_parts is not None

  return generate_task(choose_num_parts,
                       templates=templates,
                       directions=directions,
                       keep_fn=keep_fn)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(input_tokens,
                      token_id_table,
                      output_separators):
  """Creates a tf.Example message to be written to a file."""
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  input_ids = scan_vocab.encode(input_tokens, token_id_table)
  input_str = ' '.join(map(str, input_ids))

  output_tokens = translate_scan.translate(input_tokens,
                                           add_separators=output_separators)
  if output_separators:
    parts = ' '.join(output_tokens).split(scan_vocab.SEP)
    parts_ids = [scan_vocab.encode_str(part_str, token_id_table)
                 for part_str in parts]
    parts_strs = [' '.join(map(str, ids)) for ids in parts_ids]
    output_str = scan_vocab.SEP.join(parts_strs)
  else:
    output_ids = scan_vocab.encode(output_tokens, token_id_table)
    output_str = ' '.join(map(str, output_ids))

  feature = {
      'input': _bytes_feature(str.encode(input_str)),
      'output': _bytes_feature(str.encode(output_str)),
  }

  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def write_examples(filename, num_tasks, experiment, split,
                   output_separators):
  """Generate examples and write them to a file."""
  _, token_id_table = scan_vocab.build_token_tables()

  with tf.io.TFRecordWriter(filename) as writer:
    for i in range(num_tasks):
      if split in ['train', 'valid']:
        is_train = True
      elif split == 'test':
        is_train = False
      elif split == 'finetune':
        is_train = bool(i % 2)
      elif split is None and experiment == ScanExperiment.NONE.name:
        is_train = True  # This doesn't matter.
      else:
        raise ValueError('Unhandled split: {}'.format(split))
      task = sample_task(experiment, is_train)

      example = serialize_example(task, token_id_table,
                                  output_separators=output_separators)
      writer.write(example)
