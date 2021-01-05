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

# Lint as: python3
"""Library for simplifying SPARQL queries."""

import collections
import enum
import re

from absl import logging


class SimplifyFunction(enum.Enum):
  DO_NOTHING = 0
  GROUP_SUBJECTS = 1
  GROUP_SUBJECTS_AND_OBJECTS = 2
  GROUP_SUBJECTS_AND_OBJECTS_AND_SORT = 3


def get_relation(clause):
  clause = clause.strip()
  if clause.startswith('FILTER'):
    matches = re.match(r'FILTER \( (.*) != (.*) \)', clause)
    assert matches, f'Invalid FILTER clause: {clause}'
    subj, rel, obj = matches.group(1), 'not', matches.group(2)
  else:
    subj, rel, obj = clause.split(' ')
  return (subj, rel, obj)


def group_subjects(prefix, clauses):
  """This is the implementation of function F1."""
  edges = collections.defaultdict(list)
  for clause in clauses:
    subj, rel, obj = get_relation(clause)
    edges[subj].append(f'{rel} {obj}')

  output = f'{prefix} '
  for subj, rels in edges.items():
    output += subj + ' { ' + ' . '.join(rels) + ' } '
  return output.strip()


def group_subjects_and_objects(prefix, clauses, sort=False):
  """This is the implementation of F2 (sort=False) and F3 (sort=True)."""

  edges = collections.defaultdict(lambda: collections.defaultdict(list))

  for clause in clauses:
    subj, rel, obj = get_relation(clause)
    edges[subj][rel].append(obj)

  output = prefix + ' '
  subjects = edges.keys()
  if sort:
    subjects = sorted(subjects)
  for subj in subjects:
    output += subj + ' { '
    targets = edges[subj]
    rels = targets.keys()
    if sort:
      rels = sorted(rels)
    for rel in rels:
      output += rel + ' { ' + ' '.join(targets[rel]) + ' } '
    output += '} '
  return output.strip()


def rewrite(query,
            simplify_function = SimplifyFunction.DO_NOTHING):
  """Rewrites SPARQL according to the given simplifying function."""
  if simplify_function == SimplifyFunction.DO_NOTHING:
    return query
  logging.info('Rewriting %s', query)
  matches = re.match('(.*){(.*)}', query)
  assert matches, f'Invalid SPARQL: {query}'
  prefix, clauses = matches.group(1), matches.group(2).split(' . ')
  # Prefix is either 'SELECT count' or 'SELECT DISTINCT'
  prefix = prefix.split()[1].upper()

  return {
      SimplifyFunction.GROUP_SUBJECTS:
          group_subjects(prefix, clauses),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS:
          group_subjects_and_objects(prefix, clauses),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS:
          group_subjects_and_objects(prefix, clauses),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS_AND_SORT:
          group_subjects_and_objects(prefix, clauses, sort=True),
  }[simplify_function]
