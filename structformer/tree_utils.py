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

# Lint as: python3
"""Tree evaluation utils."""
import collections
import re

import nltk
from nltk.parse import DependencyGraph

from structformer.data_ptb import WORD_TAGS

# Dependency Tree


def get_dep(t, directed=True):
  """Get depth of t."""
  deps = []
  for _, node in t.nodes.items():
    word, label, head = node['word'], node['tag'], node['head']
    if label in WORD_TAGS:
      while (head != 0) and (t.nodes[head]['tag'] not in WORD_TAGS):
        head = t.nodes[head]['head']
      if head == 0:
        head_word = '<s>'
      else:
        head_word = t.nodes[head]['word'].lower()
      rel = (word.lower(), head_word)
      if directed:
        deps.append(rel)
      else:
        deps.append((min(rel), max(rel)))
  return deps


def evald(prd_list, trg_file_path, directed=True):
  """Compute UAS score."""
  with open(trg_file_path, 'r') as trg_file:
    trg_string = trg_file.read().strip()
    trg_string_list = trg_string.split('\n\n')
    try:
      trg_list = [
          DependencyGraph(t, top_relation_label='root') for t in trg_string_list
      ]
    except ValueError:

      def extract_10_cells(cells, index):
        line_index, word, lemma, tag, _, head, rel, _, _, _ = cells
        try:
          index = int(line_index)
        except ValueError:
          # index can't be parsed as an integer, use default
          pass
        return index, word, lemma, tag, tag, '', head, rel

      trg_list = [
          DependencyGraph(
              t, top_relation_label='root', cell_extractor=extract_10_cells)
          for t in trg_string_list
      ]

  correct = 0.0
  total = 0.0
  for prd, trg in zip(prd_list, trg_list):
    # assert len(prd.nodes) == len(trg.nodes)
    prd_deps = get_dep(prd, directed)
    trg_deps = get_dep(trg, directed)
    if len(prd_deps) != len(trg_deps):
      print(prd)
      print(prd_deps, len(prd_deps))
      print(trg)
      print(trg_deps, len(trg_deps))
      raise Exception

    for d in prd_deps:
      if d in trg_deps:
        correct += 1
    total += len(prd_deps)
  acc = correct / total

  if directed:
    print('DDA: %.3f' % acc)
  else:
    print('UDA: %.3f' % acc)
  return acc


# Constituency Tree


def average_depth(parse):
  """compute average tree depth."""
  depths = []
  current_depth = 0
  for token in parse.split():
    if token == '(':
      current_depth += 1
    elif token == ')':
      current_depth -= 1
    else:
      depths.append(current_depth)
  if depths:
    return float(sum(depths)) / len(depths)
  else:
    pass


def corpus_average_depth(corpus):
  """compute corpus-level average tree depth."""
  local_averages = []
  for key in corpus:
    s = corpus[key]
    if average_depth(s) is not None:
      local_averages.append(average_depth(s))
    else:
      pass
  return float(sum(local_averages)) / len(local_averages)


def mrg(tr):
  """return constituency string."""
  if isinstance(tr, str):
    return tr + ' '
  else:
    s = '( '
    for subtr in tr:
      s += mrg(subtr)
    s += ') '
    return s


def mrg_labeled(tr):
  """return labeled constituency string."""
  if isinstance(tr, nltk.Tree):
    if tr.label() in WORD_TAGS:
      return tr.leaves()[0] + ' '
    else:
      s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
      for subtr in tr:
        s += mrg_labeled(subtr)
      s += ') '
      return s
  else:
    return ''


def list2tree(node):
  """convert list instance to nltk.Tree."""
  if isinstance(node, list):
    tree = []
    for child in node:
      tree.append(list2tree(child))
    return nltk.Tree('<l>', tree)
  elif isinstance(node, dict):
    return nltk.Tree(node['tag'], [node['word']])


def build_tree(depth, sen, gap=0):
  """build constituency tree from syntactic distance."""
  assert len(depth) >= 0
  assert len(depth) == len(sen)

  if len(depth) == 1:
    parse_tree = sen[0]
  else:
    max_depth = max(depth[:-1])
    assert depth[-1] > max_depth
    parse_tree = []
    sub_sen = []
    sub_depth = []
    for d, w in zip(depth, sen):
      sub_sen.append(w)
      sub_depth.append(d)
      if d >= max_depth - gap:
        parse_tree.append(build_tree(sub_depth, sub_sen, gap))
        sub_sen = []
        sub_depth = []
  return parse_tree


def get_brackets(tree, idx=0):
  """return a set of unlabeled constituents."""
  brackets = set()
  if isinstance(tree, list) or isinstance(tree, nltk.Tree):
    for node in tree:
      node_brac, next_idx = get_brackets(node, idx)
      if next_idx - idx > 1:
        brackets.add((idx, next_idx))
        brackets.update(node_brac)
      idx = next_idx
    return brackets, idx
  else:
    return brackets, idx + 1


def spaceify(parse):
  return parse  # .replace("(", "( ").replace(")", " )")


def to_indexed_contituents(parse):
  """return unlabeled constituents."""
  if parse.count('(') != parse.count(')'):
    print(parse)
  parse = spaceify(parse)
  sp = parse.split()
  if len(sp) == 1:
    return set([(0, 1)]), 1

  backpointers = []
  indexed_constituents = set()
  word_index = 0
  first_op = -1
  for index, token in enumerate(sp):
    if token == '(':
      backpointers.append(word_index)
    elif token == ')':
      start = backpointers.pop()
      end = word_index
      constituent = (start, end)
      indexed_constituents.add(constituent)
    elif '[' in token:
      if first_op == -1:
        first_op = index
      else:
        pass
    else:
      word_index += 1

  return indexed_constituents, word_index


def to_indexed_contituents_labeled(parse):
  """return labeled constituents."""
  sp = parse.split()
  if len(sp) == 1:
    return set([(0, 1)])

  backpointers = []
  indexed_constituents = set()
  word_index = 0
  for _, token in enumerate(sp):
    if token[0] == '(':
      backpointers.append((word_index, token[1:]))
    elif token == ')':
      start, typ = backpointers.pop()
      end = word_index
      constituent = (start, end, typ)
      if end - start > 1:
        indexed_constituents.add(constituent)
    else:
      word_index += 1
  return indexed_constituents, word_index


def example_labeled_acc(c1, c2):
  """Compute the number of non-unary constituents of each type in the labeled (non-binirized) parse appear in the model output."""
  correct = collections.Counter()
  total = collections.Counter()
  for constituent in c2:
    if (constituent[0], constituent[1]) in c1:
      correct[constituent[2]] += 1
    total[constituent[2]] += 1
  return correct, total


def corpus_stats_labeled(corpus_unlabeled, corpus_labeled):
  """Compute labeled constituency accuracy."""

  correct = collections.Counter()
  total = collections.Counter()

  for key in corpus_labeled:
    c1, nwords1 = to_indexed_contituents(corpus_unlabeled[key])
    c2, nwords2 = to_indexed_contituents_labeled(corpus_labeled[key])
    assert nwords1 == nwords2
    if not c2:
      continue

    ex_correct, ex_total = example_labeled_acc(c1, c2)
    correct.update(ex_correct)
    total.update(ex_total)
  return correct, total
