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
"""Utility functions to parse CFQ examples."""

import dataclasses
import os
import pickle
import string
from typing import Any, List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

# Two types of questions and their prefixes
SPARQL_YNQ_HEADER = 'SELECT DISTINCT ?x0 WHERE'
SPARQL_WHQ_HEADER = 'SELECT count ( * ) WHERE'

# Order of the CFQ baseline models to select model-negative queries
CFQ_BASELINE_ORDER = ['lstm', 'universal', 'transformer']


def preprocess(question, query):
  """Pre-tokenize and change words to match CFQ preprocess output."""
  question = map(lambda c: f' {c} ' if c in string.punctuation else c, question)
  question = ' '.join(''.join(question).split())
  query = query.replace('\n', ' ').replace('count(*)', 'count ( * )')
  query = query.split(' ')
  query = [(t[3:] if t.startswith('ns:') else t) for t in query]
  query = [('m_' + t[2:] if len(t) <= 8 and t[:2] == 'm.' else t) for t in query
          ]
  query = ' '.join(query)
  return question, query


def load_model_output(
    model_output_dir):
  """Loads CFQ baseline model outputs."""
  if not os.path.exists(model_output_dir):
    raise ValueError('Model output does not exist.')
  model_output_dict = {}
  for name in ['train', 'dev', 'test']:
    question_fpath = os.path.join(model_output_dir, f'{name}_encode.txt')
    with open(question_fpath, 'r') as fd:
      questions = [line.strip() for line in fd.readlines()]
    for question in questions:
      model_output_dict[question] = {}

    # Each line of T2T output is the concatenation of (query, score) pairs,
    # joined by a '\t' character, and corresponds to the question in the same
    # line.
    model_output_fpaths = tf.io.gfile.glob(
        os.path.join(model_output_dir, f'{name}_decode_*.txt'))
    for fpath in model_output_fpaths:
      model_name = fpath[fpath.rindex('_') + 1:fpath.rindex('.')]
      with open(fpath, 'r') as fd:
        model_outputs = [line.strip() for line in fd.readlines()]
        assert len(model_outputs) == len(questions), f'{fpath} incomplete'
        for question, output in zip(questions, model_outputs):
          temp = output.split('\t')
          pairs = [(temp[2 * i], float(temp[2 * i + 1]))
                   for i in range(len(temp) // 2)]
          model_output_dict[question][model_name] = pairs
  return model_output_dict


class WhitespaceTokenizer:
  """Simple tokenizer. Will be replaced when tensorflow_text is available."""

  def tokenize(self, text):
    """Splits the input text by space."""
    return text.split(' ')


def load_text_encoder(vocab_fpath):
  """Loads tokenizer to convert tokens into ids."""
  with open(vocab_fpath, 'r') as fd:
    vocab = [line.strip() for line in fd.readlines()]
  tokenizer = WhitespaceTokenizer()
  text_encoder = tfds.features.text.TokenTextEncoder(
      vocab[1:],  # 0: <pad>
      oov_token='<OOV>',
      lowercase=False,
      tokenizer=tokenizer)
  return text_encoder


@dataclasses.dataclass
class Node:
  """Simple node class."""
  data: Any
  children: Optional[List['Node']] = None


def node_to_edge_list(node):
  """Converts tree to a flattened list of edges in a DFS manner.

  Args:
    node: the node to be flattened.

  Returns:
    A flattened list of edges.

  Example:
    Node(1, [Node(2), Node(3)]) -> [1, 2, 1, 3]
    Node(4, [Node(5, [Node(6)]), Node(7, [Node(8), Node(9)])])
      -> [4, 5, 5, 6, 4, 7, 7, 8, 7, 9]
  """
  if not node.children:
    return []

  output = []
  for child in node.children:
    output.extend([node.data, child.data])
    output.extend(node_to_edge_list(child))
  return output


def edge_list_to_node(edge_list):
  """Converts flattened list of edge to tree.

  Args:
    edge_list: the list of edges to build a tree from.

  Returns:
    The root of the constructed tree.

  Example:
    [1, 2, 1, 3] -> Node(1, [Node(2), Node(3)])
    [4, 5, 5, 6, 4, 7, 7, 8, 7, 9]
      -> Node(4, [Node(5, [Node(6)]), Node(7, [Node(8), Node(9)])])
  """
  if (not edge_list) or len(edge_list) % 2 != 0:
    raise ValueError('edge_list should be non-empty and even length.')
  num_edge = len(edge_list) // 2

  # Parse edges and add nodes
  root = Node(edge_list[0])
  node_dict = {edge_list[0]: root}
  for i in range(num_edge):
    parent_data, child_data = edge_list[2 * i], edge_list[2 * i + 1]
    if parent_data not in node_dict:
      raise ValueError(f'Node not found: {parent_data} in {edge_list}')
    if child_data in node_dict:
      raise ValueError(f'Cycle found: {child_data} in {edge_list}')
    node = node_dict[parent_data]
    if node.children is None:
      node.children = []
    child = Node(child_data)
    node.children.append(child)
    node_dict[child_data] = child

  return root


def extract_grammar_rule_tree(
    node_dict):
  """Generates a tree of GRAMMAR_RULE nodes from ruleTree."""
  node = None
  # If the current node is 'GRAMMAR_RULE', a new 'node' is made.
  if node_dict['ruleId']['type'] == 'GRAMMAR_RULE':
    # Remove trailing identifier
    grammar_value = node_dict['ruleId']['stringValue'][:-12]
    node = Node(grammar_value)

  subtrees = None
  if 'subTree' in node_dict:
    subtrees = [
        extract_grammar_rule_tree(child_dict)
        for child_dict in node_dict['subTree']
    ]
    subtrees = [n for n in subtrees if n]
    if not subtrees:
      subtrees = None
    elif len(subtrees) == 1:
      # Make 'subtrees' a singleton in case the current node is not a
      # 'GRAMMAR_RULE' node.
      subtrees = subtrees[0]

  if node:
    # If the current node is 'GRAMMAR_RULE', 'subtrees' becomes children of
    # 'node', and 'node' is returned.
    if isinstance(subtrees, Node):
      subtrees = [subtrees]
    node.children = subtrees
    return node
  else:
    # Otherwise, 'subtrees' is returned.
    return subtrees


def dfs_build_parse_tree(node, question_tokens,
                         question_token_idx):
  """Helper function that builds parse tree with DFS."""
  # Splits the grammar string of the node (e.g. YNQ=DID_DP_VP_INDIRECT)
  node_str = node.data[:node.data.index('=')]
  subtree_str = node.data[node.data.index('=') + 1:]

  pt_children = []
  if not node.children:
    # If the node doesn't have a child, subtree_str is question token(s).
    for token in subtree_str.split():
      question_token = question_tokens[question_token_idx]
      if token == '[ENTITY]':
        assert len(question_token) == 2 and question_token.startswith('M')
      else:
        assert token.lower() == question_token.lower()
      pt_children.append(Node(question_token_idx))
      question_token_idx += 1
  else:
    # If the node has children, recursively call with the children and find
    # tokens in subtree_str that are not covered by the children.
    child_idx = 0
    subtree_str_idx = 0

    while subtree_str_idx < len(subtree_str):
      # Whether the next token of subtree_str is child node or a question token
      parse_child = False
      if child_idx < len(node.children):
        child = node.children[child_idx]
        child_node_str = child.data[:child.data.index('=')]
        if subtree_str[subtree_str_idx:].startswith(child_node_str):
          parse_child = True

      if parse_child:
        pt_child, question_token_idx = dfs_build_parse_tree(
            child, question_tokens, question_token_idx)  # Recursive call here
        pt_children.append(pt_child)
        child_idx += 1
        subtree_str_idx += len(child_node_str) + 1
      else:
        if '_' in subtree_str[subtree_str_idx:]:
          underscore_idx = subtree_str.index('_', subtree_str_idx)
          unmatched_token = subtree_str[subtree_str_idx:underscore_idx]
          subtree_str_idx = underscore_idx + 1
        else:
          unmatched_token = subtree_str[subtree_str_idx:]
          subtree_str_idx = len(subtree_str)
        assert unmatched_token.lower(
        ) == question_tokens[question_token_idx].lower()
        pt_children.append(Node(question_token_idx))
        question_token_idx += 1

  pt_node = Node(node_str, pt_children)

  # If the node is a part of enumeration, remove tokens like "and" and ","
  # and flatten the depth of the enumerated tokens
  if pt_node.data == 'COMPLETED_AND' or pt_node.data == 'ENUMERATION':
    pt_node.children = [pt_node.children[0], pt_node.children[-1]]
    if pt_node.children[0].data == 'ENUMERATION':
      pt_node.children = pt_node.children[0].children + [pt_node.children[1]]

  # Shrink node hierarchy if the node only has one child.
  if len(pt_node.children) == 1 and pt_node.children[0] != 'COMPLETED_AND':
    pt_node = pt_node.children[0]

  return pt_node, question_token_idx


def dfs_structure_tokens(pt_node, question_token_idx,
                         structure_tokens):
  """Finds structure tokens in the parsed tree and assign token indices."""
  if isinstance(pt_node.data, str):
    structure_tokens.append(pt_node.data)
    pt_node.data = question_token_idx
    question_token_idx += 1

  if pt_node.children:
    for pt_child in pt_node.children:
      question_token_idx = dfs_structure_tokens(pt_child, question_token_idx,
                                                structure_tokens)
  return question_token_idx


def parse_question_rule_tree(rule_tree_dict,
                             question):
  """Generate parse tree from a ruleTree and corresponding question."""
  # Extracts a tree of GRAMMAR_RULE nodes from the ruleTree.
  rule_tree = extract_grammar_rule_tree(rule_tree_dict)
  question_tokens = question.split(' ')

  # Performs DFS on the rule tree and builds a parse tree
  question_token_idx = 0
  parse_tree, question_token_idx = dfs_build_parse_tree(rule_tree,
                                                        question_tokens,
                                                        question_token_idx)
  assert question_token_idx == len(
      question_tokens)  # Check all tokens are parsed

  # Assign indices to the structure tokens of constituency parsing
  structure_tokens = []
  dfs_structure_tokens(parse_tree, question_token_idx, structure_tokens)

  return parse_tree, structure_tokens


def dfs_increase_tree_value(node, cond_value, increase_value):
  """Increases all the values greater then cond_value by increase_value."""
  if node.data > cond_value:
    node.data += increase_value

  if node.children:
    for child in node.children:
      dfs_increase_tree_value(child, cond_value, increase_value)


def preprocess_question_tree(question_tree, question):
  """Shifts token indices for the tokens "'s" that will be split."""
  question_tokens = question.split()
  if "'s" in question_tokens:
    possesive_idxs = [i for i, t in enumerate(question_tokens) if t == "'s"]
    for idx in possesive_idxs[::-1]:
      dfs_increase_tree_value(question_tree, idx, 1)


def parse_sparql_predicates(predicate_tokens,
                            query_token_idx):
  """Parses SPARQL predicate string and generates list of parse tree nodes."""
  if len(predicate_tokens) < 2:
    raise ValueError(f'Predicate is too short: {" ".join(predicate_tokens)}')
  if predicate_tokens[0] != '{' or predicate_tokens[-1] != '}':
    raise ValueError(f'Predicate should be enclosed by curly braces: '
                     f'{" ".join(predicate_tokens)}')
  # Split predicate phrases by "." token
  comma_idxs = [i for i, t in enumerate(predicate_tokens) if t == '.']
  split_idxs = [0] + comma_idxs + [-1]  # Indices of "{", "}", and "." tokens
  query_token_idx += 1  # Skipping "{"

  output = []
  for i in range(len(split_idxs) - 1):
    start, end = split_idxs[i] + 1, split_idxs[i + 1]
    tokens = predicate_tokens[start:end]
    if len(tokens) == 6 and tokens[0] == 'FILTER':  # e.g. FILTER ( ?x0 != ?x1 )
      output.append(
          Node(query_token_idx + 3,
               [Node(query_token_idx + 2),
                Node(query_token_idx + 4)]))
    elif len(tokens) == 3:  # e.g. ?x0 film.editor.film M1
      output.append(
          Node(query_token_idx + 1,
               [Node(query_token_idx),
                Node(query_token_idx + 2)]))
    else:
      raise ValueError(f'Wrong predicate: {tokens}')
    query_token_idx += len(tokens) + 1  # Skipping predicate + next token
  return output, query_token_idx


def parse_sparql_query(query):
  """Generates a parse tree of a preprocessed SPARQL query."""
  # Parse header and make the first token (SELECT) the root
  if query.startswith(SPARQL_YNQ_HEADER):
    header = SPARQL_YNQ_HEADER
  elif query.startswith(SPARQL_WHQ_HEADER):
    header = SPARQL_WHQ_HEADER
  else:
    raise ValueError(f'Wrong query header: {query}')

  header_tokens = header.split(' ')
  pt_children = [Node(i) for i in list(range(1, len(header_tokens)))]
  sparql_tree = Node(0, pt_children)

  # Parse predicates and put under the "WHERE" token
  predicate_tokens = query[len(header) + 1:].split()
  query_token_idx = len(header_tokens)
  predicate_nodes, query_token_idx = parse_sparql_predicates(
      predicate_tokens, query_token_idx)
  sparql_tree.children[-1].children = predicate_nodes
  assert len(query.split()) == query_token_idx

  return sparql_tree


def sparql_query_synonym(query1,
                         query2,
                         query_tree1=None,
                         query_tree2=None):
  """Checks whether two SPARQL queries are synonym."""
  # Assume that two queries are valid and parse them
  if query_tree1 is None:
    query_tree1 = parse_sparql_query(query1)
  if query_tree2 is None:
    query_tree2 = parse_sparql_query(query2)

  # Check two queries' type (yes/no question or wh- question)
  is_query1_ynq = query1.startswith(SPARQL_YNQ_HEADER)
  is_query2_ynq = query2.startswith(SPARQL_YNQ_HEADER)
  if is_query1_ynq != is_query2_ynq:
    return False

  # Match the sets of predicates
  def get_predicates(query, query_tree):
    """Get list of predicates as triplets of tokens."""
    query_tokens = query.split()
    predicates = []
    for child in query_tree.children[-1].children:
      predicate = (child.data, child.children[0].data, child.children[1].data)
      predicate = tuple((query_tokens[i] for i in predicate))
      predicates.append(predicate)
    return predicates

  predicate_set1 = set(get_predicates(query1, query_tree1))
  predicate_set2 = set(get_predicates(query2, query_tree2))
  return predicate_set1 == predicate_set2


def load_xlink_mapping(map_fpath):
  """Loads entity cross link mapping."""
  with open(map_fpath, 'rb') as fd:
    mapping = pickle.load(fd)

  # Build mappings as list of (n-gram, group_id) tuples
  question_map, query_map = [], []
  for i, group in enumerate(mapping):
    group_id = i + 1
    question_map.extend([(ngram, group_id) for ngram in group['question_token']
                        ])
    query_map.extend([(ngram, group_id) for ngram in group['query_token']])

  # Sort mappings by the length of the n-grams
  question_map.sort(reverse=True, key=lambda x: len(x[0]))
  query_map.sort(reverse=True, key=lambda x: len(x[0]))

  return (question_map, query_map)


def convert_to_xlink_group_ids(
    question_ids, query_ids,
    xlink_mapping
):
  """Converts question and query token ids to cross link group ids."""
  question_group = [0 for _ in range(len(question_ids))]
  query_group = [0 for _ in range(len(query_ids))]

  # Find n-gram matching in a brute-force way (group id 0 means no group)
  for token_ids, group_ids, mapping in zip([question_ids, query_ids],
                                           [question_group, query_group],
                                           xlink_mapping):
    for ngram, map_group_id in mapping:
      for i in range(len(token_ids) - len(ngram) + 1):
        ngram_match = True
        for j, ngram_id in enumerate(ngram):
          if token_ids[i + j] != ngram_id or group_ids[i + j] != 0:
            ngram_match = False
            break
        if ngram_match:
          for j in range(len(ngram)):
            group_ids[i + j] = map_group_id

  return (question_group, query_group)


def print_tree(node, tokens=None, depth=0):
  """Prints tree nodes with indentation for visualization."""
  if not node:
    return

  print_str = '  ' * depth + str(node.data)
  if tokens and isinstance(node.data, int):
    print_str += ': ' + tokens[node.data]
  print(print_str)

  if node.children:
    for child in node.children:
      print_tree(child, tokens, depth + 1)
