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

"""Parser for robust fill language."""

# pylint: skip-file
import re
from aloe.rfill.utils.rfill_consts import CONSTANTS, REGEXES
from aloe.rfill.utils.rfill_grammar import RFillNode


def evaluate_prog(expr_root, inp):
  output = ''
  if isinstance(expr_root, RFillNode):
    children = [x[1] for x in expr_root.children]
  elif isinstance(expr_root, list):
    children = expr_root
  for expr in children:
    output += evaluate_expr(expr, inp)
  return output


def evaluate_expr(e, inp):
  """evaluate individual subexpr in the concat list."""

  if e.syntax == 'ConstStr':
    return CONSTANTS[e.value]
  else:
    assert e.syntax == 'SubStr'
    p = []
    for _, c in e.children:
      p.append(evaluate_pos(c, inp))

    if p[0] is None or p[1] is None:
      return ''
    if p[0] < p[1]:
      ret = inp[p[0] : p[1]]
    else:
      ret = inp[p[1] : p[0]]
    return ret


def evaluate_pos(e, inp):
  """evaluate subexpr to get position argument."""

  if e.syntax == 'ConstPos':
    if e.value >= 0:
      return e.value
    else:
      return len(inp) + e.value + 1
  else:
    assert e.syntax == 'RegPos'
    _, param = e.children[0]
    _, p1 = param.children[0]
    _, p2 = param.children[1]
    _, direct = param.children[2]

    if param.syntax == 'RegexTok':
      reg_exp = REGEXES[p1.value]
    else:
      assert param.syntax == 'ConstTok'
      reg_exp = re.escape(CONSTANTS[p1.value])
    match_positions = []
    matches = re.finditer(reg_exp, inp)
    for match in matches:
      if direct.value == 'Start':
        match_positions.append(match.start())
      else:
        assert direct.value == 'End'
        match_positions.append(match.end())
    k = p2.value
    if k < 0:
      k = len(match_positions) + k + 1

    if k < 0 or k - 1 >= len(match_positions) or len(match_positions) == 0:  # pylint: disable=g-explicit-length-test
      return None
    return match_positions[k - 1]


def _get_p_node(p):
  """Construct node from expr tuple."""

  if p[0] == 'ConstPos':
    return RFillNode(*p)

  assert p[0] == 'RegPos'
  if p[1][0] == 'ConstTok':
    c1 = RFillNode('c1', p[1][1])
    c2 = RFillNode('idx', p[1][2])
    direct = RFillNode('direct', p[1][3])
    children = [('c1', c1), ('idx', c2), ('direct', direct)]
  elif p[1][0] == 'RegexTok':
    r1 = RFillNode('r1', p[1][1])
    r2 = RFillNode('idx', p[1][2])
    direct = RFillNode('direct', p[1][3])
    children = [('r1', r1), ('idx', r2), ('direct', direct)]
  else:
    raise NotImplementedError

  c_node = RFillNode(p[1][0], subtrees=children)
  p_node = RFillNode(p[0], subtrees=[('pos_param', c_node)])
  return p_node


def _get_substr_node(e):
  assert e[0] == 'SubStr'
  p1 = _get_p_node(e[1][0])
  p2 = _get_p_node(e[1][1])
  node = RFillNode('SubStr', subtrees=[('p1', p1), ('p2', p2)])
  return node


def expr2tree(exprs):
  """Convert expr list to tree."""

  subexprs = []
  for e in exprs:
    if e[0] == 'ConstStr':
      node = RFillNode(*e)
    else:
      node = _get_substr_node(e)
    subexprs.append(node)
  expr_trees = [('subexpr', expr) for expr in subexprs]
  expr_root = RFillNode('expr_root', subtrees=expr_trees)
  return expr_root


def tree_equal(tree_1, tree_2):
  if tree_1.syntax != tree_2.syntax or tree_1.value != tree_2.value:
    return False
  if len(tree_1.children) != len(tree_2.children):
    return False
  for i in range(len(tree_1.children)):
    e1, c1 = tree_1.children[i]
    e2, c2 = tree_2.children[i]
    if e1 != e2:
      return False
    t = tree_equal(c1, c2)
    if not t:
      return False
  return True


class RobustFillParser(object):
  """Robust fill parser object."""

  def __init__(self):
    self.parsed_table = {}

  def parse(self, expr_list, extra_key=None):
    key = str(hash(str(expr_list)))
    if extra_key is not None:
      key += '-key-' + str(extra_key)
    if key not in self.parsed_table:
      self.parsed_table[key] = expr2tree(expr_list)
    return self.parsed_table[key]

  def new_game(self, input_str):
    self.input_str = input_str

  def run(self, expr_list):
    tree_root = self.parse(expr_list)
    out = evaluate_prog(tree_root, self.input_str)
    return out

