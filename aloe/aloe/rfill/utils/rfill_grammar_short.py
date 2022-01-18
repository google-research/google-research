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

# pylint: skip-file
import numpy as np
from aloe.rfill.utils.program_struct import AbsRFillNode


class RFillNode(AbsRFillNode):
  """Subclass of AbsRFillNode."""

  def get_raw_token(self):
    if self.value is None:
      return self.syntax
    return str(self.value)

  def to_tokens(self):
    if self.syntax == 'expr_root':
      toks = ['|']
      for _, c in self.children:
        sub_toks = c.to_tokens()
        if c.syntax == 'ConstStr':
          toks += [c.syntax]
        toks += sub_toks
        toks += ['|']
    elif len(self.children) == 0: # leaf
      toks = [self.get_raw_token()]
    else:
      toks = [self.get_raw_token()]
      for _, c in self.children:
        sub_toks = c.to_tokens()
        if c.syntax == 'ConstPos':
          toks += [c.syntax]
        toks += sub_toks
    return toks

  @staticmethod
  def from_tokens(tokens, is_root=True, syntax=None):
    if len(tokens) == 0:
      return None
    if is_root:
      ids_start = []
      for i, t in enumerate(tokens):
        if t == '|':
          ids_start.append(i)
      children = []
      for i in range(len(ids_start) - 1):
        sub_span = tokens[ids_start[i] + 1 : ids_start[i + 1]]
        if len(sub_span) == 2:
          stx = 'ConstStr'
          sub_span = sub_span[1:]
        else:
          stx = 'SubStr'
        ch = RFillNode.from_tokens(sub_span, is_root=False, syntax=stx)
        if ch is None:
          return None
        children.append(('subexpr', ch))
      node = RFillNode('expr_root', subtrees=children)
    elif len(tokens) == 1:
      if syntax is None:
        return None
      val = tokens[0]
      if val.lstrip('-').isnumeric():
        val = int(val)
      node = RFillNode(syntax, value=val)
    elif tokens[0] == 'SubStr':
      ids_start = [1]
      for i in range(2, len(tokens)):
        if tokens[i] == 'RegPos' or tokens[i] == 'ConstPos':
          ids_start.append(i)
      ids_start.append(len(tokens))
      if len(ids_start) != 3:
        return None
      poses = []
      for i in range(2):
        sub_span = tokens[ids_start[i] : ids_start[i + 1]]
        if len(sub_span) == 2:
          sub_span = sub_span[1:]
          stx = 'ConstPos'
        else:
          stx = 'RegPos'
        poses.append(RFillNode.from_tokens(sub_span, is_root=False, syntax=stx))
      lpos, rpos = poses
      node = RFillNode('SubStr', subtrees=[('p1', lpos), ('p2', rpos)])
    elif tokens[0] == 'RegPos':
      ch = RFillNode.from_tokens(tokens[1:], is_root=False)
      if ch is None or len(ch.children) != 3:
        return None
      node = RFillNode('RegPos', subtrees=[('pos_param', ch)])
    else:
      node = RFillNode(tokens[0])
      sub_syntax = ['idx', 'direct']
      if node.syntax == 'ConstTok':
        sub_syntax.insert(0, 'c1')
      elif node.syntax == 'RegexTok':
        sub_syntax.insert(0, 'r1')
      else:
        return None
      for i in range(1, len(tokens)):
        ch = RFillNode.from_tokens([tokens[i]], is_root=False, syntax=sub_syntax[i - 1])
        if ch is None:
          return None
        node.add_child(ch.syntax, ch)
    for t, c in node.children:
      if c is None:
        return None
    return node

RFILL_VOCAB = {
    'pad': 0,
    '-1': 1,
    '-2': 2,
    '-3': 3,
    '-4': 4,
    '0': 5,
    '1': 6,
    '2': 7,
    '3': 8,
    '4': 9,
    '5': 10,
    '6': 11,
    '7': 12,
    '8': 13,
    '9': 14,
    '10': 15,
    '11': 16,
    'ConstTok': 17,
    'RegPos': 18,
    'RegexTok': 19,
    'SubStr': 20,
    'End': 21,
    'Start': 22,
    '|': 23,
    'ConstStr': 24,
    'ConstPos': 25,
    'eos': 26
}

prod_rules = {}
prod_rules['subexpr'] = ['ConstStr', 'SubStr']
prod_rules['p1'] = ['ConstPos', 'RegPos']
prod_rules['p2'] = ['ConstPos', 'RegPos']
prod_rules['pos_param'] = ['ConstTok', 'RegexTok']

value_rules = {}
value_rules['ConstStr'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
value_rules['ConstPos'] = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
value_rules['c1'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
value_rules['r1'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
value_rules['idx'] = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
value_rules['direct'] = ['Start', 'End']

expand_rules = {}
expand_rules['SubStr'] = ['p1', 'p2']
expand_rules['RegPos'] = ['pos_param']
expand_rules['ConstTok'] = ['c1', 'idx', 'direct']
expand_rules['RegexTok'] = ['r1', 'idx', 'direct']

trans_map = {}
# start -> | -> expr
trans_map[('start', '|')] = 'expr'

# expr -> eos -> halt
trans_map[('expr', 'eos')] = 'halt'

# expr -> ConstStr -> ConstStr
trans_map[('expr', 'ConstStr')] = 'ConstStr'

# non_empty_expr -> ConstStr -> ConstStr
trans_map[('non_empty_expr', 'ConstStr')] = 'ConstStr'

# ConstStr -> ConstStr -> start
for val in value_rules['ConstStr']:
    trans_map[('ConstStr', str(val))] = 'start'

# expr -> SubStr -> p1
trans_map[('expr', 'SubStr')] = 'p1'

# non_empty_expr -> SubStr -> p1
trans_map[('non_empty_expr', 'SubStr')] = 'p1'

for p, p_next in [('p1', 'p2'), ('p2', 'start')]:
    # p -> ConstPos -> ConstPos_p
    trans_map[(p, 'ConstPos')] = 'ConstPos_%s' % p
    for val in value_rules['ConstPos']:
        trans_map[('ConstPos_%s' % p, str(val))] = p_next

    # p -> RegPos -> pos_param_p
    trans_map[(p, 'RegPos')] = 'pos_param_%s' % p

    # pos_param_p -> ConstTok -> const_p
    trans_map[('pos_param_%s' % p, 'ConstTok')] = 'const_%s' % p

    # const_p -> c1 -> idx_p
    for val in value_rules['c1']:
        trans_map[('const_%s' % p, str(val))] = 'idx_%s' % p

    # idx_p -> idx -> direct_p
    for val in value_rules['idx']:
        trans_map[('idx_%s' % p, str(val))] = 'direct_%s' % p

    # direct_p -> dir -> p_next
    for val in value_rules['direct']:
        trans_map[('direct_%s' % p, val)] = p_next

    # pos_param_p -> RegexTok -> regex_p
    trans_map[('pos_param_%s' % p, 'RegexTok')] = 'regex_%s' % p

    # regex_p -> r1 -> idx_p
    for val in value_rules['r1']:
        trans_map[('regex_%s' % p, str(val))] = 'idx_%s' % p

# halt -> eos -> halt
trans_map[('halt', 'eos')] = 'halt'
