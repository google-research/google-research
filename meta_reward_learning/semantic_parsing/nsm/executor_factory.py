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

"""Utility functions to interact with knowledge graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six
from six.moves import zip


class Executor(object):
  """Executors implements the basic subroutines and provide

  the API to the computer.
  """

  def get_api(self, config):
    """Provide API to the computer."""
    raise NotImplementedError()


def get_simple_type_hierarchy():
  type_hierarchy = {
      'entity_list': ['atom_list'],
      'list': [],
      'num_list': ['ordered_list'],
      'datetime_list': ['ordered_list'],
      'ordered_list': ['atom_list'],
      'atom_list': ['list'],
      'string_list': ['list'],
      'string_property': ['property'],
      # Atom as it doesn't have sub-parts.
      'ordered_property': ['atom_property'],
      'entity_property': ['atom_property'],
      'atom_property': ['property'],
      'datetime_property': ['ordered_property'],
      'num_property': ['ordered_property'],
      'num': [],
      'int': ['num'],
      'property': [],
      'symbol': [],
      'function': ['head'],
      'head': [],
      'primitive_function': ['function'],
      'global_primitive_function': ['primitive_function'],
      '<ERROR>': []
  }
  return type_hierarchy


class SimpleKGExecutor(Executor):
  """This executor assumes that the knowledge graph is

  encoded as a dictionary.
  """

  def __init__(self, kg_info):
    """Given a knowledge graph, the number properties and

    the datetime properties, initialize an executor that
    implements the basic subroutines.

    Args:
      kg_info: a dictionary with three keys.
    """
    self.kg = kg_info['kg']
    self.num_props = kg_info['num_props']
    self.datetime_props = kg_info['datetime_props']
    self.props = kg_info['props']

  def hop(self, entities, prop, keep_dup=False):
    """Get the property of a list of entities."""
    if keep_dup:
      result = []
    else:
      result = set()
    for ent in entities:
      try:
        if keep_dup:
          result += self.kg[ent][prop]
        else:
          result = result.union(self.kg[ent][prop])
      except KeyError:
        continue
    return list(result)

  def filter_equal(self, ents_1, ents_2, prop):
    """From ents_1, filter out the entities whose property equal to ents_2."""
    result = []
    for ent in ents_1:
      if set(self.hop([ent], prop)) == set(ents_2):
        result.append(ent)
    return result

  def filter_not_equal(self, ents_1, ents_2, prop):
    """From ents_1, filter out the entities whose property equal to ents_2."""
    result = []
    for ent in ents_1:
      if set(self.hop([ent], prop)) != set(ents_2):
        result.append(ent)
    return result

  def get_num_prop_val(self, ent, prop):
    """Get the value of an entities' number property."""
    # If there are multiple values, then take the first one.
    prop_str_list = self.hop([ent], prop)
    try:
      prop_str = prop_str_list[0]
      prop_val = float(prop_str)
    except (ValueError, IndexError):
      prop_val = None
    return prop_val

  def get_datetime_prop_val(self, ent, prop):
    """Get the value of an entities' date time property."""
    # If there are multiple values, then take the first one.
    prop_str_list = self.hop([ent], prop)
    try:
      prop_str = prop_str_list[0]
      if prop_str[0] == '-':
        sign = -1
        prop_str = prop_str[1:]
      else:
        sign = 1
      result = [float(n) for n in prop_str.replace('x', '0').split('-')]
      day = 0
      for n, unit in zip(result, [365, 30, 1]):
        day += n * unit
      day *= sign
      prop_val = day
    except (ValueError, IndexError):
      prop_val = None
    return prop_val

  def sort_select(self, entities, prop, ind):
    """Sort the entities using prop then select the i-th one."""
    if prop in self.num_props:
      get_val = self.get_num_prop_val
    elif prop in self.datetime_props:
      get_val = self.get_datetime_prop_val
    else:
      raise ValueError(prop)
    vals = []
    new_ents = []
    for ent in entities:
      val = get_val(ent, prop)
      if val is not None:
        new_ents.append(ent)
        vals.append(val)
    ent_vals = list(zip(new_ents, vals))
    best_ent_val = sorted(ent_vals, key=lambda x: x[1])[ind]
    best_score = best_ent_val[1]
    result = [ent for ent, val in ent_vals if val == best_score]
    return result
    # return [best_ent_val[0]]

  def argmax(self, entities, prop):
    return self.sort_select(entities, prop, -1)

  def argmin(self, entities, prop):
    return self.sort_select(entities, prop, 0)

  def valid_props(self,
                  source_mids,
                  token_val_dict,
                  target_mids=None,
                  condition_fn=None):
    connected_props = self.get_props(
        source_mids, target_mids, condition_fn=condition_fn)
    valid_tks = []
    for tk, prop in six.iteritems(token_val_dict):
      if prop in connected_props:
        valid_tks.append(tk)
    return valid_tks

  def is_connected(self, source_ents, target_ents, prop):
    return set(self.hop(source_ents, prop)) == set(target_ents)

  def get_props(self,
                source_ents,
                target_ents=None,
                debug=False,
                condition_fn=None):
    """Get the properties that goes from source to targets."""
    props = set()
    if condition_fn is None:
      condition_fn = self.is_connected
    if debug:
      print('=' * 100)
    for ent in source_ents:
      if debug:
        print('@' * 20)
        print(ent)
      if ent in self.kg:
        ent_props = list(self.kg[ent].keys())
        if target_ents is not None:
          for p in ent_props:
            if debug:
              print()
              print(p)
              print(self.hop([ent], p))
            # if set(self.hop([ent], p)) == set(target_ents):
            if condition_fn([ent], target_ents, p):
              props.add(p)
        else:
          props = props.union(ent_props)
    if debug:
      print('in get props')
      print(source_ents)
      print(target_ents)
      print(props)
      print('=' * 100)
    return list(props)

  def autocomplete_hop(self, exp, tokens, token_vals):
    l = len(exp)
    if l == 2:  # second argument is a property.
      source_mids = exp[1]
      token_val_dict = dict(list(zip(tokens, token_vals)))
      valid_tks = self.valid_props(source_mids, token_val_dict)
    else:
      valid_tks = tokens
    return valid_tks

  def autocomplete_argm(self, exp, tokens, token_vals, debug=False):
    l = len(exp)
    if l == 1:  # first argument has more than one entity.
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
    elif l == 2:  # second argument is a property.
      source_mids = exp[1]
      token_val_dict = dict(list(zip(tokens, token_vals)))
      valid_tks = self.valid_props(source_mids, token_val_dict)
    else:
      valid_tks = tokens
    if debug:
      print('*' * 30)
      print(exp)
      print(tokens)
      print(valid_tks)
      print('*' * 30)
    return valid_tks

  def autocomplete_filter_equal(self, exp, tokens, token_vals, debug=False):
    l = len(exp)
    if l == 1:
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
    elif l == 2:
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        # The second argument must have some connection with
        # the first argument.
        if self.get_props(exp[1], val):
          valid_tks.append(tk)
    elif l == 3:
      token_val_dict = dict(list(zip(tokens, token_vals)))
      valid_tks = self.valid_props(exp[1], token_val_dict, exp[2])
    else:
      raise ValueError('Expression is too long: {}'.format(l))

    if debug:
      print()
      print('+' * 30)
      print('in filter equal')
      print(exp)
      print(tokens)
      print(valid_tks)
      print('+' * 30)

    return valid_tks

  def get_api(self):
    func_dict = collections.OrderedDict()
    func_dict['hop'] = dict(
        name='hop',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_hop,
        value=self.hop)

    func_dict['filter_equal'] = dict(
        name='filter_equal',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_filter_equal,
        value=self.filter_equal)

    func_dict['argmax'] = dict(
        name='argmax',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_argm,
        value=self.argmax)

    func_dict['argmin'] = dict(
        name='argmin',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_argm,
        value=self.argmin)

    constant_dict = collections.OrderedDict()

    for p in self.props:
      if p in self.num_props:
        tp = 'num_property'
      elif p in self.datetime_props:
        tp = 'datetime_property'
      else:
        tp = 'string_property'

      constant_dict[p] = dict(value=p, type=tp, name=p)

    type_hierarchy = get_simple_type_hierarchy()
    return dict(
        type_hierarchy=type_hierarchy,
        func_dict=func_dict,
        constant_dict=constant_dict)


class TableExecutor(SimpleKGExecutor):
  """The executor for writing programs that processes simple Tables."""

  def __init__(self, table_info):
    super(TableExecutor, self).__init__(table_info)
    self.n_rows = len(table_info['row_ents'])

  def filter_ge(self, ents_1, nums, prop):
    """Filter out entities whose prop >= nums."""
    result = []
    for ent in ents_1:
      vals = set(self.hop([ent], prop))
      for val in vals:
        if all([(val >= x) for x in nums]):
          result.append(ent)
          break
    return result

  def filter_greater(self, ents_1, nums, prop):
    """Filter out entities whose prop > nums."""
    result = []
    for ent in ents_1:
      vals = set(self.hop([ent], prop))
      for val in vals:
        if all([(val > x) for x in nums]):
          result.append(ent)
          break
    return result

  def filter_le(self, ents_1, nums, prop):
    """Filter out entities whose prop <= nums."""
    result = []
    for ent in ents_1:
      vals = set(self.hop([ent], prop))
      for val in vals:
        if all([(val <= x) for x in nums]):
          result.append(ent)
          break
    return result

  def filter_less(self, ents_1, nums, prop):
    """Filter out entities whose prop < nums."""
    result = []
    for ent in ents_1:
      vals = set(self.hop([ent], prop))
      for val in vals:
        if all([(val < x) for x in nums]):
          result.append(ent)
          break
    return result

  def filter_str_contain_any(self, ents, string_list, prop):
    """Filter out entities whose prop contains any of the strings."""
    result = []
    for ent in ents:
      str_val_list = self.hop([ent], prop)
      assert len(str_val_list) == 1
      str_val = str_val_list[0]
      for string in string_list:
        if string in str_val:
          result.append(ent)
          break
    return result

  def filter_str_contain_not_any(self, ents, string_list, prop):
    """Filter out entities, whose prop doesn't contain any of the strings."""
    result = []
    for ent in ents:
      str_val_list = self.hop([ent], prop)
      # Make sure that entity only has one value for the prop.
      assert len(str_val_list) == 1
      str_val = str_val_list[0]
      # If any one of the string is contained by the cell,
      # then pass. Only add to the result when none of the
      # string is in the cell.
      for string in string_list:
        if string in str_val:
          break
      else:
        result.append(ent)
    return result

  def autocomplete_filter_str_contain_any(self,
                                          exp,
                                          tokens,
                                          token_vals,
                                          debug=False):
    """Auto-complete for filter_str_contain_any function."""
    l = len(exp)
    if l == 1:
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
    elif l == 2:
      # Since all the strings are in the table, no need to
      # filter any more. Keep the ones that have at least
      # one string in it.
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 0]
    elif l == 3:
      valid_tks = []
      token_val_dict = dict(list(zip(tokens, token_vals)))
      source_ents = exp[1]
      string_list = exp[2]
      for tk in tokens:
        is_valid = False
        prop = token_val_dict[tk]
        str_val_list = self.hop(source_ents, prop)
        # If one of the str_val contains any one of the
        # string, then we can use this property.
        for str_val in str_val_list:
          for string in string_list:
            if string in str_val:
              is_valid = True
              break
          if is_valid:
            valid_tks.append(tk)
            break
    else:
      raise ValueError('Expression is too long: {}'.format(l))

    if debug:
      print()
      print('+' * 30)
      print('in filter equal')
      print(exp)
      print(tokens)
      print(valid_tks)
      print('+' * 30)

    return valid_tks

  # Next and previous
  def next(self, rows):
    """Select all the rows that is right below the given rows respectively."""
    assert rows
    assert rows[0][:4] == 'row_'
    # row are in the pattern of row_0, row_1.
    row_ids = [int(row_str[4:]) for row_str in rows]
    new_row_ids = [(i + 1) for i in row_ids if i + 1 < self.n_rows]
    if new_row_ids:
      result_rows = ['row_{}'.format(i) for i in new_row_ids]
      # result_rows = ['row_{}'.format(max(new_row_ids))]
    else:
      result_rows = []
    return result_rows

  def previous(self, rows):
    """Select all the rows that is right above the given rows respectively."""
    assert rows
    assert rows[0][:4] == 'row_'
    row_ids = [int(row_str[4:]) for row_str in rows]
    new_row_ids = [(i - 1) for i in row_ids if i - 1 >= 0]
    if new_row_ids:
      result_rows = ['row_{}'.format(i) for i in new_row_ids]
      # result_rows = ['row_{}'.format(min(new_row_ids))]
    else:
      result_rows = []
    return result_rows

  def autocomplete_next(self, exp, tokens, token_vals):
    """Autocompletion for next function."""
    l = len(exp)
    if l == 1:
      # If there are any non-empty result, then it is available.
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        if len(val) > 0 and tk != 'all_rows' and self.next(val):
          valid_tks.append(tk)
    else:
      raise ValueError('Wrong length: {}.'.format(l))
    return valid_tks

  def autocomplete_previous(self, exp, tokens, token_vals):
    """Autocompletion for previous function."""
    l = len(exp)
    if l == 1:
      # If there are any non-empty result, then it is available.
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        if len(val) > 0 and tk != 'all_rows' and self.previous(val):
          valid_tks.append(tk)
    else:
      raise ValueError('Wrong length: {}.'.format(l))
    return valid_tks

  # First and last
  def first(self, rows):
    """Take the first row (the one with minimum index) in all the rows."""
    assert len(rows) > 1
    assert rows[0][:4] == 'row_'
    # Return the row with the smallest id.
    row_ids = [int(row_str[4:]) for row_str in rows]
    result_row_id = min(row_ids)
    result_rows = ['row_{}'.format(result_row_id)]
    return result_rows

  def last(self, rows):
    """Take the last row (the one with maximum index) in all the rows."""
    assert len(rows) > 1
    assert rows[0][:4] == 'row_'
    # Return the row with the largest id.
    row_ids = [int(row_str[4:]) for row_str in rows]
    result_row_id = max(row_ids)
    result_rows = ['row_{}'.format(result_row_id)]
    return result_rows

  def autocomplete_first_last(self, exp, tokens, token_vals):
    """Autocompletion for both first and last."""
    l = len(exp)
    if l == 1:
      # Only use first or last when you have more than one
      # entity.
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
    else:
      raise ValueError('Wrong length: {}.'.format(l))
    return valid_tks

  # Aggregation functions.
  def count(self, ents):
    return [len(ents)]

  def maximum(self, ents, prop):
    vals = self.hop(ents, prop)
    return [max(vals)]

  def minimum(self, ents, prop):
    vals = self.hop(ents, prop)
    try:
      result = [min(vals)]
    except Exception as e:
      print(ents, prop)
      raise e
    return result

  def mode(self, ents, prop):
    """Return the value that appears the most in the prop of the entities."""
    vals = self.hop(ents, prop, keep_dup=True)
    count_dict = {}
    for v in vals:
      if v in count_dict:
        count_dict[v] += 1
      else:
        count_dict[v] = 1
    max_count = 0
    max_val_list = []
    for val, count in six.iteritems(count_dict):
      if count > max_count:
        max_count = count
        max_val_list = [val]
      elif count == max_count:
        max_val_list.append(val)
    return max_val_list

  def sum(self, ents, prop):
    vals = self.hop(ents, prop, keep_dup=True)
    return [sum(vals)]

  def average(self, ents, prop):
    vals = self.hop(ents, prop, keep_dup=True)
    return [float(sum(vals)) / len(vals)]

  def autocomplete_aggregation(self, exp, tokens, token_vals):
    """Autocomplete for aggregation functions."""
    l = len(exp)
    if l == 1:
      # Only use aggregation when you have more than one
      # entity, otherwise just use hop.
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
    else:
      # For the second argument, all the props with the
      # right type (ordered_property) can be used.
      props = set(self.get_props(exp[1]))
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        if val in props:
          valid_tks.append(tk)
    return valid_tks

  def same(self, ents, prop, namespace):
    """Find the entities that has the prop as the given entity."""
    # Can only work with one entity.
    assert len(ents) == 1
    vals_1 = self.hop(ents, prop)
    all_rows = namespace['all_rows']['value']
    same_ents = self.filter_equal(all_rows, vals_1, prop)
    # Remove itself.
    same_ents.remove(ents[0])
    return same_ents

  def autocomplete_same(self, exp, tokens, token_vals, namespace):
    """Autocomplete for same function."""
    l = len(exp)
    if l == 1:
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) == 1]
    elif l == 2:
      props = set(self.get_props(exp[1]))
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        if val in props:
          valid_tks.append(tk)
    else:
      raise ValueError('Wrong length {}'.format(l))
    return valid_tks

  def diff(self, ents_1, ents_2, prop):
    """Return the difference of two entities in prop."""
    assert len(ents_1) == 1
    assert len(ents_2) == 1
    val_1 = self.hop(ents_1, prop)[0]
    val_2 = self.hop(ents_2, prop)[0]
    return [abs(val_1 - val_2)]

  def autocomplete_diff(self, exp, tokens, token_vals):
    """Autocomplete for diff function."""
    l = len(exp)
    if l == 1:
      valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) == 1]
      # There must be at least two valid variables to apply
      # diff.
      if len(valid_tks) < 2:
        valid_tks = []
    elif l == 2:
      valid_tks = [
          tk for tk, val in zip(tokens, token_vals)
          if (len(val) == 1 and val != exp[1])
      ]
    else:
      props = set(self.get_props(exp[1]))
      props = props.intersection(self.get_props(exp[2]))
      valid_tks = []
      for tk, val in zip(tokens, token_vals):
        if val in props:
          valid_tks.append(tk)
    return valid_tks

  def return_all_tokens(self, unused_exp, tokens, unused_token_vals):
    return tokens

  def get_api(self):
    """Get the functions, constants and type hierarchy."""
    func_dict = collections.OrderedDict()

    def hop_return_type_fn(arg1_type, arg2_type):
      if arg2_type == 'num_property':
        return 'num_list'
      elif arg2_type == 'string_property':
        return 'string_list'
      elif arg2_type == 'datetime_property':
        return 'datetime_list'
      elif arg2_type == 'entity_property':
        return 'entity_list'
      else:
        raise ValueError('Unknown type {}'.format(arg2_type))

    func_dict['hop'] = dict(
        name='hop',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type=hop_return_type_fn,
        autocomplete=self.autocomplete_hop,
        type='primitive_function',
        value=self.hop)

    # Only use filter equal for number and date and
    # entities. Use filter_str_contain for string values.
    func_dict['filter_eq'] = dict(
        name='filter_eq',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_filter_equal,
        type='primitive_function',
        value=self.filter_equal)

    func_dict['filter_not_eq'] = dict(
        name='filter_not_eq',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_filter_equal,
        type='primitive_function',
        value=self.filter_not_equal)

    func_dict['argmax'] = dict(
        name='argmax',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_argm,
        type='primitive_function',
        value=self.argmax)

    func_dict['argmin'] = dict(
        name='argmin',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_argm,
        type='primitive_function',
        value=self.argmin)

    func_dict['same'] = dict(
        name='same',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_same,
        type='global_primitive_function',
        value=self.same)

    func_dict['first'] = dict(
        name='first',
        args=[{
            'types': ['entity_list']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_first_last,
        type='primitive_function',
        value=self.first)

    func_dict['last'] = dict(
        name='last',
        args=[{
            'types': ['entity_list']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_first_last,
        type='primitive_function',
        value=self.last)

    func_dict['next'] = dict(
        name='next',
        args=[{
            'types': ['entity_list']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_next,
        type='primitive_function',
        value=self.next)

    func_dict['previous'] = dict(
        name='previous',
        args=[{
            'types': ['entity_list']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_previous,
        type='primitive_function',
        value=self.previous)

    func_dict['count'] = dict(
        name='count',
        args=[{
            'types': ['entity_list']
        }],
        return_type='num',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.count)

    func_dict['filter_str_contain_any'] = dict(
        name='filter_str_contain_any',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['string_list']
        }, {
            'types': ['string_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_filter_str_contain_any,
        type='primitive_function',
        value=self.filter_str_contain_any)

    func_dict['filter_str_contain_not_any'] = dict(
        name='filter_str_contain_not_any',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['string_list']
        }, {
            'types': ['string_property']
        }],
        return_type='entity_list',
        autocomplete=self.autocomplete_filter_str_contain_any,
        type='primitive_function',
        value=self.filter_str_contain_not_any)

    func_dict['filter_ge'] = dict(
        name='filter_ge',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_ge)

    func_dict['filter_greater'] = dict(
        name='filter_greater',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_greater)

    func_dict['filter_le'] = dict(
        name='filter_le',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_le)

    func_dict['filter_less'] = dict(
        name='filter_less',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_less)

    # aggregation functions.
    for k, f in zip(['maximum', 'minimum'], [self.maximum, self.minimum]):
      func_dict[k] = dict(
          name=k,
          args=[{
              'types': ['entity_list']
          }, {
              'types': ['ordered_property']
          }],
          return_type='ordered_list',
          autocomplete=self.autocomplete_aggregation,
          type='primitive_function',
          value=f)

    func_dict['mode'] = dict(
        name='mode',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type=hop_return_type_fn,
        autocomplete=self.autocomplete_aggregation,
        type='primitive_function',
        value=self.mode)

    func_dict['average'] = dict(
        name='average',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['num_property']
        }],
        return_type='num',
        autocomplete=self.autocomplete_aggregation,
        type='primitive_function',
        value=self.average)

    func_dict['sum'] = dict(
        name='sum',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['num_property']
        }],
        return_type='num',
        autocomplete=self.autocomplete_aggregation,
        type='primitive_function',
        value=self.sum)

    func_dict['diff'] = dict(
        name='diff',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['entity_list']
        }, {
            'types': ['num_property']
        }],
        return_type='num',
        autocomplete=self.autocomplete_diff,
        type='primitive_function',
        value=self.diff)

    constant_dict = collections.OrderedDict()

    for p in self.props:
      if p in self.num_props:
        tp = 'num_property'
      elif p in self.datetime_props:
        tp = 'datetime_property'
      elif p.split('-')[-1] == 'entity':
        tp = 'entity_property'
      else:
        tp = 'string_property'

      constant_dict[p] = dict(value=p, type=tp, name=p)

    type_hierarchy = get_simple_type_hierarchy()
    return dict(
        type_hierarchy=type_hierarchy,
        func_dict=func_dict,
        constant_dict=constant_dict)


def is_number(obj):
  return isinstance(obj, int) or isinstance(obj, float)


class WikiTableExecutor(TableExecutor):
  pass


class WikiSQLExecutor(TableExecutor):

  def __init__(self,
               table_info,
               use_filter_str_contain=True,
               use_filter_str_equal=False):
    super(TableExecutor, self).__init__(table_info)
    self.n_rows = len(table_info['row_ents'])
    self.use_filter_str_equal = use_filter_str_equal
    self.use_filter_str_contain = use_filter_str_contain

  def hop(self, entities, prop, keep_dup=True):
    """Get the property of a list of entities."""
    # Note this changes keep_dup=True as default, which is
    # different from WikiTableQuestions experiments.
    if keep_dup:
      result = []
    else:
      result = set()
    for ent in entities:
      try:
        if keep_dup:
          result += self.kg[ent][prop]
        else:
          result = result.union(self.kg[ent][prop])
      except KeyError:
        continue
    return list(result)

  def get_api(self):
    """Get the functions, constants and type hierarchy."""
    func_dict = collections.OrderedDict()

    def hop_return_type_fn(arg1_type, arg2_type):
      if arg2_type == 'num_property':
        return 'num_list'
      elif arg2_type == 'string_property':
        return 'string_list'
      elif arg2_type == 'datetime_property':
        return 'datetime_list'
      elif arg2_type == 'entity_property':
        return 'entity_list'
      else:
        raise ValueError('Unknown type {}'.format(arg2_type))

    func_dict['hop'] = dict(
        name='hop',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['property']
        }],
        return_type=hop_return_type_fn,
        autocomplete=self.autocomplete_hop,
        type='primitive_function',
        value=self.hop)

    if self.use_filter_str_equal:
      # Allow equal to work on every type.
      func_dict['filter_eq'] = dict(
          name='filter_eq',
          args=[{
              'types': ['entity_list']
          }, {
              'types': ['entity_list']
          }, {
              'types': ['property']
          }],
          return_type='entity_list',
          autocomplete=self.autocomplete_filter_equal,
          type='primitive_function',
          value=self.filter_equal)
    else:
      # Only use filter equal for number and date and
      # entities. Use filter_str_contain for string values.
      func_dict['filter_eq'] = dict(
          name='filter_eq',
          args=[{
              'types': ['entity_list']
          }, {
              'types': ['ordered_list']
          }, {
              'types': ['ordered_property']
          }],
          return_type='entity_list',
          autocomplete=self.autocomplete_filter_equal,
          type='primitive_function',
          value=self.filter_equal)

    if self.use_filter_str_contain:
      func_dict['filter_str_contain_any'] = dict(
          name='filter_str_contain_any',
          args=[{
              'types': ['entity_list']
          }, {
              'types': ['string_list']
          }, {
              'types': ['string_property']
          }],
          return_type='entity_list',
          autocomplete=self.autocomplete_filter_str_contain_any,
          type='primitive_function',
          value=self.filter_str_contain_any)

    func_dict['filter_greater'] = dict(
        name='filter_greater',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_greater)

    func_dict['filter_less'] = dict(
        name='filter_less',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['ordered_list']
        }, {
            'types': ['ordered_property']
        }],
        return_type='entity_list',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.filter_less)

    func_dict['count'] = dict(
        name='count',
        args=[{
            'types': ['entity_list']
        }],
        return_type='num',
        autocomplete=self.return_all_tokens,
        type='primitive_function',
        value=self.count)

    # aggregation functions.
    for k, f in zip(['maximum', 'minimum'], [self.maximum, self.minimum]):
      func_dict[k] = dict(
          name=k,
          args=[{
              'types': ['entity_list']
          }, {
              'types': ['ordered_property']
          }],
          return_type='ordered_list',
          autocomplete=self.autocomplete_aggregation,
          type='primitive_function',
          value=f)

    func_dict['average'] = dict(
        name='average',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['num_property']
        }],
        return_type='num',
        autocomplete=self.autocomplete_aggregation,
        type='primitive_function',
        value=self.average)

    func_dict['sum'] = dict(
        name='sum',
        args=[{
            'types': ['entity_list']
        }, {
            'types': ['num_property']
        }],
        return_type='num',
        autocomplete=self.autocomplete_aggregation,
        type='primitive_function',
        value=self.sum)

    constant_dict = collections.OrderedDict()

    for p in self.props:
      if p in self.num_props:
        tp = 'num_property'
      elif p in self.datetime_props:
        tp = 'datetime_property'
      elif p.split('-')[-1] == 'entity':
        tp = 'entity_property'
      else:
        tp = 'string_property'

      constant_dict[p] = dict(value=p, type=tp, name=p)

    type_hierarchy = get_simple_type_hierarchy()
    return dict(
        type_hierarchy=type_hierarchy,
        func_dict=func_dict,
        constant_dict=constant_dict)
