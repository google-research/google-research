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

"""Converts Multiwoz 2.1 dataset to the data format of SGD."""

import collections
import copy
import json
import os
import re

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from schema_guided_dst import schema

FLAGS = flags.FLAGS

flags.DEFINE_string('input_data_dir', '',
                    'Path of the dataset to convert from.')
flags.DEFINE_string(
    'output_dir', '',
    'Path to output directory. If not specified, generate the dialogues in the '
    'same directory as the script.')
flags.DEFINE_boolean(
    'annotate_copy_slots', False,
    'Whether to annotate slots whose value is copied from a different slot in '
    'the previous state. If true, add a new key "copy_from" in the slot '
    'annotation dict. Its value is the slot that the value is copied from.')

flags.DEFINE_string('schema_file_name', 'schema.json',
                    'Name of the schema file to use.')

_PATH_MAPPING = [('test', 'testListFile.txt'), ('dev', 'valListFile.txt'),
                 ('train', '')]

_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
# File used for correcting categorical slot values. Each line is a pair of
# the original slot value in MultiWOZ 2.1 annotation and the corrected slot
# value.
_CORRECT_FOR_STATE_PATH = os.path.join(_DIR_PATH,
                                       'correct_categorical_state_values.tsv')

_DEFAULT_SERVICE_NAME = 'all'
# "Don't care" slot value.
_DONT_CARE = 'dontcare'
_NONE_VALUE = 'none'
_INACTIVE_INTENT = 'NONE'
# Maximum number of dialogues to write in each output file.
_NUM_DIALS_PER_FILE = 512

# We try to find the span of non-categorical slot values in the dialog history,
# but sometimes there is no exact match and we choose to find the closest values
# from the utterance. If the found value is contained in the list below,
# we need to check if it is a correct match.
_FOUND_VALUES_NEED_CHECK = [
    'restaurant', 'hotel', 'museum', 'church', 'college', 'cinema', 'park',
    'guesthouses', 'guesthouse', 'great', 'from', 'hotels', 'school', 'schools',
    'guests', 'colleges', 'lodge', 'theatre', 'centre', 'bar',
    'bed and breakfast', 'train', 'station', 'gallery', 'la', 'time', 'house',
    'guest house', 'old', 'pool', 'house', 'a', 'b', 'the', 'cafe', 'cambridge',
    'hospital', 'restaurant\'s'
]

# A collection of phrases that are semantically similar to the key value, which
# is a word.
_SIMILAR_WORDS = {
    'portuguese': ['portugese', 'portugeuese'],
    '01:30': ['1 thirty p . m .'],
    '16:30': ['after 16:00'],
    'anatolia': ['anatoilia'],
    'allenbell': ['allenball'],
    'caribbean': ['carribbean'],
    'seafood': ['sea food'],
    'moroccan': ['morrocan'],
    'avalon': ['avaion'],
    'barbeque': ['bbq'],
    'american': ['americas'],
    'italian': ['pizza place'],
    'indian': ['taj tandoori'],
    'british': ['english'],
    'cambride': ['cambridge'],
    'fenditton': ['fen ditton'],
    'cafe': ['caffe'],
    'gonvile': ['gonville'],
    'shaddia': ['shaddai']
}

# A collection of phrases that are semantically similar to the key value, which
# is a phrase consisted of more than one word.
_SIMILAR_PHRASES = {
    'alexander bed and breakfast': [
        'alexander b&b', 'alexander bed and breafast',
        'alexander bed & breakfast'
    ],
    'a and b guest house': [
        'a & b guest house', 'a and b guesthouse', 'a and be guest house'
    ],
    'saint johns chop house': ['saint johns chop shop house'],
    'bridge guest house': ['bridge guesthouse'],
    'finches b and b': ['finches b & b', 'finches b&b'],
    'finches bed and breakfast': ['flinches bed and breakfast', 'finches b&b'],
    'carolina bed and breakfast': ['carolina b&b'],
    'city centre north b and b': [
        'city centre north b&b', 'city centre north b & b'
    ],
    'lan hong house': ['ian hong house', 'ian hong'],
    'ugly duckling': ['ugly ducking'],
    'sri lankan': ['sri lanken'],
    'cambridge punter': ['cambridge punte'],
    'abc theatre': ['adc theatre']
}


def _locate_boundary(phrase, text):
  """Locate the span of the phrase using exact match."""

  def _locate_token_boundary(pos, text):
    """Get the start and end index of a token that covers a certain position."""
    if pos < 0:
      raise ValueError('Pos {} should be a positive integer.'.format(pos))
    next_space = text.find(' ', pos)
    left_boundary = text.rfind(' ', 0, pos) + 1
    right_boundary = next_space if next_space != -1 else len(text)
    return left_boundary, right_boundary

  phrase = phrase.strip()
  pos_in_text = text.find(phrase)
  if pos_in_text == -1:
    return None, None

  tokens = phrase.split()
  start_idx, _ = _locate_token_boundary(pos_in_text, text)
  last_token = tokens[-1]
  find_last_token = text.find(last_token,
                              pos_in_text + len(phrase) - len(last_token))
  if find_last_token == -1:
    raise ValueError('Should find the last word for value {}'.format(phrase))
  _, end_idx = _locate_token_boundary(find_last_token, text)
  # If it's a number, the value should be exactly the same.
  if phrase.isdigit() and text[start_idx:end_idx] != phrase:
    return None, None
  # If the phrase is short, the value should be exactly the same.
  # e.g. we don't want to match "theatre" when searching for "the"
  if (len(phrase) <= 3 and len(phrase) != (end_idx - start_idx)):
    return None, None
  return start_idx, end_idx


def _locate_word(word, text, start_pos):
  """Get start and end index of a phrase that semantically equals to a word."""
  # If the word to search for contains 3 or 4 digits, correct it into time.
  obj = re.match(r'(?<!\d)\d{3,4}(?!\d)', word)

  assert start_pos <= len(text)
  if start_pos == len(text):
    return None, None
  text = text[start_pos:]
  if obj:
    if int(obj.group()) < 10000:
      word = ':'.join([obj.group(0)[:-2], obj.group(0)[-2:]])
  obj = re.match(r'^(\d+):(\d+)', word)
  if obj:
    # If word is about time, try different variations.
    # e.g. 10:15 can be written as 1015 or 10.15.
    times_to_try = [
        obj.group(0),
        obj.group(1) + obj.group(2), '.'.join([obj.group(1),
                                               obj.group(2)])
    ]
    hour = int(obj.group(1))
    minute = int(obj.group(2))
    if hour > 12:
      times_to_try.append(':'.join([str(hour - 12), obj.group(2)]))
      if minute == 0:
        times_to_try.append(str(hour - 12) + ' pm')
        times_to_try.append(str(hour - 12) + 'pm')
        times_to_try.append(str(hour - 12) + ' p . m .')
        times_to_try.append(str(hour - 12) + ' o\'clock p . m .')
        times_to_try.append(str(hour - 12) + ' o\'clock')
        times_to_try.append(str(hour) + ' o\'clock')
        times_to_try.append(str(hour - 12) + ':00')
        times_to_try.append(str(hour))
    elif (hour == 12 and minute == 0):
      times_to_try.extend(
          ['12 pm', '12pm', '12 o\'clock', '12 p . m .', '12', 'noon'])
    else:
      times_to_try.append(':'.join([str(hour + 12), obj.group(2)]))
      if int(minute) == 0:
        times_to_try.append(str(hour) + ' am')
        times_to_try.append(str(hour) + 'am')
        times_to_try.append(str(hour) + ' a . m .')
        times_to_try.append(str(hour) + ' o\'clock a . m .')
        times_to_try.append(str(hour) + ' o\'clock')
        times_to_try.append(str(hour + 12) + ':00')
        times_to_try.append(str(hour))
    if (minute == 15 or minute == 45 or minute == 30):
      times_to_try.append('after ' + str(hour) + ':' + str(minute - 15))
      if hour < 10:
        times_to_try.append('after 0' + str(hour) + ':' + str(minute - 15))
    if minute == 0:
      times_to_try.append('after ' + str(hour - 1) + ':45')
    for time_value in times_to_try:
      # Correct time like "08:15" to "8:15" to increase match possibility.
      if time_value[0] == '0':
        if len(time_value) > 2 and time_value[1] != [':']:
          time_value = time_value[1:]
      start_idx, end_idx = _locate_boundary(time_value, text)
  else:
    start_idx, end_idx = _locate_boundary(word, text)
    if start_idx is not None:
      return start_idx + start_pos, end_idx + start_pos
  # Try phrases that is similar to the word to find.
  for similar_word in _SIMILAR_WORDS.get(word, []):
    start_idx, end_idx = _locate_boundary(similar_word, text)
    if start_idx is not None:
      return start_idx + start_pos, end_idx + start_pos

  # Slot values ended with 's' can be written in different formats.
  # e.g. rosas can be written as rosa, rosa's.
  if word.endswith('s') and len(word) > 3:
    modified_words = [word[:-1] + '\'s', word[:-1]]
    for modified_word in modified_words:
      start_idx, end_idx = _locate_boundary(modified_word, text)
      if start_idx is not None:
        return start_idx + start_pos, end_idx + start_pos
  return None, None


def exists_in_prev_dialog_states(slot_value, converted_turns):
  """Whether slot value exists in the previous dialogue states."""
  for user_turn in converted_turns[::2]:
    assert user_turn['speaker'] == 'USER'
    for frame in user_turn['frames']:
      if 'state' in frame and 'slot_values' in frame['state']:
        slot_values_dict = frame['state']['slot_values']
        for slot, values_list in slot_values_dict.items():
          new_list = []
          for value in values_list:
            new_list.extend(value.split('|'))
          if slot_value in new_list:
            return frame['service'], slot, values_list
  return None, None, None


class Processor(object):
  """A processor to convert Multiwoz to the data format used in SGD."""

  def __init__(self, schemas):
    self._schemas = schemas
    # For statistically evaluating the modifications.
    # Number of non-categorical slot values in dialogue state, which needs span
    # annotations.
    self._slot_spans_num = 0
    # Dict to track the number of non-categorical slot values whose span can not
    # be found.
    self._unfound_slot_spans_num = collections.Counter()

    # Dict used to correct categorical slot values annotated in MultiWOZ 2.1.
    self._slot_value_correction_for_cat_slots = {}
    with tf.io.gfile.GFile(_CORRECT_FOR_STATE_PATH, 'r') as f:
      for line in f:
        tok_from, tok_to = line.replace('\n', '').split('\t')
        self._slot_value_correction_for_cat_slots[tok_from] = tok_to

  @property
  def unfound_slot_span_ratio(self):
    """Get the ratio of the slot spans that can't be found in the utterances."""
    ratio_dict = {
        k: float(v) / float(self._slot_spans_num)
        for k, v in self._unfound_slot_spans_num.items()
    }
    ratio_dict['total'] = float(sum(
        self._unfound_slot_spans_num.values())) / float(self._slot_spans_num)
    return ratio_dict

  def _basic_text_process(self, text, lower=True):
    # Remove redundant spaces.
    text = re.sub(r'\s+', ' ', text).strip()
    if lower:
      text = text.lower()
    return text

  def _insert_slots_annotations_to_turn(self, turn, slots_annotations_list,
                                        service_name):
    """Insert slot span annotations to a turn."""
    found_service = False
    for frame in turn['frames']:
      if frame['service'] == service_name:
        frame['slots'].extend(slots_annotations_list)
        found_service = True
        continue
    if not found_service:
      turn['frames'].append({
          'service': service_name,
          'slots': slots_annotations_list,
          'actions': []
      })
    return

  def _correct_state_value_for_noncat(self, slot, val):
    """Correct slot values for non-categorical slots."""
    val = val.strip()
    if ((val == 'cam' and slot == 'restaurant-name') or
        (val == 'friday' and slot == 'train-leaveat') or
        (val == 'bed' and slot == 'attraction-name')):
      return ''
    if val == 'portugese':
      val = 'portuguese'
    return val

  def _correct_state_value_for_cat(self, _, val):
    """Correct slot values for categorical slots."""
    val = val.strip()
    return self._slot_value_correction_for_cat_slots.get(val, val)

  def _get_intent_from_actions(self, state_value_dict, sys_actions,
                               user_actions):
    """Generate user intent by rules.

    We assume each service has only one active intent which equals to the domain
    mentioned in the current user turn.
    We use _infer_domains_from_actions to infer the list of possible domains.
    Domains that appear in the user actions and dialogue updates are prioritised
    over domains mentioned in the previous system actions.
    In the provided schema of MultiWOZ 2.1, every service contains one domain,
    so the active_intent is either "NONE" or "find_{domain}" for every service.

    Args:
      state_value_dict: a dict, key is the slot name, value is a list.
      sys_actions: a list of sys actions in the next turn.
      user_actions: a list of user actions.

    Returns:
      String, intent of the current user turn.
    """

    def _infer_domains_from_actions(state_value_dict, sys_actions,
                                    user_actions):
      """Infer the domains involved in the current turn from actions."""
      user_mentioned_domains = set()
      for user_action in user_actions:
        domain = user_action['act'].lower().split('-')[0]
        if domain not in ['general', 'booking']:
          user_mentioned_domains.add(domain)
      sys_mentioned_domains = set()
      for sys_action in sys_actions:
        domain = sys_action['act'].lower().split('-')[0]
        if domain not in ['general', 'booking']:
          sys_mentioned_domains.add(domain)
      # Compute domains whose slot values get updated in the current turn.
      state_change_domains = set()
      for slot, _ in state_value_dict.items():
        domain_name = slot.split('-')[0]
        state_change_domains.add(domain_name)
      # Infer the possible domains involved in the current turn for a certain
      # service.
      return list(user_mentioned_domains.union(state_change_domains)) or list(
          sys_mentioned_domains)

    domains = _infer_domains_from_actions(state_value_dict, sys_actions,
                                          user_actions)
    return 'find_' + domains[0] if domains else _INACTIVE_INTENT

  def _is_filled(self, slot_value):
    """Whether a slot value is filled."""
    slot_value = slot_value.lower()
    return (slot_value and slot_value != 'not mentioned' and
            slot_value != 'none')

  def _new_service_name(self, domain):
    """Get the new service_name decided by the new schema."""
    # If the schema file only contains one service, we summarize all the slots
    # into one service, otherwise, keep the domain name as the service name.
    return _DEFAULT_SERVICE_NAME if (len(
        self._schemas.services) == 1) else domain

  def _get_slot_name(self, slot_name, service_name, in_book_field=False):
    """Get the slot name that is consistent with the schema file."""
    slot_name = 'book' + slot_name if in_book_field else slot_name
    return '-'.join([service_name, slot_name]).lower()

  def _generate_dialog_states(self, frame_dict, overwrite_slot_values):
    """Get the dialog states and overwrite some of the slot values."""
    dialog_states = collections.defaultdict(dict)
    orig_dialog_states = collections.defaultdict(dict)
    for domain_name, values in frame_dict.items():
      dialog_states_of_one_domain = {}
      for k, v in values['book'].items():
        if isinstance(v, list):
          for item_dict in v:
            new_states = {
                self._get_slot_name(slot_name, domain_name, in_book_field=True):
                slot_val for slot_name, slot_val in item_dict.items()
            }
            dialog_states_of_one_domain.update(new_states)
        if isinstance(v, str) and v:
          slot_name = self._get_slot_name(k, domain_name, in_book_field=True)
          dialog_states_of_one_domain[slot_name] = v
      new_states = {
          self._get_slot_name(slot_name, domain_name): slot_val
          for slot_name, slot_val in values['semi'].items()
      }
      dialog_states_of_one_domain.update(new_states)
      # Get the new service_name that is decided by the schema. If the
      # schema file only contains one service, we summarize all the slots into
      # one service, otherwise, keep the domain name as the service name.
      new_service_name = self._new_service_name(domain_name)
      # Record the orig state values without any change.
      orig_dialog_state_of_one_domain = copy.deepcopy(
          dialog_states_of_one_domain)
      for (key, value) in orig_dialog_state_of_one_domain.items():
        if key in self._schemas.get_service_schema(
            new_service_name).slots and self._is_filled(value):
          orig_dialog_states[new_service_name][key] = value
      # Correct the slot values in the dialogue state.
      corrected_dialog_states_of_one_domain = {}
      for k, v in dialog_states_of_one_domain.items():
        if k in self._schemas.get_service_schema(
            new_service_name).categorical_slots:
          corrected_dialog_states_of_one_domain[
              k] = self._correct_state_value_for_cat(
                  k, self._basic_text_process(v))
        else:
          corrected_dialog_states_of_one_domain[
              k] = self._correct_state_value_for_noncat(
                  k, self._basic_text_process(v))
      dialog_states_of_one_domain = {
          k: v
          for k, v in corrected_dialog_states_of_one_domain.items()
          if self._is_filled(v)
      }

      # Overwrite some of the slot values and changes the slot value of a slot
      # into a list.
      for slot, value in dialog_states_of_one_domain.items():
        dialog_states_of_one_domain[slot] = [value]
        if slot in overwrite_slot_values[new_service_name]:
          if value in overwrite_slot_values[new_service_name][slot]:
            dialog_states_of_one_domain[slot] = sorted(
                overwrite_slot_values[new_service_name][slot][value])
      # Only track the slot values that are listed in the schema file. Slots
      # such as reference number, phone number are filtered out.
      for (key, value) in dialog_states_of_one_domain.items():
        if key in self._schemas.get_service_schema(new_service_name).slots:
          dialog_states[new_service_name][key] = value
    return dialog_states, orig_dialog_states

  def _get_update_states(self, prev_ds, cur_ds):
    """Get the updated dialogue states between two user turns."""
    updates = collections.defaultdict(dict)
    for service, slot_values_dict in cur_ds.items():
      if service not in prev_ds:
        updates[service] = slot_values_dict
        continue
      for slot, values in slot_values_dict.items():
        for value in values:
          if slot not in prev_ds[service] or value not in prev_ds[service][slot]:
            updates[service][slot] = updates[service].get(slot, []) + [value]
    return updates

  def _generate_slot_annotation(self, orig_utt, slot, slot_value):
    """Generate the slot span of a slot value from the utterance.

    Args:
      orig_utt: Original utterance in string.
      slot: Slot name in string.
      slot_value: Slot value to be annotated in string.

    Returns:
      slot_ann: A dict that denotes the slot name and slot spans.
      slot_value: The corrected slot value based on the utterance. It's
        unchanged if the slot value can't be found in the utterance.
    """
    slot_ann = []
    utt = orig_utt.lower()
    start_idx, end_idx = None, None
    # Check if the utterance mentions any phrases that are semantically same as
    # the slot value.
    for alias_slot_value in ([slot_value] +
                             _SIMILAR_PHRASES.get(slot_value, [])):
      start_idx, end_idx = _locate_boundary(alias_slot_value, utt)
      if start_idx is not None:
        break
    if start_idx is None:
      # Tokenize the slot value and find each of them.
      splitted_slot_values = slot_value.strip().split()
      unfound_tokens_idx = []
      search_start_idx = 0
      # Find if each token exists in the utterance.
      for i, value_tok in enumerate(splitted_slot_values):
        tok_start_idx, tok_end_idx = _locate_word(value_tok, utt,
                                                  search_start_idx)
        if tok_start_idx is not None and tok_end_idx is not None:
          # Hard coded rules
          # if the value to find is one of ['and', 'of', 'by'] and
          # there's no token prior to them having been found, we don't think
          # the value as found since they are fairly common words.
          if value_tok in ['and', 'of', 'by'] and start_idx is None:
            unfound_tokens_idx.append(i)
            continue
          if start_idx is None:
            start_idx = tok_start_idx
          search_start_idx = tok_end_idx
        else:
          unfound_tokens_idx.append(i)
      # Record the last index.
      if search_start_idx > 0:
        end_idx = search_start_idx
    if start_idx is None:
      return [], slot_value
    new_slot_value = utt[start_idx:end_idx]

    if abs(len(slot_value) - len(new_slot_value)) > 20:
      return [], slot_value
    if len(new_slot_value.split()) > (len(slot_value.strip().split()) + 2) and (
        new_slot_value not in _SIMILAR_PHRASES.get(slot_value, [])):
      return [], slot_value
    # If the value found from the utterance is one of values below and the real
    # slot value contains more than one tokens, we don't think it as a
    # successful match.
    if new_slot_value.strip() in _FOUND_VALUES_NEED_CHECK and len(
        slot_value.split()) > 1:
      return [], slot_value
    # If the value based on the utterance ends with any value below, we don't
    # annotate span of it.
    if new_slot_value.strip().split()[-1] in ['and', 'the', 'of', 'by']:
      return [], slot_value
    slot_ann.append({
        'slot': slot,
        'value': orig_utt[start_idx:end_idx],
        'exclusive_end': end_idx,
        'start': start_idx,
    })
    return slot_ann, new_slot_value

  def _update_corrected_slot_values(self, corrected_slot_values_dict,
                                    service_name, slot, slot_value,
                                    new_slot_value):
    """Update the dict that keeps track of the modified state values."""
    if slot not in corrected_slot_values_dict[service_name]:
      corrected_slot_values_dict[service_name][slot] = collections.defaultdict(
          set)
      corrected_slot_values_dict[service_name][slot][slot_value] = {slot_value}
    corrected_slot_values_dict[service_name][slot][slot_value].add(
        new_slot_value)
    return

  def _get_requested_slots_from_action(self, act_list):
    """Get user's requested slots from the action."""
    act_request = []
    for act_dict in act_list:
      if 'request' in act_dict['act'].lower():
        slot_name = act_dict['slot']
        if slot_name == 'Arrive':
          slot_name = 'arriveby'
        elif slot_name == 'Leave':
          slot_name = 'leaveat'
        act_request.append('-'.join([act_dict['act'].split('-')[0],
                                     slot_name]).lower())
    return act_request

  def _generate_actions(self, dialog_act):
    """Generate user/system actions."""
    converted_actions = collections.defaultdict(list)
    for k, pair_list in dialog_act.items():
      k_list = k.lower().strip().split('-')
      domain = k_list[0]
      service_name = self._new_service_name(domain)
      act_slot_values_dict = collections.defaultdict(list)
      for pair in pair_list:
        slot = pair[0]
        slot_value = pair[1]
        if slot != _NONE_VALUE:
          act_slot_values_dict[slot].append(slot_value)
      if not act_slot_values_dict:
        converted_actions[service_name].append({'act': k})
      for slot, values in act_slot_values_dict.items():
        converted_actions[service_name].append({
            'act': k,
            'slot': slot,
            'values': values
        })
    return converted_actions

  def _generate_dial_turns(self, turns, dial_id):
    """Generate the dialog turns and the services mentioned in the dialogue."""
    prev_dialog_states = collections.defaultdict(dict)
    corrected_slot_values = collections.defaultdict(dict)
    converted_turns = []
    appear_services = set()
    if len(turns) % 2 != 0:
      raise ValueError('dialog ended by user')
    for i in range(len(turns))[::2]:
      user_info = turns[i]
      sys_info = turns[i + 1]
      user_utt = self._basic_text_process(user_info['text'], False)
      sys_utt = self._basic_text_process(sys_info['text'], False)
      user_actions = collections.defaultdict(list)
      sys_actions = collections.defaultdict(list)
      if 'dialog_act' in user_info:
        user_actions = self._generate_actions(user_info['dialog_act'])
      if 'dialog_act' in sys_info:
        sys_actions = self._generate_actions(sys_info['dialog_act'])

      sys_turn = {
          'utterance': sys_utt,
          'speaker': 'SYSTEM',
          'frames': [],
          'turn_id': str(i + 1)
      }
      user_turn = {
          'utterance': user_utt,
          'speaker': 'USER',
          'frames': [],
          'turn_id': str(i)
      }
      dialog_states, _ = self._generate_dialog_states(sys_info['metadata'],
                                                      corrected_slot_values)
      appear_services.update(dialog_states.keys())

      # Fill in slot spans in the user turn and the previous system turn for
      # the non categorical slots.
      user_slots = collections.defaultdict(list)
      sys_slots = collections.defaultdict(list)
      update_states = self._get_update_states(prev_dialog_states, dialog_states)
      prev_sys_utt = converted_turns[-1]['utterance'] if converted_turns else ''
      for service_name, slot_values_dict in update_states.items():
        new_service_name = self._new_service_name(service_name)
        service_schema = self._schemas.get_service_schema(new_service_name)
        for slot, slot_value in slot_values_dict.items():
          assert slot_value, 'slot values shouls not be empty'
          slot_value = slot_value[0]
          if slot in service_schema.categorical_slots:
            if (slot_value not in service_schema.get_categorical_slot_values(
                slot) and slot_value not in [_DONT_CARE]):
              logging.error('Value %s not contained in slot %s, dial_id %s, ',
                            slot_value, slot, dial_id)
              dialog_states[service_name][slot] = [slot_value]
          else:
            self._slot_spans_num += 1
            if slot_value == _DONT_CARE:
              continue
            user_slot_ann, slot_value_from_user = self._generate_slot_annotation(
                user_utt, slot, slot_value)
            sys_slot_ann, slot_value_from_sys = self._generate_slot_annotation(
                prev_sys_utt, slot, slot_value)
            # Values from user utterance has a higher priority than values from
            # sys utterance. We correct the slot value of non-categorical slot
            # first based on user utterance, then system utterance.
            if user_slot_ann and slot_value_from_user != slot_value:
              if sys_slot_ann and (slot_value_from_sys == slot_value):
                user_slot_ann = None
              else:
                self._update_corrected_slot_values(corrected_slot_values,
                                                   service_name, slot,
                                                   slot_value,
                                                   slot_value_from_user)
                dialog_states[service_name][slot] = list(
                    corrected_slot_values[service_name][slot][slot_value])
            if (not user_slot_ann and sys_slot_ann and
                slot_value_from_sys != slot_value):
              self._update_corrected_slot_values(corrected_slot_values,
                                                 service_name, slot, slot_value,
                                                 slot_value_from_sys)
              dialog_states[service_name][slot] = list(
                  corrected_slot_values[service_name][slot][slot_value])
            if user_slot_ann:
              user_slots[service_name].extend(user_slot_ann)
            if sys_slot_ann:
              sys_slots[service_name].extend(sys_slot_ann)
            if not user_slot_ann and not sys_slot_ann:
              # First check if it exists in the previous dialogue states.
              from_service_name, from_slot, from_slot_values = (
                  exists_in_prev_dialog_states(slot_value, converted_turns))
              if from_service_name is not None:
                self._unfound_slot_spans_num['copy_from_prev_dialog_state'] += 1
                if FLAGS.annotate_copy_slots:
                  user_slots[service_name].append({
                      'slot': slot,
                      'copy_from': from_slot,
                      'value': from_slot_values
                  })
                continue
              # Second, trace back the dialogue history to find the span.
              for prev_turn in converted_turns[-2::-1]:
                prev_utt = prev_turn['utterance']
                prev_slot_ann, prev_slot_value = \
                    self._generate_slot_annotation(prev_utt, slot,
                                                   slot_value)
                if prev_slot_ann:
                  if prev_slot_value != slot_value:
                    self._update_corrected_slot_values(corrected_slot_values,
                                                       service_name, slot,
                                                       slot_value,
                                                       prev_slot_value)
                    dialog_states[service_name][slot] = list(
                        corrected_slot_values[service_name][slot][slot_value])
                  self._insert_slots_annotations_to_turn(
                      prev_turn, prev_slot_ann, service_name)
                  break
              self._unfound_slot_spans_num[slot] += 1
              continue
      # Fill in slot annotations for the system turn.
      for service_name in sys_slots:
        if not sys_slots[service_name]:
          continue
        self._insert_slots_annotations_to_turn(converted_turns[-1],
                                               sys_slots[service_name],
                                               service_name)
      # Generate user frames from dialog_states.
      latest_update_states = self._get_update_states(prev_dialog_states,
                                                     dialog_states)
      for service_name, slot_values_dict in dialog_states.items():
        user_intent = self._get_intent_from_actions(
            latest_update_states[service_name], sys_actions[service_name],
            user_actions[service_name])
        # Fill in values.
        user_turn['frames'].append({
            'slots': user_slots[service_name],
            'state': {
                'slot_values': {k: v for k, v in slot_values_dict.items() if v},
                'requested_slots':
                    self._get_requested_slots_from_action(
                        user_actions[service_name]),
                'active_intent':
                    user_intent
            },
            'service': service_name
        })
      non_active_services = set(self._schemas.services) - appear_services
      for service_name in non_active_services:
        user_intent = self._get_intent_from_actions({},
                                                    sys_actions[service_name],
                                                    user_actions[service_name])
        user_turn['frames'].append({
            'service': service_name,
            'slots': [],
            'state': {
                'active_intent':
                    user_intent,
                'requested_slots':
                    self._get_requested_slots_from_action(
                        user_actions[service_name]),
                'slot_values': {}
            }
        })
      converted_turns.extend([user_turn, sys_turn])
      prev_dialog_states = dialog_states
    return converted_turns, list(appear_services)

  def convert_to_dstc(self, id_list, dialogs):
    """Generate a list of dialogues in the dstc8 data format."""
    converted_dialogs = []
    for dial_id in id_list:
      converted_turns, covered_services = self._generate_dial_turns(
          dialogs[dial_id]['log'], dial_id)
      dialog = {
          'dialogue_id': dial_id,
          'services': covered_services,
          'turns': converted_turns
      }
      converted_dialogs.append(dialog)
    return converted_dialogs


def main(_):
  schema_path = os.path.join(_DIR_PATH, FLAGS.schema_file_name)
  schemas = schema.Schema(schema_path)
  processor = Processor(schemas)
  data_path = os.path.join(FLAGS.input_data_dir, 'data.json')
  with tf.io.gfile.GFile(data_path, 'r') as f:
    data = json.load(f)
  dev_test_ids = []
  output_dir = FLAGS.output_dir or _DIR_PATH
  # Generate dev and test set according to the ids listed in the files. Ids not
  # included in the dev and test id list files belong to the training set.
  for output_dir_name, file_name in _PATH_MAPPING:
    output_sub_dir = os.path.join(output_dir, output_dir_name)
    if not tf.io.gfile.exists(output_sub_dir):
      tf.io.gfile.makedirs(output_sub_dir)
    schema_path = os.path.join(output_sub_dir, 'schema.json')
    schemas.save_to_file(schema_path)
    dial_ids = []
    if file_name:
      id_list_path = os.path.join(FLAGS.input_data_dir, file_name)
      with tf.io.gfile.GFile(id_list_path) as f:
        dial_ids = [id_name.strip() for id_name in f.readlines()]
      dev_test_ids.extend(dial_ids)
    else:
      # Generate the ids for the training set.
      dial_ids = list(set(data.keys()) - set(dev_test_ids))
    converted_dials = processor.convert_to_dstc(dial_ids, data)
    logging.info('Unfound slot span ratio %s',
                 processor.unfound_slot_span_ratio)
    logging.info('Writing %d dialogs to %s', len(converted_dials),
                 output_sub_dir)
    for i in range(0, len(converted_dials), _NUM_DIALS_PER_FILE):
      file_index = int(i / _NUM_DIALS_PER_FILE) + 1
      # Create a new json file and save the dialogues.
      json_file_path = os.path.join(output_sub_dir,
                                    'dialogues_{:03d}.json'.format(file_index))
      dialogs_list = converted_dials[(file_index - 1) *
                                     _NUM_DIALS_PER_FILE:file_index *
                                     _NUM_DIALS_PER_FILE]
      with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(
            dialogs_list, f, indent=2, separators=(',', ': '), sort_keys=True)
      logging.info('Created %s with %d dialogues.', json_file_path,
                   len(dialogs_list))


if __name__ == '__main__':
  app.run(main)
