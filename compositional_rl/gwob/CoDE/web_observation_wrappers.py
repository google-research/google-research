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

"""Wrappers for mapping text-based web observations to numpy arrays."""
import datetime

from CoDE.utils import dom_attributes
from CoDE.utils import dom_element_representation
from CoDE.utils import get_dom_elements
from CoDE.utils import indexify
from CoDE.vocabulary_utils import tokenize
import gin
import numpy as np


def wrap_structured_profile(raw_profile, local_vocabulary, profile_length,
                            number_of_fields):
  """If using structured profile, wrap into numpy array.

  Args:
    raw_profile: A dictionary of key and value pairs from user profile.
    local_vocabulary: A vocabulary of tokens to ids.
    profile_length: Length of the sequences in user profiles.
    number_of_fields: Number of fields in the profile.

  Returns:
    (profile_keys, profile_values, profile_key_mask, profile_value_mask):
    Wrapped profile in numpy arrays. Each profile array is a 2D tensor with a
    shape of (number of fields, sequence length).
  """

  # tokenize keys from profile fields
  profile_key_tokens = [tokenize(token) for token in raw_profile.keys]
  # convert key tokens into ids and pad
  profile_keys = [
      indexify(tokens, local_vocabulary, profile_length)
      for tokens in profile_key_tokens
  ]
  # masks for key tokens
  profile_key_mask = np.asarray(
      [[0] * profile_length for _ in range(number_of_fields)])
  for i, token in enumerate(profile_key_tokens):
    profile_key_mask[i, 0:len(token)] = 1.0
  profile_mask = np.zeros(number_of_fields)
  profile_mask[0:len(profile_keys)] = 1.0
  # pad each field
  while len(profile_keys) < number_of_fields:
    profile_keys.append([local_vocabulary.local_vocab['NULL']] * profile_length)
  profile_keys = np.stack(profile_keys, axis=0)

  # tokenize values from profile fields
  profile_value_tokens = [
      tokenize(raw_profile[token]) for token in raw_profile.keys
  ]
  # convert value tokens into ids and pad
  profile_values = [
      indexify(token, local_vocabulary, profile_length)
      for token in profile_value_tokens
  ]
  # masks for value tokens
  profile_value_mask = np.asarray(
      [[0] * profile_length for _ in range(number_of_fields)])
  for i, token in enumerate(profile_value_tokens):
    profile_value_mask[i, 0:len(token)] = 1.0
  # pad each field
  while len(profile_values) < number_of_fields:
    profile_values.append([local_vocabulary.local_vocab['NULL']] *
                          profile_length)
  profile_values = np.stack(profile_values, axis=0)
  return profile_keys, profile_values, profile_key_mask, profile_value_mask, profile_mask


@gin.configurable
def dom_representation(obs,
                       local_vocabulary,
                       dom_attribute_sequence_length,
                       number_of_dom_attributes,
                       number_of_dom_features,
                       prev_refs,
                       number_of_dom_elements,
                       prune_refs=None,
                       mask_for_id_prefix=None):
  """Generate dom element sequence from a given observation.

  Args:
    obs: Observation from WoB environment state.
    local_vocabulary: A vocabulary of tokens to ids.
    dom_attribute_sequence_length: Maximum sequence length for attribute
      sequences.
    number_of_dom_attributes: Number of attributes in an element.
    number_of_dom_features: Number of float features for each element.
    prev_refs: References of elements from previous state.
    number_of_dom_elements: Maximum number of dom elements in the observation.
    prune_refs: If given, use only the dom elements from this ref list.
    mask_for_id_prefix: If given, this will be used to prune elements based on
      their html ids so that support set of distribution will include only
      these.

  Returns:
    (dom, dom_features, dom_elements_mask): Dom and dom feature
    sequences, with mask sequences that indicate the padding.
  """
  dom = []
  dom_mask = []
  dom_features = []
  width = np.max(
      [dom_elem.width for dom_elem in get_dom_elements(obs, prune_refs)])
  height = np.max(
      [dom_elem.height for dom_elem in get_dom_elements(obs, prune_refs)])
  initial_number_of_dom_features = 8
  mask_for_id = np.ones(number_of_dom_elements)
  for dom_elem in get_dom_elements(obs, prune_refs):
    if mask_for_id_prefix and not dom_elem.id.startswith(mask_for_id_prefix):
      mask_for_id[len(dom)] = 0.0
    rep, rep_mask = dom_element_representation(
        dom_elem,
        local_vocabulary,
        dom_attribute_sequence_length,
        num_attributes=number_of_dom_attributes)
    dom.append(rep)
    dom_mask.append(rep_mask)
    if number_of_dom_features > 0:
      dom_features.append([
          float(dom_elem.focused),
          float(dom_elem.tampered),
          float(dom_elem.ref not in prev_refs),  # is new
          float(dom_elem.tag == 'div'),
          float(dom_elem.left / width),
          float(dom_elem.top / height),
          float(dom_elem.width / width),
          float(dom_elem.height / height)
      ])

    if len(dom) >= number_of_dom_elements:
      break
  action_mask = np.zeros(number_of_dom_elements)
  action_mask[0:len(dom)] = 1.0
  action_mask = action_mask * mask_for_id
  if len(dom) < number_of_dom_elements:
    if number_of_dom_features > 0:
      computed_number_of_dom_features = len(
          dom_features[0]) if dom_features else initial_number_of_dom_features
      dom_features += [[0] * computed_number_of_dom_features] * (
          number_of_dom_elements - len(dom))
    null_pad = [local_vocabulary.local_vocab['NULL']
               ] * dom_attribute_sequence_length
    dom += [[null_pad for _ in range(number_of_dom_attributes)]] * (
        number_of_dom_elements - len(dom))
    zero_pad = [0.0] * dom_attribute_sequence_length
    dom_mask += [
        np.asarray([zero_pad for _ in range(number_of_dom_attributes)])
    ] * (
        number_of_dom_elements - len(dom_mask))
  return dom, dom_features, action_mask, dom_mask


def wrap_dom_profile_intersection(obs, profile, use_only_profile_key,
                                  number_of_dom_attributes, local_vocabulary,
                                  dom_attribute_sequence_length,
                                  number_of_fields, number_of_dom_elements):
  """Wrap intersection between profile and dom elements.

  For each profile field key and value tokens (such as ["first", "name"]) and
  for each element attribute tokens (such as ["initial", "name", ":"]), find
  their overlapping tokens.

  Args:
    obs: Observation from WoB environment state.
    profile: A user profile of a list of key and value pairs.
    use_only_profile_key: If true, use only profile key and ignore profile
      values.
    number_of_dom_attributes: Number of attributes in an element.
    local_vocabulary: A vocabulary of tokens to ids
    dom_attribute_sequence_length: Maximum sequence length for attribute
      sequences.
    number_of_fields: Number of fields in the profile.
    number_of_dom_elements: Maximum number of dom elements in the observation.

  Returns:
    [dom_profile_intersection, dom_profile_intersection_mask,
          dom_profile_jaccard_sim]: A tuple of profile and dom intersection
          words, and word based similarity between profile and dom attributes.
    Intersection is a 5D tensor of shape (number of elements, max number of
    attributes, number of profile fields, number of action types (2), max
    sequence length).
  """
  dom_profile_intersection = []
  dom_profile_intersection_mask = []
  dom_profile_jaccard_sim = []
  profile_tokenized_key, profile_tokenized_value = {}, {}
  for key in sorted(profile.keys):
    profile_tokenized_key[key] = set(tokenize(key)) - {'NULL', 'OOV', 'none'}
    if not use_only_profile_key:
      profile_tokenized_value[key] = tokenize(profile[key])
  for dom_elem in get_dom_elements(obs):
    dom_profile_intersection.append([])
    dom_profile_intersection_mask.append([])
    dom_profile_jaccard_sim.append([])
    dom_attr = dom_attributes(dom_elem, num_attributes=number_of_dom_attributes)
    for attr in dom_attr:
      attr = str(attr)
      dom_profile_intersection[-1].append([])
      dom_profile_intersection_mask[-1].append([])
      dom_profile_jaccard_sim[-1].append([])
      dom_attr_tokenized = tokenize(attr)
      dom_attr_tokenized = set(dom_attr_tokenized)
      dom_attr_tokenized = dom_attr_tokenized - {
          'NULL', 'OOV', 'none', ',', '(', ')'
      }
      for key in sorted(profile.keys):
        try:
          d = datetime.datetime.strptime(profile[key], '%m/%d/%Y')
          profile_tokenized_value[key] = [str(d.day)]
        except ValueError as _:
          pass
        profile_tokenized_value[key] = set(profile_tokenized_value[key]) - {
            'NULL', 'OOV', 'none', ',', '(', ')'
        }
        intersections_key = list(dom_attr_tokenized
                                 & profile_tokenized_key[key])
        intersections_value = list(dom_attr_tokenized
                                   & profile_tokenized_value[key])
        dom_profile_intersection[-1][-1].append([
            indexify(intersections_key, local_vocabulary,
                     dom_attribute_sequence_length),
            indexify(intersections_value, local_vocabulary,
                     dom_attribute_sequence_length)
        ])
        mask_key = np.zeros(dom_attribute_sequence_length)
        mask_key[0:len(intersections_key)] = 1.0
        mask_value = np.zeros(dom_attribute_sequence_length)
        mask_value[0:len(intersections_key)] = 1.0
        dom_profile_intersection_mask[-1][-1].append([mask_key, mask_value])
        sim_key, sim_value = 0.0, 0.0
        if dom_attr_tokenized | profile_tokenized_key[key]:
          sim_key = len(intersections_key) / (
              len(dom_attr_tokenized | profile_tokenized_key[key]))
        if dom_attr_tokenized | profile_tokenized_value[key]:
          sim_value = len(intersections_value) / (
              len(dom_attr_tokenized | profile_tokenized_value[key]))
        sim_key_profile_persp, sim_value_profile_persp = 0.0, 0.0
        sim_key_dom_persp, sim_value_dom_persp = 0.0, 0.0
        if profile_tokenized_key[key]:
          sim_key_profile_persp = (
              len(intersections_key) / len(profile_tokenized_key[key]))
          sim_value_profile_persp = (
              len(intersections_key) / len(profile_tokenized_key[key]))
        if dom_attr_tokenized:
          sim_key_dom_persp = (len(intersections_key) / len(dom_attr_tokenized))
          sim_value_dom_persp = (
              len(intersections_key) / len(dom_attr_tokenized))
        dom_profile_jaccard_sim[-1][-1].append([
            np.asarray([sim_key, sim_key_profile_persp, sim_key_dom_persp]),
            np.asarray(
                [sim_value, sim_value_profile_persp, sim_value_dom_persp])
        ])
  null_sequence = [local_vocabulary.local_vocab['NULL']] * (
      dom_attribute_sequence_length)

  # Create shorter variable names to reduce the comprehension length.
  word_map = local_vocabulary.local_vocab
  sequence_length = dom_attribute_sequence_length
  num_dom_attr = number_of_dom_attributes
  num_dom_elems = number_of_dom_elements
  dom_profile_intersection += [[[[
      null_sequence, [word_map['NULL']] * (sequence_length)
  ] for _ in range(number_of_fields)] for _ in range(num_dom_attr)]] * (
      num_dom_elems - len(dom_profile_intersection))

  dom_profile_intersection_mask += [[[
      [[0.0] * (dom_attribute_sequence_length), [0.0] * (dom_attribute_sequence_length)] for _ in range(number_of_fields)
    ] for _ in range(number_of_dom_attributes)]] * (number_of_dom_elements - len(dom_profile_intersection_mask))  # pylint: disable=line-too-long
  dom_profile_jaccard_sim += [[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                                for _ in range(number_of_fields)]
                               for _ in range(number_of_dom_attributes)]] * (
                                   number_of_dom_elements -
                                   len(dom_profile_jaccard_sim))
  return [
      np.asarray(dom_profile_intersection, dtype=np.int32),
      np.asarray(dom_profile_intersection_mask, dtype=np.float32),
      np.asarray(dom_profile_jaccard_sim, dtype=np.float32)
  ]


def dom_profile_action_mask(number_of_fields,
                            number_of_dom_elements,
                            dom_profile_acted_list=None):
  """If dom elements and profile fields are already acted on, prune them.

  Args:
    number_of_fields: Number of fields in the profile.
    number_of_dom_elements: Maximum number of dom elements in the observation.
    dom_profile_acted_list: List of pairs of dom element and profile indices
      that are already acted and should be pruned.

  Returns:
    A 2D numpy array for masking dom elements and profile fields jointly.
  """
  action_mask = np.ones((number_of_fields, number_of_dom_elements))
  if dom_profile_acted_list:
    for field, dom_element in dom_profile_acted_list:
      if not field:
        action_mask[:, dom_element] = 0.0
      elif not dom_element:
        action_mask[field, :] = 0.0
      else:
        action_mask[field, dom_element] = 0.0
  return action_mask


def wrap_observation(obs, structured_field_extractor, num_steps, step_limit,
                     use_dom_profile_intersection, number_of_dom_features,
                     local_vocabulary, profile_length, number_of_fields,
                     dom_attribute_sequence_length, number_of_dom_attributes,
                     prev_refs, number_of_dom_elements, use_only_profile_key,
                     dom_profile_acted_list):
  """Wrap a given observation into numpy arrays.

  Args:
    obs: Observation from WoB environment state.
    structured_field_extractor: Extract key and value pairs from a raw profile.
    num_steps: Number of steps taken in the environment.
    step_limit: Maximum number of steps allowed.
    use_dom_profile_intersection: Use information from intersection of profile
      fields and dom elements.
    number_of_dom_features: Number of float features for each element.
    local_vocabulary: A vocabulary of tokens to ids
    profile_length: Length of the sequences in user profiles.
    number_of_fields: Number of fields in the profile.
    dom_attribute_sequence_length: Maximum sequence length for attribute
      sequences.
    number_of_dom_attributes: Number of attributes in an element.
    prev_refs: References of elements from previous state.
    number_of_dom_elements: Maximum number of dom elements in the observation.
    use_only_profile_key: If true, use only profile key and ignore profile
      values.
    dom_profile_acted_list: List of pairs of dom element and profile indices
      that are already acted and should be pruned.

  Returns:
    {profile_key, profile_value, profile_key_mask, profile_value_mask,
    dom_elements, dom_profile_joint_mask, time_step, dom_attribute_mask,
    dom_profile_intersection, dom_profile_intersection_mask, dom_features}:
    Observations wrapped in a dictionary of numpy arrays. Each profile array is
    a 2D array with a shape of (number of fields, sequence length), each dom
    element array is
  """
  raw_profile = obs.utterance.strip()
  raw_profile = structured_field_extractor(raw_profile)
  (profile_key, profile_value, profile_key_mask, profile_value_mask,
   profile_mask) = wrap_structured_profile(raw_profile, local_vocabulary,
                                           profile_length, number_of_fields)
  dom, dom_features, action_mask, dom_mask = dom_representation(
      obs, local_vocabulary, dom_attribute_sequence_length,
      number_of_dom_attributes, number_of_dom_features, prev_refs,
      number_of_dom_elements)

  dom_profile_joint_action_mask = dom_profile_action_mask(
      number_of_fields, number_of_dom_elements,
      dom_profile_acted_list) * np.expand_dims(
          action_mask, axis=0) * np.expand_dims(
              profile_mask, axis=1)

  time_step = [float(num_steps) / float(step_limit)]

  result = {
      'profile_key':
          np.asarray(profile_key, dtype=np.int32),
      'profile_value':
          np.asarray(profile_value, dtype=np.int32),
      'profile_key_mask':
          np.asarray(profile_key_mask, dtype=np.float32),
      'profile_value_mask':
          np.asarray(profile_value_mask, dtype=np.float32),
      'dom_elements':
          np.asarray(dom, dtype=np.int32),
      'dom_elements_mask':
          np.asarray(action_mask, dtype=np.float32),
      'dom_profile_joint_mask':
          np.asarray(dom_profile_joint_action_mask, dtype=np.float32),
      'time_step':
          np.asarray(time_step, dtype=np.float32),
      'dom_attribute_mask':
          np.asarray(dom_mask, dtype=np.float32)
  }
  if use_dom_profile_intersection:
    result['dom_profile_intersection'], result[
        'dom_profile_intersection_mask'], result[
            'dom_profile_jaccard_sim'] = wrap_dom_profile_intersection(
                obs, raw_profile, use_only_profile_key,
                number_of_dom_attributes, local_vocabulary,
                dom_attribute_sequence_length, number_of_fields,
                number_of_dom_elements)
  if number_of_dom_features > 0:
    result['dom_features'] = np.asarray(dom_features, dtype=np.float32)
  return result
