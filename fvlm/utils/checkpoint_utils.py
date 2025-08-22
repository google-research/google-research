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

"""Implementation of checkpoint loading utils."""
import re
from typing import Any, Dict, Mapping, Sequence, Text, Tuple, Union

from absl import logging
import flax
from flax import jax_utils
from flax import struct
from flax import traverse_util
from flax.core import frozen_dict
from flax.training import checkpoints
import jax
import jax.numpy as jnp
from utils import clip_utils


Array = jnp.ndarray
ArrayDict = Dict[Text, Array]
LossArray = Union[Array, ArrayDict]
KeyMap = Dict[Tuple[Text], Tuple[Text]]
FrozenDict = flax.core.frozen_dict.FrozenDict


CLIP_CHECKPOINTS = (
    'resnet_50',
    'resnet_101',
    'resnet_50x4',
    'resnet_50x16',
    'resnet_50x64',
)


@struct.dataclass
class TrainState:
  step: int
  optimizer: Any
  model_state: Any


def is_chief():
  """Is this the chief host?"""
  return jax.host_id() == 0


def maybe_save_checkpoint(step,
                          train_state,
                          output_dir,
                          checkpoint_every = 2000,
                          keep_latest_n_checkpoints = 10,
                          total_train_steps = 22500):
  """Persists the model checkpoint at every `checkpoint_every` step.

  Args:
    step: The current step.
    train_state: The training state object.
    output_dir: The path to output directory.
    checkpoint_every: The interval to save checkpoint.
    keep_latest_n_checkpoints: The number of last checkpoints to keep.
    total_train_steps: The total number of training steps.
  """
  if (step % checkpoint_every == 0 or step == total_train_steps - 1):
    logging.info('Saving checkpoint for step %d', step)
    train_state = jax_utils.unreplicate(train_state)
    checkpoints.save_checkpoint_multiprocess(
        output_dir, train_state, step=step, keep=keep_latest_n_checkpoints)


def filter_variables(key,
                     filters):
  """Matches names to prefixes and filters out not matching layer names.

  For example, if we have a model that has the follow parameter:
  ('lang_model', 'encoder', 'encoder_norm', 'scale')
  i.e., it has a 'lang_model' module that has an encoder with the scale param
  and we want to restore from a checkpoint that has

  ('decoder', 'layers_5', 'encoder_decoder_attention', 'out', 'kernel')
  ('encoder', 'encoder_norm', 'scale')

  We need to filter out the 'decoder' from the checkpoint and remove the
  'lang_model' prefix from the model. This function allows those things to be
  done.

  This function takes a key, which is a list of strings, as input, e.g.,
  ('lang_model', 'encoder', 'encoder_norm', 'scale').
  The filters have the format of [('prefix_string', 'filter_string'), ...].
  The prefix strings are matched and removed from the start of the key.

  For example, a prefix of 'lang_model' would result in
  ('encoder', 'encoder_norm', 'scale').

  The filter string is used to removed unwanted variables/layers.
  For example, a filter string of 'encoder' would then take
  ('decoder', 'layers_5', 'encoder_decoder_attention', 'out', 'kernel')
  ('encoder', 'encoder_norm', 'scale')
  and result in just
  ('encoder', 'encoder_norm', 'scale')

  This will then have matching names to restore the parameters. For example use,
  see checkpoint_utils_test.py.

  Args:
    key: A list of strings used as descrbed above.
    filters: A list of tuples that have [(prefix_string, filter_string)]. Note
      that this is an AND condition. I.e., prefix_string AND filter_string must
      be true for the key to pass the condition. Note that prefix_string can be
      '', in which case the prefix is ignore and only the filter_string
      condition is checked. For examples, see checkpoint_utils_test.

  Returns:
    A tuple of [bool, str]: True if passes filter. False if filtered out.
      And the new key used to match the name of the other
      (either model or checkpoint key).
  """
  key_str = '.'.join(key)
  for prefix, var_filter in filters:
    filter_regex = re.compile(f'^{var_filter}')
    if prefix:
      prefix_regex = re.compile(f'^{prefix}')
      # If we match the prefix, we remove it.
      if re.match(prefix_regex, key_str):
        sub_key_str = re.sub(prefix_regex, '', key_str)
        if sub_key_str.startswith('.'):
          # Remove '.' if it starts with a '.'.
          # E.g. module.layer with a prefix of module results in .layer, we want
          # to remove the '.' to get just layer.
          sub_key_str = sub_key_str[1:]
        # Check filters
        if re.match(filter_regex, sub_key_str):
          # Match found.
          return True, sub_key_str
    else:
      # Check filters.
      if re.match(filter_regex, key_str):
        return True, key_str
  return False, ''


def build_assignment_map(
    target_state,
    pretrained_state,
    target_filters = (),
    pretrained_filters = (),
):
  """Build assignment map from source checkpoint to target model weights.

  This function currently only supports prefixing the pretrained and target
  state for selecting which variables to restore to and from. This allows the
  users to restore any flax.deprecated.nn module or submodule from another
  module or
  submodule. For more sophisticated checkpoint restoration, users would need to
  build their own assignment maps.

  Args:
    target_state: The train state to restore checkpoint into.
    pretrained_state: The pretrained state provided by the checkpoint.
    target_filters: A list of tuples  of the (prefix_string, filter_string) to
      select the target state to restore. The prefix string is removed from
      variable names to match the pretrained variable names. The filter_string
      selects the variables to restore.
    pretrained_filters: A list of tuples  of the (prefix_string, filter_string)
      to select the pretrained state to restore. The prefix string is removed
      from variable names to match the target variable names. The filter_string
      selects the variables to restore.

  Returns:
    assignment_map: A dictionary with two keys. One key 'weights' has value a
      dictionary with key - flattened target weights key name, and value -
      flattened restored weights key name. Another key 'batch_stats' has value a
      dictionary with key - flattened target batch stats key name, and value -
      flattened restored batch stats key name.
  """

  def assignment_map_from_param_prefix(target_params,
                                       restored_params):
    """Build assignment map from source checkpoint to target model weights.

    Args:
      target_params: The target parameters to restore checkpoint into (nested
        dictionary).
      restored_params: The pretrained parameters provided by the checkpoint
        (nested dictionary).

    Returns:
      assignment_map: A dictionary with key-value pairs as:
        key: flattened target state key name.
        value: flattened restored state key name.
    """
    flat_restored_params = traverse_util.flatten_dict(restored_params)
    flat_target_params = traverse_util.flatten_dict(target_params)

    if target_filters:
      flat_target_key_map = {}
      for k, _ in flat_target_params.items():
        belongs, key = filter_variables(k, list(target_filters))
        if belongs:
          flat_target_key_map[tuple(key.split('.'))] = k
    else:
      flat_target_key_map = {k: k for k in flat_target_params.keys()}

    if pretrained_filters:
      flat_restored_key_map = {}
      for k, _ in flat_restored_params.items():
        belongs, key = filter_variables(k, list(pretrained_filters))
        if belongs:
          flat_restored_key_map[tuple(key.split('.'))] = k
    else:
      flat_restored_key_map = {k: k for k in flat_restored_params.keys()}

    target_keys = set(flat_target_key_map.keys())
    restored_keys = set(flat_restored_key_map.keys())
    overlap_keys = target_keys.intersection(restored_keys)

    if not target_keys:
      logging.warning('There are no target keys found for %s', target_filters)
    if not restored_keys:
      logging.warning('There are no restored keys found for %s',
                      pretrained_filters)
    assignment_map = {
        flat_target_key_map[k]: flat_restored_key_map[k] for k in overlap_keys
    }
    # Check for shape equality.
    for target_key, restore_key in assignment_map.items():
      if (flat_target_params[target_key].shape !=
          flat_restored_params[restore_key].shape):
        logging.info('Target shape: %s', flat_target_params[target_key].shape)
        logging.info('Source shape: %s',
                     flat_restored_params[restore_key].shape)
        raise ValueError(
            f'Array shapes unequal ({target_key}) vs ({restore_key})!!!')

    logging.info('Target keys: %s', target_keys)
    logging.info('Pretrained keys %s', restored_keys)
    missing_target_keys = target_keys - overlap_keys
    missing_restored_keys = restored_keys - overlap_keys
    if missing_target_keys:
      raise ValueError(f'Missing target keys {missing_target_keys}!!')
    if missing_restored_keys:
      raise ValueError(f'Missing restored keys {missing_restored_keys}!!')
    return assignment_map

  restored_weights = pretrained_state['optimizer']['target']
  target_weights = frozen_dict.unfreeze(target_state.optimizer.target)
  weights_assignment_map = assignment_map_from_param_prefix(
      target_weights, restored_weights)
  if ('batch_stats' in target_state.model_state or
      'batch_stats' in pretrained_state['model_state']):
    restored_batch_stats = pretrained_state['model_state'].get(
        'batch_stats', {})
    target_batch_stats = frozen_dict.unfreeze(
        target_state.model_state.get('batch_stats', {}))
    batch_stats_assignment_map = assignment_map_from_param_prefix(
        target_batch_stats, restored_batch_stats)
  else:
    batch_stats_assignment_map = {}

  logging.info('Restoring the assignment map weights %s batch_stats %s',
               weights_assignment_map, batch_stats_assignment_map)
  return {
      'weights': weights_assignment_map,
      'batch_stats': batch_stats_assignment_map
  }


def update_from_pretrained_state(
    target_state, pretrained_state,
    assignment_map):
  """Update the TrainState from pretrained state.

  The update includes both the model weights and the batch norm parameters.

  Args:
    target_state: The train state to restore checkpoint into.
    pretrained_state: The pretrained state provided by the checkpoint.
    assignment_map: A dictionary with two keys. One key 'weights' has value a
      dictionary with key - flattened target weights key name, and value -
      flattened restored weights key name. Another key 'batch_stats' has value a
      dictionary with key - flattened target batch stats key name, and value -
      flattened restored batch stats key name.

  Returns:
    restored_target_state: The updated train state after loading the checkpoint.
  """
  # Restore model weights.
  restored_weights = pretrained_state['optimizer']['target']
  target_weights = frozen_dict.unfreeze(target_state.optimizer.target)
  flat_restored_weights = traverse_util.flatten_dict(restored_weights)
  flat_target_weights = traverse_util.flatten_dict(target_weights)

  for target_key, restore_key in assignment_map['weights'].items():
    if target_key not in flat_target_weights:
      raise ValueError(f'Target key {target_key} does not exist in the model.')
    flat_target_weights[target_key] = flat_restored_weights[restore_key]

  restored_target_weights = frozen_dict.freeze(
      traverse_util.unflatten_dict(flat_target_weights))

  # Restore model state batch normalization parameters.
  if ('batch_stats' in target_state.model_state or
      'batch_stats' in pretrained_state['model_state']):
    restored_batch_stats = pretrained_state['model_state'].get(
        'batch_stats', {})
    target_batch_stats = frozen_dict.unfreeze(
        target_state.model_state.get('batch_stats', {}))
    flat_restored_batch_stats = traverse_util.flatten_dict(restored_batch_stats)
    flat_target_batch_stats = traverse_util.flatten_dict(target_batch_stats)
    for target_key, restore_key in assignment_map['batch_stats'].items():
      if target_key not in flat_target_batch_stats:
        raise ValueError(
            f'Target key {target_key} does not exist in the batch stats.')
      flat_target_batch_stats[target_key] = (
          flat_restored_batch_stats[restore_key])
    restored_target_batch_stats = frozen_dict.freeze(
        {'batch_stats': traverse_util.unflatten_dict(flat_target_batch_stats)})
  else:
    restored_target_batch_stats = {}

  restored_optimizer = target_state.optimizer.replace(
      target=restored_target_weights)
  restored_target_state = target_state.replace(
      model_state=restored_target_batch_stats,
      optimizer=restored_optimizer)

  return restored_target_state


def load_clip_weights(
    model_name, load_vision = True):
  """Load pretrained CLIP checkpoints of vision/text encoders."""
  clip_vars = clip_utils.load_model_vars(model_name)
  model_type = 'visual' if load_vision else 'text'
  # Need the structure of what is returned here to match what is expected
  # build_assignment_map().
  params = clip_vars['params'][model_type]
  batch_stats = clip_vars['batch_stats'].get(model_type, {})
  return {
      'optimizer': {
          'target': params,
      },
      'model_state': {
          'batch_stats': batch_stats,
      }
  }


def load_pretrained_vit_v2_weights(
    checkpoint_path, load_vision = True):
  """Load ViT V2 checkpoints (pretrained in PAX) of vision/text encoders."""
  model_vars = checkpoints.restore_checkpoint(checkpoint_path, None)
  model_vars = model_vars['model_state']
  model_type = 'visual' if load_vision else 'text'
  # Need the structure of what is returned here to match what is expected
  # build_assignment_map().
  params = model_vars['params'][model_type]
  if 'batch_stats' in model_vars:
    batch_stats = model_vars['batch_stats'].get(model_type, {})
  else:
    batch_stats = {}
  return {
      'optimizer': {
          'target': params,
      },
      'model_state': {
          'batch_stats': batch_stats,
      }
  }


def load_pretrained_detector_ckpt_weights(
    checkpoint_path):
  """Load pretrained detector ckpt weights."""
  model_vars = checkpoints.restore_checkpoint(checkpoint_path, None)
  params = model_vars['optimizer']['target']
  batch_stats = {}
  return {
      'optimizer': {
          'target': params,
      },
      'model_state': {
          'batch_stats': batch_stats,
      }
  }


def load_pretrained_clip_or_vit_v2_weights(
    model_name_or_path, load_vision = True):
  """Load pretrained CLIP or ViT V2 (PAX) checkpoints."""
  if model_name_or_path in CLIP_CHECKPOINTS:
    return load_clip_weights(model_name_or_path, load_vision)
  else:
    return load_pretrained_vit_v2_weights(model_name_or_path, load_vision)


def restore_with_assignment_map(train_state, pretrained_state,
                                pretrain_target_filters,
                                pretrain_source_filters):
  """Restore the train state with assignment maps."""
  assignment_map = build_assignment_map(
      train_state,
      pretrained_state,
      target_filters=pretrain_target_filters,
      pretrained_filters=pretrain_source_filters)
  logging.info('Loading pretrained variables (format: `target: pretrained`):')
  for k, v in traverse_util.flatten_dict(assignment_map).items():
    logging.info('%s: %s', k, v)
  train_state = update_from_pretrained_state(
      train_state, pretrained_state, assignment_map=assignment_map)
  return train_state
