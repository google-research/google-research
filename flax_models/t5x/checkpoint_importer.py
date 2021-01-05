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

"""T5 Checkpoint Importer."""

import re

from flax import traverse_util
import jax
import numpy as np
import tensorflow as tf


class CheckpointTranslator:
  """Utility class for defining mapping rules from one flatdict to another.

  We assume a checkpoint is loaded as a dictionary with flattened keys of the
  form:  'name0/name1/name2/.../nameN'

  A rule is added with the 'add' decorator, which takes a regex matching rule
  and wraps a conversion function, feeding it (opts, key, val, **regex_groups)
  where opts is a dict containing apply-time keyword options for use by the
  conversion functions.
  """

  def __init__(self):
    self.rules = []

  def add(self, pattern):
    """Adds a new keyval conversion rule.

    Args:
      pattern: regex with capture groups for matching given sets of model
        variables.  We terminate all regexes with '$' to force complete matches.

    Returns:
      Translation function decorator for associating with the provided
      pattern.
    """

    def register_translation_fn_decorator(fn):
      # We force a complete match by adding end-of-string match.
      self.rules.append((re.compile(pattern + '$'), fn))
      return fn

    return register_translation_fn_decorator

  def apply(self, flatdict, **opts):
    """Applies rules to a flattened dictionary.

    Args:
      flatdict: flat-key dictionary of variables.
      **opts: additional config options for translation rules supplied at
        application time.

    Returns:
      Checkpoint data with translated key/values in flat-key dict format.
    """
    new_dict = {}
    unmatched = {}
    for k, v in flatdict.items():
      matched = False
      for rule_pat, rule_fn in self.rules:
        if rule_pat.match(k):
          groups = rule_pat.match(k).groups()
          new_k, new_v = rule_fn(opts, k, v, *groups)
          if new_k is not None:
            new_dict[new_k] = new_v
          matched = True
          break
      if not matched:
        unmatched[k] = v

    # We force every key-value pair in checkpoint to have a rule associated with
    # it.
    if unmatched:
      raise ValueError('Unmapped tensor keys exist: %s' % unmatched)

    return new_dict


# Create a translation rule set for importing T5 & T5.1.1 model checkpoints.
# -----------------------------------------------------------------------------
t5_importer = CheckpointTranslator()

# Name mappings.
SLOT_MAP = {'_slot_vc': 'v_col', '_slot_vr': 'v_row', '_slot_v': 'v'}
BLOCK_MAP = {'encoder': 'encoderblock', 'decoder': 'encoderdecoderblock'}


@t5_importer.add(r'global_step')
def global_step(opts, key, val):
  del opts, key
  return 'state/step', np.array(val, np.int32)


@t5_importer.add(r'shared/embedding(\w*)')
def shared_embeddings(opts, key, val, slot):
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/shared_embedding/embedding{suffix}'
  return newkey, val


@t5_importer.add(
    r'(encoder|decoder)/block_000/layer_000/SelfAttention/relative_attention_bias(\w*)'
)
def rel_embeddings(opts, key, val, encdec, slot):
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/{encdec}/{encdec}_relative_posemb/rel_embedding{suffix}'
  return newkey, val


@t5_importer.add(
    r'(encoder|decoder)/block_(\d+)/layer_\d+/(SelfAttention|EncDecAttention)/(q|k|v|o)(\w*)'
)
def attention_layers(opts, key, val, encdec, blocknum, attntype, qkvo, slot):
  """Attention layer."""
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  blocknum = int(blocknum)
  blockname = BLOCK_MAP[encdec]
  matrix = {'q': 'query', 'k': 'key', 'v': 'value', 'o': 'out'}[qkvo]
  attntype = {
      'SelfAttention': 'SelfAttention_0',
      'EncDecAttention': 'MultiHeadDotProductAttention_0'
  }[attntype]
  newkey = f'{prefix}/{encdec}/{blockname}_{blocknum}/{attntype}/{matrix}/kernel{suffix}'
  return newkey, val


@t5_importer.add(
    r'(encoder|decoder)/block_(\d+)/layer_\d+/DenseReluDense/(wi|wo)(?:_(\d+))?/kernel(\w*)'
)
def mlpblock(opts, key, val, encdec, blocknum, io_name, io_num, slot):
  """MLP block."""
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  blocknum = int(blocknum)
  blockname = BLOCK_MAP[encdec]
  io_num = f'_{io_num}' if io_num else ''
  newkey = f'{prefix}/{encdec}/{blockname}_{blocknum}/MlpBlock_0/{io_name}{io_num}/kernel{suffix}'
  return newkey, val


@t5_importer.add(
    r'(encoder|decoder)/block_(\d+)/layer_(\d+)/(?:layer|rms)_norm/scale(\w*)')
def layernorms(opts, key, val, encdec, blocknum, lyrnum, slot):
  """Layer norms."""
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  blocknum = int(blocknum)
  blockname = BLOCK_MAP[encdec]
  lyrnum = int(lyrnum)
  newkey = f'{prefix}/{encdec}/{blockname}_{blocknum}/LayerNorm_{lyrnum}/scale{suffix}'
  return newkey, val


@t5_importer.add(r'(encoder|decoder)/(?:final_layer|rms)_norm/scale(\w*)')
def final_layernorms(opts, key, val, encdec, slot):
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  norm = {'encoder': 'encoder_norm', 'decoder': 'encoderdecoder_norm'}[encdec]
  newkey = f'{prefix}/{encdec}/{norm}/scale{suffix}'
  return newkey, val


@t5_importer.add(r'decoder/logits/kernel(\w*)')
def final_logits(opts, key, val, slot):
  del opts, key
  prefix = 'state/param_states' if slot else 'target'
  suffix = '/' + SLOT_MAP[slot] if slot else ''
  newkey = f'{prefix}/decoder/logits_dense/kernel{suffix}'
  return newkey, val


def t5_post_process(t5_data):
  """Add dummy slots that Flax Adafactor requires but TF does not."""
  updates = {}
  for k in t5_data:
    if k.startswith('target'):
      state_leaf = 'state/param_states' + k[len('target'):]
      updates[state_leaf + '/m'] = np.zeros((1,), np.float32)
      if state_leaf + '/v' in t5_data:
        updates[state_leaf + '/v_row'] = np.zeros((1,), np.float32)
        updates[state_leaf + '/v_col'] = np.zeros((1,), np.float32)
      elif state_leaf + '/v_row' in t5_data:
        updates[state_leaf + '/v'] = np.zeros((1,), np.float32)
  t5_data.update(**updates)
  return t5_data


# Load checkpoint, translate, and update flax optimizer and model.
# -----------------------------------------------------------------------------
def load_tf_ckpt(path):
  """Load a TF checkpoint as a flat dictionary of numpy arrays."""
  ckpt_reader = tf.train.load_checkpoint(path)
  ckpt_shape_map = ckpt_reader.get_variable_to_shape_map()
  datamap = {k: ckpt_reader.get_tensor(k) for k in ckpt_shape_map}
  return datamap


def update_optimizer(optimizer, t5_data):
  """Update flax optimizer for T5 model."""
  optimizer_data = traverse_util.flatten_dict(optimizer.state_dict())
  optimizer_data = {'/'.join(k): v for k, v in optimizer_data.items()}
  # Shape check.
  for k, v in jax.tree_map(np.shape, t5_data).items():
    if np.shape(optimizer_data[k]) != v:
      raise ValueError(
          f'Variable {k} has shape {v} != {np.shape(optimizer_data[k])}')
  # Dtype check template optimizer against imported T5 arrays.
  for k, v in jax.tree_map(lambda x: x.dtype, t5_data).items():
    if optimizer_data[k].dtype != v:
      raise ValueError(
          f'Variable {k} has dtype {v} != {optimizer_data[k].dtype}')
  optimizer_data = t5_data
  optimizer_data = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in optimizer_data.items()})
  return optimizer.restore_state(optimizer_data)


def restore_from_t5_checkpoint(optimizer, path):
  """Load T5 checkpoint and update Adafactor optimizer and T5 model from it.

  We require that the final translated checkpoint structure exactly matches
  that of the Flax Adafactor + Transformer data, up to shape agreement of
  the leaves.

  Args:
    optimizer: Flax Adafactor Optimizer for T5 transformer encoder-decoder.
    path: Path to T5 model.

  Returns:
    Adafactor optimizer updated with parameters and optimizer state from
    T5 checkpoint.
  """
  ckpt_data = load_tf_ckpt(path)
  t5_data = t5_importer.apply(ckpt_data)
  t5_data = t5_post_process(t5_data)
  return update_optimizer(optimizer, t5_data)
