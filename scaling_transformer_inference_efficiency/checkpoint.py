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

"""Checkpoint loading from PaLM checkpoints.

Minimal-dependency alternative to t5x checkpoint loading. Unlike t5x checkpoint
loading, this module doesn't require a Flax module hierarchy to load into.
Instead it loads directly into the `Checkpoint` class.
"""

import copy
from functools import partial  # pylint: disable=g-importing-member
import os
from typing import Any, NewType
from typing import Optional, Union

from flax import serialization
from flax import struct
from flax import traverse_util
import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from seqio.vocabularies import SentencePieceVocabulary
from seqio.vocabularies import Vocabulary
import tensorstore

 import gfile

# A ts.Spec that has been converted to a dict using `ts.Spec.to_json`
TsSpecDict = NewType('TsSpecDict', dict)


@struct.dataclass
class HParams:
  """Hyperparameters for a PaLM model."""
  layers: int = struct.field(
      pytree_node=False,
  )
  embed: int = struct.field(
      pytree_node=False,
  )
  ff: int = struct.field(
      pytree_node=False,
  )
  heads: int = struct.field(
      pytree_node=False,
  )
  qkv: int = struct.field(
      pytree_node=False,
  )
  max_len: int = struct.field(
      pytree_node=False,
  )
  vocab: int = struct.field(
      pytree_node=False,
  )
  padded_heads: Optional[int] = struct.field(pytree_node=False, default=0)

  @property
  def q_wi_per_head(self):
    """In the fused q_wi layer, dimension size per head."""
    assert self.ff % self.heads == 0
    # fuses W and V from swiGLU, and the q computation
    # note both W and V have output dim .ff, not
    # 2/3 ff as in https://arxiv.org/pdf/2002.05202.pdf
    return (self.ff * 2 // (self.heads-self.padded_heads)) + self.qkv

  @property
  def o_wo_per_head(self):
    """In the fused o_wo layer, dimension size per head."""
    assert self.ff % self.heads == 0
    # fuse ff->e and projection layer of self-attention
    return (self.ff // (self.heads-self.padded_heads)) + self.qkv


HParams.TOY = HParams(
    layers=4,
    embed=32,
    ff=4 * 32,
    heads=16,
    qkv=2,
    max_len=8096,
    vocab=256,
)

HParams.PALM_8B = HParams(
    layers=32,
    embed=4096,
    ff=4 * 4096,
    heads=16,
    qkv=256,
    max_len=8096,
    vocab=256128,
)

HParams.PALM_62B = HParams(
    layers=64,
    embed=8192,
    ff=4 * 8192,
    heads=32,
    qkv=256,
    max_len=8096,
    vocab=256128,
)

HParams.PALM_540B = HParams(
    layers=118,
    embed=18432,
    ff=4 * 18432,
    heads=64,  # actually 48, but 16 padded
    qkv=256,
    max_len=8096,
    vocab=256128,
    padded_heads=16,
)

HParams.TURING_NLG = HParams(
    layers=105,
    embed=20480,
    ff=4 * 20480,
    heads=128,
    qkv=160,
    max_len=8096,
    vocab=51200,
)

HParams.PALM_540B_64HEADS = HParams.PALM_540B.replace(heads=64)

HParams.PALM_540B_128HEADS = HParams.PALM_540B.replace(heads=128)

HParams.PALM_540B_256HEADS = HParams.PALM_540B.replace(heads=256)


@struct.dataclass
class CheckpointSpec:
  """Identifies a particular checkpoint file and its format."""
  hparams: HParams
  dir: str
  transpose_scan_axis: bool  # True if layers not saved as the leading axis




def load_vocab():
  """Loads PaLM vocab - only works internally."""
  # vocab loaded internally at google
  unused = r.sp_model
  del unused
  return r


def tensorstore_leaf(_, value):
  """Detect if the node is a serialized tensorstore spec.

  Args:
    _: The unused name of the current item.
    value: The value of the possible leaf.

  Returns:
    True if the value represents a tensorstore spec, False otherwise.
  """
  # It is a tensorstore leaf if it at least has `driver`, `kvstore` and
  # `metadata` in its keys, sometime they have additional ones like `dtype` or
  # `transform`.
  return set(value.keys()) >= {'driver', 'kvstore', 'metadata'}


def flatten_state_dict(state_dict, keep_empty_nodes = False):
  """Flatten a dictionary until an array or tensorstore is reached.

  Args:
    state_dict: Optimizer state as nested dictionary.
    keep_empty_nodes: Whether to keep empty node, for example, empty param
      states from simple optimizers or non-touched parameter states in a
      multioptimizer.

  Returns:
    Flattened dictionary, though keeping tensor store state unflattened.
  """
  return traverse_util.flatten_dict(
      state_dict,
      is_leaf=tensorstore_leaf,
      keep_empty_nodes=keep_empty_nodes,
      sep='/')


PyTree = Any
FlatCheckpointDict = Any


def parse_checkpoint(checkpoint_path):
  """Returns the pytree from a checkpoint file path."""
  with gfile.Open(checkpoint_path, 'rb') as fp:
    raw_contents = fp.read()
    if raw_contents.startswith(b'model_checkpoint_path'):
      raise ValueError(
          'Attempting to restore a TensorFlow checkpoint as a native T5X '
          f'checkpoint. Path: {checkpoint_path}')
    return serialization.msgpack_restore(raw_contents)


def flatten_checkpoint(parsed_checkpoint,
                       keep_empty_nodes = True):
  """Flattens a PyTree till an array or TensorStore is reached."""
  return flatten_state_dict(
      parsed_checkpoint, keep_empty_nodes=keep_empty_nodes)


def load_checkpoint(checkpoint_path):
  """Actually load the checkpoint."""
  flat_checkpoint_dict = flatten_checkpoint(
      parse_checkpoint(checkpoint_path), keep_empty_nodes=True)
  return flat_checkpoint_dict


def add_path_to_spec(spec, checkpoint_directory):
  """Adds the given path to the spec and returns a deepcopy of the spec."""
  # From pathways.tensorstore_utils
  spec = copy.deepcopy(spec)
  spec['kvstore']['path'] = os.path.join(checkpoint_directory,
                                         spec['kvstore']['path'])
  return spec


def var(spec, checkpoint_dict, key,
        transpose):
  """Loads a variable stored in a tensorstore checkpoint."""
  if key not in checkpoint_dict:
    raise ValueError(f'could not find {key} in checkpoint. '
                     f'Available keys: {checkpoint_dict.keys()}')
  val = checkpoint_dict[key]
  if isinstance(val, np.ndarray):
    ts = tensorstore.array(val).spec()
  else:
    spec_dict = add_path_to_spec(val, spec.dir)
    # Rank is a required property for transpose operations downstream
    # and not included in some PaLM checkpoints
    if 'rank' not in spec_dict:
      spec_dict['rank'] = len(spec_dict['metadata']['shape'])
    ts = tensorstore.Spec(spec_dict)

  if transpose and spec.transpose_scan_axis:
    ts = ts[tensorstore.d[0].transpose[1]]
  return ts


def check_shape(ts, expected):
  if ts.shape != expected.shape:
    raise ValueError(
        f'Unexpected shape in checkpoint: {ts.shape} vs {expected.shape}')


def load_one(ts, sh):
  v = tensorstore.open(ts, read=True).result()
  v = v.read().result()
  v = v.astype(sh.dtype, copy=False)
  return v


@struct.dataclass
class Checkpoint:
  """The contents of a checkpoint, without any postprocessing.

  Typically this is stored in host DRAM, or produced lazily as a
  tensorstore.Spec or a core.ShapedArray.
  """
  q_wi: Union[np.ndarray, core.ShapedArray]
  kv: Union[np.ndarray, core.ShapedArray]
  o_wo: Union[np.ndarray, core.ShapedArray]
  layernorm_scale: Union[np.ndarray, core.ShapedArray]
  embedding: Union[np.ndarray, core.ShapedArray]

  @classmethod
  def make_shaped_arrays(cls, h):
    """Creates a Checkpoint populated by zero-footprint core.ShapedArray."""
    return Checkpoint(
        q_wi=core.ShapedArray(
            (h.layers, h.embed, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.bfloat16),
        kv=core.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.bfloat16),
        o_wo=core.ShapedArray(
            (h.layers, h.heads - h.padded_heads, h.o_wo_per_head, h.embed),
            jnp.bfloat16),
        layernorm_scale=core.ShapedArray((h.layers, h.embed), jnp.float32),
        embedding=core.ShapedArray((h.vocab, h.embed), jnp.bfloat16),
    )

  @classmethod
  def load_spec(cls, spec):
    """Loads checkpoint metadata, returning tensorstore Spec for tensors."""
    checkpoint_dict = load_checkpoint(os.path.join(spec.dir, 'checkpoint'))

    load_var = partial(var, spec, checkpoint_dict)

    result = Checkpoint(
        q_wi=load_var(
            'optimizer/target/decoder/decoder/q_wi_fused/kernel',
            transpose=True),
        kv=load_var(
            'optimizer/target/decoder/decoder/kv_fused/kernel', transpose=True),
        o_wo=load_var(
            'optimizer/target/decoder/decoder/o_wo_fused/kernel',
            transpose=True),
        layernorm_scale=load_var(
            'optimizer/target/decoder/decoder/layer_norm/scale',
            transpose=True),
        embedding=load_var(
            'optimizer/target/decoder/token_embedder/embedding',
            transpose=False),
    )

    jax.tree_util.tree_map(
        check_shape,
        result,
        Checkpoint.make_shaped_arrays(spec.hparams),
        is_leaf=lambda v: isinstance(v, tensorstore.Spec))
    return result

  @classmethod
  def load_unsharded_to_host(cls, spec):
    """Loads checkpoint data into host memory, unsharded."""
    ts = Checkpoint.load_spec(spec)
    shapes = Checkpoint.make_shaped_arrays(spec.hparams)
    return jax.tree_util.tree_map(
        load_one, ts, shapes, is_leaf=lambda v: isinstance(v, tensorstore.Spec))

  @classmethod
  def init_zero(cls, h):
    """Initializes a checkpoint with all zeros."""
    shapes = Checkpoint.make_shaped_arrays(h)
    return jax.tree_util.tree_map(lambda s: np.zeros(s.shape, s.dtype), shapes)


@struct.dataclass
class QuantizedCheckpoint:
  """The contents of a checkpoint, without any postprocessing.

  Typically this is stored in host DRAM, or produced lazily as a
  tensorstore.Spec or a core.ShapedArray.

  Separate quantized and non-quantized checkpoints to reduce code branching.
  """
  q_wi: Union[np.ndarray, core.ShapedArray]
  q_wi_scale: Union[np.ndarray, core.ShapedArray]
  kv: Union[np.ndarray, core.ShapedArray]
  kv_scale: Union[np.ndarray, core.ShapedArray]
  o_wo: Union[np.ndarray, core.ShapedArray]
  o_wo_scale: Union[np.ndarray, core.ShapedArray]
  layernorm_scale: Union[np.ndarray, core.ShapedArray]
  embedding: Union[np.ndarray, core.ShapedArray]

  @classmethod
  def make_shaped_arrays(cls, h):
    """Creates a Checkpoint populated by zero-footprint core.ShapedArray."""
    return QuantizedCheckpoint(
        q_wi=core.ShapedArray(
            (h.layers, h.embed, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.int8),
        q_wi_scale=core.ShapedArray(
            (h.layers, 1, h.heads - h.padded_heads, h.q_wi_per_head),
            jnp.bfloat16),
        kv=core.ShapedArray((h.layers, h.embed, 1, 2 * h.qkv), jnp.int8),
        kv_scale=core.ShapedArray((h.layers, 1, 1, 2 * h.qkv), jnp.bfloat16),
        o_wo=core.ShapedArray(
            (h.layers, h.heads - h.padded_heads, h.o_wo_per_head, h.embed),
            jnp.int8),
        o_wo_scale=core.ShapedArray((h.layers, 1, 1, h.embed), jnp.bfloat16),
        layernorm_scale=core.ShapedArray((h.layers, h.embed), jnp.float32),
        embedding=core.ShapedArray((h.vocab, h.embed), jnp.bfloat16),
    )

  @classmethod
  def load_spec(cls, spec):
    """Loads checkpoint metadata, returning tensorstore Spec for tensors."""
    checkpoint_dict = load_checkpoint(os.path.join(spec.dir, 'checkpoint'))

    load_var = partial(var, spec, checkpoint_dict)

    result = QuantizedCheckpoint(
        q_wi=load_var(
            'optimizer/target/decoder/decoder/q_wi_fused/qkernel',
            transpose=spec.transpose_scan_axis),
        q_wi_scale=load_var(
            'optimizer/target/decoder/decoder/q_wi_fused/qscale',
            transpose=spec.transpose_scan_axis,
        ),
        kv=load_var(
            'optimizer/target/decoder/decoder/kv_fused/qkernel',
            transpose=spec.transpose_scan_axis),
        kv_scale=load_var(
            'optimizer/target/decoder/decoder/kv_fused/qscale',
            transpose=spec.transpose_scan_axis),
        o_wo=load_var(
            'optimizer/target/decoder/decoder/o_wo_fused/qkernel',
            transpose=spec.transpose_scan_axis),
        o_wo_scale=load_var(
            'optimizer/target/decoder/decoder/o_wo_fused/qscale',
            transpose=spec.transpose_scan_axis),
        layernorm_scale=load_var(
            'optimizer/target/decoder/decoder/layer_norm/scale',
            transpose=spec.transpose_scan_axis),
        embedding=load_var(
            'optimizer/target/decoder/token_embedder/embedding',
            transpose=False),
    )

    jax.tree_util.tree_map(
        check_shape,
        result,
        QuantizedCheckpoint.make_shaped_arrays(spec.hparams),
        is_leaf=lambda v: isinstance(v, tensorstore.Spec))
    return result

  @classmethod
  def load_unsharded_to_host(cls,
                             spec):
    """Loads checkpoint data into host memory, unsharded."""
    ts = QuantizedCheckpoint.load_spec(spec)
    shapes = QuantizedCheckpoint.make_shaped_arrays(spec.hparams)
    return jax.tree_util.tree_map(
        load_one, ts, shapes, is_leaf=lambda v: isinstance(v, tensorstore.Spec))

  @classmethod
  def init_zero(cls, h):
    """Initializes a checkpoint with all zeros."""
    shapes = QuantizedCheckpoint.make_shaped_arrays(h)
    return jax.tree_util.tree_map(lambda s: np.zeros(s.shape, s.dtype), shapes)
