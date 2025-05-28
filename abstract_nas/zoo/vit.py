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

"""Vision Transformer implementation.

Two implementations of vision transformers:

spatial=False is the standard implementation, where the patches are flattened
  before the position embedding is added, and the encoder receives the 1D
  sequence of patches.

spatial=True is an implementation which retains the spatial information. The
  patches are not flattened, so that the position embedding is 2D and the
  encoder attends over the 2D grid of values. The output of the encoder is
  flattened before the final Dense layer for the logits.

These implementations are equivalent, but yield different representations for
seeding a NAS search.

Parameter names are chosen to match those found in
//learning/brain/research/dune/experimental/big_vision/models/vit.py
"""

import functools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType

Tensor = Any


def append_op(ops,
              op_name,
              op_type,
              input_names = None,
              input_kwargs = None,
              op_kwargs = None,
              num_outputs = 1):
  """Convenience function for append to a sequence of ops."""
  if input_names is None:
    input_names = [ops[-1].name]
  ops.append(
      new_op(op_name=op_name,
             op_type=op_type,
             input_names=input_names,
             input_kwargs=input_kwargs,
             op_kwargs=op_kwargs if op_kwargs else {},
             num_outputs=num_outputs))


def mlp_block(dropout, mlp_factor):
  """MLP block in the encoder block."""
  ops = []
  append = functools.partial(append_op, ops)
  use_dropout = dropout > 1e-3

  input_name = "input"
  append(op_name="dense0",
         op_type=OpType.DENSE,
         op_kwargs={
             "features": f"S:-1*{mlp_factor}",
             "kernel_init": "I:xavier_uniform",
             "bias_init": "I:normal:stddev:1e-6"
         },
         input_names=[input_name])

  append(op_name="gelu1",
         op_type=OpType.GELU)
  if use_dropout:
    append(op_name="dropout",
           op_type=OpType.DROPOUT,
           op_kwargs={"rate": dropout})
  append(op_name="dense2",
         op_type=OpType.DENSE,
         op_kwargs={
             "features": f"S:-1%{mlp_factor}",
             "kernel_init": "I:xavier_uniform",
             "bias_init": "I:normal:stddev:1e-6"
         })
  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return Block(name=f"mlp_block{'_dropout' if use_dropout else ''}",
               graph=graph, constants={})


def mhdpa_block(spatial):
  """Multi headed dot product (self) attention block in encoder block."""
  input_name = "input"
  constants = {"num_heads": None, "head_dim": None}
  width = "S:input:-1"
  num_heads = "K:num_heads"
  head_dim = "K:head_dim"

  ops = []
  append = functools.partial(append_op, ops)

  append(
      op_name="value/pre",
      op_type=OpType.DENSE,
      op_kwargs={
          "features": "S:-1",
          "kernel_init": "I:xavier_uniform",
          "bias_init": "I:zeros"
      },
      input_names=[input_name])
  append(
      op_name="key/pre",
      op_type=OpType.DENSE,
      op_kwargs={
          "features": "S:-1",
          "kernel_init": "I:xavier_uniform",
          "bias_init": "I:zeros"
      },
      input_names=[input_name])
  append(
      op_name="query/pre",
      op_type=OpType.DENSE,
      op_kwargs={
          "features": "S:-1",
          "kernel_init": "I:xavier_uniform",
          "bias_init": "I:zeros"
      },
      input_names=[input_name])

  # spatial:  [b, h, w, width] -> [b, h*w, num_heads, head_dim]
  # original: [b, h*w, width]  -> [b, h*w, num_heads, head_dim]
  new_shape = ["B", -1, num_heads, head_dim]
  append(
      op_name="query",
      op_type=OpType.RESHAPE,
      input_kwargs={"new_shape": new_shape},
      input_names=["query/pre"])
  append(
      op_name="key",
      op_type=OpType.RESHAPE,
      input_kwargs={"new_shape": new_shape},
      input_names=["key/pre"])
  append(
      op_name="value",
      op_type=OpType.RESHAPE,
      input_kwargs={"new_shape": new_shape},
      input_names=["value/pre"])

  append(op_name="query/scale",
         op_type=OpType.SCALAR_MUL,
         input_names=["query"])

  # attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
  append(
      op_name="attn_weight",
      op_type=OpType.EINSUM,
      input_kwargs={"sum": "...qhd,...khd->...hqk"},
      input_names=["query/scale", "key"])

  append(
      op_name="attn_weight/softmax",
      op_type=OpType.SOFTMAX,
      input_kwargs={"axis": -1})

  # attn_values = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
  append(
      op_name="attn_value",
      op_type=OpType.EINSUM,
      input_kwargs={"sum": "...hqk,...khd->...qhd"},
      input_names=["attn_weight/softmax", "value"])

  # back to the original inputs dimensions
  if spatial:
    # [b, h*w, num_heads, head_dim] -> [b, h, w, width]
    new_shape = ["B", "S:input:1", "S:input:2", width]
  else:
    # [b, h*w, num_heads, head_dim] -> [b, h*w, width]
    new_shape = ["B", -1, width]

  append(
      op_name="attn_value/reshape",
      op_type=OpType.RESHAPE,
      input_kwargs={"new_shape": new_shape})
  append(
      op_name="out",
      op_type=OpType.DENSE,
      op_kwargs={
          "features": "S:-1",
          "kernel_init": "I:xavier_uniform",
          "bias_init": "I:zeros",
      })

  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return Block(name=f"mhdpa_block{'_spatial' if spatial else ''}",
               graph=graph, constants=constants)


def encoder_block(
    input_name,
    block_id,
    block,
    dropout,
):
  """Returns an encoder block."""

  prefix = f"encoder{block_id}"

  ops = []
  append = functools.partial(append_op, ops)
  use_dropout = dropout > 1e-3

  res_input = input_name

  append(op_name=f"{prefix}/layernorm0",
         op_type=OpType.LAYER_NORM,
         input_names=[input_name])

  block = block.instantiate(
      input_names=[ops[-1].name],
      instance_id=block_id)
  ops.extend(block.graph.ops)
  constants = block.constants

  if use_dropout:
    append(op_name=f"{prefix}/dropout1",
           op_type=OpType.DROPOUT,
           op_kwargs={"rate": dropout})
  append(op_name=f"{prefix}/residual1",
         op_type=OpType.ADD,
         input_names=[res_input, ops[-1].name])

  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return graph, constants, block


def encoder(
    input_name,
    blocks,
    dropout,
):
  """Encoder for ViT."""
  ops = []
  constants = {}
  new_blocks = []
  append = functools.partial(append_op, ops)

  encoder_input_name = input_name
  for block_id, block in enumerate(blocks):
    graph, new_constants, new_block = encoder_block(
        input_name=encoder_input_name,
        block_id=block_id,
        block=block,
        dropout=dropout)
    encoder_input_name = graph.output_names[0]
    constants.update(new_constants)
    new_blocks.append(new_block)
    ops.extend(graph.ops)
  append(op_name="transformer/encoder_norm",
         op_type=OpType.LAYER_NORM)
  output_name = ops[-1].name
  graph = new_graph(
      input_names=[input_name], output_names=[output_name], ops=ops)
  return graph, constants, new_blocks


def vit(
    blocks,
    patch_size,
    image_size,
    width,
    dropout,
    num_classes,
    spatial,
):
  """Graph for ViT."""
  assert image_size % patch_size == 0
  ops = []
  append = functools.partial(append_op, ops)
  use_dropout = dropout > 1e-3

  append(op_name="embedding",
         op_type=OpType.CONV,
         op_kwargs={
             "features": width,
             "kernel_size": [patch_size, patch_size],
             "strides": [patch_size, patch_size],
             "padding": "VALID",
         },
         input_names=["input"])

  if spatial:
    # could also be [1, sequence_size, sequence_size, width]
    pos_embedding_shape = [
        1, f"S:{ops[-1].name}:1", f"S:{ops[-1].name}:2", f"S:{ops[-1].name}:3"
    ]
  else:
    append(
        op_name="reshape",
        op_type=OpType.RESHAPE,
        input_kwargs={"new_shape": ["B", -1, "S:-1"]})

    # could also be [1, sequence_size**2, width]
    pos_embedding_shape = [1, f"S:{ops[-1].name}:1", f"S:{ops[-1].name}:2"]

  append(op_name="transformer/pos_embedding",
         op_type=OpType.PARAM,
         input_kwargs={
             "shape": pos_embedding_shape,
             "init_fn": f"I:normal:stddev:{1/math.sqrt(width):.03f}"
         },
         input_names=[])
  append(op_name="transformer/pos_embedding/add",
         op_type=OpType.ADD,
         input_names=[ops[-2].name, ops[-1].name])
  if use_dropout:
    append(op_name="transformer/dropout",
           op_type=OpType.DROPOUT,
           op_kwargs={"rate": dropout})

  graph, constants, blocks = encoder(
      input_name=ops[-1].name,
      blocks=blocks,
      dropout=dropout)
  ops.extend(graph.ops)

  if spatial:
    append(op_name="reshape",
           op_type=OpType.RESHAPE,
           input_kwargs={"new_shape": ["B", -1, "S:-1"]})

  append(op_name="gap",
         op_type=OpType.MEAN,
         input_kwargs={"axis": 1})
  append(op_name="head",
         op_type=OpType.DENSE,
         op_kwargs={
             "features": num_classes,
             "kernel_init": "I:zeros"
         })

  graph = new_graph(input_names=["input"], output_names=["head"], ops=ops)
  return graph, constants, blocks


def base_vit(
    patch_size,
    width,
    depth,
    mlp_dim,
    num_heads,
    image_size = 224,
    num_classes = 1000,
    dropout = 0.0,
    spatial = False,
    blocks = None
):
  """Base graph for ViT."""

  if blocks is None:
    assert width % num_heads == 0
    head_dim = width // num_heads

    assert mlp_dim % width == 0
    mlp_factor = mlp_dim // width

    mhdpa_def = mhdpa_block(spatial)
    mlp_def = mlp_block(dropout, mlp_factor)
    blocks = []
    for layer in range(depth):
      mhdpa_constants = {
          "num_heads": num_heads,
          "head_dim": head_dim
      }
      mhdpa_layer = mhdpa_def.instantiate(
          input_names=["input"],
          instance_id=layer,
          constants=mhdpa_constants)
      blocks.append(mhdpa_layer)

      mlp_layer = mlp_def.instantiate(
          input_names=["input"],
          instance_id=layer)
      blocks.append(mlp_layer)

  return vit(blocks, patch_size, image_size,
             width, dropout, num_classes, spatial)


VIT_PARAMS = {
    "Ti/16": {
        "width": 192,      # output feature dimension
        "depth": 12,       # number of (mhdpa, mlp) layers
        "mlp_dim": 768,    # intermediate feature dimension of mlp
        "num_heads": 3,    # number of attention heads
                           # attention head dimension = width // num_heads
        "patch_size": 16,  # patch size for extraction
    },
    "S/16": {
        "width": 384,
        "depth": 12,
        "mlp_dim": 1536,
        "num_heads": 6,
        "patch_size": 16,
    },
    "B/16": {
        "width": 768,
        "depth": 12,
        "mlp_dim": 3072,
        "num_heads": 12,
        "patch_size": 16,
    },
    "L/16": {
        "width": 1024,
        "depth": 24,
        "mlp_dim": 4096,
        "num_heads": 16,
        "patch_size": 16,
    }
}

ViT_Ti16 = functools.partial(
    base_vit, **VIT_PARAMS["Ti/16"])
ViT_S16 = functools.partial(
    base_vit, **VIT_PARAMS["S/16"])
ViT_B16 = functools.partial(
    base_vit, **VIT_PARAMS["B/16"])
ViT_L16 = functools.partial(
    base_vit, **VIT_PARAMS["L/16"])
