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

"""ResNet V1.

This follows flax/examples/imagenet/models.py @ 44ee6f2.
"""

import functools
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType

Tensor = Any

DEFAULT_OP_KWARGS = {
    OpType.CONV: {"use_bias": False},
    OpType.BATCH_NORM: {
        "momentum": 0.9,
        "epsilon": 1e-5
    }
}


def append_op(ops,  # pylint: disable=dangerous-default-value
              op_name,
              op_type,
              input_names = None,
              input_kwargs = None,
              op_kwargs = {},
              num_outputs = 1):
  """Convenience function for append to a sequence of ops."""
  if not input_names:
    input_names = [ops[-1].name]
  default_op_kwargs = DEFAULT_OP_KWARGS.get(op_type, {})
  ops.append(
      new_op(op_name=op_name,
             op_type=op_type,
             input_names=input_names,
             input_kwargs=input_kwargs,
             op_kwargs={**default_op_kwargs, **op_kwargs},
             num_outputs=num_outputs))


def basic_block(stride, filter_mul):
  """Returns a ResNetv1 basic block."""
  input_name = "input"
  ops = []
  append = functools.partial(append_op, ops)

  out_filters = "S:-1"
  if filter_mul > 1:
    out_filters = f"{out_filters}*{filter_mul}"

  append(op_name="conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": out_filters,
             "kernel_size": 3,
             "strides": stride
         },
         input_names=[input_name])
  append(op_name="bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="relu2",
         op_type=OpType.RELU)
  append(op_name="conv3",
         op_type=OpType.CONV,
         op_kwargs={
             "features": "S:-1",
             "kernel_size": 3
         })
  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return Block(name=f"resnet_stride{stride}_filtermul{filter_mul}_basic",
               graph=graph)


def bottleneck_block(stride, filter_mul):
  """Returns a ResNetv1 bottleneck block."""
  input_name = "input"
  ops = []
  append = functools.partial(append_op, ops)

  in_filters = "S:-1"
  if 4 > filter_mul:
    assert 4 % filter_mul == 0
    bottleneck_factor = 4 // filter_mul
    bottleneck_filters = f"{in_filters}%{bottleneck_factor}"
  elif filter_mul > 4:
    assert filter_mul % 4 == 0
    bottleneck_factor = filter_mul // 4
    bottleneck_filters = f"{in_filters}*{bottleneck_factor}"
  else:
    bottleneck_filters = in_filters

  append(op_name="conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": bottleneck_filters,
             "kernel_size": 1,
         },
         input_names=[input_name])
  append(op_name="bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="relu2",
         op_type=OpType.RELU)
  append(op_name="conv3",
         op_type=OpType.CONV,
         op_kwargs={
             "features": "S:-1",
             "kernel_size": 3,
             "strides": stride
         })
  append(op_name="bn4",
         op_type=OpType.BATCH_NORM)
  append(op_name="relu5",
         op_type=OpType.RELU)
  append(op_name="conv6",
         op_type=OpType.CONV,
         op_kwargs={
             "features": "S:-1*4",
             "kernel_size": 1,
         })
  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return Block(
      name=f"resnet_stride{stride}_filtermul{filter_mul}_bottleneck",
      graph=graph)


def residual_connection(
    skip_name,
    input_name,
    prefix,
    filter_mul,
    strides = 1):
  """Returns a sequence of ops making a residual connection.

  skip_name refers to the original output (e.g., x), and input_name refers to
  the layer output (e.g., f(x)). Then this computes
    maybe_proj(x) + f(x)
  where maybe_proj is a projection that is added only when necessary.

  Args:
    skip_name: the name of the original output
    input_name: the name of the layer output
    prefix: a prefix for unique names in this residual connection
    filter_mul: the number of filters in input_name divided by the number of
      filters in skip_name
    strides: the striding from skip_name to input_name
  """

  ops = []
  append = functools.partial(append_op, ops)

  append(op_name=f"{prefix}/bn",
         op_type=OpType.BATCH_NORM,
         input_names=[input_name],
         op_kwargs={"scale_init": "I:zeros"})
  input_name = ops[-1].name

  if filter_mul > 1 or strides != 1:
    out_filters = "S:-1"
    if filter_mul > 1:
      out_filters = f"{out_filters}*{filter_mul}"
    append(op_name=f"{prefix}/proj/conv0",
           op_type=OpType.CONV,
           op_kwargs={
               "features": out_filters,
               "kernel_size": 1,
               "strides": strides,
           },
           input_names=[skip_name])
    append(op_name=f"{prefix}/proj/bn1",
           op_type=OpType.BATCH_NORM)
    skip_name = ops[-1].name

  append(op_name=f"{prefix}/skip/add0",
         op_type=OpType.ADD,
         input_names=[input_name, skip_name])
  append(op_name=f"{prefix}/skip/relu1",
         op_type=OpType.RELU)

  return ops, ops[-1].name


def resnet_layer(
    block,
    input_name,
    block_id,
    layer_id,
    filter_mul,
    strides = 1):
  """Returns a ResNetv1 layer."""

  instance_id = int(f"{block_id}{layer_id}")
  prefix = f"resnet{instance_id}"

  block = block.instantiate(input_names=[input_name], instance_id=instance_id)
  constants = block.constants

  residual_ops, output_name = residual_connection(
      input_name, block.graph.output_names[0], prefix, filter_mul, strides)

  ops = list(block.graph.ops) + list(residual_ops)
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return graph, constants, block


BLOCK_FNS = {
    "bottleneck": bottleneck_block,
    "basic": basic_block,
}

INIT_CONV_OP_KWARGS = {
    # 32x32 resolution (cifar-10)
    "small": {
        "kernel_size": 3,
        "strides": 1,
        "padding": "SAME",  # [(1, 1), (1, 1)]
    },

    # 224x224 resolution (imagenet)
    "large": {
        "kernel_size": 7,
        "strides": 2,
        "padding": "SAME",  # [(3, 3), (3, 3)]
    }
}


def create_resnet_blocks(block_type,
                         stage_sizes):
  """Creates the default resnet architecture."""
  block_fn = BLOCK_FNS[block_type]
  in_filters = 64
  num_filters = 64  # dummy value
  blocks = []
  for i, block_size in enumerate(stage_sizes):
    out_filters = num_filters * 2 ** i
    if block_type == "bottleneck":
      out_filters = out_filters * 4
    for j in range(block_size):
      strides = 2 if i > 0 and j == 0 else 1
      assert out_filters >= in_filters
      assert out_filters % in_filters == 0
      filter_mul = out_filters // in_filters
      block = block_fn(strides, filter_mul)
      blocks.append(block)
      in_filters = out_filters
  return blocks


def _extract_block_info(block_name):
  m = re.match(r".+_stride([0-9]+)_filtermul([0-9]+)_(.+)", block_name)
  assert m
  strides = int(m.group(1))
  filter_mul = int(m.group(2))
  block_type = m.group(3)
  return block_type, strides, filter_mul


def resnet(
    num_classes,
    input_resolution,
    num_filters,
    blocks
):
  """Returns a graph for ResNet V1."""

  ops = []
  constants = {}
  new_blocks = []
  append = functools.partial(append_op, ops)

  append(op_name="init/conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": num_filters,
             **INIT_CONV_OP_KWARGS[input_resolution],
         },
         input_names=["input"])
  append(op_name="init/bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="init/relu2",
         op_type=OpType.RELU)
  append(op_name="init/maxpool3",
         op_type=OpType.MAX_POOL,
         input_kwargs={
             "window_shape": 3,
             "strides": 2,
             "padding": "SAME",
         })
  # vvv !!!THIS LAYER IS NON STANDARD!!! vvv
  append(op_name="proj4",
         op_type=OpType.CONV,
         op_kwargs={
             "features": num_filters,
             "kernel_size": 1,
         })
  # ^^^ !!!THIS LAYER IS NON STANDARD!!! ^^^

  input_name = ops[-1].name
  block_id = 1
  layer_id = 0
  for block in blocks:
    _, strides, filter_mul = _extract_block_info(block.name)

    if strides == 2 and layer_id:
      block_id += 1
      layer_id = 1
    else:
      layer_id += 1

    graph, new_constants, new_block = resnet_layer(
        block=block, input_name=input_name, block_id=block_id,
        layer_id=layer_id, filter_mul=filter_mul,
        strides=strides)

    input_name = graph.output_names[0]
    constants.update(new_constants)
    new_blocks.append(new_block)
    ops.extend(graph.ops)

  append(
      op_name="fc/mean", op_type=OpType.MEAN, input_kwargs={
          "axis": (1, 2),
      })
  append(
      op_name="fc/dense",
      op_type=OpType.DENSE,
      op_kwargs={
          "features": num_classes,
          "kernel_init": "I:zeros",
      })
  graph = new_graph(input_names=["input"], output_names=["fc/dense"], ops=ops)
  return graph, constants, new_blocks


ResNet18 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[2, 2, 2, 2], block_type="basic"))
ResNet34 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[3, 4, 6, 3], block_type="basic"))
ResNet50 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[3, 4, 6, 3], block_type="bottleneck"))
ResNet101 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[3, 4, 23, 3], block_type="bottleneck"))
ResNet152 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[3, 8, 36, 3], block_type="bottleneck"))
ResNet200 = functools.partial(
    resnet,
    num_filters=64,
    blocks=create_resnet_blocks(
        stage_sizes=[3, 24, 36, 3], block_type="bottleneck"))
