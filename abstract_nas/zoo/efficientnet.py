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

"""Efficientnet model.

This follows //learning/faster_training/sidewinder/efficientnet/efficientnet.py
"""

import dataclasses
import functools
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging

from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType

Tensor = Any


@dataclasses.dataclass
class BlockConfig:
  """Class that contains configuration parameters for a single block."""
  input_filters: int = 0
  output_filters: int = 0
  kernel_size: int = 3
  num_repeat: int = 1
  expand_ratio: int = 1
  strides: int = 1


@dataclasses.dataclass
class ModelConfig:
  """Class that contains configuration parameters for the model."""
  width_coefficient: float = 1.0
  depth_coefficient: float = 1.0
  resolution: int = 224
  dropout_rate: float = 0.2
  blocks: Tuple[BlockConfig, Ellipsis] = (
      # (input_filters, output_filters, kernel_size, num_repeat,
      #  expand_ratio, strides)
      # pylint: disable=bad-whitespace
      BlockConfig(32, 16, 3, 1, 1, 1),
      BlockConfig(16, 24, 3, 2, 6, 2),
      BlockConfig(24, 40, 5, 2, 6, 2),
      BlockConfig(40, 80, 3, 3, 6, 2),
      BlockConfig(80, 112, 5, 3, 6, 1),
      BlockConfig(112, 192, 5, 4, 6, 2),
      BlockConfig(192, 320, 3, 1, 6, 1),
      # pylint: enable=bad-whitespace
  )


def round_filters(filters,
                  config):
  """Returns rounded number of filters based on width coefficient."""
  width_coefficient = config.width_coefficient
  min_depth = divisor = 8
  orig_filters = filters

  if not width_coefficient:
    return filters

  filters *= width_coefficient
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  logging.info("round_filter input=%s output=%s", orig_filters, new_filters)
  return int(new_filters)


def round_repeats(repeats, depth_coefficient):
  """Returns rounded number of repeats based on depth coefficient."""
  return math.ceil(depth_coefficient * repeats)


DEFAULT_OP_KWARGS = {
    OpType.CONV: {
        "kernel_init": "I:variance_scaling:"
                       "scale:2.0:"
                       "mode:fan_out:"
                       "distribution:truncated_normal",
        "use_bias": False,
    },
    OpType.DENSE: {
        "kernel_init": "I:variance_scaling:"
                       "scale:0.33333:"
                       "mode:fan_out:"
                       "distribution:uniform"
    },
    OpType.BATCH_NORM: {
        "momentum": 0.99,
        "epsilon": 1e-3
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


def squeeze_excite(input_name, squeeze_factor):
  """Returns a squeeze-excite block."""
  ops = []
  append = functools.partial(append_op, ops)
  append(op_name="se/pool0",
         op_type=OpType.AVG_POOL,
         input_kwargs={"window_shape": 0},
         input_names=[input_name])
  append(op_name="se/dense1",
         op_type=OpType.DENSE,
         op_kwargs={"features": f"S:-1%{squeeze_factor}"})
  append(op_name="se/swish2",
         op_type=OpType.SWISH)
  append(op_name="se/dense3",
         op_type=OpType.DENSE,
         op_kwargs={"features": f"S:-1*{squeeze_factor}"})
  append(op_name="se/sigmoid4",
         op_type=OpType.SIGMOID)
  append(op_name="se/mul5",
         op_type=OpType.MUL,
         input_names=[input_name, ops[-1].name])
  return ops


def mb_conv_block(expand_ratio, stride, kernel_size,
                  output_filters):
  """Returns an Efficientnet MBConvBlock."""
  input_name = "input"
  ops = []
  append = functools.partial(append_op, ops)

  # Expand.
  if expand_ratio > 1:
    append(op_name="mb_conv/expand/conv0",
           op_type=OpType.CONV,
           op_kwargs={
               "features": f"S:-1*{expand_ratio}",
               "kernel_size": 1,
               "strides": 1,
           },
           input_names=[input_name])
    append(op_name="mb_conv/expand/bn1",
           op_type=OpType.BATCH_NORM)
    append(op_name="mb_conv/expand/swish2",
           op_type=OpType.SWISH)
    input_name = ops[-1].name

  # Depthwise conv.
  append(op_name="mb_conv/dw/conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": "S:-1",
             "feature_group_count": "S:-1",
             "kernel_size": kernel_size,
             "strides": stride,
         },
         input_names=[input_name])
  append(op_name="mb_conv/dw/bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="mb_conv/dw/swish2",
         op_type=OpType.SWISH)

  # Squeeze and excitation.
  input_name = ops[-1].name
  se_ops = squeeze_excite(input_name, expand_ratio * 4)
  ops.extend(se_ops)

  # Output.
  append(op_name="mb_conv/output/conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": "S:input:-1" if not output_filters else output_filters,
             "kernel_size": 1,
             "strides": 1,
         })
  append(op_name="mb_conv/output/bn1",
         op_type=OpType.BATCH_NORM)
  output_name = ops[-1].name
  graph = new_graph(input_names=["input"], output_names=[output_name],
                    ops=ops)
  return Block(
      name=f"mbconv_expand{expand_ratio}_stride{stride}_kernel{kernel_size}_"
           f"outputfilters{output_filters}_",
      graph=graph)


def mbconv_layer(
    block, input_name, block_id, output_filters,
    stride,
    layer_drop_rate):
  """Returns a MBConv layer."""

  prefix = f"mbconv{block_id}"

  block = block.instantiate(input_names=[input_name], instance_id=block_id)
  constants = block.constants
  ops = list(block.graph.ops)

  if not output_filters and stride == 1:
    append_op(ops,
              op_name=f"{prefix}/skip",
              op_type=OpType.ADD,
              input_kwargs={"layer_drop_rate": layer_drop_rate},
              input_names=[ops[-1].name, input_name])

  output_name = ops[-1].name
  graph = new_graph(input_names=[input_name], output_names=[output_name],
                    ops=ops)
  return graph, constants, block


def create_efficietnet_blocks(config):
  """Creates the default resnet architecture."""
  depth_coefficient = config.depth_coefficient
  blocks = []
  for block in config.blocks:
    assert block.num_repeat > 0
    # Update block input and output filters based on depth multiplier
    input_filters = round_filters(block.input_filters, config)
    output_filters = round_filters(block.output_filters, config)
    num_repeat = round_repeats(block.num_repeat, depth_coefficient)

    # The first block needs to take care of stride and filter size increase
    output_filters = 0 if output_filters == input_filters else output_filters
    block_def = mb_conv_block(block.expand_ratio, block.strides,
                              block.kernel_size, output_filters)
    blocks.append(block_def)
    if num_repeat > 1:
      for _ in range(num_repeat - 1):
        block_def = mb_conv_block(block.expand_ratio, 1,
                                  block.kernel_size, 0)
        blocks.append(block_def)
  return blocks


def _extract_block_info(block_name):
  m = re.fullmatch(r".+expand([0-9]+)_stride([0-9]+)_kernel([0-9]+)_"
                   r"outputfilters([0-9]+)_.*", block_name)
  logging.info(block_name)
  assert m
  expand_ratio = int(m.group(1))
  stride = int(m.group(2))
  kernel_size = int(m.group(3))
  output_filters = int(m.group(4))
  return expand_ratio, stride, kernel_size, output_filters


def efficietnet(
    num_classes,
    config,
    blocks
):
  """Returns a graph for ResNet V1."""

  drop_connect_rate = .2

  ops = []
  constants = {}
  new_blocks = []
  append = functools.partial(append_op, ops)

  stem_filters = round_filters(32, config)
  append(op_name="stem/conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": stem_filters,
             "kernel_size": 3,
             "strides": 2,
         },
         input_names=["input"])
  append(op_name="stem/bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="stem/swish2",
         op_type=OpType.SWISH)

  input_name = ops[-1].name
  block_num = 0
  num_blocks_total = len(blocks)
  for block in blocks:
    drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
    _, stride, _, output_filters = _extract_block_info(block.name)

    graph, new_constants, new_block = mbconv_layer(
        block=block, input_name=input_name, block_id=block_num,
        output_filters=output_filters, stride=stride, layer_drop_rate=drop_rate)

    input_name = graph.output_names[0]
    constants.update(new_constants)
    new_blocks.append(new_block)
    ops.extend(graph.ops)

    block_num += 1

  top_filters = round_filters(1280, config)
  append(op_name="head/conv0",
         op_type=OpType.CONV,
         op_kwargs={
             "features": top_filters,
             "kernel_size": 1,
             "strides": 1,
         })
  append(op_name="head/bn1",
         op_type=OpType.BATCH_NORM)
  append(op_name="head/swish2",
         op_type=OpType.SWISH)
  append(op_name="head/pool3",
         op_type=OpType.AVG_POOL,
         input_kwargs={"window_shape": 0})
  if config.dropout_rate and config.dropout_rate > 0:
    append(op_name="head/dropout4",
           op_type=OpType.DROPOUT,
           op_kwargs={"rate": config.dropout_rate})
  append(
      op_name="head/dense5",
      op_type=OpType.DENSE,
      op_kwargs={"features": num_classes})
  append(
      op_name="head/out",
      op_type=OpType.FLATTEN)
  graph = new_graph(input_names=["input"], output_names=["head/out"],
                    ops=ops)
  return graph, constants, new_blocks


MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    "efficientnet-b0": ModelConfig(1.0, 1.0, 224, 0.2),
    "efficientnet-b1": ModelConfig(1.0, 1.1, 240, 0.2),
    "efficientnet-b2": ModelConfig(1.1, 1.2, 260, 0.3),
    "efficientnet-b3": ModelConfig(1.2, 1.4, 300, 0.3),
    "efficientnet-b4": ModelConfig(1.4, 1.8, 380, 0.4),
    "efficientnet-b5": ModelConfig(1.6, 2.2, 456, 0.4),
    "efficientnet-b6": ModelConfig(1.8, 2.6, 528, 0.5),
    "efficientnet-b7": ModelConfig(2.0, 3.1, 600, 0.5),
    "efficientnet-b8": ModelConfig(2.2, 3.6, 672, 0.5),
    "efficientnet-l2": ModelConfig(4.3, 5.3, 800, 0.5),
}


EfficientNetB0 = functools.partial(
    efficietnet,
    config=MODEL_CONFIGS["efficientnet-b0"],
    blocks=create_efficietnet_blocks(MODEL_CONFIGS["efficientnet-b0"]))


EfficientNetB1 = functools.partial(
    efficietnet,
    config=MODEL_CONFIGS["efficientnet-b1"],
    blocks=create_efficietnet_blocks(MODEL_CONFIGS["efficientnet-b1"]))
