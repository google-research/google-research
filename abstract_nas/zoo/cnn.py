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

"""Small CNN for MNIST.

A small 2-layer CNN achieving 99% on MNIST.
"""

import functools
from typing import Any, Dict, Optional, Sequence, Tuple

from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import OpType

Tensor = Any


def conv_block():
  """Makes a conv block parameterized by the number of features."""
  ops = [
      new_op(
          op_name="conv",
          op_type=OpType.CONV,
          op_kwargs={
              "features": "S:-1*2",
              "kernel_size": 3
          },
          input_names=["input"]),
      new_op(
          op_name="relu",
          op_type=OpType.RELU,
          input_names=["conv"]),
      new_op(
          op_name="avg_pool",
          op_type=OpType.AVG_POOL,
          input_names=["relu"],
          input_kwargs={
              "window_shape": 2,
              "strides": 2
          }),
  ]

  graph = new_graph(input_names=["input"], output_names=["avg_pool"], ops=ops)
  return Block(name="conv_layer", graph=graph)

BLOCK_TYPES = [conv_block, conv_block]
BLOCK_NAMES = ["conv_layer", "conv_layer"]


def conv_net(
    in_features,
    out_features,
    num_classes,
    blocks = None
):
  """Graph for 3-layer CNN."""
  if not blocks:
    blocks = [block_type() for block_type in BLOCK_TYPES]

  input_name = "input"
  new_blocks = []
  ops = [
      new_op(
          op_name="proj",
          op_type=OpType.CONV,
          op_kwargs={
              "features": in_features,
              "kernel_size": 1,
          },
          input_names=[input_name])
  ]
  constants = {}

  block_input_name = ops[-1].name
  for idx, block in enumerate(blocks):
    block = block.instantiate(input_names=[block_input_name],
                              instance_id=idx)
    new_blocks.append(block)
    constants.update(block.constants)
    ops.extend(block.graph.ops)
    block_input_name = ops[-1].name

  constants.update({"out_features": out_features, "num_classes": num_classes})
  ops.extend([
      new_op(
          op_name="flatten",
          op_type=OpType.FLATTEN,
          input_names=[ops[-1].name]),
      new_op(
          op_name="fc/dense",
          op_type=OpType.DENSE,
          op_kwargs={"features": "K:out_features"},
          input_names=["flatten"]),
      new_op(op_name="fc/relu", op_type=OpType.RELU, input_names=["fc/dense"]),
      new_op(
          op_name="fc/logits",
          op_type=OpType.DENSE,
          op_kwargs={"features": "K:num_classes"},
          input_names=["fc/relu"])
  ])
  graph = new_graph(
      input_names=[input_name], output_names=["fc/logits"], ops=ops)
  return graph, constants, new_blocks


CifarNet = functools.partial(
    conv_net, in_features=32, out_features=256, num_classes=10)

