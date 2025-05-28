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

"""Abstract class for a sequential subgraph synthesizer.

Sequential synthesizers only synthesize subgraphs which have a single input and
output, and also do not branch (i.e., for each op, exactly one output tensor is
consumed immediately by the following op, and any additional outputs are then
discarded).
"""

import copy
import math
from typing import Optional, Sequence, Tuple

from absl import logging

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.abstract.shape import ShapeProperty
from abstract_nas.model import subgraph
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.utils import split_div_mul
from abstract_nas.synthesis.base import AbstractSynthesizer


class AbstractSequentialSynthesizer(AbstractSynthesizer):
  """Base class for sequential synthesizer."""

  def __init__(self,
               subgraphs_and_props,
               generation,
               abstract = True):
    super().__init__(subgraphs_and_props, generation, abstract)

    input_name = None
    output_name = None
    num_ops = None
    for subg, _ in self.subgraphs_and_props:
      if num_ops and num_ops != len(subg.subgraph):
        raise ValueError("Subgraphs have different length.")
      num_ops = len(subg.subgraph)

      input_names = subg.input_names
      if len(input_names) != 1:
        raise ValueError("Sequential synthesizer subgraph must have exactly "
                         f"one input, but found {len(input_names)}.")
      if not input_name:
        input_name = input_names[0]
      else:
        if input_name != input_names[0]:
          raise ValueError("Subgraphs are not identical.")

      output_names = subg.output_names
      if len(output_names) != 1:
        raise ValueError("Sequential synthesizer subgraph must have exactly "
                         f"one output, but found {len(output_names)}.")
      if not output_name:
        output_name = output_names[0]
      elif output_name != output_names[0]:
        raise ValueError("Subgraphs are not identical.")

    self.output_name = output_name
    self.input_name = input_name
    self.num_ops = num_ops

    self.output_features_mul = None
    self.output_features_div = None
    self.output_features_const = None
    self.init_output_features()

  def init_output_features(self):
    output_features_mul = None
    output_features_div = None
    output_features_const = None
    for _, props in self.subgraphs_and_props:
      cur_output_features_mul = None
      cur_output_features_div = None
      cur_output_features_const = None
      for prop in props:
        # Can't use isinstance because LinopProperty subclasses ShapeProperty.
        if type(prop) is not ShapeProperty:  # pylint: disable=unidiomatic-typecheck
          continue
        logging.info("Inferring...")
        prop: ShapeProperty
        input_shapes = prop.input_shapes
        output_shapes = prop.output_shapes
        assert len(input_shapes) == 1
        if not output_shapes:
          logging.info("Shape property mutated.")
          continue
        assert len(output_shapes) == 1
        input_shape = input_shapes[self.input_name]
        output_shape = output_shapes[self.output_name]
        input_features = input_shape[-1]
        output_features = output_shape[-1]
        logging.info("Input to output: %d %d", input_features, output_features)

        indivisible = False
        if output_features > input_features:
          indivisible = output_features % input_features != 0
          cur_output_features_mul = output_features // input_features
        elif output_features < input_features:
          indivisible = input_features % output_features != 0
          cur_output_features_div = input_features // output_features
        else:
          cur_output_features_mul = 1

        if indivisible:
          assert not output_features_div
          assert not output_features_mul
          # If there is only one subgraph, then we can just synthesize a
          # concrete value for the output features.
          if len(self.subgraphs_and_props) == 1:
            cur_output_features_const = output_features
            if output_features_const:
              assert output_features_const == cur_output_features_const
            output_features_const = cur_output_features_const
          else:
            raise ValueError(f"Output features {output_features} "
                             "are not an integer multiple or divisor of "
                             f"input features {input_features}")
        else:
          assert not output_features_const
          if cur_output_features_div:
            if (output_features_div and
                cur_output_features_div != output_features_div):
              raise ValueError(f"Inferred different output_features_div. "
                               f"Previous: {cur_output_features_div}. "
                               f"Current: {output_features_div}")
            output_features_div = cur_output_features_div
          if cur_output_features_mul:
            if output_features_mul and cur_output_features_mul != output_features_mul:
              raise ValueError(f"Inferred different output_features_mul. "
                               f"Previous: {cur_output_features_mul}. "
                               f"Current: {output_features_mul}")
            output_features_mul = cur_output_features_mul

    if output_features_div:
      self.output_features_mul = 1
      self.output_features_div = output_features_div
    elif output_features_mul:
      assert not output_features_div
      self.output_features_mul = output_features_mul
      self.output_features_div = 1
    elif output_features_const:
      assert not output_features_div
      assert not output_features_mul
      self.output_features_const = output_features_const
    logging.info("Inferred output features: %s.", self.get_output_features([]))

  def get_output_features(
      self,
      subg):
    if self.output_features_const:
      return str(self.output_features_const)

    if not self.output_features_mul:
      return

    total_mul = 1
    total_div = 1
    for op in subg:
      if op.type == OpType.DENSE or op.type == OpType.CONV:
        features = op.op_kwargs["features"]
        v, div, mul = split_div_mul(features)
        if not isinstance(v, str) or v != "S:0:-1" and v != "S:-1":
          return
        total_mul *= mul
        total_div *= div

    required_mul = self.output_features_mul * total_div
    required_div = self.output_features_div * total_mul
    gcd = math.gcd(required_mul, required_div)
    required_mul //= gcd
    required_div //= gcd

    output_features = "S:-1"
    if required_div > 1:
      output_features = f"{output_features}%{required_div}"
    if required_mul > 1:
      output_features = f"{output_features}*{required_mul}"
    return output_features

  def make_subgraph_spec(
      self,
      subg,
      adjust_features = False):
    """Converts a list of ops into a (sequential) subgraph spec."""
    input_name = self.input_name
    subgraph_spec = []

    # Check to see if need to unique-ify ops.
    op_names_unique = True
    idxs = []
    for op in subg:
      splits = op.name.split("/")
      idx = splits[-1]
      if len(splits) == 1 or idx in idxs or not idx.isdigit():
        op_names_unique = False
        break
      idxs.append(idx)

    for idx, op in enumerate(subg):
      op = copy.deepcopy(op)
      if not op_names_unique:
        op.name = f"{op.name}/{idx}"
      op.input_names = [input_name]
      input_name = op.name + ":0"
      subgraph_spec.append(subgraph.SubgraphNode(op))
    subgraph_spec[-1].output_names = [self.output_name]

    if adjust_features:
      output_features = self.get_output_features(subg)
      if output_features:
        for node in subgraph_spec[::-1]:
          if node.op.type == OpType.DENSE or node.op.type == OpType.CONV:
            node.op.op_kwargs["features"] = output_features
            break
    return subgraph_spec
