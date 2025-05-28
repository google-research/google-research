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

"""Class for a sequential synthesizer using the primer search space."""

from __future__ import annotations

import copy
import random
import sys
import traceback
from typing import List, Sequence, Tuple, Optional

from absl import logging

from abstract_nas.abstract.base import AbstractProperty
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.subgraph import SubgraphModel
from abstract_nas.synthesis.random_enum_sequential import RandomEnumerativeSequentialSynthesizer


def log_exc():
  exc_type, exc_value, exc_traceback = sys.exc_info()
  logging.info("".join(
      traceback.format_exception(exc_type, exc_value, exc_traceback)))

MUTABLE_OPS = [OpType.DENSE, OpType.CONV, OpType.GROUP_NORM,
               OpType.AVG_POOL, OpType.MAX_POOL]


class PrimerSequentialSynthesizer(RandomEnumerativeSequentialSynthesizer):
  """Synthesizer that uses the primer search primitives.

  This synthesizer only works for sequential subgraphs (see sequential.py).

  The mutations are:
  - Delete an op
  - Insert an op
  - Delete and insert an op
  - Mutate field
  - Swap ops
  """

  def __init__(self,
               subgraphs_and_props,
               generation,
               abstract = True,
               max_len = -1,
               max_delta = -1,
               min_len = 0,
               min_delta = -1,
               use_automl_zero = False):
    new_subgraph_and_props = []
    for subg, _ in subgraphs_and_props:
      new_subgraph_and_props.append((subg, []))
    super().__init__(new_subgraph_and_props, generation, abstract, max_len,
                     max_delta, min_len, min_delta)
    self.use_automl_zero = use_automl_zero

  def synthesize(self):
    """Returns a new subgraph."""
    subgraph_spec = self.subgraphs_and_props[0][0].subgraph
    subg_ops = [copy.deepcopy(node.op) for node in subgraph_spec]

    mutations = [
        self.delete,
        self.insert,
        self.mutate_field,
        lambda x: self.insert(self.delete(x)),
        self.swap]

    if self.use_automl_zero:
      mutations.append(lambda _: self.randomize())

    # Certain mutations may not be applicable for the selected subgraph, and
    # they will return None (e.g., if the subgraph is of size 1, we cannot
    # swap). So loop through all mutations in a random order until we find a
    # mutation that is applicable.
    random.shuffle(mutations)
    mutated_subg_ops = None
    while mutations and mutated_subg_ops is None:
      mutation = mutations.pop()
      mutated_subg_ops = mutation(subg_ops)
    if mutated_subg_ops is None:
      raise ValueError("Synthesis failed.")
    subg_ops = mutated_subg_ops

    prefix = f"gen{self.generation}/"
    if not subg_ops:
      subg_ops.append(new_op("dummy", OpType.IDENTITY, [self.input_name]))
    for op in subg_ops:
      op.name = f"{prefix}{op.type.name.lower()}"
    subgraph_spec = self.make_subgraph_spec(subg_ops)
    return self.make_subgraph_models(subgraph_spec)

  def delete(self, subg_ops):
    pos = random.randrange(len(subg_ops))
    logging.info("deleting %s", subg_ops[pos].name)
    del subg_ops[pos]
    return subg_ops

  def insert(self, subg_ops):
    pos = random.randrange(len(subg_ops) + 1)
    ops = self.op_enumerator(full=True)
    ops = list(ops)
    op = random.choice(ops)
    logging.info("inserting %s\n"
                 "    op_kwargs=%s\n"
                 "    input_kwargs=%s\n",
                 op.name, op.op_kwargs, op.input_kwargs)
    subg_ops.insert(pos, op)
    return subg_ops

  def mutate_field(self, subg_ops):
    mutable = [op for op in subg_ops if op.type in MUTABLE_OPS]
    if not mutable: return None
    op: Op = random.choice(mutable)
    logging.info("mutating %s", op.name)

    op_kwargs_dict, input_kwargs_dict = self.all_kwargs_for_op_type(
        self.kwarg_defaults, full=True, op_type=op.type)
    keys_to_choose = []
    for kwargs_dict in [op_kwargs_dict, input_kwargs_dict]:
      for k, v in kwargs_dict.items():
        if v and len(v) > 1:
          keys_to_choose.append(k)
    keys_to_choose = list(set(keys_to_choose))
    if not keys_to_choose:
      logging.info("No fields to mutate.")
      return None
    key = random.choice(keys_to_choose)

    if key in op_kwargs_dict:
      value = random.choice(list(op_kwargs_dict[key]))
      op.op_kwargs[key] = value
    else:
      value = random.choice(list(input_kwargs_dict[key]))
      op.input_kwargs[key] = value
    logging.info("mutated %s\n"
                 "    op_kwargs=%s\n"
                 "    input_kwargs=%s\n",
                 op.name, op.op_kwargs, op.input_kwargs)
    return subg_ops

  def swap(self, subg_ops):
    if len(subg_ops) == 1: return None
    pos1 = random.randrange(len(subg_ops))
    pos2 = random.randrange(len(subg_ops) - 1)
    if pos1 == pos2:
      pos2 = len(subg_ops) - 1
    logging.info("swapping %s and %s", subg_ops[pos1].name, subg_ops[pos2].name)
    op = subg_ops[pos1]
    subg_ops[pos1] = subg_ops[pos2]
    subg_ops[pos2] = op
    return subg_ops

  def randomize(self):
    logging.info("randomizing")
    # We initialize the synthesizer without any properties, so the call to super
    # will use the random enumerative strategy to synthesize a random subgraph.
    subg_models = super().synthesize()
    subg_ops = [copy.deepcopy(node.op) for node in subg_models[0].subgraph]
    return subg_ops
