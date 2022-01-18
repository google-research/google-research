# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Helper methods to create, save and interact with hlo proto objects."""
import os.path
import re
from typing import Any, Callable, Sequence, Text, Tuple, Union
from flax import linen as nn
import jax
import jax.numpy as jnp


# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.service import hlo_pb2
# pylint: enable=g-direct-tensorflow-import


def load_hlo_proto_from_jax_fn(
    fn,
    *fn_args,
    **fn_kwargs):
  """Loads HLO proto object from jax function.

  Args:
    fn: a jax function.
    *fn_args: Arguments to fn.
    **fn_kwargs: Keyword arguments to fn.

  Returns:
    An HloModuleProto object.

  """
  computation = jax.xla_computation(fn)(*fn_args, **fn_kwargs)
  serialized_hlo = computation.as_serialized_hlo_module_proto()
  hlo_module_proto = hlo_pb2.HloModuleProto.FromString(serialized_hlo)
  return hlo_module_proto


def load_hlo_proto_from_model(
    model,  #
    state,
    input_shapes,
    **kwargs):
  """Loads HLO proto object from flax model.

  Args:
    model: a flax model.
    state: for model with stateful context, state of the model.
    input_shapes: Sequence of shapes of model input tensor(s). Order of input
      shapes must correspond to order in which model accepts the inputs.
    **kwargs: Addition arguments forwarded to model's `apply` function.

  Returns:
    An HloModuleProto object.

  """

  ones_shape_list = [jnp.ones(shape) for shape in input_shapes]
  return load_hlo_proto_from_jax_fn(lambda *x: model.apply(state, *x, **kwargs),
                                    *ones_shape_list)


# TODO(shivaniagrawal): use strict pytype here.
def output_hlo(computation, file_path):
  """Saves HLO for the given xla computation to given file path.

  The file format is determined by the file_path extension, i.e. for .txt file
  saves the hlo as text, and for .pb file save the hlo module proto binary.

  Args:
    computation: wrapped jax.xla_computation
    file_path: file path to write the HLO to.
  """
  hlo_module_proto_str = computation.as_serialized_hlo_module_proto()
  hlo_txt = computation.as_hlo_text()
  output_hlo_to_file(hlo_module_proto_str, hlo_txt, file_path)




def count_ops_in_hlo_proto(hlo_proto,
                           ops_regex):
  """Counts specific ops in hlo proto, whose names match the provided pattern.

  Args:
    hlo_proto: an HloModuleProto object.
    ops_regex: a string regex to filter ops with matching names.

  Returns:
    Count of matching ops.

  """
  ops_count = 0
  for computation in hlo_proto.computations:
    for instr in computation.instructions:
      if re.match(ops_regex, instr.name) is not None:
        ops_count += 1
  return ops_count
