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

"""Model class that executes a graph.proto.

The graph.proto is just a list of op specifications in the computation graph.
This class converts the proto specification into an executable / trainable JAX
flax.linen model.
"""

from collections import abc
import functools
from typing import Any, Dict, Optional, Sequence, Union

import flax.linen as nn
from jax import lax
from jax import random
import jax.numpy as jnp

from abstract_nas.model.concrete import Graph
from abstract_nas.model.concrete import new_op
from abstract_nas.model.concrete import Op
from abstract_nas.model.concrete import OpType
from abstract_nas.model.utils import split_div_mul
from abstract_nas.utils import canonicalize_tensor_name

Tensor = Any


def _to_float(maybe_float):
  """Safe cast to float."""
  try:
    return float(maybe_float)
  except ValueError:
    return maybe_float


def _kv_to_float(kv, keys):
  """Casts all values in kv with a key in keys to float."""
  for k, v in kv.items():
    if k in keys:
      kv[k] = _to_float(v)
  return kv


def _to_int(maybe_iter):
  """Casts both iterables and single items to integer."""
  if not isinstance(maybe_iter, str) and isinstance(maybe_iter, abc.Iterable):
    return tuple([_to_int(a) for a in maybe_iter])
  try:
    return int(maybe_iter)
  except ValueError:
    return maybe_iter


def _kv_to_int(kv, keys):
  """Casts all values in kv with a key in keys to int."""
  for k, v in kv.items():
    if k in keys:
      kv[k] = _to_int(v)
  return kv


def _resolve_symbolic(maybe_iter, input_values,
                      intermediate_values):
  """Resolves symbolic constants to their concrete values.

  Some of the properties specified by the Graph.proto may use placeholder values
  that are only known at run time (e.g., batch_size), e.g., reshaping a
  tensor that has a batch dimension. To enable this, we allow special symbolic
  constants that are replaced by concrete values at run time.

  In addition to regular integer constants, we support three types of symbolic
  constants:
    Named integers constants, shapes, and initializers.

  The parse rules are as follows:
    Regular integer value
      "#" => #

    A single "B" is handled by the reshape operator as the batch size.
      "B" => "B"

    Initializers are prepended with "I"
      An initializer with args (all arguments are cast to float if possible)
        "I:init_fn:kw1:arg1:kw2:arg2:..." => init_fn(kw1=arg1, kw2=arg2, ...)

    Named integer constants are prepended with "K"
      "K:T" => intermediate_value[T]

    Shapes are prepended with "S"
      The #th dim of the first input value
        "S:#" => input_values[0].shape[#]
      The #th dim of the $th input value)
        "S:$:#" => input_values[$].shape[#]
      The #th dim of the input with name T as found in intermediate_value
        "S:T:#" => intermediate_value[T].shape[#]

    Either named or shape constants may have a suffix which is parsed as a
    mul/div factor
      "TX*#" => parse("TX") * #
      "TX%#" => parse("TX") / #

  This function supports both iterables as well as single values.

  Args:
    maybe_iter: the property string (or list of property strings) to concretize
    input_values: the input_values to the op
    intermediate_values: all intermediate values produced by prior ops in the
      graph.

  Returns:
    a concretized version of maybe_iter

  Raises:
    ValueError: if the div factor does not divide the shape
  """
  if not isinstance(maybe_iter, str) and isinstance(maybe_iter, abc.Iterable):
    return tuple([
        _resolve_symbolic(a, input_values, intermediate_values)
        for a in maybe_iter
    ])
  v = maybe_iter

  # Return numeric.
  if isinstance(v, float) or isinstance(v, int):
    return v

  # Try parsing to numeric.
  try:
    return int(v)
  except (TypeError, ValueError):
    pass
  try:
    return float(v)
  except (TypeError, ValueError):
    pass

  # Only strings are symbolic.
  if not isinstance(v, str): return v

  if v == "B":
    return v

  if ":" not in v:
    raise ValueError(f"Cannot parse value as symbolic: {v}.")
  k, v = v.split(":", maxsplit=1)

  # parsing as init
  if k == "I":
    v = v.split(":")
    name, args = v[0], v[1:]
    kwargs = {k: _to_float(v) for k, v in zip(args[::2], args[1::2])}
    init_fn = getattr(nn.initializers, name)

    if name == "zeros" or name == "ones":
      if kwargs:
        raise ValueError(f"nn.initializers.{name} does not accept kwargs")
      return init_fn
    else:
      return init_fn(**kwargs)

  else:
    if k not in ["K", "S"]:
      raise ValueError(f"Type {k} of symbolic constant {v} not recognized.")

  v, div, mul = split_div_mul(v)

  # parsing as shape
  if k == "S":
    if ":" not in v:
      key = int(v)
      v = input_values[0].shape[key]
    else:
      arr = v.split(":")
      key1, key2 = ":".join(arr[:-1]), arr[-1]
      key2 = int(key2)
      try:
        key1 = int(key1)
        v = input_values[key1].shape[key2]
      except ValueError:
        key1 = canonicalize_tensor_name(key1)
        v = intermediate_values[key1].shape[key2]

  # parsing as named integer constant
  else:
    assert k == "K"
    v = int(intermediate_values[v])

  if v % div != 0:
    raise ValueError(f"Div {div} not a factor of value {v}")
  return v // div * mul


def _kv_resolve_symbolic(kv,
                         keys,
                         input_values = None,
                         intermediate_values = None):
  """Concretizes all symbolic constants in kv with a key in keys."""
  for k, v in kv.items():
    if k in keys:
      kv[k] = _resolve_symbolic(v, input_values, intermediate_values)
  return kv


class Model(nn.Module):
  """Model class for executing a graph."""
  graph: Graph
  constants: Optional[Dict[str, Tensor]] = None

  def __hash__(self):
    """A hash for the Model.

    This implementation enables jax.jit to treat two Models which execute the
    same graph as the same function.

    Returns:
      The hash for the Model.
    """
    return self.graph.__hash__()

  def __eq__(self, other):
    """Checks for equality of underlying graphs, ignoring constants."""
    if not isinstance(other, Model):
      return False
    return self.graph == other.graph

  def exec_init(self, key, value, **_):
    """Initializes the input values for the computation graph."""
    return value

  def exec_op(self, op, input_values,
              deterministic, training, **_):
    """Executes an op according to the normal concrete semantics."""
    input_kwargs: Dict[str, Any] = op.input_kwargs
    op_kwargs: Dict[str, Any] = op.op_kwargs
    op_type = op.type
    if "name" not in op_kwargs:
      raise ValueError("Op kwargs must contain a name.")
    op_name = op_kwargs["name"]

    if op_type == OpType.NONE:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      assert len(op_kwargs) == 1
      output_values = [lax.stop_gradient(input_value)]

    elif op_type == OpType.IDENTITY:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      assert len(op_kwargs) == 1
      output_values = [input_value]

    # nn.linear

    elif op_type == OpType.DENSE:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      output_values = [nn.Dense(**op_kwargs)(input_value)]

    elif op_type == OpType.DENSE_GENERAL:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      assert 2 <= len(op_kwargs) <= 7
      output_values = [nn.DenseGeneral(**op_kwargs)(input_value)]

    elif op_type == OpType.CONV:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs

      ks = op_kwargs["kernel_size"]
      if isinstance(ks, int):
        op_kwargs["kernel_size"] = (ks,) * (input_value.ndim - 2)

      output_values = [nn.Conv(**op_kwargs)(input_value)]

    # others

    elif op_type == OpType.MUL:
      assert len(input_values) == 2
      assert not input_kwargs
      assert len(op_kwargs) == 1  # name
      output_values = [input_values[0] * input_values[1]]

    elif op_type in [OpType.ADD, OpType.STOCH_DEPTH]:
      assert len(op_kwargs) == 1  # name

      input_value = input_values[0]
      if "layer_drop_rate" in input_kwargs:
        assert len(input_kwargs) == 1
        survival_rate = 1 - input_kwargs["layer_drop_rate"]
        if survival_rate == 1.0 or deterministic:
          pass
        else:
          # Reuse dropout's rng stream.
          rng = self.make_rng("dropout")
          mask_shape = [input_value.shape[0]] + [1] * (input_value.ndim - 1)
          mask = random.bernoulli(rng, p=survival_rate, shape=mask_shape)
          mask = jnp.tile(mask, [1] + list(input_value.shape[1:]))
          input_value = lax.select(mask, input_value / survival_rate,
                                   jnp.zeros_like(input_value))
      else:
        assert not input_kwargs
        assert op_type == OpType.ADD

      if op_type == OpType.ADD:
        assert len(input_values) == 2
        output_values = [input_value + input_values[1]]
      else:
        assert len(input_values) == 1
        output_values = [input_value]

    elif op_type == OpType.SCALAR_MUL:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) <= 1
      assert len(op_kwargs) == 1  # name
      if "const" in input_kwargs:
        c = input_kwargs["const"]
      else:
        c = 1 / jnp.sqrt(input_values[0].shape[-1])
      output_values = [input_values[0] * c]

    elif op_type == OpType.SCALAR_ADD:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) <= 1
      assert len(op_kwargs) == 1  # name
      assert "const" in input_kwargs
      c = input_kwargs["const"]
      output_values = [input_values[0] + c]

    elif op_type == OpType.DOT_GENERAL:
      assert len(input_values) == 2
      assert 0 < len(input_kwargs) <= 3
      assert len(op_kwargs) == 1  # name
      output_values = [
          lax.dot_general(input_values[0], input_values[1], **input_kwargs)
      ]

    elif op_type == OpType.EINSUM:
      assert len(input_values) == 2
      assert len(input_kwargs) == 1
      assert "sum" in input_kwargs
      output_values = [
          jnp.einsum(input_kwargs["sum"], input_values[0], input_values[1])
      ]

    # nn.attention

    elif op_type == OpType.SELF_ATTENTION:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      output_values = [
          nn.SelfAttention(**op_kwargs,
                           deterministic=deterministic)(input_value)
      ]

    # nn.activation

    elif op_type in [OpType.RELU, OpType.GELU, OpType.SWISH, OpType.SIGMOID]:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      fn = {
          OpType.RELU: nn.relu,
          OpType.GELU: nn.gelu,
          OpType.SWISH: nn.swish,
          OpType.SIGMOID: nn.sigmoid
      }[op_type]
      output_values = [fn(input_value)]

    elif op_type == OpType.SOFTMAX:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) <= 1
      output_values = [nn.softmax(input_value, **input_kwargs)]

    # nn.normalization

    elif op_type == OpType.BATCH_NORM:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) <= 1
      add_kwargs = {}
      if "use_running_average" not in input_kwargs:
        add_kwargs = {"use_running_average": not training}
      else:
        add_kwargs = {}
      output_values = [
          nn.BatchNorm(**op_kwargs)(input_value, **input_kwargs, **add_kwargs)
      ]

    elif op_type == OpType.LAYER_NORM:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      output_values = [nn.LayerNorm(**op_kwargs)(input_value)]

    elif op_type == OpType.GROUP_NORM:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      output_values = [nn.GroupNorm(**op_kwargs)(input_value)]

    # reshape operators

    elif op_type == OpType.RESHAPE:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert 0 < len(input_kwargs) < 3
      new_shape = input_kwargs.pop("new_shape")
      if new_shape[0] == "B":
        new_shape = (input_value.shape[0],) + new_shape[1:]
      output_values = [jnp.reshape(input_value, new_shape, **input_kwargs)]

    elif op_type == OpType.FLATTEN:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert not input_kwargs
      new_shape = (input_value.shape[0], -1)
      output_values = [jnp.reshape(input_value, new_shape)]

    elif op_type == OpType.TRANSPOSE:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) == 1
      assert len(op_kwargs) == 1  # name
      output_values = [jnp.transpose(input_value, **input_kwargs)]

    # nn.stochastic

    elif op_type == OpType.DROPOUT:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert len(input_kwargs) <= 1
      output_values = [
          nn.Dropout(**op_kwargs)(
              input_value, deterministic=deterministic, **input_kwargs)
      ]

    # nn.pooling

    elif op_type == OpType.AVG_POOL or op_type == OpType.MAX_POOL:
      op_fn = nn.avg_pool if op_type == OpType.AVG_POOL else nn.max_pool
      assert len(input_values) == 1
      input_value = input_values[0]
      assert input_kwargs

      ws = input_kwargs["window_shape"]
      if isinstance(ws, int):
        ws = [ws] * (input_value.ndim - 2)
      new_ws = []
      for window_dim_shape, dim_shape in zip(ws, input_value.shape[1:]):
        if window_dim_shape == 0:
          new_ws.append(dim_shape)
        else:
          new_ws.append(window_dim_shape)
      input_kwargs["window_shape"] = tuple(new_ws)

      if "strides" in input_kwargs:
        s = input_kwargs["strides"]
        if isinstance(s, int):
          input_kwargs["strides"] = (s,) * (input_value.ndim - 2)

      output_values = [op_fn(input_value, **input_kwargs)]

    elif op_type == OpType.MEAN:
      assert len(input_values) == 1
      input_value = input_values[0]
      assert input_kwargs
      output_values = [jnp.mean(input_value, **input_kwargs)]

    # new param

    elif op_type == OpType.PARAM:
      assert not input_values
      assert 0 < len(input_kwargs) <= 2
      init_fn = input_kwargs.pop("init_fn")

      init_fn_with_kwargs = functools.partial(init_fn, **input_kwargs)
      output_values = [self.param(op_name, init_fn_with_kwargs)]

    else:
      raise ValueError(f"op_type {op_type} not supported...")

    return output_values

  def resolve_op(self, op, intermediate_values,
                 **_):
    """Resolves an op with possibly symbolic arguments to a concrete op."""
    op_name = op.name.lower()
    op_type = op.type

    input_names = op.input_names
    input_values = [intermediate_values[key.lower()] for key in input_names]

    input_kwargs: Dict[str, Any] = op.input_kwargs
    op_kwargs: Dict[str, Any] = op.op_kwargs
    op_kwargs["name"] = op_name

    if op_type == OpType.NONE:
      pass

    elif op_type == OpType.IDENTITY:
      pass

    # nn.linear

    elif op_type == OpType.DENSE:
      _kv_resolve_symbolic(op_kwargs, ["kernel_init", "bias_init"])
      _kv_resolve_symbolic(op_kwargs, ["features"], input_values,
                           intermediate_values)

    elif op_type == OpType.DENSE_GENERAL:
      _kv_to_int(op_kwargs, ["axis", "batch_dims"])
      _kv_resolve_symbolic(op_kwargs, ["kernel_init", "bias_init"])
      _kv_resolve_symbolic(op_kwargs, ["features"], input_values,
                           intermediate_values)

    elif op_type == OpType.CONV:
      _kv_to_int(op_kwargs, [
          "kernel_size",
          "strides",
          "input_dilation",
          "kernel_dilation",
          "padding",
      ])
      _kv_resolve_symbolic(op_kwargs, ["kernel_init", "bias_init"])
      _kv_resolve_symbolic(op_kwargs, ["features", "feature_group_count"],
                           input_values, intermediate_values)

    # others

    elif op_type == OpType.ADD:
      _kv_to_float(op_kwargs, ["layer_drop_rate"])

    elif op_type == OpType.SCALAR_ADD:
      _kv_to_float(input_kwargs, ["const"])

    elif op_type == OpType.MUL:
      pass

    elif op_type == OpType.SCALAR_MUL:
      _kv_to_float(input_kwargs, ["const"])

    elif op_type == OpType.DOT_GENERAL:
      _kv_to_int(input_kwargs, ["dimension_numbers"])

    elif op_type == OpType.EINSUM:
      pass

    # nn.attention

    elif op_type == OpType.SELF_ATTENTION:
      _kv_resolve_symbolic(op_kwargs, ["kernel_init", "bias_init"])
      _kv_resolve_symbolic(op_kwargs,
                           ["num_heads", "qkv_features", "out_features"],
                           input_values, intermediate_values)

    # nn.activation

    elif op_type in [OpType.RELU, OpType.GELU, OpType.SWISH, OpType.SIGMOID]:
      pass

    elif op_type == OpType.SOFTMAX:
      _kv_to_int(input_kwargs, ["axis"])

    # nn.normalization

    elif op_type == OpType.BATCH_NORM:
      _kv_to_int(op_kwargs, ["axis"])
      _kv_resolve_symbolic(op_kwargs, ["scale_init", "bias_init"])

    elif op_type == OpType.LAYER_NORM:
      pass

    elif op_type == OpType.GROUP_NORM:
      _kv_resolve_symbolic(op_kwargs, ["num_groups", "group_size"],
                           input_values, intermediate_values)

    # reshape operators

    elif op_type == OpType.RESHAPE:
      _kv_resolve_symbolic(input_kwargs, ["new_shape"], input_values,
                           intermediate_values)
      _kv_to_int(input_kwargs, ["new_shape"])

    elif op_type == OpType.FLATTEN:
      pass

    elif op_type == OpType.TRANSPOSE:
      _kv_to_int(input_kwargs, ["axes"])

    # nn.stochastic

    elif op_type == OpType.DROPOUT:
      _kv_to_int(op_kwargs, ["broadcast_dims"])
      _kv_to_float(op_kwargs, ["rate"])

    elif op_type == OpType.STOCH_DEPTH:
      _kv_to_float(op_kwargs, ["layer_drop_rate"])

    # nn.pooling

    elif op_type == OpType.AVG_POOL:
      _kv_to_int(input_kwargs, ["window_shape", "strides"])

    elif op_type == OpType.MAX_POOL:
      _kv_to_int(input_kwargs, ["window_shape", "strides"])

    elif op_type == OpType.MEAN:
      _kv_to_int(input_kwargs, ["axis"])

    # new param

    elif op_type == OpType.PARAM:
      _kv_to_int(input_kwargs, ["shape"])
      _kv_resolve_symbolic(input_kwargs, ["shape", "init_fn"], input_values,
                           intermediate_values)

    else:
      raise ValueError(f"op_type {op_type} not supported...")

    return new_op(
        op_name,
        op_type,
        input_names,
        input_kwargs,
        op_kwargs,
        num_outputs=op.num_outputs)

  @nn.compact
  def __call__(self, inp, deterministic=True,
               training=False, **context):
    if (len(self.graph.input_names) > 1 or
        len(self.graph.output_names) > 1) and not isinstance(inp, dict):
      raise ValueError(
          "Inputs must be dictionary for graph with multiple inputs and/or "
          "outputs.")

    tensor_input_output = (
        len(self.graph.input_names) == 1 and
        len(self.graph.output_names) == 1 and not isinstance(inp, dict))
    if tensor_input_output:
      input_dict = {self.graph.input_names[0]: inp}
    else:
      input_dict = inp
      if len(input_dict) < len(self.graph.input_names):
        raise ValueError(f"Expected {len(self.graph.input_names)} inputs, "
                         f"received only {len(input_dict)} inputs.")
      for input_name in self.graph.input_names:
        if input_name not in input_dict:
          raise ValueError(f"Expected input {input_name} not provided.")

    intermediate_values = dict(self.constants) if self.constants else {}
    for k, v in input_dict.items():
      key = canonicalize_tensor_name(k)
      intermediate_values[key] = self.exec_init(k, v, **context)

    for op in self.graph.ops:
      op = self.resolve_op(op, intermediate_values, **context)

      input_names = op.input_names
      input_values = [intermediate_values[key.lower()] for key in input_names]
      output_values = self.exec_op(op, input_values, deterministic, training,
                                   **context)
      if len(output_values) != op.num_outputs:
        raise RuntimeError(f"op_type {op.type} expected {op.num_outputs} "
                           f"outputs, produced {len(output_values)} outputs...")

      for idx, output_value in enumerate(output_values):
        intermediate_values[f"{op.name.lower()}:{idx}"] = output_value

    if not self.graph.output_names:
      for k in input_dict.keys():
        k = canonicalize_tensor_name(k)
        if k in intermediate_values:
          del intermediate_values[k]
      if self.constants:
        for k in self.constants.keys():
          if k in intermediate_values:
            del intermediate_values[k]
      return intermediate_values
    if tensor_input_output:
      output_name = self.graph.output_names[0]
      k = canonicalize_tensor_name(output_name)
      outputs = intermediate_values[k]
    else:
      outputs = {}
      for output_name in self.graph.output_names:
        k = canonicalize_tensor_name(output_name)
        outputs[output_name] = intermediate_values[k]

    return outputs
