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

"""A jax.vmap-style wrapper over tf.vectorized_map."""
import functools
import inspect
from typing import Any, Mapping, Sequence, Callable, Union, Tuple

import tensorflow as tf

_TensorTuple = Union[tf.Tensor, Tuple[tf.Tensor, Ellipsis]]


def _vmap_single_arg(func,
                     arg_name,
                     kwargs):

  if arg_name not in kwargs:
    raise ValueError('No value is provided for %s in %s' % (arg_name, kwargs))
  pargs = {k: v for k, v in kwargs.items() if k != arg_name}
  pfn = lambda x: functools.partial(func, **pargs)(**{arg_name: x})
  return tf.vectorized_map(pfn, kwargs[arg_name])


def _pack_tensors_across_batch(
    args):
  """Concats a tuple of batched tensors into a single batched tensor."""

  batch_sizes = [tf.shape(arg)[0] for arg in args]
  for other_batch_size in batch_sizes[1:]:
    tf.debugging.assert_equal(
        batch_sizes[0], other_batch_size,
        'Not all arguments have the same batch dim')

  batch_size = batch_sizes[0]
  arg_shapes = [arg.shape[1:] for arg in args]
  flat_arg_tensors = [tf.reshape(arg, (batch_size, -1)) for arg in args]
  packed_tensor = tf.concat(flat_arg_tensors, axis=1)
  return packed_tensor, arg_shapes


def _unpack_single_tensor(
    single_packed_tensor,
    arg_shapes):

  flat_dims = [tf.reduce_prod(x) for x in arg_shapes]
  arg_split = tf.split(single_packed_tensor, flat_dims, axis=0)
  unpacked = [tf.reshape(t, d) for t, d in zip(arg_split, arg_shapes)]
  return unpacked


def _vmap_call(func,
               args_names,
               kwargs):
  """Calls func vectorized along args in `args_names` on data in `kwargs`.

  It flattens all arguments in args_names into [batch_size, -1], concats them,
  and then vectorizes a function that splits and applies as a function with
  a single vectorized argument called `____special_argument_name`.

  Arguments:
    func: The function to auto-vectorize.
    args_names: Names of arguments to auto-vectorize (tensors).
    kwargs: The values of arguments to execute the vectorized function on.

  Returns:
    The result of applying a vectorized func on data in kwargs.
  """

  if len(args_names) == 1:
    return _vmap_single_arg(func, args_names[0], kwargs)

  for name in args_names:
    if name not in kwargs:
      raise ValueError('No value is provided for %s in %s' % (name, kwargs))

  args = [kwargs[name] for name in args_names]
  packed_tensor, arg_shapes = _pack_tensors_across_batch(args)
  # the name of the argument to auto-vectorize in
  special_arg_name = '____special_argument_name'
  assert special_arg_name not in args_names
  call_args = {k: v for k, v in kwargs.items() if k not in args_names}
  call_args[special_arg_name] = packed_tensor

  def _merged_args_func(**merged_kwargs):
    special_arg = merged_kwargs.pop(special_arg_name)
    unpacked_tensors = _unpack_single_tensor(special_arg, arg_shapes)
    merged_kwargs.update(dict(zip(args_names, unpacked_tensors)))
    return func(**merged_kwargs)

  return _vmap_single_arg(_merged_args_func, special_arg_name, call_args)


def vmap(func,
         args_names):
  """Returns a function auto-vectorized along `args_names`.

  For example, for `func(x, y, z)` following two computations are eqvivalent,
  but the first is much faster:

    func_vec = vmap(func, ['x', 'z'])
    output1 = func_vec(a, b, c)
    output2 = tf.stack([func(x, b, z) for x, z in zip(a, c)])

  Auto-vectorization can be applied repeatedly. The resulting function is
  auto-differentiable. `vmap` inherits limitations of `tf.vectorized_map`,
  namely limited support for branching and no support for state-full
  computations.

  Arguments:
    func: The function to auto-vectorize.
    args_names: Name(s) of tensor argument(s) to auto-vectorize.

  Returns:
    A function vectorized along args_names.
  """
  if not args_names:
    return func

  if isinstance(args_names, str):
    return vmap(func, [args_names])
  elif isinstance(args_names, list) and isinstance(args_names[0], str):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
      full_kwargs = inspect.signature(func).bind(*args, **kwargs).arguments
      return _vmap_call(func, args_names, full_kwargs)
    return new_func
  else:
    raise ValueError('unexpected arg specification')
