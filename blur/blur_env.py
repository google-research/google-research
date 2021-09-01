# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Describes computation ennvironment for switching between jax and tf."""

import contextlib
import dataclasses as dc
from typing import Any
from typing import Callable, Union, Tuple, Dict, Text

import jax
import jax.numpy as jp
import numpy as np
import tensorflow.compat.v1 as tf

Tensor = Union[tf.Tensor, np.ndarray, jp.array]

# Data comes as a dict with values containing a tuple of images and labels.
DataInputType = Dict[Text, Tuple[tf.Tensor, tf.Tensor]]
NP_FLOATING_TYPE = np.float32


@dc.dataclass
class Env:
  """Environment allowing for interop between different accelerator backends."""

  atanh: Callable[[Tensor], Tensor]
  # Returns function that advances dataset by 1 (e.g. next(iter(ds))
  iterator_next: Callable[[tf.data.Dataset], Callable[[], Any]]
  argmax: Callable[Ellipsis, Tensor]
  relu: Callable[[Tensor], Tensor]
  tanh: Callable[[Tensor], Tensor]
  relu: Callable[[Tensor], Tensor]
  abs: Callable[[Tensor], Tensor]
  sign: Callable[[Tensor], Tensor]
  ones_like: Callable[Ellipsis, Any]
  zeros_like: Callable[Ellipsis, Any]
  zeros: Callable[Ellipsis, Any]
  identity: Callable[Ellipsis, Any]
  concat: Callable[Ellipsis, Any]
  stack: Callable[Ellipsis, Any]
  std: Callable[Ellipsis, Any]
  variance: Callable[Ellipsis, Any]
  mean: Callable[Ellipsis, Any]
  ndims: Callable[[Tensor], int]
  einsum: Callable[Ellipsis, Any]
  to_tensor: Callable[Ellipsis, Any]
  transpose: Callable[Ellipsis, Any]
  exp: Callable[[Tensor], Any]
  name_scope: Callable[[str], Any]
  sqrt: Callable[[Tensor], Any]
  sum: Callable[Ellipsis, Any]

  def relu_tanh(self, x):
    """Relu tanh - a somewhat arbitrary choice of positive activation.

    The goal is to have strong non-linearity around 0, and smooth saturation
    on the  positive side.

    Args:
      x: tensor

    Returns:
      result.
    """
    return self.relu(self.tanh(x))

  def concat_row(self, x, v=1):
    """Concatenates constant row to x in channel dimension."""
    v = self.to_tensor(v, x.dtype)
    return self.concat([x, v * self.ones_like(x[Ellipsis, 0:1, :], dtype=x.dtype)],
                       axis=-2)

  def right_pad_shape(self, inp, to):
    """Appends dimensions of size 1 to inp to match rank of "to"."""
    num_dims = self.ndims(to) - self.ndims(inp)
    assert num_dims >= 0
    extra_dims = (None,) * num_dims
    return self.to_tensor(inp, dtype=to.dtype)[(Ellipsis, *extra_dims)]


@contextlib.contextmanager
def ctx_name_scope(name):
  yield name


def ctx_identity(x, name):
  with ctx_name_scope(name):
    return x


def eager_next(ds):
  it = iter(ds)
  return lambda: next(it)

jp_env = Env(
    atanh=jp.arctanh,
    tanh=jp.tanh,
    relu=jax.nn.relu,
    abs=jp.abs,
    sign=jp.sign,
    ndims=jp.ndim,
    identity=ctx_identity,
    zeros=jp.zeros,
    zeros_like=jp.zeros_like,
    ones_like=jp.ones_like,
    to_tensor=jp.array,
    transpose=jp.transpose,
    concat=jp.concatenate,
    stack=jp.stack,
    mean=jp.mean,
    std=jp.std,
    variance=jp.var,
    argmax=jp.argmax,
    einsum=jp.einsum,
    iterator_next=eager_next,
    exp=jp.exp,
    sum=jp.sum,
    sqrt=jp.sqrt,
    name_scope=ctx_name_scope)

tf_env = Env(
    atanh=tf.math.atanh,
    tanh=tf.math.tanh,
    relu=tf.nn.relu,
    abs=tf.math.abs,
    sign=tf.math.sign,
    ndims=lambda x: tf.shape(x).shape[0],
    identity=tf.identity,
    zeros=tf.zeros,
    zeros_like=tf.zeros_like,
    ones_like=tf.ones_like,
    to_tensor=tf.convert_to_tensor,
    transpose=tf.transpose,
    concat=tf.concat,
    stack=tf.stack,
    mean=tf.reduce_mean,
    std=tf.math.reduce_std,
    variance=tf.math.reduce_variance,
    argmax=tf.argmax,
    einsum=tf.einsum,
    iterator_next=lambda ds: tf.data.make_one_shot_iterator(ds).get_next,
    exp=tf.exp,
    sum=tf.reduce_sum,
    sqrt=tf.sqrt,
    name_scope=tf.name_scope)
