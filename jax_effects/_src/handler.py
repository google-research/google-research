# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Effect handler registration.

This file implements support for registering effect handlers via the Handler and
ParameterizedHandler classes.
"""

import collections
from collections.abc import Sequence, Mapping
import contextlib
import functools
from typing import Any, Callable, Generic, TypeAlias, TypeVar

import jax
from jax import api_util
from jax import core as jax_core
from jax.extend import linear_util as lu
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp


################################################################################
# Handler primitives
################################################################################

# Effect handler primitive.
handler_p = jax_core.Primitive('handler')


def is_handler(primitive):
  return primitive is handler_p


def handler_start(
    arg, name, parameterized, handler_impl
):
  """Starts an effect handler scope."""
  flat_args, in_tree = jax.tree_util.tree_flatten(arg)
  return handler_p.bind(
      *flat_args,
      name=name,
      parameterized=parameterized,
      in_tree=in_tree,
      handler_impl=handler_impl,
  )


@handler_p.def_abstract_eval
def _(*args, **kwargs):
  del args, kwargs
  return jax_core.ShapedArray(shape=(), dtype=jnp.bool_)


@handler_p.def_impl
def _(*args, **kwargs):
  del args, kwargs
  return False


def handler_p_batching_rule(batched_args, batch_dims, *, name, **kwargs):
  del batched_args, batch_dims, name, kwargs
  return [], []


batching.primitive_batchers[handler_p] = handler_p_batching_rule


handler_end_p = jax_core.Primitive('handler_end')


def handler_end(name):
  """Ends an effect handler scope."""
  handler_end_p.bind(name=name)


@handler_end_p.def_abstract_eval
def _(name):
  del name
  return jax_core.ShapedArray(shape=(), dtype=jnp.bool_)


@handler_end_p.def_impl
def _(name):
  del name
  return False


def handler_end_batching_rule(batched_args, batch_dims, *, name):
  del batched_args, batch_dims, name
  return [], []


batching.primitive_batchers[handler_end_p] = (
    handler_end_batching_rule
)


class HandlerReturn(jax_core.Primitive):
  """A handler_return primitive."""


def is_handler_return(primitive):
  return isinstance(primitive, HandlerReturn)


def _check_handler_return(
    ctx_factory, *in_atoms, return_fun_jaxpr, **kwargs
):
  """Type-checking rule for handler_return primitive."""
  del ctx_factory, in_atoms, kwargs
  # TODO(jax-effects-team): To check input types, need to support both before
  # and after explicit effect-parameter-passing.
  # in_avals = [x.aval for x in in_atoms]
  # body_in_avals = [x.aval for x in return_fun_jaxpr.invars]
  # if list(in_avals) != list(body_in_avals):
  #   raise jax_core.JaxprTypeError('handler_return in_avals mismatch')
  jax_core.check_jaxpr(return_fun_jaxpr.jaxpr)
  out_avals = [v.aval for v in return_fun_jaxpr.jaxpr.outvars]
  return out_avals, return_fun_jaxpr.effects


def make_handler_return(name, return_fun_jaxpr):
  """Creates a new `handler_return` primitive and function."""
  handler_return_p = HandlerReturn(f'{name}_return')
  handler_return_p.multiple_results = True

  jax_core.custom_typechecks[handler_return_p] = _check_handler_return

  def handler_return(args):
    flat_args, _ = jax.tree_util.tree_flatten(args)
    return handler_return_p.bind(
        *flat_args,
        name=name,
        return_fun_jaxpr=return_fun_jaxpr,
    )

  @handler_return_p.def_abstract_eval
  def _(
      *args, name, return_fun_jaxpr
  ):
    del args, name
    return [v.aval for v in return_fun_jaxpr.jaxpr.outvars]

  return handler_return


delimited_handler_p = jax_core.Primitive('delimited_handler')
delimited_handler_p.multiple_results = True


def _check_delimited_handler(
    ctx_factory, *in_atoms, body_jaxpr, **kwargs
):
  del ctx_factory, kwargs
  in_avals = [x.aval for x in in_atoms]
  body_in_avals = [x.aval for x in body_jaxpr.invars]
  if list(in_avals) != list(body_in_avals):
    raise jax_core.JaxprTypeError('delimited_handler in_avals mismatch')
  jax_core.check_jaxpr(body_jaxpr)
  out_avals = [v.aval for v in body_jaxpr.outvars]
  return out_avals, body_jaxpr.effects


jax_core.custom_typechecks[delimited_handler_p] = _check_delimited_handler


@delimited_handler_p.def_abstract_eval
def _(*args, body_jaxpr, **kwargs):
  del args, kwargs
  return [v.aval for v in body_jaxpr.outvars]


def is_delimited_handler(primitive):
  return primitive is delimited_handler_p

################################################################################
# Effect handler stack
################################################################################


@lu.cache
def initial_style_jaxpr(fun, in_tree, in_avals):
  """Converts a wrapped function (taking kwargs) to a Jaxpr."""
  wrapped_fun, out_tree = jax.api_util.flatten_fun(fun, in_tree)
  debug = pe.debug_info(
      fun, in_tree, out_tree, has_kwargs=True, traced_for='jax_effects'
  )
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals, debug)
  return jaxpr, consts, out_tree()


# Unique object, used as the default value for `arg` in `Handler.__init__`.
Unspecified: TypeAlias = type('NotSpecified', (), {})  # type: ignore
UNSPECIFIED = Unspecified()

_DEFAULT_HANDLER_ARGUMENT = object()


T = TypeVar('T')
Result = TypeVar('Result')
ReturnFn = Callable[[Result], Any] | Callable[[T, Result], Any]


handler_name_counter = collections.defaultdict(int)


def gensym_handler_name(s):
  """Generates a unique handler name."""
  if s in handler_name_counter:
    handler_name = f'{s}_{handler_name_counter[s]}'
  else:
    handler_name = s
  handler_name_counter[s] += 1
  return handler_name


class Handler(contextlib.ContextDecorator, Generic[Result]):
  """A context manager to provide effect handlers for effect operations."""

  name: str
  result: Result | None

  def __init__(
      self,
      arg = _DEFAULT_HANDLER_ARGUMENT,
      parameterized = None,
      return_fun = None,
      **kwargs,
  ):
    """Creates a new handler.

    Args:
      arg: Optional handler argument.
      parameterized: Boolean value specifying whether the handler is
        parameterized, for type checking. If unspecified, parameterized-ness is
        inferred based on whether `arg` is specified.
      return_fun: Return clause function for transforming the handler result.
      **kwargs: Handler implementation functions.
    """
    self.name = gensym_handler_name('_'.join(kwargs.keys()))
    # Infer whether handler is parameterized based on argument.
    if arg is _DEFAULT_HANDLER_ARGUMENT:
      self.arg = ()
      self.parameterized = False
    else:
      self.arg = arg
      self.parameterized = True
    if parameterized is not None and parameterized != self.parameterized:
      raise ValueError(
          'Handler parameterized mismatch: specified '
          f'parameterized={parameterized} but inferred '
          f'parameterized={self.parameterized}'
      )
    del arg
    if return_fun is not None:
      self.return_fun = return_fun
    else:
      if self.parameterized:
        self.return_fun = lambda p, x: (p, x)
      else:
        self.return_fun = lambda x: x
    self.result = None
    self.kwargs = kwargs

  def __enter__(self):
    # Start handler scope.
    handler_start(
        self.arg,
        name=self.name,
        parameterized=self.parameterized,
        handler_impl=self.kwargs,
    )
    return self

  # NOTE(jax-effects-team): Calling `return` within a Handler `with` block may
  # behave unexpectedly.
  def __exit__(self, *exc):
    # If user did not set handler result, simply end handler scope.
    if self.result is None:
      handler_end(self.name)
      return False
    # Otherwise, apply handler return function.
    if not self.parameterized:
      self.result = self.return_fun(self.result)
    else:
      # Convert handler return function to a Jaxpr.
      args = (self.arg, self.result)
      kwargs = {}
      flat_args, in_tree = jax.tree_util.tree_flatten((args, kwargs))
      in_avals = tuple(api_util.shaped_abstractify(arg) for arg in flat_args)
      return_fn_jaxpr, consts, out_tree = initial_style_jaxpr(
          lu.wrap_init(self.return_fun), in_tree, in_avals
      )
      return_fn_closed_jaxpr = jax_core.ClosedJaxpr(return_fn_jaxpr, consts)
      # Create a handler_return equation from the return function Jaxpr.
      handler_return = make_handler_return(self.name, return_fn_closed_jaxpr)
      out_flat = handler_return(self.result)
      self.result = jax.tree_util.tree_unflatten(out_tree, out_flat)
    handler_end(self.name)
    return False


ParameterizedHandler = functools.partial(
    Handler,
    parameterized=True,
)

