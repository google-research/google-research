# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Effect handling core."""

# pylint: disable=cell-var-from-loop,g-bad-todo

import abc
from collections.abc import Mapping, MutableMapping, Sequence, Set
import dataclasses
import sys
import typing
from typing import Any, Callable, Iterable, ParamSpec, TypeVar

import jax
from jax import api_util
from jax import config
from jax import core as jax_core
from jax import util as jax_util
from jax.extend import source_info_util
from jax.interpreters import ad
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import rich
import rich.console

from jax_effects._src import handler
from jax_effects._src import logging_utils

CONSOLE = rich.console.Console()
PRINT_EFFECTS = False
PP_SETTINGS = jax_core.JaxprPpSettings(print_effects=PRINT_EFFECTS)

# High recursion limit currently needed for recursive functions like
# `translate_loss_eqns`. This could be removed once we rewrite recursive
# functions using loops.
sys.setrecursionlimit(10_000)

Handler = handler.Handler
ParameterizedHandler = handler.ParameterizedHandler

T = TypeVar('T')
P = ParamSpec('P')
Cont = Callable[[T], Any]
LossCont = Callable[[T], Any]

THETA_SUFFIX = '_θ'

print_jaxpr = logging_utils.print_jaxpr

################################################################################
# General effect registration
################################################################################

effect_primitives: dict[str, jax_core.Primitive] = {}

Aval = jax_core.AbstractValue
AbstractEvalFn = Callable[P, Aval | Sequence[Aval]]
GensymFn = Callable[[Aval], jax_core.Var]
GensymFnVar = Callable[[jax_core.Atom], jax_core.Atom]


def identity(x):
  return x


def infer_effect_name(obj: Any) -> str | None:
  """Infers the effect name from a Python object."""
  if not callable(obj):
    return None
  if not hasattr(obj, '__name__'):
    return None
  name = obj.__name__
  if name == '<lambda>':
    return None
  return name


@typing.overload
def effect(
    f: Callable[P, T],
    *,
    name: str | None = None,
    abstract_eval: AbstractEvalFn = identity,
    out_tree: jax.tree_util.PyTreeDef | None = None,
) -> Callable[P, T]:
  ...


@typing.overload
def effect(
    f: None = None,
    *,
    name: str | None = None,
    abstract_eval: AbstractEvalFn = identity,
    out_tree: jax.tree_util.PyTreeDef | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
  ...


class EffectPrimitive(jax.core.Primitive):
  """An effect operation primitive."""


def effect(
    f: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    abstract_eval: AbstractEvalFn = identity,
    out_tree: jax.tree_util.PyTreeDef | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
  """Decorator function for defining an effect operation."""

  def decorator(f: Callable[P, T]) -> Callable[P, T]:
    nonlocal name
    name = name or infer_effect_name(f)

    if name is None:
      raise ValueError(
          f'Effect name could not be inferred from {f}; '
          'please provide an explicit value'
      )

    if name in effect_primitives:
      raise ValueError(
          f'Effect {name} already exists: {effect_primitives[name]}'
      )

    new_effect_p = EffectPrimitive(name)

    effect_primitives[name] = new_effect_p

    @new_effect_p.def_abstract_eval
    def _(*args, name: str, in_tree: jax.tree_util.PyTreeDef):
      del name, in_tree
      if not args:
        out_avals = abstract_eval()
      else:
        flat_args, _ = jax.tree_util.tree_flatten(args)
        in_avals = list(api_util.shaped_abstractify(arg) for arg in flat_args)
        if len(in_avals) == 1:
          in_avals = in_avals[0]
        out_avals = abstract_eval(in_avals)

      if isinstance(out_avals, Sequence):
        new_effect_p.multiple_results = True

      return out_avals

    # TODO(jax_effects-team): Document the behavior when `name` specifies a
    # handler that is not in scope.

    def effect_operation(*args, name: str = name, **kwargs):
      """Perform an effect operation.

      Args:
        *args: The argument to the effect operation.
        name: A string tag used to identify a specific named handler. The
          default value is the name of the primitive operation itself.
        **kwargs: Named parameters for the JAX effect primitive.

      Returns:
        Values corresponding to the `abstract_eval` function for the effect
        operations.
      """
      if not args:
        _, in_tree = jax.tree_util.tree_flatten(args)
        result = new_effect_p.bind(name=name, in_tree=in_tree, **kwargs)
        if out_tree is None:
          return result
        else:
          return jax.tree_util.tree_unflatten(out_tree, result)

      if len(args) == 1:
        [args] = args
      flat_args, in_tree = jax.tree_util.tree_flatten(args)
      result = new_effect_p.bind(
          *flat_args, name=name, in_tree=in_tree, **kwargs
      )
      if out_tree is None:
        return result
      else:
        return jax.tree_util.tree_unflatten(out_tree, result)

    effect_operation.name = name
    effect_operation.primitive = new_effect_p

    return effect_operation

  if f is None:
    return decorator
  else:
    return decorator(f)


def is_effect_op(primitive: jax_core.Primitive) -> bool:
  return isinstance(primitive, EffectPrimitive)


################################################################################
# Jaxpr utilities
################################################################################


def pp_ctx_factory(jaxpr: jax_core.Jaxpr) -> jax_core.JaxprPpContext:
  ctx = jax_core.JaxprPpContext()
  jax_core.pp_jaxpr(jaxpr, ctx, PP_SETTINGS)
  return ctx


def is_literal_equal(lhs: jax_core.Atom, rhs: jax_core.Atom):
  if not isinstance(lhs, jax_core.Literal) or not isinstance(
      rhs, jax_core.Literal
  ):
    return False
  return lhs.val == rhs.val and lhs.aval == rhs.aval


def free_vars_in_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
    bounded_vars: set[jax_core.Var] | None = None,
) -> list[jax_core.Var]:
  """Return all free variables in JaxprEqns, given bounded variables."""
  if bounded_vars is None:
    bounded_vars = set()
  result = []
  for eqn in eqns:
    for in_var in eqn.invars:
      if isinstance(in_var, jax_core.Literal):
        continue
      if in_var not in bounded_vars and in_var not in result:
        result.append(in_var)
    for v in eqn.params.values():
      if isinstance(v, jax_core.Jaxpr):
        result.extend(free_vars_jaxpr(v))
      elif isinstance(v, jax_core.ClosedJaxpr):
        result.extend(free_vars_jaxpr(v.jaxpr))
    bounded_vars |= set(eqn.outvars)
  return result


def free_vars_jaxpr(
    jaxpr: jax_core.Jaxpr,
    bounded_vars: Set[jax_core.Var] = frozenset(),
) -> list[jax_core.Var]:
  """Return all free variables in a Jaxpr, given a set of bounded variables."""
  all_bounded_vars = set(jaxpr.invars) | set(jaxpr.constvars)
  if bounded_vars:
    all_bounded_vars |= bounded_vars
  result = []
  result.extend(free_vars_in_eqns(jaxpr.eqns, bounded_vars=all_bounded_vars))
  for outvar in jaxpr.outvars:
    if isinstance(outvar, jax_core.Literal):
      continue
    if outvar not in all_bounded_vars and outvar not in result:
      result.append(outvar)
  return result


def zeros_like_avals(avals):
  return tuple(ad.zeros_like_aval(v) for v in avals)


def get_tracer_ids(xs):
  return tuple(id(x) for x in xs)


def dce_jaxpr(jaxpr: jax_core.Jaxpr) -> jax_core.Jaxpr:
  new_jaxpr, _ = pe.dce_jaxpr(
      jaxpr, used_outputs=[True] * len(jaxpr.outvars), instantiate=True
  )
  return new_jaxpr


def dce_closed_jaxpr(jaxpr: jax_core.ClosedJaxpr) -> jax_core.ClosedJaxpr:
  return jaxpr.map_jaxpr(dce_jaxpr)


def filter_indices(l: Sequence[T], indices: Sequence[int]) -> Sequence[T]:
  result = []
  index_set = set(indices)
  for i, x in enumerate(l):
    if i in index_set:
      continue
    result.append(x)
  return result


################################################################################
# Scope delimitation
################################################################################


def substitute_eqn(
    eqn: jax_core.JaxprEqn,
    substitute_fn,
    gensym: GensymFn,
    free_vars: Sequence[jax_core.Var] = (),
) -> jax_core.JaxprEqn:
  """Substitutes variables in `eqn` using `substitute_fn`."""
  params = {}
  for k, v in eqn.params.items():
    if isinstance(v, jax_core.Var):
      v = substitute_fn(v)
    elif isinstance(v, jax_core.Jaxpr):
      v = clone_jaxpr(v, gensym=gensym, free_vars=free_vars)
    elif isinstance(v, jax_core.ClosedJaxpr):
      v = v.map_jaxpr(
          lambda jaxpr: clone_jaxpr(jaxpr, gensym=gensym, free_vars=free_vars)
      )
    params[k] = v
  return eqn.replace(
      invars=[substitute_fn(v) for v in eqn.invars],
      outvars=[substitute_fn(v) for v in eqn.outvars],
      params=params,
  )


def substitute_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
    substitute_fn,
    gensym: GensymFn,
    free_vars: Sequence[jax_core.Var] = (),
) -> Sequence[jax_core.JaxprEqn]:
  """Substitutes variables in `eqns` using `substitute_fn`."""
  return [
      substitute_eqn(e, substitute_fn, gensym=gensym, free_vars=free_vars)
      for e in eqns
  ]


def gensym_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
    gensym: GensymFn,
    mapping: MutableMapping[jax_core.Var, jax_core.Var] | None = None,
    free_vars: Sequence[jax_core.Var] = (),
) -> tuple[
    Sequence[jax_core.JaxprEqn], Callable[[jax_core.Atom], jax_core.Var]
]:
  """Renames all variables (except `free_vars`) in `eqn` using `gensym`."""
  if mapping is None:
    mapping: dict[jax_core.Var, jax_core.Var] = {}

  def substitute(v: jax_core.Atom) -> jax_core.Var:
    if isinstance(v, jax_core.Literal):
      return v
    if free_vars and v in free_vars:
      return v
    if v in mapping:
      return mapping[v]
    else:
      return mapping.setdefault(v, gensym(v.aval))

  new_eqns = []
  for eqn in eqns:
    new_eqn = substitute_eqn(
        eqn, substitute, gensym=gensym, free_vars=free_vars
    )
    new_eqns.append(new_eqn)

  return new_eqns, substitute


def clone_jaxpr(
    jaxpr: jax_core.Jaxpr,
    gensym: GensymFn,
    free_vars: Sequence[jax_core.Var] | None = None,
) -> jax_core.Jaxpr:
  """Clones a Jaxpr, renaming all variables except those in `free_vars`."""
  mapping = {}

  def substitute(v: jax_core.Atom) -> jax_core.Var:
    if isinstance(v, jax_core.Literal):
      return v
    if free_vars and v in free_vars:
      return v
    if v in mapping:
      return mapping[v]
    else:
      return mapping.setdefault(v, gensym(v.aval))

  new_constvars = [substitute(v) for v in jaxpr.constvars]
  new_invars = [substitute(v) for v in jaxpr.invars]
  for prevvar, newvar in jax_util.safe_zip(jaxpr.constvars, new_constvars):
    mapping[prevvar] = newvar
  for prevvar, newvar in jax_util.safe_zip(jaxpr.invars, new_invars):
    mapping[prevvar] = newvar
  new_eqns, substitute = gensym_eqns(
      eqns=jaxpr.eqns,
      gensym=gensym,
      free_vars=free_vars,
      mapping=mapping,
  )
  for prevvar, newvar in mapping.items():
    newvar.suffix = prevvar.suffix
  return jaxpr.replace(
      constvars=new_constvars,
      invars=new_invars,
      outvars=[substitute(v) for v in jaxpr.outvars],
      eqns=new_eqns,
  )


def find_handler_end_eqn(op: str, eqns: Sequence[jax_core.JaxprEqn]) -> int:
  for i, eqn in enumerate(eqns):
    if eqn.primitive is handler.handler_end_p and op == eqn.params['name']:
      return i
  raise ValueError(f'While tracking handler {op}, no end marker found.')


def foreach_subjaxprs(
    eqn: jax_core.JaxprEqn,
    foreach_fn: Callable[[jax_core.Jaxpr], None],
):
  """Applies `foreach_fn` to all subjaxprs in `eqn`."""
  for v in eqn.params.values():
    if isinstance(v, jax_core.Jaxpr):
      foreach_fn(v)
    elif isinstance(v, jax_core.ClosedJaxpr):
      foreach_fn(v.jaxpr)
  if eqn.primitive is handler.delimited_handler_p:
    for handler_impl_jaxpr in eqn.params['handler_impl'].values():
      foreach_fn(handler_impl_jaxpr)


def transform_subjaxprs(
    eqn: jax_core.JaxprEqn,
    transform_fn: Callable[[jax_core.Jaxpr], jax_core.Jaxpr],
    skip_params: Set[str] = frozenset(),
) -> jax_core.JaxprEqn:
  """Transforms all subjaxprs in `eqn` using `transform_fn`."""
  new_params = eqn.params
  for k, v in eqn.params.items():
    if k in skip_params:
      continue
    if isinstance(v, jax_core.Jaxpr):
      v = transform_fn(v)
    elif isinstance(v, jax_core.ClosedJaxpr):
      v = v.map_jaxpr(transform_fn)
    elif eqn.primitive is handler.delimited_handler_p and k == 'handler_impl':
      new_v = {}
      for handler_k, handler_impl in v.items():
        new_v[handler_k] = handler_impl.map_jaxpr(transform_fn)
      v = new_v
    new_params[k] = v
  return eqn.replace(params=new_params)


def delimited_handler_transform(jaxpr: jax_core.Jaxpr) -> jax_core.Jaxpr:
  """Transforms `handler` equations into `delimited_handler` equations.

  Args:
    jaxpr: A Jaxpr with `handler` and `handler_end` equations,
      representing handler scopes as the contained range of equations.

  Returns:
    A Jaxpr with `delimited_handler` equations, where handler scopes are
    explicitly represented by `scope_jaxpr` Jaxprs. No more `handler` and
    `handler_end` equations exist.
  """
  new_eqns = [
      transform_subjaxprs(eqn, delimited_handler_transform)
      for eqn in jaxpr.eqns
  ]

  # Find the first handler.
  handler_eqn_index = None
  handler_eqn = None
  for i, eqn in enumerate(new_eqns):
    if handler.is_handler(eqn.primitive):
      handler_eqn_index = i
      handler_eqn = eqn
      break

  if handler_eqn_index is None:
    return jaxpr.replace(eqns=new_eqns)

  handler_args = handler_eqn.invars
  handler_params = handler_eqn.params
  handler_name = handler_params['name']
  handler_parameterized = handler_params['parameterized']
  handler_arg_tree = handler_params['in_tree']
  handler_implementations = handler_params['handler_impl']

  scope_start_index = handler_eqn_index + 1
  scope_end_index = find_handler_end_eqn(handler_name, new_eqns)
  scope_eqns = new_eqns[scope_start_index:scope_end_index]
  remaining_eqns = new_eqns[scope_end_index + 1 :]

  gensym = jax_core.gensym([jaxpr])
  scope_new_eqns, scope_substitute = gensym_eqns(scope_eqns, gensym)
  scope_free_vars = free_vars_in_eqns(
      scope_eqns,
      bounded_vars=set(
          v for v in handler_args if not isinstance(v, jax_core.Literal)
      ),
  )
  all_bounded_vars = set(jaxpr.invars)
  scope_call_outvars = free_vars_in_eqns(
      remaining_eqns, bounded_vars=all_bounded_vars
  )
  scope_call_outvars.extend(
      v
      for v in jaxpr.outvars
      if not isinstance(v, jax_core.Literal)
      and v not in scope_call_outvars
      and v not in all_bounded_vars
  )
  scope_outvars = [scope_substitute(v) for v in scope_call_outvars]

  scope_call_invars = handler_args + scope_free_vars
  scope_invars = [
      scope_substitute(v)
      if not isinstance(v, jax_core.Literal)
      else gensym(v.aval)
      for v in scope_call_invars
  ]
  for i in range(len(handler_args)):
    assert not scope_invars[i].suffix
    scope_invars[i].suffix = THETA_SUFFIX

  scope_jaxpr = jaxpr.replace(
      constvars=(),
      invars=scope_invars,
      outvars=scope_outvars,
      eqns=scope_new_eqns,
  )
  # Recursively transform all remaining `handler` equations to
  # `delimited_handler`.
  scope_jaxpr = delimited_handler_transform(scope_jaxpr)
  delimited_handler_eqn = jax_core.new_jaxpr_eqn(
      invars=scope_call_invars,
      outvars=scope_call_outvars,
      primitive=handler.delimited_handler_p,
      params=dict(
          name=handler_name,
          body_jaxpr=scope_jaxpr,
          parameterized=handler_parameterized,
          args=handler_args,
          in_tree=handler_arg_tree,
          handler_impl=handler_implementations,
      ),
      effects=scope_jaxpr.effects,
      source_info=handler_eqn.source_info,
  )

  transformed_eqns = (
      new_eqns[:handler_eqn_index]
      + [delimited_handler_eqn]
      + new_eqns[scope_end_index + 1 :]
  )
  return jaxpr.replace(eqns=transformed_eqns)


################################################################################
# Handler concretization
################################################################################


@dataclasses.dataclass
class EffectOpSignature:
  """An effect operation signature: input abstract values and PyTree."""
  in_avals: Sequence[Aval]
  in_tree: jax.tree_util.PyTreeDef


def collect_effect_op_signatures(
    jaxpr: jax_core.Jaxpr,
    effect_op_signatures: MutableMapping[str, EffectOpSignature] | None = None,
) -> Mapping[str, EffectOpSignature]:
  """Collects argument types for all effect operations in `jaxpr`."""

  if effect_op_signatures is None:
    effect_op_signatures = {}

  eqns = jaxpr.eqns
  for eqn in eqns:
    if is_effect_op(eqn.primitive):
      in_avals = tuple(v.aval.strip_weak_type() for v in eqn.invars)
      effect_name = eqn.params['name']
      in_tree = eqn.params['in_tree']
      signature = EffectOpSignature(in_avals, in_tree)
      # Argument effect types must be stable.
      if effect_name in effect_op_signatures:
        if signature != effect_op_signatures[effect_name]:
          raise ValueError(
              f'Effect {effect_name} signature is not stable: '
              f'{signature} vs {effect_op_signatures[effect_name]}'
          )
      effect_op_signatures[effect_name] = signature
    else:
      foreach_subjaxprs(
          eqn,
          lambda jaxpr: collect_effect_op_signatures(
              jaxpr, effect_op_signatures
          ),
      )

  return effect_op_signatures


class CallWithContinuationPrimitive(jax_core.Primitive):
  """A call_k primitive."""
  outvals: Sequence[jax_core.AbstractValue]


class CallWithLossContinuationPrimitive(jax_core.Primitive):
  """A call_lk primitive."""


Loss = jax_core.ShapedArray((), jnp.float32)


def is_call_k(primitive: jax_core.Primitive) -> bool:
  return isinstance(primitive, CallWithContinuationPrimitive)


def is_call_lk(primitive: jax_core.Primitive) -> bool:
  return isinstance(primitive, CallWithLossContinuationPrimitive)


def make_call_k(
    name: str, abstract_eval_outvals: Sequence[jax_core.AbstractValue]
):
  """Makes a fresh call_k primitive."""
  call_k_p = CallWithContinuationPrimitive('call_k')
  call_k_p.outvals = abstract_eval_outvals

  def call_k(*arg):
    flat_args, _ = jax.tree_util.tree_flatten(arg)
    return call_k_p.bind(*flat_args, name=name)

  call_k.primitive = call_k_p

  @call_k_p.def_abstract_eval
  def _(*args, name):
    del args, name
    outval_count = len(abstract_eval_outvals)
    if outval_count == 0:
      call_k_p.multiple_results = True
      return ()
    elif outval_count == 1:
      call_k_p.multiple_results = False
      return abstract_eval_outvals[0]
    else:
      call_k_p.multiple_results = True
      return abstract_eval_outvals

  return call_k


def make_call_lk(name: str, call_k: Callable[..., Any]):
  """Makes a fresh call_lk primitive."""
  call_lk_p = CallWithLossContinuationPrimitive('call_lk')

  def call_lk(*arg):
    flat_args, in_tree = jax.tree_util.tree_flatten(arg)
    return call_lk_p.bind(*flat_args, name=name, in_tree=in_tree)

  def call_lk_jvp(primals, tangents, name, in_tree):
    def call_lk_wrapper(*args, **kwargs):
      # Note: this wrapper does nothing except call `k`.
      # The loss transformation changes the "call_lk" Jaxpr to return a loss
      # value.
      call_k(*args, **kwargs)
      return ()

    original_result = call_lk_p.bind(*primals, name=name, in_tree=in_tree)
    call_lk_jaxpr = jax.make_jaxpr(call_lk_wrapper)(*primals)
    tangent_outs = jvp_staged(call_lk_jaxpr, primals, tangents)
    return original_result, tangent_outs[0]

  ad.primitive_jvps[call_lk_p] = call_lk_jvp

  @call_lk_p.def_abstract_eval
  def _(*args, **kwargs):
    del args, kwargs
    return Loss

  call_lk.primitive = call_lk_p

  return call_lk


def make_call_k_and_call_lk(
    name: str,
    abstract_eval_outvals: Sequence[jax_core.AbstractValue],
):
  """Makes fresh primitives and functions for call_k and call_lk."""

  call_k = make_call_k(name=name, abstract_eval_outvals=abstract_eval_outvals)
  call_lk = make_call_lk(name=name, call_k=call_k)

  call_lk.primitive.call_k_primitive = call_k.primitive

  return call_k, call_lk


jvp_staged_p = jax_core.Primitive('jvp_staged')
jvp_staged_p.multiple_results = True


def jvp_staged(jaxpr, primals, tangents):
  flat_primals, _ = jax.tree_util.tree_flatten(primals)
  flat_tangents, tangents_tree = jax.tree_util.tree_flatten(tangents)
  flat_args = flat_primals + flat_tangents
  # Bind primals and tangents as arguments.
  return jvp_staged_p.bind(
      *flat_args,
      jaxpr=jaxpr,
      num_primals=len(flat_primals),
      tangents_tree=tangents_tree,
  )


# Note: currently specialized to loss continuations, which always return a
# scalar loss.
@jvp_staged_p.def_abstract_eval
def _(*args, **kwargs):
  del args, kwargs
  return (Loss, Loss)


def jvp_staged_transpose(cotangents, *args, jaxpr, num_primals, tangents_tree):
  flat_primals = args[:num_primals]
  primal_zeros = [ad.Zero.from_value(x) for x in flat_primals]
  cotangents_out = primal_zeros + linear_transpose_staged(
      cotangents,
      args,
      jaxpr=jaxpr,
      num_primals=num_primals,
      tangents_tree=tangents_tree,
  )
  return cotangents_out


ad.primitive_transposes[jvp_staged_p] = jvp_staged_transpose


linear_transpose_staged_p = jax_core.Primitive('linear_transpose_staged')
linear_transpose_staged_p.multiple_results = True


def linear_transpose_staged(
    cotangents, args, *, jaxpr, num_primals, tangents_tree
):
  flat_cotangents, cotangents_tree = jax.tree_util.tree_flatten(cotangents)
  flat_primals, flat_tangents = jax_util.split_list(args, [num_primals])
  tangents = jax.tree_util.tree_unflatten(tangents_tree, flat_tangents)
  return linear_transpose_staged_p.bind(
      *(flat_primals + flat_cotangents),
      jaxpr=jaxpr,
      num_primals=num_primals,
      tangents=tangents,
      cotangents_tree=cotangents_tree,
  )


@linear_transpose_staged_p.def_abstract_eval
def _(*args, tangents, **kwargs):
  del args, kwargs
  # Note: skipping ad.Zero is needed to match typing rules.
  tangent_avals = [x.aval for x in tangents if not isinstance(x, ad.Zero)]
  return tangent_avals


vmap_staged_p = jax_core.Primitive('vmap_staged')
vmap_staged_p.multiple_results = True


def vmap_staged(jaxpr, batched_args, batch_dim):
  return vmap_staged_p.bind(
      *batched_args,
      jaxpr=jaxpr,
      batch_dim=batch_dim,
  )


def convert_captured_vars_to_constvars(
    closed_jaxpr: jax_core.ClosedJaxpr,
    ctx: jax_core.JaxprPpContext,
) -> tuple[jax_core.ClosedJaxpr, Sequence[jax_core.Var]]:
  """Converts a ClosedJaxpr that captures variables to one with constvars.

  This is needed because JAX does not support Jaxprs that capture variables from
  outer scope.

  Args:
    closed_jaxpr: The ClosedJaxpr to convert.
    ctx: The pretty-printing context, for debug printing.

  Returns:
    Tuple of a new ClosedJaxpr with free variables converted to constvars, with
    the free variables.
  """
  jaxpr: jax_core.Jaxpr = closed_jaxpr.jaxpr
  jaxpr = inline_closed_call(jaxpr, ctx)
  free_vars = free_vars_jaxpr(jaxpr)
  new_jaxpr = jaxpr.replace(constvars=jaxpr.constvars + free_vars)
  free_var_consts = [ad.zeros_like_aval(v.aval) for v in free_vars]
  new_closed_jaxpr = closed_jaxpr.replace(
      jaxpr=new_jaxpr, consts=closed_jaxpr.consts + free_var_consts
  )
  return new_closed_jaxpr, free_vars


def convert_constvars_to_invars_jaxpr(
    jaxpr: jax_core.Jaxpr, n: int
) -> jax_core.Jaxpr:
  """Moves the n last constvars in `jaxpr` to the end of invars."""
  if config.jax_enable_checks:
    jax_core.check_jaxpr(jaxpr)
  constvars, converted_invars = jax_util.split_list(jaxpr.constvars, [-n])
  dbg = jaxpr.debug_info and jaxpr.debug_info._replace(
      arg_names=(None,) * len(constvars) + jaxpr.debug_info.arg_names
  )
  lifted_jaxpr = jax_core.Jaxpr(
      constvars=constvars,
      invars=jaxpr.invars + converted_invars,
      outvars=jaxpr.outvars,
      eqns=jaxpr.eqns,
      effects=jaxpr.effects,
      debug_info=dbg,
  )
  if config.jax_enable_checks:
    jax_core.check_jaxpr(jaxpr)
  return lifted_jaxpr


def convert_constvars_to_invars_closed_jaxpr(
    closed_jaxpr: jax_core.ClosedJaxpr, n: int
) -> jax_core.ClosedJaxpr:
  """Moves constvars and consts to the start of invars."""
  new_jaxpr = convert_constvars_to_invars_jaxpr(closed_jaxpr.jaxpr, n)
  new_consts, _ = jax_util.split_list(closed_jaxpr.consts, [-n])
  return closed_jaxpr.replace(jaxpr=new_jaxpr, consts=new_consts)


def lower_linear_transpose_staged(
    eqn: jax_core.JaxprEqn,
    ctx: jax_core.JaxprPpContext,
) -> jax_core.JaxprEqn:
  """Transforms linear_transpose_staged into a regular closed_call."""
  if eqn.primitive is not linear_transpose_staged_p:
    raise ValueError(f'Expected linear_transpose_staged_p equation, got: {eqn}')

  print_jaxpr(
      jax_core.pp_eqn(eqn, ctx, PP_SETTINGS),
      title='[bold dark_orange]linear_transpose_staged eqn',
      panel=False,
  )

  jaxpr: jax_core.ClosedJaxpr = eqn.params['jaxpr']
  # Convert captured variables to constvars in Jaxpr.
  jaxpr, free_vars = convert_captured_vars_to_constvars(jaxpr, ctx)
  jaxpr_fun = jax_core.jaxpr_as_fun(jaxpr)
  print_jaxpr(jaxpr, title='linear transpose source jaxpr', panel=False)

  num_primals = eqn.params['num_primals']
  primal_invars = eqn.invars[:num_primals]
  primal_avals = [v.aval for v in primal_invars]
  primal_zeros = zeros_like_avals(primal_avals)

  def wrapper_jaxpr_fun(jaxpr_fun):
    def fn(*args, **kwargs):
      # Assert that Jaxpr only produces a single result.
      (result,) = jaxpr_fun(*args, **kwargs)
      return result

    return fn

  # Linearize to get JVP function.
  tangents = eqn.params['tangents']
  call_transpose_outvars = []
  grad_idx = []
  outvar_index = 0
  for i, tangent in enumerate(tangents):
    if isinstance(tangent, ad.Zero):
      pass
    else:
      call_transpose_outvars.append(eqn.outvars[outvar_index])
      grad_idx.append(i)
      outvar_index += 1

  grad_fn = jax.grad(wrapper_jaxpr_fun(jaxpr_fun), argnums=grad_idx)
  grad_fn_jaxpr = jax.make_jaxpr(grad_fn)(*primal_zeros)
  grad_fn_jaxpr = convert_constvars_to_invars_closed_jaxpr(
      grad_fn_jaxpr, n=len(free_vars)
  )

  call_transpose_eqn = jax_core.new_jaxpr_eqn(
      invars=primal_invars + free_vars,
      outvars=call_transpose_outvars,
      primitive=jax_core.closed_call_p,
      params=dict(call_jaxpr=grad_fn_jaxpr),
      effects=grad_fn_jaxpr.effects,
      source_info=eqn.source_info,
  )
  return call_transpose_eqn


staged_transform_primitives = set()
staged_transform_primitives.add(linear_transpose_staged_p)


def move_const_from_tracer_id_list(
    closed_jaxpr: jax_core.ClosedJaxpr, tracer_ids: Sequence[int]
) -> jax_core.ClosedJaxpr:
  """Converts consts to invars in `closed_jaxpr` based on `tracer_ids`."""
  jaxpr = closed_jaxpr.jaxpr
  constvar_ids = get_tracer_ids(closed_jaxpr.consts)
  constvar_indices_to_move = []
  unhandled_tracer_id = []
  for tracer_id in tracer_ids:
    if tracer_id not in constvar_ids:
      unhandled_tracer_id.append(tracer_id)
    else:
      constvar_indices_to_move.append(constvar_ids.index(tracer_id))
  constvars_to_move = [jaxpr.constvars[i] for i in constvar_indices_to_move]
  new_jaxpr = jaxpr.replace(
      constvars=filter_indices(jaxpr.constvars, constvar_indices_to_move),
      invars=constvars_to_move + jaxpr.invars,
  )
  new_consts = filter_indices(closed_jaxpr.consts, constvar_indices_to_move)
  return closed_jaxpr.replace(jaxpr=new_jaxpr, consts=new_consts)


def concretize_handlers(
    handler_impl: Mapping[str, Callable[..., Any]],
    effect_op_signatures: MutableMapping[str, EffectOpSignature],
    handler_args: Sequence[jax_core.Atom],
    handler_arg_tree: jax.tree_util.PyTreeDef,
    k_abstract_eval_outvals: Sequence[Aval],
) -> Mapping[str, jax_core.ClosedJaxpr]:
  """Converts a mapping of handler Python functions to handler Jaxprs."""
  # Create dummy handler function arguments.
  handler_arg_zeros = tuple(
      ad.zeros_like_aval(h.aval)
      if isinstance(h, (jax_core.Tracer, jax_core.Var, jax_core.Literal))
      else h
      for h in handler_args
  )

  # Transform handler functions from Python to Jaxprs.
  concrete_handlers_by_name = {}
  for effect_name, handler_fun_raw in handler_impl.items():
    CONSOLE.print(f'[bold green]Concretizing effect: {effect_name}')

    # Warn for unused effects.
    if effect_name not in effect_op_signatures:
      CONSOLE.print(
          f'Warning: handler_impl {effect_name} not used!',
          style='bold on bright_red',
      )
      continue
    effect_op_signature = effect_op_signatures[effect_name]

    if isinstance(handler_fun_raw, jax_core.Jaxpr):
      raise ValueError('Expecting ClosedJaxpr handler implementation only')
    if isinstance(handler_fun_raw, jax_core.ClosedJaxpr):
      concrete_handlers_by_name[effect_name] = handler_fun_raw
      continue

    handler_fun = handler_fun_raw
    # Form handler arguments.
    call_k, call_lk = make_call_k_and_call_lk(
        effect_name,
        abstract_eval_outvals=k_abstract_eval_outvals,
    )
    handler_arg_avals = tuple(v.aval for v in handler_args)
    effect_arg_avals = effect_op_signature.in_avals
    effect_in_tree = effect_op_signature.in_tree
    effect_arg_zero = jax.tree_util.tree_unflatten(
        effect_in_tree, zeros_like_avals(effect_arg_avals)
    )
    handler_arg_zero = jax.tree_util.tree_unflatten(
        handler_arg_tree, zeros_like_avals(handler_arg_avals)
    )
    if handler_arg_avals:
      handler_concrete_args = (handler_arg_zero, effect_arg_zero)
    else:
      handler_concrete_args = (effect_arg_zero,)
    make_handler_jaxpr = jax.make_jaxpr(
        handler_fun,
        static_argnums=(
            len(handler_concrete_args),
            len(handler_concrete_args) + 1,
        ),
    )
    raw_handler_jaxpr = make_handler_jaxpr(
        *handler_concrete_args,
        call_k,
        call_lk,
    )
    handler_jaxpr = move_const_from_tracer_id_list(
        raw_handler_jaxpr, get_tracer_ids(handler_arg_zeros)
    )
    for i in range(len(handler_arg_avals)):
      handler_jaxpr.jaxpr.invars[i].suffix = THETA_SUFFIX
    # TODO(jax-effects-team): Store continuation argument types here?
    concrete_handlers_by_name[effect_name] = handler_jaxpr
    collect_effect_op_signatures(handler_jaxpr.jaxpr, effect_op_signatures)

  return concrete_handlers_by_name


def concretize_delimited_handler_eqn(
    eqn: jax_core.JaxprEqn,
    effect_op_signatures: MutableMapping[str, EffectOpSignature],
) -> jax_core.JaxprEqn:
  """Transforms `delimited_handler_p` handler functions from Python to Jaxprs.

  Args:
    eqn: A `delimited_handler_p` equation where all handler functions are Python
      functions (as originally defined and registered in source code).
    effect_op_signatures: Mapping from effect operation names to signatures.

  Returns:
    A `delimited_handler_p` equation where all handler functions are transformed
    to Jaxprs (strictly speaking, Jaxpr-producing functions).
  """
  if eqn.primitive is not handler.delimited_handler_p:
    raise ValueError(f'Expected delimited_handler_p equation, got: {eqn}')

  # First, recurse on body Jaxpr.
  # Handler implementation may contain nested handlers whose handler
  # implementation may contain nested handlers, etc.
  body_jaxpr = eqn.params['body_jaxpr']
  translated_body_jaxpr = concretize_handlers_transform(
      body_jaxpr, effect_op_signatures
  )

  # Then, concretize handler implementations.
  handler_args = eqn.params['args']
  handler_arg_tree = eqn.params['in_tree']
  handler_impl = eqn.params['handler_impl']
  k_abstract_eval_outvals = tuple(x.aval for x in eqn.outvars)

  concrete_handlers_by_name = concretize_handlers(
      handler_impl=handler_impl,
      effect_op_signatures=effect_op_signatures,
      handler_args=handler_args,
      handler_arg_tree=handler_arg_tree,
      k_abstract_eval_outvals=k_abstract_eval_outvals,
  )

  return eqn.replace(
      params=dict(
          eqn.params,
          handler_impl=concrete_handlers_by_name,
          body_jaxpr=translated_body_jaxpr,
      )
  )


def concretize_handlers_transform(
    jaxpr: jax_core.Jaxpr,
    effect_op_signatures: MutableMapping[str, EffectOpSignature],
) -> jax_core.Jaxpr:
  """"Transforms handler functions from Python to Jaxprs."""
  transformed_eqns = []
  for eqn in jaxpr.eqns:
    if handler.is_handler(eqn.primitive):
      print_jaxpr(eqn, title='Unexpected handler equation')
      raise ValueError(f'Unexpected handler equation: {eqn}')
    if handler.is_delimited_handler(eqn.primitive):
      concretized_eqn = concretize_delimited_handler_eqn(
          eqn, effect_op_signatures
      )
      transformed_eqns.append(
          transform_subjaxprs(
              concretized_eqn,
              lambda jaxpr: concretize_handlers_transform(
                  # Note: applying `delimited_handler_transform` here is crucial
                  # for turning `handler_p` into `delimited_handler_p` in
                  # `handler_impl` dicts before `concretize_handlers_transform`
                  # runs.
                  delimited_handler_transform(jaxpr),
                  effect_op_signatures,
              ),
          )
      )
    else:
      transformed_eqns.append(
          transform_subjaxprs(
              eqn,
              lambda jaxpr: concretize_handlers_transform(
                  jaxpr, effect_op_signatures
              ),
          )
      )
  return jaxpr.replace(eqns=transformed_eqns)


################################################################################
# Explicit parameterized effects
################################################################################


def explicit_parameterized_effect_transform_primitive(
    primitive: jax_core.Primitive,
    handler_arg_count: int,
    handler_state_outvals: Sequence[Aval],
) -> jax_core.Primitive:
  """Transforms effect op primitive to take and return state explicitly."""
  primitive_type = type(primitive)
  transformed_primitive = primitive_type(f'{primitive.name}_with_state')
  transformed_primitive.multiple_results = True
  transformed_primitive.call_primitive = primitive.call_primitive
  transformed_primitive.map_primitive = primitive.map_primitive

  @transformed_primitive.def_effectful_abstract_eval
  def _(*args, **kwargs):
    true_args = args[handler_arg_count:]
    outvals, effects = primitive.abstract_eval(*true_args, **kwargs)
    if not isinstance(outvals, Sequence):
      outvals = [outvals]
    return list(handler_state_outvals) + list(outvals), effects

  return transformed_primitive


@dataclasses.dataclass
class HandlerStack:
  """A handler of handler functions."""
  gensym: GensymFn
  handler_eqn_stack: list[jax_core.JaxprEqn] = dataclasses.field(
      default_factory=list
  )
  handler_args_by_handler_eqn: dict[int, list[jax_core.Var]] = (
      dataclasses.field(default_factory=dict)
  )

  def push_handler(self, handler_eqn: jax_core.JaxprEqn) -> 'HandlerStack':
    assert handler.is_delimited_handler(handler_eqn.primitive)
    handler_args: Sequence[jax_core.Atom] = handler_eqn.params['args']
    body_jaxpr: jax_core.ClosedJaxpr = handler_eqn.params['body_jaxpr']
    body_args = body_jaxpr.invars[: len(handler_args)]
    new_handler_eqn_stack = [handler_eqn] + self.handler_eqn_stack
    # NOTE: Explicitly share handler arguments between handler stacks.
    # TODO(jax-effects-team): Is this safe?
    new_handler_args_by_handler_eqn = self.handler_args_by_handler_eqn
    new_handler_args_by_handler_eqn[id(handler_eqn)] = body_args
    return HandlerStack(
        gensym=self.gensym,
        handler_eqn_stack=new_handler_eqn_stack,
        handler_args_by_handler_eqn=new_handler_args_by_handler_eqn,
    )

  def get_handler_state_vars(
      self, handler_eqn: jax_core.JaxprEqn
  ) -> list[jax_core.Var]:
    # TODO(jax-effects-team): Consider optimizing potentially expensive check
    # below.
    assert handler_eqn in self.handler_eqn_stack
    return self.handler_args_by_handler_eqn[id(handler_eqn)]

  def set_handler_state_vars(
      self,
      handler_eqn: jax_core.JaxprEqn,
      handler_state_vars: Sequence[jax_core.Var],
  ) -> None:
    assert handler.is_delimited_handler(handler_eqn.primitive)
    # TODO(jax-effects-team): Consider optimizing potentially expensive check
    # below.
    assert handler_eqn in self.handler_eqn_stack
    self.handler_args_by_handler_eqn[id(handler_eqn)] = handler_state_vars

  def get_and_update_handler_state_vars(
      self, handler_eqn: jax_core.JaxprEqn
  ) -> tuple[list[jax_core.Var], list[jax_core.Var]]:
    # TODO(jax-effects-team): Consider optimizing potentially expensive check
    # below.
    assert handler_eqn in self.handler_eqn_stack
    handler_state_invars = self.handler_args_by_handler_eqn[id(handler_eqn)]
    handler_state_outvars = [self.gensym(v.aval) for v in handler_state_invars]
    self.handler_args_by_handler_eqn[id(handler_eqn)] = handler_state_outvars
    return handler_state_invars, handler_state_outvars

  def get_defining_handler(self, effect_name: str) -> jax_core.JaxprEqn:
    defining_handler_eqn = None
    for handler_eqn in self.handler_eqn_stack:
      if effect_name in handler_eqn.params['handler_impl']:
        defining_handler_eqn = handler_eqn
        break
    assert defining_handler_eqn is not None
    return defining_handler_eqn


def explicit_effect_parameter_passing_effect_op_eqn(
    body_eqn: jax_core.JaxprEqn,
    handler_stack: HandlerStack,
) -> jax_core.JaxprEqn:
  """Transforms an effect op equation to do explicit effect parameter passing.

  Args:
    body_eqn: An effect op equation.
    handler_stack: The handler handler.

  Returns:
    A modified version of `body_eqn` where parameter state is passed explicitly.
  """
  effect_name = body_eqn.params['name']
  CONSOLE.print(
      '[bold green]Explicit effect-parameter passing for: '
      f'[on white]{effect_name}'
  )

  defining_handler_eqn = handler_stack.get_defining_handler(effect_name)
  handler_state_invars, handler_state_outvars = (
      handler_stack.get_and_update_handler_state_vars(defining_handler_eqn)
  )

  new_invars = handler_state_invars + body_eqn.invars
  new_outvars = handler_state_outvars + body_eqn.outvars
  new_primitive = explicit_parameterized_effect_transform_primitive(
      body_eqn.primitive,
      handler_arg_count=len(handler_state_invars),
      handler_state_outvals=[v.aval for v in handler_state_outvars],
  )
  return body_eqn.replace(
      invars=new_invars, outvars=new_outvars, primitive=new_primitive
  )


def explicit_effect_parameter_passing_effect_op_eqns(
    body_eqns: Sequence[jax_core.JaxprEqn],
    handler_stack: HandlerStack,
) -> Sequence[jax_core.JaxprEqn]:
  """Transforms a sequence of equations to do explicit effect parameter passing.

  Args:
    body_eqns: An sequence of Jaxpr equations.
    handler_stack: The handler handler.

  Returns:
    A modified version of `body_eqns` where parameter state is passed
    explicitly.
  """
  new_body_eqns = []
  for body_eqn in body_eqns:
    body_eqn: jax_core.JaxprEqn
    if handler.is_delimited_handler(body_eqn.primitive):
      new_body_eqns.append(
          explicit_effect_parameter_passing_handler_eqn(
              body_eqn,
              handler_stack=handler_stack,
          )
      )
    elif is_effect_op(body_eqn.primitive):
      new_body_eqn = explicit_effect_parameter_passing_effect_op_eqn(
          body_eqn,
          handler_stack=handler_stack,
      )
      new_body_eqns.append(new_body_eqn)
    else:
      new_body_eqn = transform_subjaxprs(
          body_eqn,
          lambda jaxpr: explicit_handler_parameter_passing_transform(
              jaxpr, handler_stack
          ),
      )
      new_body_eqns.append(body_eqn)
  return new_body_eqns


def explicit_effect_parameter_passing_handler_eqn(
    eqn: jax_core.JaxprEqn,
    handler_stack: HandlerStack,
) -> jax_core.JaxprEqn:
  """Transforms `delimited_handler` handler functions from Python to Jaxprs.

  Args:
    eqn: A parameterized `delimited_handler` equation where effect operations in
      the body take parameter state implicitly.
    handler_stack: The handler handler.

  Returns:
    A `delimited_handler` equation where effect operations in the body take
    parameter state explicitly.
  """
  if eqn.primitive is not handler.delimited_handler_p:
    raise ValueError(f'Expected delimited_handler equation, got: {eqn}')

  body_jaxpr: jax_core.Jaxpr = eqn.params['body_jaxpr']
  handler_impl = eqn.params['handler_impl']
  parameterized: bool = eqn.params['parameterized']

  handler_stack = handler_stack.push_handler(eqn)
  gensym = handler_stack.gensym

  if not parameterized:
    return eqn

  mapping = {}

  def substitute(v: jax_core.Atom) -> jax_core.Var:
    if isinstance(v, jax_core.Literal):
      return v
    result = mapping.get(v, v)
    return result if result == v else substitute(result)

  new_body_eqns = []
  for body_eqn in body_jaxpr.eqns:
    body_eqn: jax_core.JaxprEqn
    if handler.is_delimited_handler(body_eqn.primitive):
      new_body_eqns.append(
          explicit_effect_parameter_passing_handler_eqn(
              body_eqn,
              handler_stack=handler_stack,
          )
      )
    elif is_effect_op(body_eqn.primitive):
      new_body_eqn = explicit_effect_parameter_passing_effect_op_eqn(
          body_eqn,
          handler_stack=handler_stack,
      )
      new_body_eqns.append(new_body_eqn)
    elif handler.is_handler_return(body_eqn.primitive):
      args = eqn.params['args']
      handler_args = body_jaxpr.invars[: len(args)]
      handler_arg_count = len(handler_args)

      new_invars = handler_args + body_eqn.invars
      new_outvars = body_eqn.outvars
      new_primitive = explicit_parameterized_effect_transform_primitive(
          body_eqn.primitive,
          handler_arg_count=handler_arg_count,
          handler_state_outvals=[],
      )
      new_body_eqn = body_eqn.replace(
          invars=new_invars, outvars=new_outvars, primitive=new_primitive
      )
      new_body_eqns.append(new_body_eqn)
    else:
      new_body_eqns.append(substitute_eqn(body_eqn, substitute, gensym))
  new_body_jaxpr = body_jaxpr.replace(
      outvars=[substitute(v) for v in body_jaxpr.outvars],
      eqns=new_body_eqns,
  )

  # Transform handlers to do explicit parameter passing.
  new_handler_impl = {}
  for handler_name, handler_jaxpr in handler_impl.items():
    new_handler_jaxpr_eqns = explicit_effect_parameter_passing_effect_op_eqns(
        handler_jaxpr.eqns,
        handler_stack=handler_stack,
    )
    new_handler_jaxpr = handler_jaxpr.map_jaxpr(
        lambda jaxpr: jaxpr.replace(eqns=new_handler_jaxpr_eqns)
    )
    new_handler_impl[handler_name] = new_handler_jaxpr

  return eqn.replace(
      params=dict(
          eqn.params,
          body_jaxpr=new_body_jaxpr,
          handler_impl=new_handler_impl,
      )
  )


def explicit_handler_parameter_passing_transform(
    jaxpr: jax_core.Jaxpr,
    handler_stack: HandlerStack | None = None,
) -> jax_core.Jaxpr:
  """Transforms Jaxpr so that effect operations use explicit parameter state."""
  gensym = jax_core.gensym([jaxpr], suffix=THETA_SUFFIX)
  if handler_stack is None:
    handler_stack = HandlerStack(gensym=gensym)
  transformed_eqns = []
  for eqn in jaxpr.eqns:
    if eqn.primitive is handler.delimited_handler_p:
      transformed_eqns.append(
          explicit_effect_parameter_passing_handler_eqn(
              eqn,
              handler_stack=handler_stack,
          )
      )
    else:
      transformed_eqns.append(
          transform_subjaxprs(
              eqn,
              lambda jaxpr: explicit_handler_parameter_passing_transform(
                  jaxpr, handler_stack
              ),
          )
      )
  return jaxpr.replace(eqns=transformed_eqns)


################################################################################
# Effect-handling transformation
################################################################################


def check_call_eqn(eqn: jax_core.JaxprEqn) -> bool:
  call_jaxpr = eqn.params['call_jaxpr']
  if len(eqn.invars) != len(call_jaxpr.jaxpr.invars):
    raise ValueError(
        f'call primitive argument count mismatch: {len(eqn.invars)} (eqn) vs '
        f'{len(call_jaxpr.jaxpr.invars)} (Jaxpr)'
    )
  if len(eqn.outvars) != len(call_jaxpr.jaxpr.outvars):
    raise ValueError(
        f'call primitive result count mismatch: {len(eqn.outvars)} (eqn) vs '
        f'{len(call_jaxpr.jaxpr.outvars)} (Jaxpr)'
    )


def effect_operation_lowering_transform(
    handler_stack: HandlerStack,
    body_eqns: Sequence[jax_core.JaxprEqn],
    parameterized: bool,
    body_outvars: Sequence[jax_core.Atom],
    effects,
    debug_info,
    handler_fns_by_name: Mapping[str, jax_core.Jaxpr],
    gensym,
    ctx,
    global_mapping: dict[jax_core.Atom, jax_core.Atom],
) -> tuple[Sequence[jax_core.JaxprEqn], Sequence[jax_core.Atom]]:
  """Returns new equations and outvars after effect operation lowering."""
  handler_eqn_stack = handler_stack.handler_eqn_stack
  for handler_eqn in handler_eqn_stack:
    assert handler.is_delimited_handler(handler_eqn.primitive)
  assert len(handler_eqn_stack) == len(
      set(id(eqn) for eqn in handler_eqn_stack)
  )

  this_handler = handler_eqn_stack[0]
  this_handler_impl = this_handler.params['handler_impl']
  this_handler_arg_count = len(this_handler.params['args'])

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    if isinstance(v, jax_core.Literal):
      return v
    result = global_mapping.get(v, v)
    return result if result == v else substitute(result)

  unhandled_effect_op_eqns = []
  new_body_eqns = []
  for i, body_eqn in enumerate(body_eqns):
    body_eqn: jax_core.JaxprEqn
    if is_effect_op(body_eqn.primitive):
      effect_name = body_eqn.params['name']
      CONSOLE.print(effect_name, style='bold on bright_magenta')

      # For effects not handled by this handler, keep them untransformed.
      if effect_name not in this_handler_impl:
        unhandled_effect_op_eqns.append(body_eqn)
        new_body_eqn = substitute_eqn(
            body_eqn, substitute, gensym, free_vars=()
        )
        new_body_eqns.append(new_body_eqn)
        continue

      # Otherwise, transform the effect operation into a handler call.
      handler_jaxpr: jax_core.ClosedJaxpr = handler_fns_by_name[effect_name]

      # TODO(jax-effects-team): Avoid cloning the first time as an optimization.
      handler_jaxpr = handler_jaxpr.map_jaxpr(
          lambda jaxpr: clone_jaxpr(
              jaxpr,
              gensym,
              free_vars=free_vars_jaxpr(jaxpr),
          )
      )

      # Construct continuation, to be used in handler calls.
      has_call_k_eqn = any(
          is_call_k(eqn.primitive) or is_call_lk(eqn.primitive)
          for eqn in subjaxpr_eqns(handler_jaxpr.eqns)
      )
      if not has_call_k_eqn:
        lowered_handler_jaxpr = handler_jaxpr
      else:
        k_invars = body_eqn.outvars
        k_invars_new = [gensym(v.aval) for v in k_invars]
        for old_var, new_var in jax_util.safe_zip(k_invars, k_invars_new):
          new_var.suffix = old_var.suffix
        k_mapping = {key: value for key, value in zip(k_invars, k_invars_new)}
        k_mapping.update(global_mapping)

        def k_substitute(v: jax_core.Atom) -> jax_core.Atom:
          if isinstance(v, jax_core.Literal):
            return v
          return k_mapping.get(v, v)

        k_eqns = body_eqns[i + 1 :]
        k_eqns = substitute_eqns(k_eqns, k_substitute, gensym)
        global_mapping.update(k_mapping)
        k_new_eqns, k_new_outvars = effect_operation_lowering_transform(
            handler_stack=handler_stack,
            body_eqns=k_eqns,
            parameterized=parameterized,
            body_outvars=body_outvars,
            effects=effects,
            debug_info=debug_info,
            handler_fns_by_name=handler_fns_by_name,
            gensym=gensym,
            ctx=ctx,
            global_mapping=global_mapping,
        )

        global_mapping.update(k_mapping)

        # If there is a nested continuation, forward results from inner
        # continuation.
        if (
            len(k_new_eqns) >= 1
            and k_new_eqns[-1].primitive == jax_core.closed_call_p
        ):
          k_outvars_final = k_new_eqns[-1].outvars
          k_outvars_final = [k_substitute(v) for v in k_outvars_final]
        else:
          k_outvars = k_new_outvars
          k_outvars_new = [k_substitute(v) for v in k_outvars]
          k_outvars_final = k_outvars_new

        k_jaxpr = jax_core.Jaxpr(
            constvars=(),
            invars=k_invars_new,
            outvars=k_outvars_final,
            eqns=k_new_eqns,
            effects=effects,
            debug_info=debug_info,
        )
        k_jaxpr_free_vars = free_vars_jaxpr(k_jaxpr)

        def lower_handler_jaxpr(
            handler_jaxpr: jax_core.Jaxpr,
            k_jaxpr: jax_core.Jaxpr,
        ) -> jax_core.Jaxpr:
          # Clone handler body.
          lowered_handler_eqns = []
          first_call_k = True
          for handler_eqn in handler_jaxpr.eqns:
            handler_eqn: jax_core.JaxprEqn
            if is_call_k(handler_eqn.primitive):
              call_k_invars = handler_eqn.invars
              call_k_outvars = handler_eqn.outvars
              if not first_call_k:
                k_jaxpr = clone_jaxpr(
                    k_jaxpr, gensym, free_vars=k_jaxpr_free_vars
                )
              call_k_eqn = jax_core.new_jaxpr_eqn(
                  invars=call_k_invars,
                  outvars=call_k_outvars,
                  primitive=jax_core.closed_call_p,
                  params=dict(
                      call_jaxpr=jax_core.ClosedJaxpr(k_jaxpr, consts=())
                  ),
                  effects=k_jaxpr.effects,
                  source_info=body_eqn.source_info,
              )
              first_call_k = False
              check_call_eqn(call_k_eqn)
              lowered_handler_eqns.append(call_k_eqn)
            elif is_call_lk(handler_eqn.primitive):
              raise ValueError(
                  'call_lk should not exist after loss transformation'
              )
            elif handler_eqn.primitive in staged_transform_primitives:
              lowered_handler_eqn = transform_subjaxprs(
                  handler_eqn,
                  lambda jaxpr: lower_handler_jaxpr(jaxpr, k_jaxpr=k_jaxpr),
              )
              if lowered_handler_eqn.primitive is linear_transpose_staged_p:
                call_transpose_eqn = lower_linear_transpose_staged(
                    lowered_handler_eqn, ctx
                )
                lowered_handler_eqns.append(call_transpose_eqn)
              else:
                raise ValueError(f'staged_transform_primitive: {handler_eqn}')
            else:
              lowered_handler_eqns.append(handler_eqn)

          return handler_jaxpr.replace(
              eqns=lowered_handler_eqns,
              effects=handler_jaxpr.effects | effects,
          )

        lowered_handler_jaxpr = handler_jaxpr.map_jaxpr(
            lambda jaxpr: lower_handler_jaxpr(jaxpr, k_jaxpr)
        )
        global_mapping.update(k_mapping)

      # Construct handler call.
      lowered_effect_op_invars = [substitute(v) for v in body_eqn.invars]
      lowered_effect_op_outvars = [
          gensym(v.aval) for v in lowered_handler_jaxpr.jaxpr.outvars
      ]
      lowered_effect_op = jax_core.new_jaxpr_eqn(
          invars=lowered_effect_op_invars,
          outvars=lowered_effect_op_outvars,
          primitive=jax_core.closed_call_p,
          params=dict(call_jaxpr=lowered_handler_jaxpr),
          effects=lowered_handler_jaxpr.effects,
          source_info=body_eqn.source_info,
      )
      check_call_eqn(lowered_effect_op)
      # Replace effect op with a continuation call.
      new_body_eqns.append(lowered_effect_op)
      # Directly return lowered effect op outvars as final outvars.
      new_body_outvars = lowered_effect_op.outvars
      return new_body_eqns, new_body_outvars
    elif handler.is_delimited_handler(body_eqn.primitive):
      raise ValueError(
          'All delimited_handler eqns should have been transformed away'
      )
    elif is_call_k(body_eqn.primitive):
      raise ValueError('All call_k eqns should have been transformed away')
    elif handler.is_handler_return(body_eqn.primitive):
      # NOTE(jax-effects-team): Enabling this assertion requires filtering out
      # loss accumulation equations.
      # assert i == len(body_eqns) - 1, (
      #     f'Handler return should be last equation in handler body: {i} vs'
      #     f' {len(body_eqns)}'
      # )
      handler_final_state_vars = handler_stack.get_handler_state_vars(
          this_handler
      )
      handler_return_new_invars = (
          handler_final_state_vars + body_eqn.invars[this_handler_arg_count:]
      )
      new_body_eqn = body_eqn.replace(invars=handler_return_new_invars)
      inlined_eqns, _ = inline_closed_call_eqn(
          new_body_eqn, ctx, global_mapping, gensym
      )
      new_body_eqns.extend(inlined_eqns)
    else:
      new_body_eqn = substitute_eqn(
          body_eqn, substitute, gensym=gensym, free_vars=()
      )
      new_body_eqns.append(new_body_eqn)
  final_body_outvars = [substitute(v) for v in body_outvars]
  return new_body_eqns, final_body_outvars


def subjaxpr_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
) -> Iterable[jax_core.JaxprEqn]:
  """Yields recursively all equations in `eqns`, including from subjaxprs."""
  for eqn in eqns:
    yield eqn
    for subjaxpr in jax_core.jaxprs_in_params(eqn.params):
      yield from subjaxpr_eqns(subjaxpr.eqns)


def delimited_handler_lowering_transform(
    eqn: jax_core.JaxprEqn,
    handler_stack: HandlerStack,
    ctx: jax_core.JaxprPpContext,
    handler_fns_by_name: Mapping[str, jax_core.Jaxpr] | None,
    global_mapping: dict[jax_core.Atom, jax_core.Atom],
    gensym: GensymFn,
) -> tuple[Sequence[jax_core.JaxprEqn], Sequence[jax_core.Atom]]:
  """Effect-handling transformation for `delimited_handler` equation.

  Args:
    eqn: A `delimited_handler` Jaxpr equation.
    handler_stack: The handler handler.
    ctx: Pretty-printing context for debugging purposes.
    handler_fns_by_name: A mapping from handler name to handler function.
    global_mapping: Mapping from old variables to new ones.
    gensym: A function that generates a new variable name.

  Returns:
    Lowered Jaxpr equations and outvars.
  """
  assert handler.is_delimited_handler(eqn.primitive)
  body_jaxpr: jax_core.Jaxpr = eqn.params['body_jaxpr']
  name = eqn.params['name']
  args = eqn.params['args']
  handler_arg_count = len(args)
  handler_body_args = body_jaxpr.invars[:handler_arg_count]
  handler_impl = handler_fns_by_name or {}
  handler_impl |= eqn.params['handler_impl']
  parameterized = eqn.params['parameterized']

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    if isinstance(v, jax_core.Literal):
      return v
    result = global_mapping.get(v, v)
    return result if result == v else substitute(result)

  for old_invar, new_invar in jax_util.safe_zip(body_jaxpr.invars, eqn.invars):
    global_mapping[old_invar] = new_invar

  free_vars_body = free_vars_jaxpr(body_jaxpr)

  # Handle all nested handler operations.
  body_eqns = []
  for body_eqn in body_jaxpr.eqns:
    if handler.is_delimited_handler(body_eqn.primitive):
      new_lowered_handler_eqns, new_handler_outvars = (
          delimited_handler_lowering_transform(
              eqn=body_eqn,
              handler_stack=handler_stack,
              ctx=ctx,
              handler_fns_by_name=handler_fns_by_name,
              global_mapping=global_mapping,
              gensym=gensym,
          )
      )
      # Inline lowered equations.
      new_lowered_handler_eqns, _, _ = (
          inline_closed_call_eqns(
              new_lowered_handler_eqns,
              ctx,
              gensym=gensym,
              mapping=global_mapping,
              free_vars=free_vars_body,
          )
      )
      new_lowered_handler_eqns = [
          substitute_eqn(eqn, substitute, gensym, free_vars=free_vars_body)
          for eqn in new_lowered_handler_eqns
      ]
      new_handler_outvars = [substitute(v) for v in new_handler_outvars]
      # Update mapping for handled equations.
      for old_outvar, new_outvar in jax_util.safe_zip(
          body_eqn.outvars, new_handler_outvars
      ):
        if isinstance(old_outvar, jax_core.Literal):
          continue
        global_mapping[old_outvar] = new_outvar
      for old_invar, new_invar in jax_util.safe_zip(
          body_eqn.params['body_jaxpr'].invars, body_eqn.invars
      ):
        global_mapping[old_invar] = new_invar
      body_eqns.extend(new_lowered_handler_eqns)
    else:
      body_eqn = substitute_eqn(
          body_eqn, substitute, gensym=gensym, free_vars=free_vars_body
      )
      body_eqns.append(body_eqn)

  new_handler_stack = handler_stack.push_handler(eqn)
  if parameterized:
    handler_state_outvars = handler_body_args
    for body_eqn in reversed(body_eqns):
      if not is_effect_op(body_eqn.primitive):
        continue
      effect_name = body_eqn.params['name']
      if effect_name not in eqn.params['handler_impl']:
        continue
      # Found effect operation handled by the current handler.
      handler_state_outvars = body_eqn.outvars[:handler_arg_count]
      if len(handler_state_outvars) != len(handler_body_args):
        print_jaxpr(jax_core.pp_eqn(body_eqn, ctx, PP_SETTINGS))
        raise ValueError('Effect op does not have correct handler parameters')
      break
    # Set handler state outvars for return.
    new_handler_stack.set_handler_state_vars(eqn, handler_state_outvars)

  new_body_eqns, new_body_outvars = effect_operation_lowering_transform(
      handler_stack=new_handler_stack,
      body_eqns=body_eqns,
      parameterized=parameterized,
      body_outvars=body_jaxpr.outvars,
      effects=body_jaxpr.effects,
      debug_info=body_jaxpr.debug_info,
      handler_fns_by_name=handler_impl,
      gensym=gensym,
      ctx=ctx,
      global_mapping=global_mapping,
  )
  if len(new_body_outvars) != len(body_jaxpr.outvars):
    raise ValueError(
        f'new_body_outvars ({len(new_body_outvars)}) != '
        f'body_jaxpr.outvars ({len(body_jaxpr.outvars)})'
    )

  delimited_handler_invars = body_jaxpr.invars
  if new_body_eqns:
    if new_body_eqns[-1].primitive is not jax_core.closed_call_p:
      # NOTE: This code path only triggers for interleaved effects, when
      # delimited_handler contains no effect ops (handle-able by the current
      # handler).
      print_jaxpr(
          jax_core.pp_eqns(new_body_eqns, ctx, PP_SETTINGS),
          title=(
              f'[bold red]delimited_handler "{name}" body does not end in '
              'closed_call'
          ),
      )
      print_jaxpr(
          jax_core.pp_vars(new_body_outvars, ctx),
          title=f'[bold red]delimited_handler new_body_outvars for {name}',
      )

      assert any(is_effect_op(e.primitive) for e in new_body_eqns), (
          'Expect this case when there is at least one effect op not '
          'handleable by this handler'
      )
      return new_body_eqns, new_body_outvars

    call_handler_eqn = new_body_eqns[-1]
    assert call_handler_eqn.primitive is jax_core.closed_call_p
    delimited_handler_outvars = new_body_outvars
  else:
    delimited_handler_outvars = new_body_outvars
  new_body_jaxpr = body_jaxpr.replace(
      invars=delimited_handler_invars,
      outvars=delimited_handler_outvars,
      eqns=new_body_eqns,
  )
  lowered_delimited_handler_call = eqn.replace(
      primitive=jax_core.closed_call_p,
      params=dict(call_jaxpr=jax_core.ClosedJaxpr(new_body_jaxpr, consts=())),
      effects=body_jaxpr.effects,
  )

  check_call_eqn(lowered_delimited_handler_call)
  final_outvars = lowered_delimited_handler_call.outvars
  assert len(final_outvars) == len(eqn.outvars)
  return [lowered_delimited_handler_call], final_outvars


def effect_handling_transform(
    jaxpr: jax_core.Jaxpr,
    ctx: jax_core.JaxprPpContext | None = None,
    diagnose_unhandled_effects: bool = True,
) -> jax_core.Jaxpr:
  """Effect-handling transformation.

  Args:
    jaxpr: Jaxpr with effect operations within `delimited_handler` scope.
    ctx: Pretty-printing context for debugging purposes.
    diagnose_unhandled_effects: If True, diagnose unhandled effect operations in
      `jaxpr`.

  Returns:
    A Jaxpr with all effect operations handled.
  """

  if ctx is None:
    ctx = pp_ctx_factory(jaxpr)

  gensym = jax_core.gensym([jaxpr], suffix="'")

  new_eqns = []
  new_outvars = jaxpr.outvars
  mapping = {}

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    if isinstance(v, jax_core.Literal):
      return v
    result = mapping.get(v, v)
    return result if result == v else substitute(result)

  for eqn in jaxpr.eqns:
    if eqn.primitive is handler.delimited_handler_p:
      handler_impls = eqn.params['handler_impl']
      for k, handler_impl in handler_impls.items():
        handled_impl = handler_impl.map_jaxpr(
            lambda x: effect_handling_transform(x, ctx)
        )
        handler_impls[k] = handled_impl
      eqn = eqn.replace(params=dict(eqn.params, handler_impl=handler_impls))
      lowered_eqns, lowered_eqn_outvars = delimited_handler_lowering_transform(
          eqn=eqn,
          handler_stack=HandlerStack(gensym=gensym),
          ctx=ctx,
          handler_fns_by_name=None,
          global_mapping=mapping,
          gensym=gensym,
      )
      assert len(lowered_eqn_outvars) == len(eqn.outvars)
      for old_outvar, new_outvar in jax_util.safe_zip(
          eqn.outvars, lowered_eqn_outvars
      ):
        mapping[old_outvar] = new_outvar
      new_eqns.extend(lowered_eqns)
    else:
      new_eqn = transform_subjaxprs(
          eqn,
          lambda jaxpr: effect_handling_transform(
              jaxpr, ctx, diagnose_unhandled_effects=False
          ),
      )
      new_eqns.append(new_eqn)

  unhandled_effect_op_eqns = [
      eqn for eqn in new_eqns if is_effect_op(eqn.primitive)
  ]
  if diagnose_unhandled_effects and unhandled_effect_op_eqns:
    srcs = [
        source_info_util.summarize(eqn.source_info)
        for eqn in unhandled_effect_op_eqns
    ]
    src = '\n'.join(srcs)
    effect_names = [eqn.params['name'] for eqn in unhandled_effect_op_eqns]
    CONSOLE.print(
        f'There are unhandled effect operations: {effect_names}. '
        'A Handler value defining these effects must be provided.\n'
        f'From sources:\n{src}',
        style='bold on red',
    )
    raise ValueError(
        f'There are unhandled effect operations: {effect_names}. '
        'A Handler value defining these effects must be provided.\n'
        f'From sources:\n{src}'
    )

  new_outvars = [substitute(v) for v in new_outvars]
  return jaxpr.replace(eqns=new_eqns, outvars=new_outvars)


################################################################################
# Inlining
################################################################################


def inline_closed_call_eqn(
    call_eqn: jax_core.JaxprEqn,
    parent_ctx: jax_core.JaxprPpContext,
    mapping: MutableMapping[jax_core.Var, jax_core.Atom],
    gensym: GensymFn,
    free_vars: Sequence[jax_core.Var] = (),
) -> tuple[
    Sequence[jax_core.JaxprEqn], Callable[[jax_core.Atom], jax_core.Atom]
]:
  """Inlines a closed_call equation."""
  del parent_ctx

  if 'call_jaxpr' in call_eqn.params:
    call_jaxpr = call_eqn.params['call_jaxpr']
  elif handler.is_handler_return(call_eqn.primitive):
    call_jaxpr = call_eqn.params['return_fun_jaxpr']
  else:
    raise ValueError(
        f'Unrecognized call equation type cannot be inlined: {call_eqn}'
    )
  assert isinstance(call_jaxpr, jax_core.ClosedJaxpr)
  source_constvars = call_jaxpr.jaxpr.constvars
  target_consts = call_jaxpr.consts
  source_invars = call_eqn.invars
  target_invars = call_jaxpr.jaxpr.invars
  source_outvars = call_eqn.outvars
  target_outvars = call_jaxpr.jaxpr.outvars

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    # Handle literals and constant values.
    if not isinstance(v, jax_core.Var):
      return v
    result = mapping.get(v, v)
    return result if result == v else substitute(result)

  substitute.mapping = mapping

  for s, t in jax_util.safe_zip(source_constvars, target_consts):
    literal = jax_core.Literal(val=t, aval=s.aval)
    mapping[s] = literal

  if len(source_invars) != len(target_invars):
    raise ValueError(
        'inline_closed_call_eqn: invars length mismatch, '
        f'{len(source_invars)} (eqn) vs {len(target_invars)} (Jaxpr)'
    )
  for s, t in jax_util.safe_zip(source_invars, target_invars):
    mapping[t] = s

  new_eqns = substitute_eqns(
      call_jaxpr.eqns,
      substitute,
      gensym=gensym,
      free_vars=list(free_vars) + call_jaxpr.jaxpr.constvars,
  )

  for s, t in jax_util.safe_zip(source_outvars, target_outvars):
    mapping[s] = t

  return new_eqns, substitute


def inline_closed_call_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
    ctx: jax_core.JaxprPpContext,
    gensym: GensymFn,
    mapping: MutableMapping[jax_core.Var, jax_core.Atom] | None = None,
    free_vars: Sequence[jax_core.Var] = (),
) -> tuple[
    Sequence[jax_core.JaxprEqn],
    jax_core.Effects,
    Callable[[jax_core.Atom], jax_core.Atom],
]:
  """Inlines all closed_call equations in `eqns`."""
  if mapping is None:
    mapping = {}

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    if isinstance(v, jax_core.Literal):
      return v
    result = mapping.get(v, v)
    return result if result == v else substitute(result)

  new_eqns = []
  new_effects = set()
  for eqn in eqns:
    eqn_free_vars = list(free_vars)
    foreach_subjaxprs(
        eqn, lambda jaxpr: eqn_free_vars.extend(free_vars_jaxpr(jaxpr))
    )

    if eqn.primitive is jax_core.closed_call_p:
      new_call_jaxpr = eqn.params['call_jaxpr'].map_jaxpr(
          lambda jaxpr: inline_closed_call(
              jaxpr, ctx, mapping, free_vars=eqn_free_vars
          ),
      )
      new_effects |= new_call_jaxpr.effects
      new_eqn = eqn.replace(params=dict(eqn.params, call_jaxpr=new_call_jaxpr))
      inlined_eqns, _ = inline_closed_call_eqn(
          new_eqn, ctx, mapping, gensym, free_vars=eqn_free_vars
      )
      new_eqns.extend(inlined_eqns)
    else:
      new_eqn = transform_subjaxprs(
          eqn,
          lambda jaxpr: inline_closed_call(
              jaxpr, ctx, mapping, free_vars=eqn_free_vars
          ),
      )
      new_eqn = substitute_eqn(new_eqn, substitute, gensym, free_vars=free_vars)
      new_eqns.append(new_eqn)

  return new_eqns, new_effects, substitute


def inline_closed_call(
    jaxpr: jax_core.Jaxpr,
    ctx: jax_core.JaxprPpContext | None = None,
    mapping: MutableMapping[jax_core.Var, jax_core.Atom] | None = None,
    free_vars: Sequence[jax_core.Var] = (),
) -> jax_core.Jaxpr:
  """Transforms a Jaxpr by recursively inlining all closed_call equations."""
  if ctx is None:
    ctx = pp_ctx_factory(jaxpr)

  if mapping is None:
    mapping = {}

  def substitute(v: jax_core.Atom) -> jax_core.Atom:
    if isinstance(v, jax_core.Literal):
      return v
    result = mapping.get(v, v)
    return result if result == v else substitute(result)

  gensym = jax_core.gensym([jaxpr], suffix='_inline')
  new_eqns, new_effects, _ = inline_closed_call_eqns(
      eqns=jaxpr.eqns,
      ctx=ctx,
      gensym=gensym,
      mapping=mapping,
      free_vars=free_vars,
  )

  new_jaxpr = jaxpr.replace(
      eqns=new_eqns,
      outvars=[substitute(v) for v in jaxpr.outvars],
      effects=new_effects | jaxpr.effects,
  )
  return new_jaxpr


class JaxprTransformation(abc.ABC):
  """A Jaxpr transformation."""

  @classmethod
  @abc.abstractmethod
  def name(cls) -> str:
    """Returns the transformation name."""

  @abc.abstractmethod
  def transform_jaxpr(self, jaxpr: jax_core.Jaxpr) -> jax_core.Jaxpr:
    """Transforms a Jaxpr."""

  def split_results(self, outvals: Sequence[Any]) -> tuple[Any, Sequence[Any]]:
    """Splits transformed Jaxpr outputs into auxiliary and original results."""
    return None, outvals

  def combine_results(self, aux: Any, out_tree: Any) -> Any:
    """Transforms effect-handling transformation results."""
    if aux is not None:
      raise ValueError(
          f'Unexpected non-None auxiliary value: {aux}. '
          'Consider overriding `finalize_results`.'
      )
    return out_tree
