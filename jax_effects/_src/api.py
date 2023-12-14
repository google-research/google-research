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

"""jax-effects APIs.

This file defines the main APIs for jax-effects, including:
- @effect: Decorator for declaring effect operations.
- Handler, ParameterizedHandler: Effect handler registration.
- effectify: Effect-handling transformation.
"""

from collections.abc import Sequence
import contextlib
import copy
import functools
import os
from typing import Callable, Iterator, ParamSpec, TypeVar

import jax
from jax import api_util
from jax import core as jax_core
from jax.extend import linear_util as lu

from jax_effects._src import choix
from jax_effects._src import core
from jax_effects._src import handler
from jax_effects._src import logging_utils

PP_SETTINGS = core.PP_SETTINGS
Handler = core.Handler
ParameterizedHandler = core.ParameterizedHandler

print_jaxpr = logging_utils.print_jaxpr
print_panel = logging_utils.print_panel

################################################################################
# General effect registration
################################################################################

Aval = jax_core.AbstractValue
# Function from input AbstractValues to output AbstractValue(s).
AbstractEvalFn = Callable[Ellipsis, Aval | Sequence[Aval]]
GensymFn = Callable[[Aval], jax_core.Var]
GensymFnVar = Callable[[jax_core.Atom], jax_core.Atom]

effect = core.effect
Aval = core.Aval
Handler = core.Handler
JaxprTransformation = core.JaxprTransformation

loss = choix.loss


################################################################################
# Transformation API
################################################################################


T = TypeVar('T')
P = ParamSpec('P')


def redirect_output(hide_output = True):
  """Returns a decorator that redirects stdout and stderr to /dev/null."""

  def decorator(f):
    def wrapper_fun(*args, **kwargs):
      with open(os.devnull, 'w') as devnull:
        with contextlib.ExitStack() as stack:
          if hide_output:
            stack.enter_context(contextlib.redirect_stdout(devnull))
            stack.enter_context(contextlib.redirect_stderr(devnull))
          return f(*args, **kwargs)

    return wrapper_fun

  return decorator


@contextlib.contextmanager
def enable_legacy_prng_key():
  """Context manager for enabling jax.config.jax_legacy_prng_key."""
  # Version check: exit early if jax.config.jax_legacy_prng_key does not exist.
  if not hasattr(jax.config, 'jax_legacy_prng_key'):
    yield
    return
  prev = jax.config.jax_legacy_prng_key
  jax.config.update('jax_legacy_prng_key', 'allow')
  try:
    yield
  finally:
    jax.config.update('jax_legacy_prng_key', prev)


@contextlib.contextmanager
def jax_enable_checks():
  """Context manager for enabling jax.config.jax_enable_checks."""
  prev = jax.config.jax_enable_checks
  jax.config.update('jax_enable_checks', True)
  try:
    yield
  finally:
    jax.config.update('jax_enable_checks', prev)


def effectify(
    f,
    hooks = (),
    verbose = False,
):
  """Effect-handling transformation.

  Args:
    f: The function to transform.
    hooks: A sequence of additional Jaxpr transformations to apply.
    verbose: If True, show debugging output.

  Returns:
    An effect-handling version of `f`.
  """

  @jax_enable_checks()
  @enable_legacy_prng_key()
  @redirect_output(hide_output=not verbose)
  def transformed_fun(*args, **kwargs):
    # Get source Jaxpr.
    flat_args, in_tree = jax.tree_util.tree_flatten((args, kwargs))
    in_avals = tuple(api_util.shaped_abstractify(arg) for arg in flat_args)
    jaxpr, consts, out_tree = handler.initial_style_jaxpr(
        lu.wrap_init(f), in_tree, in_avals
    )
    print_jaxpr(jaxpr, title='Source Jaxpr')
    print_jaxpr(consts, title='Source Jaxpr (consts)')

    # Pre-Stage: collect argument types for effect ops.
    effect_op_signatures_prev = None
    effect_op_signatures = core.collect_effect_op_signatures(jaxpr)
    print_panel(effect_op_signatures, title='Effect op signatures')

    while effect_op_signatures != effect_op_signatures_prev:
      effect_op_signatures_prev = copy.deepcopy(effect_op_signatures)

      # Stage 0: make handler scopes explicit.
      # Source Jaxpr -> delimiter_handler Jaxpr.
      delimited_handler_jaxpr = core.delimited_handler_transform(jaxpr)
      print_jaxpr(delimited_handler_jaxpr, title='Delimited handler Jaxpr')
      jaxpr = delimited_handler_jaxpr
      jax_core.check_jaxpr(delimited_handler_jaxpr)

      # Stage 1: concretize handler functions (from Python functions to Jaxprs).
      # -> delimiter_handler Jaxpr with handler implementations as Jaxprs.
      concrete_handler_jaxpr = core.concretize_handlers_transform(
          jaxpr, effect_op_signatures
      )
      print_jaxpr(concrete_handler_jaxpr, title='Concrete handler Jaxpr')
      jaxpr = concrete_handler_jaxpr
      jax_core.check_jaxpr(concrete_handler_jaxpr)

      effect_op_signatures = core.collect_effect_op_signatures(jaxpr)
      print_panel(effect_op_signatures, title='Effect op signatures')

    # Assert that Jaxpr has no `handler_p` equations after concretization.
    def assert_no_handler_eqn(jaxpr):
      for eqn in jaxpr.eqns:
        if handler.is_handler(eqn.primitive):
          assert False, 'Found handler equation'
        core.foreach_subjaxprs(eqn, assert_no_handler_eqn)

    assert_no_handler_eqn(jaxpr)

    # Stage 2: make parameterized handler operations pass state explicitly.
    # delimiter_handler Jaxpr -> explicit handler parameter passing Jaxpr.
    explicit_handler_parameter_passing_jaxpr = (
        core.explicit_handler_parameter_passing_transform(jaxpr)
    )
    print_jaxpr(
        explicit_handler_parameter_passing_jaxpr,
        title=(
            'Explicit effect-parameter-passing Jaxpr '
            '(parameterized handler canonicalization)'
        ),
    )
    jaxpr = explicit_handler_parameter_passing_jaxpr

    # Stage 3: custom transformation hooks.
    for hook in hooks:
      jaxpr = hook.transform_jaxpr(jaxpr)
      print_jaxpr(jaxpr, title=hook.name())

    ctx = core.pp_ctx_factory(jaxpr)

    # Stage 4: effect handling.
    lowered_jaxpr = core.effect_handling_transform(jaxpr)
    print_jaxpr(
        jax_core.pp_jaxpr(lowered_jaxpr, ctx, PP_SETTINGS),
        title='Effect-handled Jaxpr',
    )
    jaxpr = lowered_jaxpr

    # Stage 5: inline calls to legalize subjaxprs that capture free variables.
    jaxpr = core.inline_closed_call(jaxpr)
    print_jaxpr(
        jax_core.pp_jaxpr(jaxpr, ctx, PP_SETTINGS),
        title='Inlined Jaxpr',
    )
    jax_core.check_jaxpr(jaxpr)
    assert len(jaxpr.invars) == len(flat_args)

    # Transform results.
    out_flat = jax_core.eval_jaxpr(jaxpr, consts, *flat_args)
    hook_auxs = []
    for hook in hooks:
      aux, out_flat = hook.split_results(out_flat)
      hook_auxs.append(aux)
    result = jax.tree_util.tree_unflatten(out_tree, out_flat)
    for hook, aux in jax.util.safe_zip(hooks, hook_auxs):
      result = hook.combine_results(aux, result)
    return result

  return transformed_fun


effectify_with_loss = functools.partial(
    effectify,
    hooks=[choix.LossTransformation()],
)
