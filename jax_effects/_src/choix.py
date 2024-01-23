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

"""Choice-based learning transformation."""

from collections.abc import Sequence
import dataclasses
from typing import Any, NamedTuple

import jax
from jax import core as jax_core
from jax.interpreters import ad
import jax.numpy as jnp

from jax_effects._src import core
from jax_effects._src import handler
from jax_effects._src import logging_utils


CONSOLE = logging_utils.CONSOLE
PP_SETTINGS = core.PP_SETTINGS

Loss = jax_core.ShapedArray((), jnp.float32)
loss_zero = jax_core.Literal(0.0, Loss)

print_jaxpr = logging_utils.print_jaxpr

################################################################################
# Transformation hook
################################################################################


class ValueWithLoss(NamedTuple):
  """A value with an associated loss."""

  loss: float
  value: Any


class LossTransformation(core.JaxprTransformation):
  """Loss effect transformation."""

  @classmethod
  def name(cls) -> str:
    return 'Loss-translation'

  def transform_jaxpr(self, jaxpr: jax_core.Jaxpr) -> jax_core.Jaxpr:
    return loss_translation_transform(jaxpr, g=make_zero_loss_continuation())

  def split_results(self, outvals: Sequence[Any]) -> tuple[Any, Any]:
    """Splits loss value from other result values."""
    # The loss value is the first outval.
    loss_value, *out_flat = outvals
    return loss_value, out_flat

  def combine_results(self, aux: Any, out_tree: Any) -> Any:
    """Combines loss value with other result values."""
    return ValueWithLoss(loss=aux, value=out_tree)

################################################################################
# Loss effect operation
################################################################################


# NOTE: `loss` is a special effect operation primitive.
#
# It is intentionally not registered via `@effect` for now because it should be
# skipped by the effect operation transformation because its primitive does not
# have (or need) a handler. We could extend `@effect` to support this later.
loss_p = jax_core.Primitive('loss')


LossDtype = jnp.float32
Loss = jax_core.ShapedArray((), LossDtype)


def loss(x) -> None:
  loss_p.bind(LossDtype(x))


@loss_p.def_impl
def _(x):
  return x


@loss_p.def_abstract_eval
def _(x):
  return jax_core.ShapedArray((), x.dtype)


def is_loss(primitive: jax_core.Primitive) -> bool:
  return primitive is loss_p


################################################################################
# Loss translation
################################################################################


def create_add_eqn(result, x, y):
  """Create result = x + y in jax."""
  # TODO(jax-effects-team): Create dedicated loss accumulation primitive.
  # We need special semantics for loss accumulation that effect handling
  # transform can recognize.
  return jax_core.new_jaxpr_eqn(
      invars=[x, y],
      outvars=[result],
      primitive=ad.add_jaxvals_p,
      params={},
      effects=set(),
      source_info=None,
  )


@dataclasses.dataclass
class PartialLossCont:
  """A partial loss continuation.

  captured_vars should contain a list of variables defined in the enclosing
  context.
  """

  jaxpr: jax_core.Jaxpr
  captured_vars: list[jax_core.Var]
  is_zero_loss_continuation: bool


def combine_loss_continuations(
    f: PartialLossCont, g: PartialLossCont
) -> PartialLossCont:
  """Loss continuation composition rule.

  f wthen g = lambda y: let loss, x = f(y) in loss + g(x)

  Args:
    f: The first loss continuation.
    g: The second loss continuation.

  Returns:
    A loss continuation that sequences `f` and `g` and returns their combined
    loss.
  """
  f_jaxpr = f.jaxpr
  g_jaxpr = g.jaxpr

  gensym = jax_core.gensym((f_jaxpr, g_jaxpr))
  gensym_loss = jax_core.gensym((f_jaxpr, g_jaxpr), suffix='_loss')

  loss_cont_arg = f_jaxpr.invars[len(f.captured_vars) :]
  combined_invars = f.captured_vars + g.captured_vars + loss_cont_arg
  combined_captured_vars = f.captured_vars + g.captured_vars
  f_call_invars = f.captured_vars + loss_cont_arg
  f_call_outvars = [gensym(v.aval) for v in f_jaxpr.outvars]
  f_call_eqn = jax_core.new_jaxpr_eqn(
      invars=f_call_invars,
      outvars=f_call_outvars,
      primitive=jax_core.closed_call_p,
      params=dict(call_jaxpr=jax_core.ClosedJaxpr(f_jaxpr, ())),
      effects=f_jaxpr.effects,
      source_info=None,
  )

  # If `g` is a zero loss continuation, return loss of `f` directly.
  if g.is_zero_loss_continuation:
    return PartialLossCont(
        jaxpr=jax_core.Jaxpr(
            constvars=(),
            invars=combined_invars,
            outvars=(f_call_outvars[0],),
            eqns=(f_call_eqn,),
            debug_info=f_jaxpr.debug_info,
        ),
        captured_vars=combined_captured_vars,
        is_zero_loss_continuation=False,
    )

  g_call_invars = g.captured_vars + f_call_outvars[1:]
  g_call_outvars = [gensym(v.aval) for v in g_jaxpr.outvars]
  g_call_eqn = jax_core.new_jaxpr_eqn(
      invars=g_call_invars,
      outvars=g_call_outvars,
      primitive=jax_core.closed_call_p,
      params=dict(call_jaxpr=jax_core.ClosedJaxpr(g_jaxpr, ())),
      effects=g_jaxpr.effects,
      source_info=None,
  )

  loss_add_eqn = create_add_eqn(
      gensym_loss(Loss), f_call_outvars[0], g_call_outvars[0]
  )

  combined_eqns = (f_call_eqn, g_call_eqn, loss_add_eqn)
  combined_outvars = loss_add_eqn.outvars
  combined_jaxpr = jax_core.Jaxpr(
      constvars=(),
      invars=combined_invars,
      outvars=combined_outvars,
      eqns=combined_eqns,
      debug_info=f_jaxpr.debug_info,
  )

  return PartialLossCont(
      jaxpr=combined_jaxpr,
      captured_vars=combined_captured_vars,
      is_zero_loss_continuation=False,
  )


def translate_loss_delimited_handler(
    eqn: jax_core.JaxprEqn,
    g: PartialLossCont,
    gensym_suffix,
    ctx: jax_core.JaxprPpContext,
) -> jax_core.JaxprEqn:
  """Loss translation rule for delimited handlers.

  T(with nh from p handle nf, g) = with T(nh, g) from p handle T(nf, 0)
  lk = k(arg).1 + g(k(arg).2).

  Args:
    eqn: The `delimited_handler` equation.
    g: The loss continuation.
    gensym_suffix: The gensym function.
    ctx: The pretty-printing context, for debug printing.

  Returns:
    A loss-translated `delimited_handler` equation.
  """
  if eqn.primitive is not handler.delimited_handler_p:
    raise ValueError(f'Expected delimited_handler equation, got: {eqn}')

  handler_impl = eqn.params['handler_impl']
  loss_translated_handler_impl = {}
  for effect_name, handler_impl_jaxpr in handler_impl.items():
    translated_handler_jaxpr = handler_impl_jaxpr.map_jaxpr(
        lambda jaxpr: loss_translation_transform(jaxpr, g, ctx=ctx)
    )
    loss_translated_handler_impl[effect_name] = translated_handler_jaxpr
  body_jaxpr = eqn.params['body_jaxpr']
  translated_body_jaxpr = loss_translation_transform(
      body_jaxpr, make_zero_loss_continuation()
  )
  return eqn.replace(
      outvars=[gensym_suffix('_loss')(Loss)] + eqn.outvars,
      params=dict(
          eqn.params,
          handler_impl=loss_translated_handler_impl,
          body_jaxpr=translated_body_jaxpr,
      ),
  )


@dataclasses.dataclass
class LossTranslationResult:
  new_eqns: Sequence[jax_core.JaxprEqn]
  loss_outvar: jax_core.Atom
  outvars: Sequence[jax_core.Atom]

  def pp(self, ctx: jax_core.JaxprPpContext) -> str:
    new_eqns_str = jax_core.pp_eqns(self.new_eqns, ctx, PP_SETTINGS)
    loss_outvar_str = jax_core.pp_var(self.loss_outvar, ctx)
    outvars_str = jax_core.pp_vars(self.outvars, ctx)
    return f"""LossTranslationResult(
new_eqns={new_eqns_str},
loss_outvar={loss_outvar_str},
outvars=[{outvars_str}]
)"""


def loss_transform_primitive(
    primitive: jax_core.Primitive,
) -> jax_core.Primitive:
  """Transforms primitive to return loss."""
  primitive_type = type(primitive)
  transformed_primitive = primitive_type(f'{primitive.name}_with_loss')
  transformed_primitive.multiple_results = True
  transformed_primitive.call_primitive = primitive.call_primitive
  transformed_primitive.map_primitive = primitive.map_primitive

  @transformed_primitive.def_impl
  def _(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError()

  @transformed_primitive.def_effectful_abstract_eval
  def _(*args, **kwargs):
    outvals, effects = primitive.abstract_eval(*args, **kwargs)
    if isinstance(outvals, Sequence):
      return (Loss,) + tuple(outvals), effects
    else:
      return (Loss, outvals), effects

  return transformed_primitive


# TODO(jax-effects-team): Encapsulate loss transformation into a class.
# Class members: `PartialLossCont` (built), `gensym_suffix`.
def translate_loss_eqn(
    eqn: jax_core.JaxprEqn,
    g: PartialLossCont,
    gensym_suffix,
    ctx: jax_core.JaxprPpContext,
) -> LossTranslationResult:
  """Loss translation rule for a single Jaxpr equation.

  Args:
    eqn: The Jaxpr equation to translate.
    g: The loss continuation.
    gensym_suffix: The gensym function.
    ctx: The pretty-printing context, for debug printing.

  Returns:
    Loss translation result for `eqn`.
  """
  if is_loss(eqn.primitive):
    return LossTranslationResult((), eqn.invars[0], eqn.outvars)
  # TODO(jax-effects-team): Fix this rule.
  elif core.is_call_k(eqn.primitive):
    new_eqn_outvars = [gensym_suffix('_loss')(Loss)] + eqn.outvars
    new_eqn = eqn.replace(
        primitive=loss_transform_primitive(eqn.primitive),
        outvars=new_eqn_outvars,
    )
    return LossTranslationResult(
        [new_eqn],
        new_eqn.outvars[0],
        new_eqn.outvars[1:],
    )
  # Transform `call_lk` primitives.
  # Target Jaxpr should have no more `call_lk` operations.
  elif core.is_call_lk(eqn.primitive):
    CONSOLE.print('Found call_lk in loss transformation', style='bold yellow')
    print(jax_core.pp_eqn(eqn, ctx, PP_SETTINGS))
    call_k_primitive = eqn.primitive.call_k_primitive  # pytype: disable=attribute-error
    call_k_eqn_outvars = [
        gensym_suffix('')(aval) for aval in call_k_primitive.outvals
    ]
    call_k_eqn = eqn.replace(
        primitive=call_k_primitive,
        outvars=call_k_eqn_outvars,
    )

    CONSOLE.print('Converted call_lk to call_k', style='bold yellow')
    print(jax_core.pp_eqn(call_k_eqn, ctx, PP_SETTINGS))
    result = translate_loss_eqns(
        [call_k_eqn], outvars=[], g=g, gensym_suffix=gensym_suffix, ctx=ctx
    )
    CONSOLE.print('Transformed call_lk result', style='bold yellow')
    print(result.pp(ctx))
    assert len(eqn.outvars) == 1
    assert result.new_eqns[0].outvars[0] == result.loss_outvar
    # Reassign `call_k` loss outvar to `call_lk` outvar, to ensure outvar usages
    # are correct.
    result.new_eqns[0].outvars[0] = eqn.outvars[0]
    result.loss_outvar = loss_zero
    return result
  elif eqn.primitive == jax.lax.while_p:
    # Update body jaxpr.
    body_jaxpr: jax_core.ClosedJaxpr = eqn.params['body_jaxpr']
    body_num_consts = eqn.params['body_nconsts']
    new_body_jaxpr = body_jaxpr.map_jaxpr(
        lambda jaxpr: loss_translation_transform(jaxpr, g, ctx=ctx)
    )
    new_body_invars = (
        new_body_jaxpr.jaxpr.invars[:body_num_consts]
        + [gensym_suffix('_loss')(Loss)]
        + new_body_jaxpr.jaxpr.invars[body_num_consts:]
    )
    new_body_jaxpr = new_body_jaxpr.map_jaxpr(
        lambda jaxpr: jaxpr.replace(invars=new_body_invars)
    )

    # Update cond jaxpr.
    cond_jaxpr: jax_core.ClosedJaxpr = eqn.params['cond_jaxpr']
    cond_num_consts = eqn.params['cond_nconsts']
    new_cond_invars = [gensym_suffix('_loss')(Loss)] + cond_jaxpr.jaxpr.invars
    new_cond_invars = (
        cond_jaxpr.jaxpr.invars[:cond_num_consts]
        + [gensym_suffix('_loss')(Loss)]
        + cond_jaxpr.jaxpr.invars[cond_num_consts:]
    )
    new_cond_jaxpr = cond_jaxpr.map_jaxpr(
        lambda jaxpr: jaxpr.replace(invars=new_cond_invars)
    )

    # Update equation.
    new_eqn_invars = list(eqn.invars)
    new_eqn_invars.insert(body_num_consts, loss_zero)
    new_eqn_outvars = [gensym_suffix('_loss')(Loss)] + eqn.outvars
    new_eqn = eqn.replace(
        invars=new_eqn_invars,
        outvars=new_eqn_outvars,
        params=dict(
            eqn.params, body_jaxpr=new_body_jaxpr, cond_jaxpr=new_cond_jaxpr
        ),
    )
    return LossTranslationResult(
        [new_eqn],
        new_eqn.outvars[0],
        new_eqn.outvars[1:],
    )
  elif eqn.primitive is jax.lax.scan_p:
    # Update body jaxpr.
    body_jaxpr: jax_core.ClosedJaxpr = eqn.params['jaxpr']
    num_consts = eqn.params['num_consts']
    num_carry = eqn.params['num_carry']

    new_body_jaxpr = body_jaxpr.map_jaxpr(
        lambda jaxpr: loss_translation_transform(jaxpr, g, ctx=ctx)
    )
    new_body_invars = (
        new_body_jaxpr.jaxpr.invars[:num_consts]
        + [gensym_suffix('_loss')(Loss)]
        + new_body_jaxpr.jaxpr.invars[num_consts:]
    )
    new_body_jaxpr = new_body_jaxpr.map_jaxpr(
        lambda jaxpr: jaxpr.replace(
            invars=new_body_invars,
        )
    )

    # Update equation.
    new_eqn_invars = list(eqn.invars)
    new_eqn_invars.insert(num_consts, loss_zero)
    new_eqn_outvars = [gensym_suffix('_loss')(Loss)] + eqn.outvars
    new_eqn = eqn.replace(
        invars=new_eqn_invars,
        outvars=new_eqn_outvars,
        params=dict(
            eqn.params,
            jaxpr=new_body_jaxpr,
            linear=(False,) + eqn.params['linear'],
            num_carry=1 + num_carry,
        ),
    )
    return LossTranslationResult(
        [new_eqn],
        new_eqn.outvars[0],
        new_eqn.outvars[1:],
    )
  elif handler.is_handler_return(eqn.primitive):
    return LossTranslationResult([eqn], loss_zero, eqn.outvars)
  elif handler.is_delimited_handler(eqn.primitive):
    translated_eqn = translate_loss_delimited_handler(
        eqn, g, gensym_suffix, ctx=ctx
    )
    return LossTranslationResult(
        [translated_eqn], translated_eqn.outvars[0], translated_eqn.outvars[1:]
    )
  elif eqn.primitive in core.staged_transform_primitives:
    new_eqn = core.transform_subjaxprs(
        eqn, lambda jaxpr: loss_translation_transform(jaxpr, g, ctx=ctx)
    )
    return LossTranslationResult([new_eqn], loss_zero, new_eqn.outvars)
  else:
    return LossTranslationResult([eqn], loss_zero, eqn.outvars)


def translate_loss_eqns(
    eqns: Sequence[jax_core.JaxprEqn],
    outvars: Sequence[jax_core.Atom],
    g: PartialLossCont,
    gensym_suffix,
    ctx: jax_core.JaxprPpContext,
) -> LossTranslationResult:
  """Loss translation rule for Jaxpr equations.

  Definition of wlet:
    wlet x = e in e'  ==  let loss, x = e in (loss + e'.1, e'.2)

  Definition of wthen:
    e wthen lambda x: e'  ==  let loss, x = e in loss + (lambda x: e') x

  Translation rule for let
  let x = at in nf  ==  wlet x = T(at, lambda x: T(nf, g) wthen g) in T(nf, g)

  Args:
    eqns: The Jaxpr equations to translate.
    outvars: The original Jaxpr outvars.
    g: The loss continuation.
    gensym_suffix: The gensym function.
    ctx: The pretty-printing context, for debug printing.

  Returns:
    Loss translation result for `eqns`.
  """
  if not eqns:
    return LossTranslationResult((), loss_zero, outvars)

  translated_continuation = translate_loss_eqns(
      eqns[1:], outvars, g, gensym_suffix=gensym_suffix, ctx=ctx
  )

  result_eqns = []
  eqn = eqns[0]

  translated_remaining_eqns = translated_continuation.new_eqns
  continuation_loss_var = translated_continuation.loss_outvar
  continuation_outvars = translated_continuation.outvars
  continuation_outvars = [
      v if not isinstance(v, jax_core.DropVar) else jax_core.Literal(0, v.aval)
      for v in continuation_outvars
  ]
  loss_continuation_free_vars = core.free_vars_in_eqns(
      translated_remaining_eqns, set(eqn.outvars)
  )
  remaining_loss_continuation = jax_core.Jaxpr(
      constvars=(),
      invars=loss_continuation_free_vars + eqn.outvars,
      outvars=(continuation_loss_var, *continuation_outvars),
      eqns=translated_remaining_eqns,
      debug_info=g.jaxpr.debug_info,
  )
  # remaining_loss_continuation = dce_jaxpr(remaining_loss_continuation)

  # T(nf, g)
  remaining_loss_continuation_t = PartialLossCont(
      remaining_loss_continuation,
      captured_vars=loss_continuation_free_vars,
      is_zero_loss_continuation=False,
  )

  # T(at, lambda x: T(nf, g) wthen g) in T(nf, g)
  translated_eqn_loss = translate_loss_eqn(
      eqn,
      combine_loss_continuations(remaining_loss_continuation_t, g),
      gensym_suffix=gensym_suffix,
      ctx=ctx,
  )
  loss_outvar = translated_eqn_loss.loss_outvar
  result_eqns.extend(translated_eqn_loss.new_eqns)
  result_eqns.extend(translated_remaining_eqns)

  if core.is_literal_equal(loss_outvar, loss_zero):
    return LossTranslationResult(
        result_eqns, continuation_loss_var, continuation_outvars
    )
  if core.is_literal_equal(continuation_loss_var, loss_zero):
    return LossTranslationResult(result_eqns, loss_outvar, continuation_outvars)

  loss_result_var = gensym_suffix('_loss')(Loss)
  add_eqn = create_add_eqn(loss_result_var, continuation_loss_var, loss_outvar)
  result_eqns.append(add_eqn)
  return LossTranslationResult(
      result_eqns, loss_result_var, continuation_outvars
  )


def make_zero_loss_continuation() -> PartialLossCont:
  res = jax_core.Jaxpr(
      constvars=(),
      invars=(),
      outvars=(loss_zero,),
      eqns=(),
  )
  return PartialLossCont(
      jaxpr=res, captured_vars=[], is_zero_loss_continuation=True
  )


def loss_translation_transform(
    jaxpr: jax_core.Jaxpr,
    g: PartialLossCont,
    ctx: jax_core.JaxprPpContext | None = None,
) -> jax_core.Jaxpr:
  """Loss translation transform for `Jaxpr`."""
  if ctx is None:
    ctx = core.pp_ctx_factory(jaxpr)
  gensym_suffix = lambda suffix: jax_core.gensym((jaxpr,), suffix=suffix)
  translation_result = translate_loss_eqns(
      jaxpr.eqns,
      outvars=jaxpr.outvars,
      g=g,
      gensym_suffix=gensym_suffix,
      ctx=ctx,
  )
  transformed_eqns = translation_result.new_eqns
  loss_var = translation_result.loss_outvar
  return jaxpr.replace(
      eqns=transformed_eqns, outvars=(loss_var,) + tuple(jaxpr.outvars)
  )
