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

"""Misc utilities."""

import functools
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp

PRNGKey = Any


def random_geometric(key, continue_log_prob):
  """Sample a geometric r.v. with given log prob of continuing."""
  return jnp.floor(jnp.log(jax.random.uniform(key)) / continue_log_prob).astype(
      jnp.int32
  )


def tree_choose(choice, trees):
  return jax.tree.map(
      lambda *args: jnp.choose(choice, list(args), mode="clip"), *trees
  )


def tree_where(gate, true_tree, false_tree):
  return jax.tree.map(lambda t, f: jnp.where(gate, t, f), true_tree, false_tree)


def tree_select(condlist, choicelist, default=None):
  if default is None:

    def mk_default(x):
      x = jnp.array(x)
      return jnp.full(x.shape, -12345, x.dtype)

    default = jax.tree.map(mk_default, choicelist[0])
  return jax.tree.map(
      lambda d, *args: jnp.select(condlist, list(args), default=d),
      default,
      *choicelist,
  )


def previous_active_value(values, active, default=0, inclusive=False):
  """Produces an array containing the previous active value at each position.

  If values is
    [1, 2, 3, 4, 5, 6, 7]
  and active is
    [0, 1, 1, 0, 0, 1, 0],
  returns
    [0, 0, 2, 3, 3, 3, 6],

  such that each element of the return value is a copy of the most recently
  encountered active value from left to right.

  Args:
    values: Array of values to index into.
    active: Boolean mask indicating which are active.
    default: Default value if there is no previous value.
    inclusive: Whether the current value counts as the "most recent" active
      value for an index that is active.

  Returns:
    Array containing the previous active value at each position.
  """

  # Associative scan with state (most_recent_value, has_some_active) accumulated
  # across spans. This will produce the desired output but shifted, so that
  # it produces the most recent inclusive active value at each position but
  # possibly including the current position.

  def step(prevstate, nextstate):
    pval, pactive = prevstate
    nval, nactive = nextstate
    return jnp.where(nactive, nval, pval), (pactive | nactive)

  inclusive_result, has_active = jax.lax.associative_scan(
      step, (values, active.astype(jnp.int32)))
  inclusive_result = jnp.where(has_active, inclusive_result, default)
  if inclusive:
    return inclusive_result
  result = jnp.concatenate([jnp.array(default)[None], inclusive_result[:-1]])
  return result


def next_active_index(active, inclusive=False):
  """Produces an array containing the next active index at each position.

  If active is
    [0, 1, 1, 0, 0, 1, 0],
  returns
    [1, 2, 5, 5, 5, 7, 7].

  Args:
    active: Boolean mask indicating which are active.
    inclusive: Whether the current index counts as the "most recent" active
      value for an index that is active.

  Returns:
    Array containing the previous active index at each position.
  """

  def step(nextstate, prevstate):
    pval, pactive = prevstate
    nval, nactive = nextstate
    return jnp.where(pactive, pval, nval), (pactive | nactive)

  inclusive_result, has_active = jax.lax.associative_scan(
      step, (jnp.arange(active.shape[0]), active.astype(jnp.int32)),
      reverse=True)
  inclusive_result = jnp.where(has_active, inclusive_result, active.shape[0])
  if inclusive:
    return inclusive_result
  result = jnp.concatenate(
      [inclusive_result[1:],
       jnp.array(active.shape[0])[None]])
  return result


def rejection_sample(fn,
                     rng,
                     max_rejects = None):
  """Helper to run rejection sampling."""

  def rejection_cond(rejection_state):
    i, is_good, _, _ = rejection_state
    if max_rejects is not None:
      return (~is_good) & (i < max_rejects)
    return ~is_good

  def rejection_body(rejection_state):
    i, _, _, rng = rejection_state
    next_rng, key = jax.random.split(rng)
    is_good, output = fn(key)
    return i + 1, is_good, output, next_rng

  _, shapes_and_types = jax.eval_shape(fn, rng)
  dummy_output = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype),
                              shapes_and_types)

  _, _, good_output, _ = jax.lax.while_loop(rejection_cond, rejection_body,
                                            (0, False, dummy_output, rng))
  return good_output


def vmap_with_kwargs(fun,
                     positional_axes=0,
                     out_axes=0,
                     **kwargs_axes):
  """Wrapper around jax.vmap that supports specifying axes for kwargs.

  For instance, if we have

    def foo(w, x, y, z):
      ...

  then instead of writing `jax.vmap(foo, in_axes=(0, 1, (2, 3), None))` and
  calling with only positional arguments, you can instead write

    vmap_with_kwargs(foo, positional_axes=(0,), x_axis=1, y_axes=(2, 3))

  and call it with x, y, and z as keyword arguments.

  Args:
    fun: Function to vmap.
    positional_axes: Input axes for positional arguments; like `in_axes` for
      jax.vmap. If not provided, all positional arguments will be vmapped across
      their first dimension.
    out_axes: Output axes; see jax.vmap.
    **kwargs_axes: Input axes for keyword arguments, which works the same way as
      in_axes does for positional arguments. Each keyword argument should have
      the suffix "_axis" or "_axes", corresponding to the axis to vectorize the
      keyword argument along. Any missing kwargs will be assumed to be
      broadcasted (i.e. it is as if they were given an axis of None).

  Returns:
    Batched/vectorized version of `fun`; see jax.vmap. Note that the positional
    and keyword arguments used to call this batched version must match the
    positional and keyword axis specifications passed in to vmap_with_kwargs.
  """

  known_kw_axes = {}
  for keyword, value in kwargs_axes.items():
    if not (keyword.endswith("_axis") or keyword.endswith("_axes")):
      raise ValueError(
          f"Keyword argument {keyword} does not end in '_axis' or '_axes'")
    known_kw_axes[keyword[:-5]] = value

  @functools.wraps(fun)
  def apply(args, known_kw_axes, extra_kwargs):
    return fun(*args, **known_kw_axes, **extra_kwargs)

  mapped = jax.vmap(
      apply, in_axes=(positional_axes, known_kw_axes, None), out_axes=out_axes)

  @functools.wraps(mapped)
  def wrapper(*args, **kwargs):
    return mapped(args, {k: v for k, v in kwargs.items() if k in known_kw_axes},
                  {k: v for k, v in kwargs.items() if k not in known_kw_axes})

  return wrapper
