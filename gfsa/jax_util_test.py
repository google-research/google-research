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

# Lint as: python3
"""Tests for gfsa.jax_util."""

import pickle
from typing import Any, Tuple, Union
from absl.testing import absltest
import dataclasses
import flax
import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
from gfsa import jax_util


class JaxUtilTest(absltest.TestCase):

  def test_synthesize_dataclass(self):

    @dataclasses.dataclass
    class Inner:
      x: jax_util.NDArray
      y: int
      z: Any

    @dataclasses.dataclass
    class Outer:
      a: str
      b: Inner

    synthesized = jax_util.synthesize_dataclass(Outer)

    self.assertEqual(synthesized,
                     Outer(
                         a="",
                         b=Inner(
                             x=jax_util.LeafPlaceholder(jax_util.NDArray),
                             y=0,
                             z=jax_util.LeafPlaceholder(Any))))  # type:ignore

  def test_pickle_placeholder(self):
    placeholder = jax_util.LeafPlaceholder(Union[str, int])
    roundtrip = pickle.loads(pickle.dumps(placeholder))
    self.assertEqual(roundtrip,
                     jax_util.LeafPlaceholder("typing.Union[str, int]"))

    placeholder = jax_util.LeafPlaceholder(Any)
    roundtrip = pickle.loads(pickle.dumps(placeholder))
    self.assertEqual(roundtrip, jax_util.LeafPlaceholder("typing.Any"))

  def test_vmap_with_kwargs(self):

    def foo(x, y, z, w):
      return jnp.sum(x + y[0] + z["bar"] + w)

    foo_vmapped = jax_util.vmap_with_kwargs(
        foo, positional_axes=(0,), y_axis=1, z_axes={"bar": 2})
    result = foo_vmapped(
        jnp.full((5, 2, 2), 1.),
        y=[jnp.full((2, 5, 2), 2.)],
        z={"bar": jnp.full((2, 2, 5), 3.)},
        w=jnp.full((2, 2), 4.))
    np.testing.assert_allclose(result, np.full((5,), 40.))

  def test_np_or_jnp(self):
    self.assertIs(jax_util.np_or_jnp(1.), np)
    self.assertIs(jax_util.np_or_jnp(np.array(1.)), np)
    self.assertIs(jax_util.np_or_jnp(jnp.array(1.)), jnp)

    def trace_check(x):
      self.assertIs(jax_util.np_or_jnp(x), jnp)

    jax.make_jaxpr(trace_check)(np.array(1.))

  def test_pad_to(self):
    arr = np.arange(15).reshape((3, 5))
    padded = jax_util.pad_to(arr, 7, 1)
    expected = np.array([
        [0, 1, 2, 3, 4, 0, 0],
        [5, 6, 7, 8, 9, 0, 0],
        [10, 11, 12, 13, 14, 0, 0],
    ])
    np.testing.assert_equal(padded, expected)

  def test_register_dataclass_pytree(self):

    @jax_util.register_dataclass_pytree
    @dataclasses.dataclass
    class Foo:
      a: jax_util.NDArray
      b: Tuple[jax_util.NDArray, jax_util.NDArray]

    @jax_util.register_dataclass_pytree
    @dataclasses.dataclass
    class Bar:
      x: Foo
      y: Foo

    bar = Bar(
        x=Foo(a=jnp.array([1, 2]), b=(jnp.array(3.), jnp.array(4.))),
        y=Foo(a=jnp.array([5, 6]), b=(jnp.array(7.), jnp.array(8.))),
    )

    bar_plus_ten = jax.tree_map(lambda x: x + 10, bar)
    expected = Bar(
        x=Foo(a=jnp.array([11, 12]), b=(jnp.array(13.), jnp.array(14.))),
        y=Foo(a=jnp.array([15, 16]), b=(jnp.array(17.), jnp.array(18.))),
    )

    jax.test_util.check_close(bar_plus_ten, expected)
    self.assertIsInstance(bar_plus_ten, Bar)
    self.assertIsInstance(bar_plus_ten.x, Foo)
    self.assertIsInstance(bar_plus_ten.y, Foo)

    bar_dict = flax.serialization.to_state_dict(bar)
    jax.test_util.check_close(
        flax.serialization.from_state_dict(bar_plus_ten, bar_dict), bar)


if __name__ == "__main__":
  absltest.main()
