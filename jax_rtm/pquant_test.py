# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Unit tests for the physical quantity dimension tracking DSL (pquant)."""

# pylint: disable=g-import-not-at-top

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_rtm import pquant
import numpy as np

jax.config.update("jax_enable_x64", True)


class PQuantTest(absltest.TestCase):

  def test_quantity_creation_and_repr(self):
    q = pquant.Quantity(jnp.array(1.0), (1, 2, 3, 4))
    self.assertEqual(q.value, 1.0)
    self.assertEqual(q.dimensions, (1, 2, 3, 4))
    self.assertEqual(repr(q), "Quantity(val=1.0, dims=[1,2,3,4])")

    # Test repr with float dimensions
    q_float = pquant.Quantity(jnp.array(1.0), (1.5, 2.0, -3.1, 0.0))
    self.assertEqual(repr(q_float), "Quantity(val=1.0, dims=[1.5,2,-3.1,0])")

  def test_addition(self):
    # Same dimensions
    q1 = pquant.Length(1.0)
    q2 = pquant.Length(2.0)
    res = q1 + q2
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Dimension mismatch
    q3 = pquant.Mass(1.0)
    with self.assertRaises(ValueError):
      _ = q1 + q3

    # Dimensionless + scalar
    q_dimless = pquant.Dimensionless(1.0)
    res = q_dimless + 2.0
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

    res = 2.0 + q_dimless
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

    # Temperature + scalar (offset allowed)
    q_temp = pquant.Temperature(273.15)
    res = q_temp + 10.0
    self.assertEqual(res.value, 283.15)
    self.assertEqual(res.dimensions, pquant.Temperature.dimensions)

    res = 10.0 + q_temp
    self.assertEqual(res.value, 283.15)
    self.assertEqual(res.dimensions, pquant.Temperature.dimensions)

    # Quantity + 0.0 (identity allowed for any dimension)
    res = q1 + 0.0
    self.assertEqual(res.value, 1.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    res = 0.0 + q1
    self.assertEqual(res.value, 1.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Quantity + non-zero scalar (disallowed if not dimless/temp)
    with self.assertRaises(ValueError):
      _ = q1 + 2.0
    with self.assertRaises(ValueError):
      _ = 2.0 + q1

  def test_subtraction(self):
    # Same dimensions
    q1 = pquant.Length(5.0)
    q2 = pquant.Length(2.0)
    res = q1 - q2
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Dimension mismatch
    q3 = pquant.Mass(1.0)
    with self.assertRaises(ValueError):
      _ = q1 - q3

    # Dimensionless - scalar
    q_dimless = pquant.Dimensionless(5.0)
    res = q_dimless - 2.0
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

    res = 10.0 - q_dimless
    self.assertEqual(res.value, 5.0)
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

    # Temperature - scalar
    q_temp = pquant.Temperature(273.15)
    res = q_temp - 10.0
    self.assertEqual(res.value, 263.15)
    self.assertEqual(res.dimensions, pquant.Temperature.dimensions)

    res = 283.15 - q_temp
    self.assertEqual(res.value, 10.0)
    self.assertEqual(res.dimensions, pquant.Temperature.dimensions)

    # Quantity - 0.0
    res = q1 - 0.0
    self.assertEqual(res.value, 5.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # 0.0 - Quantity
    res = 0.0 - q1
    self.assertEqual(res.value, -5.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Quantity - non-zero scalar
    with self.assertRaises(ValueError):
      _ = q1 - 2.0
    with self.assertRaises(ValueError):
      _ = 2.0 - q1

  def test_multiplication(self):
    q1 = pquant.Length(2.0)
    q2 = pquant.Mass(3.0)

    # Quantity * Quantity
    res = q1 * q2
    self.assertEqual(res.value, 6.0)
    self.assertEqual(res.dimensions, (1, 1, 0, 0))  # Mass * Length

    # Quantity * Scalar
    res = q1 * 3.0
    self.assertEqual(res.value, 6.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Scalar * Quantity
    res = 3.0 * q1
    self.assertEqual(res.value, 6.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

  def test_division(self):
    q1 = pquant.Length(6.0)
    q2 = pquant.Time(2.0)

    # Quantity / Quantity
    res = q1 / q2
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, (0, 1, -1, 0))  # Length / Time (Velocity)

    # Quantity / Scalar
    res = q1 / 2.0
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Scalar / Quantity
    res = 12.0 / q1
    self.assertEqual(res.value, 2.0)
    self.assertEqual(res.dimensions, (0, -1, 0, 0))  # 1 / Length

  def test_power(self):
    q = pquant.Length(3.0)

    # Integer power
    res = q**2
    self.assertEqual(res.value, 9.0)
    self.assertEqual(res.dimensions, pquant.Area.dimensions)

    # Float power
    res = q**0.5
    self.assertAlmostEqual(float(res.value), 3.0**0.5)
    self.assertEqual(res.dimensions, (0, 0.5, 0, 0))

    # Negative power
    res = q**-1
    self.assertAlmostEqual(float(res.value), 1.0 / 3.0)
    self.assertEqual(res.dimensions, (0, -1, 0, 0))

    # Invalid exponent
    with self.assertRaises(ValueError):
      _ = q**q

  def test_comparisons(self):
    q1 = pquant.Length(2.0)
    q2 = pquant.Length(3.0)
    q3 = pquant.Mass(2.0)

    # Same dimensions
    self.assertLess(q1, q2)
    self.assertLessEqual(q1, q2)
    self.assertGreater(q2, q1)
    self.assertGreaterEqual(q2, q1)
    self.assertNotEqual(q1, q2)

    # Dimension mismatch
    with self.assertRaises(ValueError):
      _ = q1 < q3
    with self.assertRaises(ValueError):
      _ = q1 <= q3
    with self.assertRaises(ValueError):
      _ = q1 > q3
    with self.assertRaises(ValueError):
      _ = q1 >= q3
    # __eq__ should return False, not raise
    self.assertNotEqual(q1, q3)

    # Comparison with 0.0
    self.assertGreater(q1, 0.0)
    self.assertGreaterEqual(q1, 0.0)
    self.assertLess(0.0, q1)
    self.assertLessEqual(0.0, q1)
    self.assertNotEqual(q1, 0.0)

    # Comparison with non-zero scalar
    with self.assertRaises(ValueError):
      _ = q1 > 1.0
    self.assertNotEqual(q1, 1.0)

    # Dimensionless comparison with scalar
    q_dimless = pquant.Dimensionless(2.0)
    self.assertGreater(q_dimless, 1.0)
    self.assertEqual(q_dimless, 2.0)
    self.assertNotEqual(q_dimless, 3.0)

  def test_unary_operators(self):
    q = pquant.Length(2.0)
    self.assertEqual((-q).value, -2.0)
    self.assertEqual((-q).dimensions, pquant.Length.dimensions)

    self.assertEqual((+q).value, 2.0)
    self.assertEqual((+q).dimensions, pquant.Length.dimensions)

    q_neg = pquant.Length(-3.0)
    self.assertEqual(abs(q_neg).value, 3.0)
    self.assertEqual(abs(q_neg).dimensions, pquant.Length.dimensions)

  def test_bool_coercion(self):
    # Scalar Quantity
    self.assertTrue(bool(pquant.Dimensionless(1.0)))
    self.assertFalse(bool(pquant.Dimensionless(0.0)))
    self.assertTrue(bool(pquant.Length(2.0)))

    # Array Quantity
    q_array_ok = pquant.Length(jnp.array([1.0]))
    self.assertTrue(bool(q_array_ok))

    q_array_bad = pquant.Length(jnp.array([1.0, 2.0]))
    with self.assertRaises(TypeError):
      _ = bool(q_array_bad)

  def test_array_interface(self):
    q = pquant.Length(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    self.assertEqual(q.shape, (2, 2))
    self.assertEqual(q.ndim, 2)
    self.assertEqual(q.dtype, jnp.float64)
    self.assertLen(q, 2)

    # Indexing
    sub_q = q[0]
    self.assertIsInstance(sub_q, pquant.Quantity)
    np.testing.assert_allclose(sub_q.value, jnp.array([1.0, 2.0]))
    self.assertEqual(sub_q.dimensions, pquant.Length.dimensions)

  def test_jax_pytree(self):
    q = pquant.Length(2.0)

    # Flatten
    leaves, treedef = jax.tree_util.tree_flatten(q)
    self.assertLen(leaves, 1)
    self.assertEqual(leaves[0], 2.0)

    # Unflatten
    q_recon = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertIsInstance(q_recon, pquant.Quantity)
    self.assertEqual(q_recon.value, 2.0)
    self.assertEqual(q_recon.dimensions, pquant.Length.dimensions)

    # JIT compatibility
    @jax.jit
    def f(x):
      return x * 2.0

    res = f(q)
    self.assertIsInstance(res, pquant.Quantity)
    self.assertEqual(res.value, 4.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # VMAP compatibility
    q_batch = pquant.Length(jnp.array([1.0, 2.0, 3.0]))

    @jax.vmap
    def g(x):
      return x * 2.0

    res_batch = g(q_batch)
    self.assertIsInstance(res_batch, pquant.Quantity)
    np.testing.assert_allclose(res_batch.value, jnp.array([2.0, 4.0, 6.0]))
    self.assertEqual(res_batch.dimensions, pquant.Length.dimensions)

  def test_factory_functions(self):
    self.assertEqual(
        pquant.Dimensionless(1.0).dimensions,
        pquant.Dimensionless.dimensions,
    )
    self.assertEqual(
        pquant.Length(1.0).dimensions, pquant.Length.dimensions
    )
    self.assertEqual(
        pquant.Area(1.0).dimensions, pquant.Area.dimensions
    )
    self.assertEqual(
        pquant.Volume(1.0).dimensions, pquant.Volume.dimensions
    )
    self.assertEqual(
        pquant.Mass(1.0).dimensions, pquant.Mass.dimensions
    )
    self.assertEqual(
        pquant.Time(1.0).dimensions, pquant.Time.dimensions
    )
    self.assertEqual(
        pquant.Temperature(1.0).dimensions,
        pquant.Temperature.dimensions,
    )
    self.assertEqual(
        pquant.Density(1.0).dimensions, pquant.Density.dimensions
    )
    self.assertEqual(
        pquant.NumberConcentration(1.0).dimensions,
        pquant.NumberConcentration.dimensions,
    )
    self.assertEqual(
        pquant.MassExtinction(1.0).dimensions,
        pquant.MassExtinction.dimensions,
    )
    self.assertEqual(
        pquant.IWP(1.0).dimensions, pquant.IWP.dimensions
    )
    self.assertEqual(
        pquant.InverseIWP(1.0).dimensions,
        pquant.InverseIWP.dimensions,
    )
    self.assertEqual(
        pquant.SpectralRadiance(1.0).dimensions,
        pquant.SpectralRadiance.dimensions,
    )

    # TemperaturePower (dynamic)
    self.assertEqual(
        pquant.TemperaturePower(1.0, 1.5).dimensions, (0, 0, 0, 1.5)
    )
    self.assertEqual(pquant.TemperaturePower(1.0, -2).dimensions, (0, 0, 0, -2))

  def test_recursive_helpers(self):
    q1 = pquant.Length(1.0)
    q2 = pquant.Mass(2.0)

    struct = {"a": q1, "b": [q2, 3.0], "c": (4.0, {"d": q1})}

    # contains_pq
    self.assertTrue(pquant.contains_pq(struct))
    self.assertTrue(pquant.contains_pq([1.0, q1]))
    self.assertFalse(pquant.contains_pq([1.0, 2.0, {"a": 3.0}]))

    # unwrap_pq
    unwrapped = pquant.unwrap_pq(struct)
    self.assertEqual(
        unwrapped, {"a": 1.0, "b": [2.0, 3.0], "c": (4.0, {"d": 1.0})}
    )

    # get_pq_leaves
    leaves = pquant.get_pq_leaves(struct)
    self.assertLen(leaves, 5)
    self.assertEqual(leaves, [q1, q2, 3.0, 4.0, q1])

  def test_jnp_wrapper_linear_ops(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)

    q1 = pquant.Length(1.0)
    q2 = pquant.Length(2.0)
    q3 = pquant.Mass(1.0)

    # add
    res = wrapped_jnp.add(q1, q2)
    self.assertEqual(res.value, 3.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    with self.assertRaises(ValueError):
      _ = wrapped_jnp.add(q1, q3)

    # maximum / minimum
    res = wrapped_jnp.maximum(q1, q2)
    self.assertEqual(res.value, 2.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    with self.assertRaises(ValueError):
      _ = wrapped_jnp.maximum(q1, q3)

    # clip
    res = wrapped_jnp.clip(q1, 0.0, q2)
    self.assertEqual(res.value, 1.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

  def test_jnp_wrapper_where(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    cond = jnp.array([True, False])
    q1 = pquant.Length(jnp.array([1.0, 2.0]))
    q2 = pquant.Length(jnp.array([3.0, 4.0]))
    q3 = pquant.Mass(jnp.array([5.0, 6.0]))

    # Matching dimensions
    res = wrapped_jnp.where(cond, q1, q2)
    np.testing.assert_allclose(res.value, jnp.array([1.0, 4.0]))
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # Dimension mismatch
    with self.assertRaises(ValueError):
      _ = wrapped_jnp.where(cond, q1, q3)

  def test_jnp_wrapper_transcendental(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    q = pquant.Dimensionless(1.0)

    # Should work for dimensionless
    res = wrapped_jnp.exp(q)
    self.assertAlmostEqual(float(res.value), np.exp(1.0))
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

    # Should work for non-dimensionless (forces to dimensionless)
    q_len = pquant.Length(1.0)
    res = wrapped_jnp.exp(q_len)
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

  def test_jnp_wrapper_interp(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    x = jnp.array([1.5])
    xp = jnp.array([1.0, 2.0])
    fp = pquant.Length(jnp.array([10.0, 20.0]))

    res = wrapped_jnp.interp(x, xp, fp)
    np.testing.assert_allclose(res.value, jnp.array([15.0]))
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

  def test_jnp_wrapper_stacking(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    q1 = pquant.Length(1.0)
    q2 = pquant.Length(2.0)
    q3 = pquant.Mass(3.0)

    # stack matching
    res = wrapped_jnp.stack([q1, q2])
    np.testing.assert_allclose(res.value, jnp.array([1.0, 2.0]))
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # stack mismatch
    with self.assertRaises(ValueError):
      _ = wrapped_jnp.stack([q1, q3])

  def test_jnp_wrapper_reductions(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    q = pquant.Length(jnp.array([1.0, 2.0, 3.0]))

    # sum
    res = wrapped_jnp.sum(q)
    self.assertEqual(res.value, 6.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # mean
    res = wrapped_jnp.mean(q)
    self.assertEqual(res.value, 2.0)
    self.assertEqual(res.dimensions, pquant.Length.dimensions)

    # var (variance should double dimensions)
    res = wrapped_jnp.var(q)
    self.assertAlmostEqual(float(res.value), 2.0 / 3.0)
    self.assertEqual(res.dimensions, pquant.Area.dimensions)

  def test_jnp_wrapper_matrix_multiplication(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)

    # Vector dot product
    q1 = pquant.Length(jnp.array([1.0, 2.0]))
    q2 = pquant.Mass(jnp.array([3.0, 4.0]))
    res = wrapped_jnp.dot(q1, q2)
    self.assertEqual(res.value, 11.0)
    self.assertEqual(res.dimensions, (1, 1, 0, 0))

    # Matmul
    a = pquant.Area(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    b = pquant.Density(jnp.array([[5.0, 6.0], [7.0, 8.0]]))
    res = wrapped_jnp.matmul(a, b)
    expected_val = jnp.matmul(a.value, b.value)
    np.testing.assert_allclose(res.value, expected_val)
    self.assertEqual(res.dimensions, (1, -1, 0, 0))

  def test_nn_wrapper(self):
    wrapped_jax = pquant.JaxWrapper(jax)
    q = pquant.Length(1.0)

    # jax.nn activation functions should return dimensionless
    res = wrapped_jax.nn.sigmoid(q)
    self.assertAlmostEqual(float(res.value), float(jax.nn.sigmoid(1.0)))
    self.assertEqual(res.dimensions, pquant.Dimensionless.dimensions)

  def test_instrument_module(self):
    # Create a dummy module to instrument
    class DummyModule:
      jax = jax
      jnp = jnp

    dummy = DummyModule()
    dummy.__name__ = "Dummy"

    pquant.instrument_module(dummy)

    self.assertIsInstance(dummy.jnp, pquant.JNPWrapper)
    self.assertIsInstance(dummy.jax, pquant.JaxWrapper)

  def test_jnp_wrapper_nanvar_and_fallback(self):
    wrapped_jnp = pquant.JNPWrapper(jnp)
    q = pquant.Length(jnp.array([1.0, 2.0, jnp.nan]))
    res = wrapped_jnp.nanvar(q)
    self.assertAlmostEqual(float(res.value), 0.25)
    self.assertEqual(res.dimensions, pquant.Area.dimensions)

    res_kw = wrapped_jnp.var(a=pquant.Length(jnp.array([1.0, 3.0])))
    self.assertAlmostEqual(float(res_kw.value), 1.0)
    self.assertEqual(res_kw.dimensions, pquant.Area.dimensions)

    res_raw = wrapped_jnp.var(jnp.array([1.0, 3.0]))
    self.assertNotIsInstance(res_raw, pquant.Quantity)

    res_dot_raw = wrapped_jnp.dot(
        pquant.Length([1.0, 2.0]), jnp.array([3.0, 4.0])
    )
    self.assertEqual(res_dot_raw.dimensions, pquant.Length.dimensions)


if __name__ == "__main__":
  absltest.main()
