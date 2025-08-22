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

from absl.testing import parameterized
import jax
from jax import numpy as jnp
import tensorflow as tf

from . import impls
from . import primitives


def _jaxpr_has_primitive(jaxpr, prim_name):
  """A reimplementation of the fun of the same name in jax._src_dispatch."""
  for eqn in jaxpr.eqns:
    if prim_name in eqn.primitive.name:
      return True
    for subjaxpr in jax.core.subjaxprs(jaxpr):
      if _jaxpr_has_primitive(subjaxpr, prim_name):
        return True
  return False


class PrimitivesActingOnArraysTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._n_clients = 100
    self._impl_defs = impls.PlacedComputations(
        {'clients': self._n_clients},
    )
    self._primdefs, _ = primitives.register_primitives(
        {'clients': self._n_clients},
    )

  def test_broadcast_clients_evaluation(self):
    fn = self._primdefs['broadcast_clients']
    # Check that this function is callable.
    self.assertAllClose(fn(jnp.array(1.0)), jnp.ones(shape=[self._n_clients]))
    # Check that it's jittable.
    self.assertAllClose(
        jax.jit(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )
    # Check that its forward-diffable.
    self.assertAllClose(
        jax.jacfwd(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )
    # Also that it's reverse-diffable.
    self.assertAllClose(
        jax.jacrev(fn)(jnp.array(1.0)), jnp.ones(shape=[self._n_clients])
    )

  def test_broadcast_clients_closure_under_fad(self):
    fn = self._primdefs['broadcast_clients']
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(jnp.array(1.0))
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'broadcast_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(jnp.array(1.0))
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'sum_from_clients'))

  def test_sum_from_clients_evaluation(self):
    fn = self._primdefs['sum_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    # Check that this function is callable.
    self.assertAllClose(fn(clients_ones), jnp.array([1.0 * self._n_clients]))
    # Check that it's jittable.
    self.assertAllClose(
        jax.jit(fn)(clients_ones), jnp.array([1.0 * self._n_clients])
    )
    # Check that its forward-diffable.
    self.assertAllClose(
        jax.jacfwd(fn)(clients_ones), jnp.ones(shape=[1, 100, 1])
    )
    # Check that its reverse-diffable.
    self.assertAllClose(
        jax.jacrev(fn)(clients_ones), jnp.ones(shape=[1, self._n_clients, 1])
    )

  def test_broadcast_and_sum_from_clients_eval(self):
    fn = self._primdefs['sum_from_clients']

    def _broadcast_then_sum(x):
      broadcasted_x = self._primdefs['broadcast_clients'](x)
      return fn(broadcasted_x)

    # This thing corresponds to fwd-mode AD in our paper.
    self.assertAllClose(
        jax.jacfwd(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0 * self._n_clients]]),
    )

    # And here's reverse-ad.
    self.assertAllClose(
        jax.jacrev(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0 * self._n_clients]]),
    )

  def test_sum_from_clients_closure_under_fad(self):
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fn = self._primdefs['sum_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'sum_from_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'broadcast_clients'))

  def test_mean_from_clients_eval(self):
    fn = self._primdefs['mean_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    # Check that this function is callable.
    self.assertAllClose(fn(clients_ones), jnp.array([1.0]))
    # Check that it's jittable.
    self.assertAllClose(jax.jit(fn)(clients_ones), jnp.array([1.0]))
    # Check that its forward-diffable.
    self.assertAllClose(
        jax.jacfwd(fn)(clients_ones),
        1 / self._n_clients * jnp.ones(shape=[1, self._n_clients, 1]),
    )

  def test_broadcast_then_mean_from_clients_eval(self):
    fn = self._primdefs['mean_from_clients']

    def _broadcast_then_sum(x):
      broadcasted_x = self._primdefs['broadcast_clients'](x)
      return fn(broadcasted_x)

    # Again, let's do the forward-mode, reverse-mode checks.
    self.assertAllClose(
        jax.jacfwd(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0]]),
    )
    self.assertAllClose(
        jax.jacrev(_broadcast_then_sum)(jnp.array([1.0])),
        jnp.array([[1.0]]),
    )

  def test_mean_from_clients_closure_under_fad(self):
    # Check that the forward and reverse-mode derivatives generate the expected
    # primitives.
    fn = self._primdefs['mean_from_clients']
    clients_ones = jnp.ones(shape=[self._n_clients, 1])
    fwd_mode_jaxpr = jax.make_jaxpr(jax.jacfwd(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(fwd_mode_jaxpr, 'mean_from_clients'))
    rev_mode_jaxpr = jax.make_jaxpr(jax.jacrev(fn))(clients_ones)
    self.assertTrue(_jaxpr_has_primitive(rev_mode_jaxpr, 'broadcast_clients'))

  @parameterized.named_parameters(
      (
          'broadcast',
          'broadcast_clients',
          lambda _: jnp.array(1.0),
          lambda n: jnp.ones(shape=[n]),
      ),
      (
          'sum',
          'sum_from_clients',
          lambda n: jnp.ones(shape=[n, 1]),
          lambda n: jnp.ones(shape=[1, n, 1]),
      ),
      (
          'mean',
          'mean_from_clients',
          lambda n: jnp.ones(shape=[n, 1]),
          lambda n: 1 / n * jnp.ones(shape=[1, n, 1]),
      ),
  )
  def test_broadcast_clients_reverse_ad_with_symbolic_zero_and_jit(
      self, prim_name, arg_fn, result_fn
  ):
    fn = self._primdefs[prim_name]

    @jax.jit
    def duplicate_prim_result(x):
      return fn(x), fn(x)

    @jax.jit
    def ignore_prim_result(x):
      # Ignoring one result from this tuple-returning function triggers
      # reverse evaluation with a symbolic zero cotangent argument.
      y, _ = duplicate_prim_result(x)
      return y

    jac = jax.jacrev(ignore_prim_result)
    self.assertAllClose(
        jac(arg_fn(self._n_clients)), result_fn(self._n_clients)
    )


if __name__ == '__main__':
  tf.test.main()
