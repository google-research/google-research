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

"""Tests for kernel transformations for the linear attention."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp

from imp.max.modeling import kernel_transformation as kt
from imp.max.modeling import stochastic as st

FIRST_RAND_SEED = 143567883590
SECOND_RAND_SEED = 847392817892
THIRD_RAND_SEED = 5939874023


class KernelTransformationAttentionTest(parameterized.TestCase):

  def test_estimator_accuracy_favorplus(self):

    # query -> [batch_size, length, num_heads, features]
    # key -> [batch_size, length, num_heads, features]
    # value -> [batch_size, length, num_heads, features]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, v_dim)
    query = random.normal(random.key(FIRST_RAND_SEED), shape_query)
    key = random.normal(random.key(SECOND_RAND_SEED), shape_key)
    value = random.normal(random.key(THIRD_RAND_SEED), shape_value)
    projection_matrix = st.get_gaussian_orth_rand_mat(
        random.key(0), nb_random_features, qk_dim
    )
    exact_attention_tensor = jnp.einsum('...LHD, ...THD->...LTH', query, key)
    exact_attention_tensor /= jnp.sqrt(qk_dim)
    exact_attention_tensor = jax.nn.softmax(exact_attention_tensor, axis=-2)
    exact_result = jnp.einsum(
        '...LTH, ...THD->...LHD', exact_attention_tensor, value
    )
    query_prime = kt.exp_softmax_kernel_transformation(
        query, key, True, projection_matrix
    )
    key_prime = kt.exp_softmax_kernel_transformation(
        key, query, False, projection_matrix
    )
    kv_tensor = jnp.einsum('...LHM, ...LHD->...LHMD', key_prime, value)
    approx_result = jnp.einsum(
        '...LHM, ...LHMD->...LHD', query_prime, kv_tensor
    )

    max_error = 1.05
    error = jnp.abs((exact_result - approx_result) / exact_result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)

  def test_estimator_accuracy_favorplusplus(self):

    # query -> [batch_size, length, num_heads, features]
    # key -> [batch_size, length, num_heads, features]
    # value -> [batch_size, length, num_heads, features]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, v_dim)
    query = random.normal(random.key(FIRST_RAND_SEED), shape_query)
    key = random.normal(random.key(SECOND_RAND_SEED), shape_key)
    value = random.normal(random.key(THIRD_RAND_SEED), shape_value)
    projection_matrix = st.get_gaussian_orth_rand_mat(
        random.key(0), nb_random_features, qk_dim
    )
    exact_attention_tensor = jnp.einsum('...LHD, ...THD->...LTH', query, key)
    exact_attention_tensor /= jnp.sqrt(qk_dim)
    exact_attention_tensor = jax.nn.softmax(exact_attention_tensor, axis=-2)
    exact_result = jnp.einsum(
        '...LTH, ...THD->...LHD', exact_attention_tensor, value
    )
    query_prime = kt.expplus_softmax_kernel_transformation(
        query, key, True, projection_matrix
    )
    key_prime = kt.expplus_softmax_kernel_transformation(
        key, query, False, projection_matrix
    )
    kv_tensor = jnp.einsum('...LHM, ...LHD->...LHMD', key_prime, value)
    approx_result = jnp.einsum(
        '...LHM, ...LHMD->...LHD', query_prime, kv_tensor
    )

    max_error = 1.05
    error = jnp.abs((exact_result - approx_result) / exact_result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)

  def test_estimator_accuracy_hyperbolic(self):

    # query -> [batch_size, length, num_heads, features]
    # key -> [batch_size, length, num_heads, features]
    # value -> [batch_size, length, num_heads, features]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, v_dim)
    query = random.normal(random.key(FIRST_RAND_SEED), shape_query)
    key = random.normal(random.key(SECOND_RAND_SEED), shape_key)
    value = random.normal(random.key(THIRD_RAND_SEED), shape_value)
    projection_matrix = st.get_gaussian_orth_rand_mat(
        random.key(0), nb_random_features, qk_dim
    )
    exact_attention_tensor = jnp.einsum('...LHD, ...THD->...LTH', query, key)
    exact_attention_tensor /= jnp.sqrt(qk_dim)
    exact_attention_tensor = jax.nn.softmax(exact_attention_tensor, axis=-2)
    exact_result = jnp.einsum(
        '...LTH, ...THD->...LHD', exact_attention_tensor, value
    )
    query_prime = kt.hyp_softmax_kernel_transformation(
        query, key, True, projection_matrix
    )
    key_prime = kt.hyp_softmax_kernel_transformation(
        key, query, False, projection_matrix
    )
    kv_tensor = jnp.einsum('...LHM, ...LHD->...LHMD', key_prime, value)
    approx_result = jnp.einsum(
        '...LHM, ...LHMD->...LHD', query_prime, kv_tensor
    )

    max_error = 1.2
    error = jnp.abs((exact_result - approx_result) / exact_result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)

  def test_estimator_accuracy_asymmetric_rfs(self):

    # query -> [batch_size, length, num_heads, features]
    # key -> [batch_size, length, num_heads, features]
    # value -> [batch_size, length, num_heads, features]

    qk_dim = 8
    v_dim = 10
    batch_size = 1
    length = 2
    num_heads = 1
    nb_random_features = 10000
    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, v_dim)
    query = random.normal(random.key(FIRST_RAND_SEED), shape_query)
    key = random.normal(random.key(SECOND_RAND_SEED), shape_key)
    value = random.normal(random.key(THIRD_RAND_SEED), shape_value)
    projection_matrix = st.get_gaussian_orth_rand_mat(
        random.key(0), nb_random_features, qk_dim
    )
    exact_attention_tensor = jnp.einsum('...LHD, ...THD->...LTH', query, key)
    exact_attention_tensor /= jnp.sqrt(qk_dim)
    exact_attention_tensor = jax.nn.softmax(exact_attention_tensor, axis=-2)
    exact_result = jnp.einsum(
        '...LTH, ...THD->...LHD', exact_attention_tensor, value
    )
    query_prime = kt.exparf_softmax_kernel_transformation(
        query, key, True, projection_matrix
    )
    key_prime = kt.exparf_softmax_kernel_transformation(
        key, query, False, projection_matrix
    )
    kv_tensor = jnp.einsum('...LHM, ...LHD->...LHMD', key_prime, value)
    approx_result = jnp.einsum(
        '...LHM, ...LHMD->...LHD', query_prime, kv_tensor
    )

    max_error = 1.05
    error = jnp.abs((exact_result - approx_result) / exact_result)
    self.assertLess(jnp.max(jnp.abs(error)), max_error)

if __name__ == '__main__':
  absltest.main()
