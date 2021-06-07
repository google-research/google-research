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

"""Tests for aqt.jax.primitives."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as onp

from aqt.jax import primitives
from aqt.jax import test_utils

fp32 = onp.float32
test_utils.configure_jax()


class PrimitivesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='floor_clip_uint1',
          prec=1,
      ),
      dict(
          testcase_name='floor_clip_uint4',
          prec=4,
      ),
      dict(
          testcase_name='floor_clip_uint8',
          prec=8,
      ),
      dict(
          testcase_name='floor_clip_uint16',
          prec=16,
      ),
  )
  def test_floor_and_clip_to_unsigned_int(self, prec):
    x = jnp.array(fp32(2.0**5 * onp.random.uniform(0, 1.0, size=(10, 1))))
    y = primitives.floor_and_clip_to_unsigned_int(x, prec=prec, dtype=x.dtype)
    self.assertGreaterEqual(onp.min(y), 0.0)
    self.assertLessEqual(onp.max(y), fp32(2**prec - 1))
    onp.testing.assert_allclose(y, onp.around(y))

  @parameterized.named_parameters(
      dict(testcase_name='round_clip_sint1', prec=1),
      dict(testcase_name='round_clip_sint4', prec=4),
      dict(testcase_name='round_clip_sint8', prec=8),
      dict(testcase_name='round_clip_sint16', prec=16))
  def test_round_and_clip_to_signed_int(self, prec):
    np_x = fp32(2.0**5 * onp.random.uniform(-1.0, 1.0, size=(10, 1)))
    x = jnp.array(np_x)
    y = primitives.round_and_clip_to_signed_int(x, prec=prec, dtype=x.dtype)
    ubound = fp32(2.0**(prec - 1) - 1)
    lbound = fp32(-2.0**(prec - 1) + 1)
    self.assertGreaterEqual(onp.min(y), lbound)
    self.assertLessEqual(onp.max(y), ubound)
    onp.testing.assert_allclose(y, onp.around(y))
    onp.testing.assert_allclose(
        y, onp.clip(onp.floor(np_x + 0.5), a_min=lbound, a_max=ubound))

  def test_round_and_clip_to_signed_int_half_shift(self):
    # centers_2bit = [-1.5, -0.5, 0.5, 1.5]
    q_inp = [-2.1, -1.9, -1.1, -0.9, -0.1, +0.1, +0.9, +1.1, +1.9, +2.1]
    q2bit = [-1.5, -1.5, -1.5, -0.5, -0.5, +0.5, +0.5, +1.5, +1.5, +1.5]
    q1bit = [-0.5, -0.5, -0.5, -0.5, -0.5, +0.5, +0.5, +0.5, +0.5, +0.5]
    q2gra = [+0.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +0.0]
    q1gra = [+0.0, +0.0, +0.0, +1.0, +1.0, +1.0, +1.0, +0.0, +0.0, +0.0]
    q_inp = jnp.array(q_inp)
    q2bit = jnp.array(q2bit)
    q1bit = jnp.array(q1bit)
    q2gra = jnp.array(q2gra)
    q1gra = jnp.array(q1gra)

    def quantize1bit(x):
      return primitives.round_and_clip_to_signed_int(
          x, prec=1, half_shift=True, dtype=jnp.float32)

    def quantize2bit(x):
      return primitives.round_and_clip_to_signed_int(
          x, prec=2, half_shift=True, dtype=jnp.float32)

    quantize1bit_grad = jax.vmap(jax.grad(quantize1bit))
    quantize2bit_grad = jax.vmap(jax.grad(quantize2bit))

    # TODO(lew): Resolve the error and use self.assertEqual instead.
    # ValueError: The truth value of an array with more than one element is
    # ambiguous. Use a.any() or a.all()
    def assert_equal(a, b):
      self.assertEqual(jnp.sum(jnp.abs(a - b)), 0.0)

    assert_equal(quantize1bit(q_inp), q1bit)
    assert_equal(quantize2bit(q_inp), q2bit)
    assert_equal(quantize1bit_grad(q_inp), q1gra)
    assert_equal(quantize2bit_grad(q_inp), q2gra)

  @parameterized.named_parameters(
      dict(
          testcase_name='uint8_scale_grow_range',
          prec=8,
          x_shape=(10, 1),
          x_min=3.0,
          x_max=10.0,
          axis=None,
      ),
      dict(
          testcase_name='uint8_scale_shrink_range',
          prec=8,
          x_shape=(10, 1),
          x_min=2.0,
          x_max=1000.0,
          axis=None,
      ),
      dict(
          testcase_name='uint8_scale_grow_range_per_ch',
          prec=8,
          x_shape=(2, 3, 3, 2),
          x_min=2.0,
          x_max=10.0,
          axis=(0, 1, 2),
      ),
      dict(
          testcase_name='uint4_input_all_zeros',
          prec=4,
          x_shape=(10, 1),
          x_min=0.0,
          x_max=0.0,
          axis=None,
      ),
      dict(
          testcase_name='uint1',
          prec=1,
          x_shape=(10, 1),
          x_min=2.0,
          x_max=10.0,
          axis=None,
      ),
  )
  def test_unsigned_int_scale(self, prec, x_shape, x_min, x_max, axis):
    x = jnp.array(fp32(onp.random.uniform(x_min, x_max, size=x_shape)))
    scale = primitives.unsigned_int_scale(x, prec=prec, axis=axis)
    scaled_x = jnp.multiply(x, scale)
    ubound = fp32(2.0**prec)
    if jnp.max(x) == 0:
      self.assertEqual(jnp.max(scaled_x), 0)
    else:
      if axis is None:
        self.assertAlmostEqual(jnp.max(scaled_x), ubound, places=3)
      else:  # TODO(wanglisa): confirm equiv of onp.testing.assert_allclose
        max_values = jnp.max(scaled_x, axis=axis)
        onp.testing.assert_allclose(max_values,
                                    ubound * jnp.ones(max_values.shape))

  @parameterized.named_parameters(
      dict(
          testcase_name='sint8_scale_grow_pos_range',
          prec=8,
          x_shape=(10, 1),
          x_min=0.0,
          x_max=10.0,
          cnstr=jnp.max,
          axis=None,
          bound=fp32(2.0**7 - 1),
      ),
      dict(
          testcase_name='sint8_scale_shrink_pos_range',
          prec=8,
          x_shape=(10, 1),
          x_min=0.0,
          x_max=1000.0,
          cnstr=jnp.max,
          axis=None,
          bound=fp32(2.0**7 - 1),
      ),
      dict(
          testcase_name='sint8_scale_grow_neg_range',
          prec=8,
          x_shape=(10, 1),
          x_min=-10.0,
          x_max=0.0,
          cnstr=jnp.min,
          axis=None,
          bound=fp32(-2.0**7 + 1),
      ),
      dict(
          testcase_name='sint8_scale_shrink_neg_range',
          prec=8,
          x_shape=(10, 1),
          x_min=-1000.0,
          x_max=0.0,
          cnstr=jnp.min,
          axis=None,
          bound=fp32(-2.0**7 + 1),
      ),
      dict(
          testcase_name='sint4_scale_shrink_neg_range_per_ch',
          prec=4,
          x_shape=(2, 3, 3, 2),
          x_min=-1000.0,
          x_max=0.0,
          cnstr=jnp.min,
          axis=(0, 1, 2),
          bound=fp32(-2.0**3 + 1),
      ),
      dict(
          testcase_name='sint4_scale_input_all_zeros',
          prec=4,
          x_shape=(2, 3, 3, 2),
          x_min=0.0,
          x_max=0.0,
          cnstr=jnp.min,
          axis=(0, 1, 2),
          bound=fp32(0.0),
      ),
      dict(
          testcase_name='sint1_pos',
          prec=1,
          x_shape=(10, 1),
          x_min=2.0,
          x_max=10.0,
          cnstr=jnp.max,
          axis=None,
          bound=fp32(0.25),
      ),
      dict(
          testcase_name='sint1_neg',
          prec=1,
          x_shape=(10, 1),
          x_min=-10.0,
          x_max=2.0,
          cnstr=jnp.min,
          axis=None,
          bound=fp32(-0.25),
      ),
  )
  def test_signed_int_scale(self, prec, x_shape, x_min, x_max, cnstr, axis,
                            bound):
    x = jnp.array(fp32(onp.random.uniform(x_min, x_max, size=x_shape)))
    scale = primitives.signed_int_scale(x, prec=prec, axis=axis)
    scaled_x = jnp.multiply(x, scale)
    if jnp.max(abs(x)) == 0:
      self.assertEqual(jnp.max(abs(scaled_x)), 0)
    else:
      if axis is None:
        self.assertAlmostEqual(cnstr(scaled_x), bound, places=4)
      else:  # TODO(wanglisa): confirm equiv of onp.testing.assert_allclose
        x_cnstr = cnstr(scaled_x, axis=axis)
        onp.testing.assert_allclose(x_cnstr, bound * jnp.ones(x_cnstr.shape))

  def test_signed_int_scale_missing_named_arg(self):
    x = jnp.ones((2, 1))
    prec = 4
    with self.assertRaises(TypeError):
      _ = primitives.signed_int_scale(x, prec, axis=None)

  @parameterized.named_parameters(
      dict(testcase_name='round', op_with_ste=primitives.round_with_gradient),
      dict(testcase_name='floor', op_with_ste=primitives.floor_with_gradient))
  def test_ste(self, op_with_ste):
    x = jnp.array([1.2, 3.8])
    op_grad = jax.grad(lambda x: jnp.sum(op_with_ste(x)))
    onp.testing.assert_array_equal(op_grad(x), 1.0)

  def test_grad_of_round_and_clip_to_signed_int(self):
    x = jnp.array([-1000.5, 5.2, 130.7])
    # Since -1000.5 and 900.3 are outside the 8-bit signed clipping range of
    # [-127, 127], we expect their gradient to be 0. Since the rounding and
    # int-casting use the STE, we expect the gradient of 5.2 to be 1.
    @jax.grad
    def grad_fn(x):
      return jnp.sum(
          primitives.round_and_clip_to_signed_int(x, prec=8, dtype=x.dtype))

    onp.testing.assert_array_equal(grad_fn(x), [0.0, 1.0, 0.0])

  def test_grad_of_round_and_clip_to_unsigned_int(self):
    x = jnp.array([-1.5, 5.2, 300.2])
    # Since -1.5 and 300.2 are outside the 8-bit unsigned clipping range of [0,
    # 255], we expect their gradient to be 0. Since the rounding and int-casting
    # use the STE, we expect the gradient of 5.2 to be 1.
    @jax.grad
    def grad_fn(x):
      return jnp.sum(
          primitives.floor_and_clip_to_unsigned_int(x, prec=8, dtype=x.dtype))

    onp.testing.assert_array_equal(grad_fn(x), [0.0, 1.0, 0.0])


if __name__ == '__main__':
  absltest.main()
