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

"""Tests for realnvp_model.

We test against the PyTorch DPI implementation at
https://github.com/HeSunPU/DPI.
"""
import unittest
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from score_prior.posterior_sampling import losses
from score_prior.posterior_sampling import realnvp_model

_INP_DIM = _OUT_DIM = 8
_BATCH_SIZE = 2
_LEARNING_RATE = 1e-3
_GRAD_CLIP = 1e-2

_ATOL = 1e-4


class RealnvpModelTest(unittest.TestCase):

  def test_affine_coupling_init_output(self):
    """Test output of initialized AffineCoupling module."""
    model = realnvp_model.AffineCoupling(
        _OUT_DIM, seqfrac=4, train=True)

    # Initialize model params and states.
    init_rng = jax.random.PRNGKey(1)
    z = jax.random.normal(init_rng, (_BATCH_SIZE, _INP_DIM))
    variables = model.init(init_rng, z, reverse=True)
    model_state, _ = flax.core.pop(variables, 'params')

    # Check output with initialized variables.
    inp = np.random.RandomState(0).normal(size=(_BATCH_SIZE, _INP_DIM))
    (out, logdet), model_state = model.apply(
        variables, inp, reverse=True, mutable=list(model_state.keys()))
    batch_stats = model_state['batch_stats']
    ref_init_out = np.array(
        [[1.7640524, 0.4001572, 0.978738, 2.2408931,
          1.867558, -0.9772779, 0.95008844, -0.1513572],
         [-0.10321885, 0.41059852, 0.14404356, 1.4542735,
          0.7610377, 0.12167501, 0.44386324, 0.33367434]]
    )
    np.testing.assert_allclose(out, ref_init_out, atol=_ATOL)
    np.testing.assert_allclose(logdet, np.zeros(2), atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['BatchNorm_0']['mean'], np.array([-0.00013112]), atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['BatchNorm_0']['var'], np.array([0.9000001]), atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['BatchNorm_1']['mean'], np.array([2.0003292e-5]),
        atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['BatchNorm_1']['var'], np.array([0.90000004]), atol=_ATOL)

  def test_affine_coupling_train(self):
    orders, reverse_orders = realnvp_model.get_orders(
        _OUT_DIM, n_flow=1)
    model = realnvp_model.RealNVP(
        _OUT_DIM, n_flow=1, orders=orders, reverse_orders=reverse_orders,
        seqfrac=4)

    # Check permutation list.
    ref_orders = [6, 2, 1, 7, 3, 0, 5, 4]
    ref_reverse_orders = [5, 2, 1, 4, 7, 6, 0, 3]
    self.assertSequenceEqual(list(model.orders[0]), ref_orders)
    self.assertSequenceEqual(list(model.reverse_orders[0]), ref_reverse_orders)

    def loss_fn(params, states, inp):
      variables = {'params': params, **states}
      (out, logdet), new_states = model.apply(
          variables, inp, reverse=True, mutable=list(states.keys()))
      loss = jnp.mean(out) - jnp.mean(logdet)
      return loss, (new_states, logdet)

    # Initialize model params and states.
    init_rng = jax.random.PRNGKey(1)
    z = jax.random.normal(init_rng, (_BATCH_SIZE, _INP_DIM))
    variables = model.init(init_rng, z, reverse=True)
    model_state, params = flax.core.pop(variables, 'params')

    # Initialize optimizer.
    optimizer = optax.adam(_LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Train.
    val_and_grad_fn = jax.jit(
        jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
    for i in range(100):
      inp = np.random.RandomState(i).normal(size=(_BATCH_SIZE, _INP_DIM))
      (loss, (model_state, logdet)), grad = val_and_grad_fn(
          params, model_state, inp)

      grad = losses.clip_grad(grad, grad_clip=_GRAD_CLIP)
      # Apply updates.
      updates, opt_state = optimizer.update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)

    ref_params = {
        'actnorm1': {
            'actnorm_loc': np.array([0.09861294]),
            'actnorm_log_scale_inv': np.array([0.09954955])
        },
        'actnorm2': {
            'actnorm_loc': np.array([0.10173539]),
            'actnorm_log_scale_inv': np.array([0.09912212])
        },
        'coupling1': {
            'BatchNorm_0': {
                'bias': np.array([0.00033231]),
                'scale': np.array([0.99909914])
            },
            'BatchNorm_1': {
                'bias': np.array([-0.13938409]),
                'scale': np.array([0.9991157])
            },
            'Dense_0': {
                'bias': np.array([-0.00020559]),
                'kernel': np.array(
                    [[0.03396557, -0.04279518, 0.03619083, -0.01210138]]).T
            },
            'Dense_1': {
                'bias': np.array([-0.00657576]),
                'kernel': np.array([[0.00066323]])
            },
            'ZeroFC_0': {
                'Dense_0': {
                    'bias': np.array(
                        [-0.10463703, -0.10620482, -0.1065182, -0.10537734,
                         0.10711132, 0.10711132, 0.10711132, 0.10711132]),
                    'kernel': np.array(
                        [[0.13773677, 0.1382578, 0.1384332, 0.13812076,
                          -0.13817428, -0.13817428, -0.13817428, -0.13817428]]),
                },
                'fc_scale': np.array(
                    [0.13764872, 0.13744526, 0.13728197, 0.13702656,
                     0.1384482, 0.1384482, 0.1384482, 0.1384482]),
            }
        },
        'coupling2': {
            'BatchNorm_0': {
                'bias': np.array([-0.01188065]),
                'scale': np.array([0.99897784])
            },
            'BatchNorm_1': {
                'bias': np.array([-0.14261203]),
                'scale': np.array([0.99741536])
            },
            'Dense_0': {
                'bias': np.array([-0.01446989]),
                'kernel': np.array(
                    [[-0.01981652, 0.03414352, -0.03213985, -0.04945593]]).T
            },
            'Dense_1': {
                'bias': np.array([-0.01164357]),
                'kernel': np.array([[0.08740333]])
            },
            'ZeroFC_0': {
                'Dense_0': {
                    'bias': np.array(
                        [-0.10447567, -0.10459194, -0.10409055, -0.10433649,
                         0.10877459, 0.10877461, 0.10877457, 0.1087746]),
                    'kernel': np.array(
                        [[0.12204264, 0.1265513, 0.12713438, 0.11383463,
                          -0.13898923, -0.13898928, -0.13898923, -0.13898925]]),
                },
                'fc_scale': np.array(
                    [0.13644303, 0.13680908, 0.13640362, 0.13696161,
                     0.13934673, 0.1393468, 0.13934673, 0.1393468]),
            }
        }
    }

    # Check loss.
    np.testing.assert_almost_equal(
        loss.item(), -3.2552779, decimal=-np.log10(_ATOL))
    # Check logdet.
    np.testing.assert_allclose(
        logdet, np.array([3.0265014, 3.0134401]), atol=_ATOL)
    # Check trained params.
    for param, ref_param in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(ref_params)):
      np.testing.assert_allclose(param, ref_param, atol=_ATOL)

    # Check batch statistics.
    ref_batch_stats = {
        'flow_1': {
            'coupling1': {
                'BatchNorm_0': {
                    'mean': np.array([0.02607972]),
                    'var': np.array([0.00116053]),
                },
                'BatchNorm_1': {
                    'mean': np.array([-6.573511e-5]),
                    'var': np.array([2.6561373e-5]),
                }
            },
            'coupling2': {
                'BatchNorm_0': {
                    'mean': np.array([0.02712584]),
                    'var': np.array([0.00241244]),
                },
                'BatchNorm_1': {
                    'mean': np.array([0.00494728]),
                    'var': np.array([0.00019153]),
                }
            }
        }
    }
    batch_stats = model_state['batch_stats']

    # Only check running means because running variances are computed
    # differently between PyTorch and Flax. PyTorch/TF use Bessel-corrected
    # variance, whereas Flax does not. Additionally, PyTorch/TF and Flax use
    # different algorithms for computing batch variance, leading to numerical
    # differences.
    np.testing.assert_allclose(
        batch_stats['flow_1']['coupling1']['BatchNorm_0']['mean'],
        ref_batch_stats['flow_1']['coupling1']['BatchNorm_0']['mean'],
        atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['flow_1']['coupling1']['BatchNorm_1']['mean'],
        ref_batch_stats['flow_1']['coupling1']['BatchNorm_1']['mean'],
        atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['flow_1']['coupling2']['BatchNorm_0']['mean'],
        ref_batch_stats['flow_1']['coupling2']['BatchNorm_0']['mean'],
        atol=_ATOL)
    np.testing.assert_allclose(
        batch_stats['flow_1']['coupling2']['BatchNorm_1']['mean'],
        ref_batch_stats['flow_1']['coupling2']['BatchNorm_1']['mean'],
        atol=_ATOL)


if __name__ == '__main__':
  unittest.main()
