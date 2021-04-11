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

# pylint: skip-file

from . import unet
from . import equations
from . import multigrid as MG
import functools
import jax

from jax.tree_util import Partial
import jax.numpy as np
from jax import lax


def _lin_iter(carry, i):
  A, x = carry
  return (A, A(x)), 0


@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, 0))
def losses_lin(lin_op, mesh, params, n, k, x0, b):
  multigrid = lambda x: MG._V_Cycle(x, b.reshape(mesh.shape), 3, 'R', k=k)
  iterator = lambda x: multigrid(x.reshape(mesh.shape)).ravel() + \
      lin_op(params, multigrid(x.reshape(mesh.shape)).ravel() - x)
  A = Partial(iterator)
  (_, x_opt), _ = jax.lax.scan(_lin_iter, (A, x0), np.arange(n))
  new_matvec = lambda x: mesh.matvec_helmholtz(k, 1.0, equations.make_mask,
                                               equations.make_mask_dual, x)
  return np.linalg.norm(new_matvec(x_opt) - b) * 10000000 / np.linalg.norm(b)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def loss_lin(lin_op, n, mesh, params, inputs, bs, x=0, k=0, **kwargs):
  return np.mean(losses_lin(lin_op, mesh, params, n, k, inputs, bs))


@functools.partial(jax.vmap, in_axes=(None, None, None, None, 0, 0, 0))
def losses_lin_supervised(lin_op, mesh, params, n, x0, b, solution):
  multigrid = lambda x: MG._V_Cycle(x, b.reshape(mesh.shape), 3, 'R')
  iterator = lambda x: multigrid(x.reshape(mesh.shape)).ravel() + \
      lin_op(params, multigrid(x.reshape(mesh.shape)).ravel() - x)
  A = Partial(iterator)
  (_, x_opt), _ = jax.lax.scan(_lin_iter, (A, x0), np.arange(n))
  return np.linalg.norm(x_opt - solution) * 10000000 / np.linalg.norm(solution)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def loss_lin_supervised(lin_op,
                        n,
                        mesh,
                        params,
                        inputs,
                        bs,
                        *args,
                        solutions = None,
                        **kwargs):
  return np.mean(
      losses_lin_supervised(lin_op, mesh, params, n, inputs, bs, solutions))


class LinUNet(unet.UNet):

  def __init__(self, *args, **kwargs):
    super(LinUNet, self).__init__(*args, **kwargs)
    self.loss = loss_lin_supervised
    self.test_loss = loss_lin

  @functools.partial(jax.jit, static_argnums=(0,))
  def preconditioner(self, params, x, k=0.0):
    correction = lambda x: self.net_apply(params, x.reshape(
        1, self.n, self.n, 1)).ravel()
    return correction(x)

  @functools.partial(jax.jit, static_argnums=(0,))
  def approximate_inverse(self, inputs):
    return self.preconditioner(self.opt_params,
                               inputs.reshape(1, self.n, self.n, 1)).reshape(-1)
