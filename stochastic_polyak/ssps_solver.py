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

"""Stochastic System Batch Polyak solver."""
# pylint: disable=invalid-name
import dataclasses
from typing import Any
from typing import Callable
from typing import NamedTuple

import jax
from jax import flatten_util
import jax.numpy as jnp

import jaxopt
from jaxopt import base
from jaxopt import tree_util

from jaxopt._src import linear_solve


class SystemStochasticPolyakState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  # value: float
  aux: Any


@dataclasses.dataclass
class SystemStochasticPolyak:
  """System Batch Stochastic Polyak solver.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model, ``*args`` and ``**kwargs`` are
      additional arguments.
    momentum: momentum parameter, between 0 and 1
    delta: damping coefficient
    learning_rate: a learning rate to use
    has_aux: whether ``fun`` outputs one (False) or more values (True). When
      True it will be assumed by default that ``fun(...)[0]`` is the objective
      value. The auxiliary outputs are stored in ``state.aux``.
    choose_update: variant to use. They are mathematically equivalent
      but have different runtimes.
    update: perform one update of the method
  """
  fun: Callable
  momentum: float = 0.0
  delta: float = 0.0
  learning_rate: float = 1.0
  has_aux: bool = False
  choose_update: int = 0

  def init(self, init_params):
    """Initialize the ``(params, state)`` pair.

    Args:
      init_params: pytree containing the initial parameters.
    Return type: base.OptStep

    Returns:
      (params, state)
    """
    # state = SystemStochasticPolyakState(iter_num=0, value=jnp.inf, aux=None)
    state = SystemStochasticPolyakState(iter_num=0, aux=None)
    return base.OptStep(params=init_params, state=state)

  def update_pytrees_CG(self, params, state,
                        epoch, data, *args, **kwargs):
    """Solves one iteration of the system Polyak solver calling directly CG.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      epoch: int.
      data: a batch of data.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type: base.OptStep

    Returns:
      (params, state)
    """
    del epoch, args, kwargs  # unused

    def losses(params):
      # Currently, self.fun returns the losses BEFORE the mean reduction.
      return self.fun(params, data)[0]

    # TODO(rmgower): avoid recomputing the auxiliary output (metrics)
    aux = self.fun(params, data)[1]

    # get Jacobian transpose operator
    Jt = jax.vjp(losses, params)[1]

    @jax.jit
    def matvec(u):
      """Matrix-vector product.

      Args:
        u:  vectors of length batch_size

      Returns:
        K: vector (J J^T + delta * I)u  = J(J^T(u)) +delta * u
      """
      ## Important: This is slow
      Jtu = Jt(u)  # evaluate Jacobian transpose vector product
      # evaluate Jacobian-vector product
      JJtu = jax.jvp(losses, (params,), (Jtu[0],))[1]
      deltau = self.delta*u
      return JJtu+deltau

    ## Solve the small linear system (J J^T +delta * I)x = -loss
    ## Warning: This is the bottleneck cost
    rhs = -losses(params)
    cg_sol = linear_solve.solve_cg(matvec, rhs, init=None, maxiter=20)

    ## Builds final solution w = w - J^T(J J^T +delta * I)^{-1}loss
    rhs = -losses(params)
    Jtsol = Jt(cg_sol)[0]
    new_params = tree_util.tree_add(params, Jtsol)

    if state.iter_num % 10 == 0:
      print('Number of iterations', state.iter_num,
            '. Objective function value: ', jnp.mean(-rhs))
    new_state = SystemStochasticPolyakState(
        iter_num=state.iter_num + 1, aux=aux)

    return base.OptStep(params=new_params, state=new_state)

  def update_jacrev_arrays_CG(self, params,
                              state, data,
                              *args, **kwargs):
    """Perform the update using jacrev and CG."""

    del args, kwargs  # unused
    # Currently the fastest implementation.
    batch_size = data['label'].shape[0]
    _, unravel_pytree = flatten_util.ravel_pytree(params)
    values = jnp.zeros((batch_size))
    @jax.jit
    def losses(params):
      # Currently, self.fun returns the losses BEFORE the mean reduction.
      return self.fun(params, data)[0]
    # TODO(rmgower): avoid recomputing the auxiliary output (metrics)
    aux = self.fun(params, data)[1]

    @jax.jit
    def matvec_array(u):  # Computes the product  (J J^T +delta * I)u
      out = grads@(u@grads)+self.delta*u
      return out

    def jacobian_builder(losses, params):
      grads_tree = jax.jacrev(losses)(params)
      grads, _ = flatten_util.ravel_pytree(grads_tree)
      grads = jnp.reshape(grads, (batch_size, int(grads.shape[0]/batch_size)))
      return grads
    ## Important: This is the bottleneck cost of this update!
    grads = jacobian_builder(losses, params)

    values = losses(params)

    # Solving  v =(J J^T +delta * I)^{-1}loss
    v = linear_solve.solve_cg(matvec_array, values, init=None, maxiter=10)

    ## Builds final update v= J^T(J J^T +delta * I)^{-1}loss
    v = v@grads

    v_tree = unravel_pytree(v)
    new_params = tree_util.tree_add_scalar_mul(params, -1.0, v_tree)
    value = jnp.mean(values)

    if state.iter_num % 10 == 0:
      print('Number of iterations', state.iter_num,
            '. Objective function value: ', value)

    new_state = SystemStochasticPolyakState(
        # iter_num=state.iter_num + 1, value=value, aux=aux)
        iter_num=state.iter_num + 1,
        aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def update_arrays_CG(self, params, state,
                       data, *args, **kwargs):
    """Perform the update using CG."""

    del kwargs  # unused
    batch_size = data['label'].shape[0]
    _, unravel_pytree = flatten_util.ravel_pytree(params)
    values = jnp.zeros((batch_size))

    @jax.jit
    def loss_sample(image, label):
      tmp_kwargs = {'data': {'image': image, 'label': label}}
      # compute a gradient on a single image/label pair
      if self.has_aux:
        # we only store the last value of aux
        (value_i,
         aux), grad_i = self._value_and_grad_fun(params, *args, **tmp_kwargs)
      else:
        value_i, grad_i = self._value_and_grad_fun(params, *args, **tmp_kwargs)
        aux = None
      grad_i_flatten, _ = flatten_util.ravel_pytree(grad_i)
      return value_i, aux, grad_i_flatten

    @jax.jit
    def matvec_array(u):
      """Computes the product  (J J^T +delta * I)u ."""
      out = grads@(u@grads)+self.delta*u
      return out
    # We add a new axis on data and labels so they have the correct
    # shape after vectorization by vmap, which removes the batch dimension
    ## Important: This is the bottleneck cost of this update!
    expand_data = jnp.expand_dims(data['image'], axis=1)
    expand_labels = jnp.expand_dims(data['label'], axis=1)
    values, aux, grads = jax.vmap(
        loss_sample, in_axes=(0, 0))(expand_data, expand_labels)
    grads = jax.vmap(loss_sample, in_axes=(0, 0))(expand_data, expand_labels)[2]

    # Solving  v =(J J^T +delta * I)^{-1}loss
    v = linear_solve.solve_cg(matvec_array, values, init=None, maxiter=20)
    ## Builds final update v= J^T(J J^T +delta * I)^{-1}loss
    v = v@grads

    v_tree = unravel_pytree(v)
    new_params = tree_util.tree_add_scalar_mul(params, -1.0, v_tree)
    value = jnp.mean(values)

    if state.iter_num % 10 == 0:
      print('Number of iterations', state.iter_num,
            '. Objective function value: ', value)

    new_state = SystemStochasticPolyakState(
        # iter_num=state.iter_num + 1, value=value, aux=aux)
        iter_num=state.iter_num + 1,
        aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def update_arrays_lstsq(self, params, state,
                          data, *args, **kwargs):
    """Perform the update using a least square solver."""

    del kwargs  # unused
    # This version makes use of the least-squares solver jnp.linalg.lstsq
    # which has two problems
    # 1. It is too slow because it computes a full svd (overkill) to solve
    # the systems
    # 2. It has no support for regularization
    batch_size = data['label'].shape[0]
    _, unravel_pytree = flatten_util.ravel_pytree(params)
    values = jnp.zeros((batch_size))

    @jax.jit
    def loss_sample(image, label):
      tmp_kwargs = {'data': {'image': image, 'label': label}}
      # compute a gradient on a single image/label pair
      if self.has_aux:
        # we only store the last value of aux
        (value_i,
         aux), grad_i = self._value_and_grad_fun(params, *args, **tmp_kwargs)
      else:
        value_i, grad_i = self._value_and_grad_fun(params, *args, **tmp_kwargs)
        aux = None
      grad_i_flatten, _ = flatten_util.ravel_pytree(grad_i)
      return value_i, aux, grad_i_flatten

    # we add a new axis on data and labels so they have the correct
    # shape after vectorization by vmap, which removes the batch dimension
    expand_data = jnp.expand_dims(data['image'], axis=1)
    expand_labels = jnp.expand_dims(data['label'], axis=1)
    values, aux, grads = jax.vmap(
        loss_sample, in_axes=(0, 0))(expand_data, expand_labels)
    grads = jax.vmap(loss_sample, in_axes=(0, 0))(expand_data, expand_labels)[2]

    # This is too slow. Need faster implementation
    v = jnp.linalg.lstsq(grads, values)[0]

    v_tree = unravel_pytree(v)
    new_params = tree_util.tree_add_scalar_mul(params, -1.0, v_tree)
    value = jnp.mean(values)

    if state.iter_num % 10 == 0:
      print('Number of iterations', state.iter_num,
            '. Objective function value: ', value)

    new_state = SystemStochasticPolyakState(
        # iter_num=state.iter_num + 1, value=value, aux=aux)
        iter_num=state.iter_num + 1,
        aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def update_pytrees_QP(self, params, state,
                        epoch, data, *args, **kwargs):
    """Performs one iteration of the system Polyak solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      epoch: int, epoch number.
      data: a batch of data.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type: base.OptStep

    Returns:
      (params, state)
    """

    del epoch, args, kwargs  # unused
    # The output of losses(params) is of shape size(batch).
    # Therefore, losses is a function from size(params) to size(batch).
    # Therefore the Jacobian is of shape size(batch) x size(params).
    def losses(params):
      # Currently, self.fun returns the losses BEFORE the mean reduction.
      return self.fun(params, data)[0]

    # TODO(rmgower): avoid recomputing the auxiliary output (metrics)
    aux = self.fun(params, data)[1]

    # Solves 0.5 ||w - w^t||^2 s.t. A w = b
    # where A is of shape size(batch) x size(params) and contains the gradients
    #       b is of shape size(batch) and b[i] = A w^t - loss_values[i]
    #       w = params

    # This is equivalent to solving 0.5 w^T Q w + <c, w> s.t. A w = b,
    # where Q = Identity and c = -w^t.
    def matvec_Q(params_Q, u):
      del params_Q  # ignored
      return u

    def matvec_A(params_A, u):
      del params_A  # ignored
      return jax.jvp(losses, (params,), (u,))[1]

    # Since A is the Jacobian of losses, A w^t is a JVP.
    # This computes the JVP and loss values along the way.
    loss_values, Awt = jax.jvp(losses, (params,), (params,))
    b = Awt - loss_values
    # Rob: Double wrong! Check again
    c = tree_util.tree_scalar_mul(-1.0, params)
    params_obj = (None, c)
    params_eq = (None, b)

    # Rob: Solves for primal and dual variables,
    # thus solves very large linear system.
    qp = jaxopt.QuadraticProgramming(
        matvec_Q=matvec_Q, matvec_A=matvec_A, maxiter=10)
    res = qp.run(params_obj=params_obj, params_eq=params_eq).params
    # res contains both the primal and dual variables
    # but we only need the primal variable.
    new_params = res[0]

    new_state = SystemStochasticPolyakState(
        iter_num=state.iter_num + 1, aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def __post_init__(self):
    # The different updates and ordered by how slow they our. update_arrays_CG
    # is the fastest, but the process keep getting killed!
    # So I suspect that there is an memory issue of some sort.
    if self.choose_update == 1:
      # Fastest so far
      self.update = self.update_arrays_CG
    elif self.choose_update == 2:
      # Second fastest, close contender
      self.update = self.update_jacrev_arrays_CG
    elif self.choose_update == 3:
      # Third, but most stable, the process doesn't get killed!
      self.update = self.update_pytrees_CG
    elif self.choose_update == 4:
      # Too slow
      self.update = self.update_arrays_lstsq
    else:
      # Too slow
      self.update = self.update_pytrees_QP

    # Pre-compile useful functions.
    def fun(params, data):
      losses, aux = self.fun(params, data)  # assumes has_aux = True
      return jnp.mean(losses), aux

    self._value_and_grad_fun = jax.value_and_grad(fun, has_aux=self.has_aux)
