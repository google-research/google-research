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

"""Flax Optimizer api."""

import dataclasses
from typing import Any, List, Tuple, Optional
import warnings

from flax import jax_utils
from flax import serialization
from flax import struct
from flax import traverse_util

import jax
import jax.numpy as jnp

# Backwards compatibility symbol import.
ModelParamTraversal = traverse_util.ModelParamTraversal


@struct.dataclass
class OptimizerState:
  step: jnp.ndarray
  param_states: Any


class OptimizerDef:
  """Base class for an optimizer defintion, which specifies the initialization and gradient application logic.

  See docstring of :class:`Optimizer` for more details.
  """

  def __init__(self, hyper_params):
    self.hyper_params = hyper_params
    warnings.warn(
        'Use `optax` instead of `flax.optim`. Refer to the update guide '
        'https://flax.readthedocs.io/en/latest/advanced_topics/optax_update_guide.html '
        'for detailed instructions.\n'
        '\n'
        '`flax.optim` was removed in public Flax 0.6.0', DeprecationWarning)

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply a gradient for a single parameter.

    Args:
      step: the current step of the optimizer.
      hyper_params: a named tuple of hyper parameters.
      param: the parameter that should be updated.
      state: a named tuple containing the state for this parameter
      grad: the gradient tensor for the parameter.
    Returns:
      A tuple containing the new parameter and the new state.
    """
    raise NotImplementedError()

  def init_param_state(self, param):
    """Initializes the state for a parameter.

    Args:
      param: the parameter for which to initialize the state.
    Returns:
      A named tuple containing the initial optimization state for the parameter.
    """
    raise NotImplementedError()

  def apply_gradient(self, hyper_params, params, state, grads):
    """Applies a gradient for a set of parameters.

    Args:
      hyper_params: a named tuple of hyper parameters.
      params: the parameters that should be updated.
      state: a named tuple containing the state of the optimizer
      grads: the gradient tensors for the parameters.
    Returns:
      A tuple containing the new parameters and the new optimizer state.
    """
    step = state.step
    params_flat, treedef = jax.tree_util.tree_flatten(params)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)
    out = [self.apply_param_gradient(step, hyper_params, param, state, grad)
           for param, state, grad in zip(params_flat, states_flat, grads_flat)]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params = jax.tree_util.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_util.tree_unflatten(treedef, new_states_flat)
    new_state = OptimizerState(step + 1, new_param_states)
    return new_params, new_state

  def init_state(self, params):
    param_states = jax.tree_util.tree_map(self.init_param_state, params)
    state = OptimizerState(jnp.asarray(0, dtype=jnp.int32), param_states)
    return state

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from Optimizer apply_gradient to create the
    hyper parameters for a specific optimization step.

    Args:
      **hyper_param_overrides: the hyper parameters updates
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      The new hyper parameters.
    """
    hp = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hp = hp.replace(**hyper_param_overrides)
    return hp

  def create(self, target, focus = None):
    """Creates a new optimizer for the given target.

    See docstring of :class:`Optimizer` for more details.

    Args:
      target: the object to be optimized. This is typically a variable dict
        returned by `flax.linen.Module.init()`, but it can also be a container
        of variables dicts, e.g. `(v1, v2)` and  `('var1': v1, 'var2': v2)`
        are valid inputs as well.
      focus: a `flax.traverse_util.Traversal` that selects which subset of
        the target is optimized. See docstring of :class:`MultiOptimizer`
        for an example of how to define a `Traversal` object.
    Returns:
      An instance of `Optimizer`.
    """
    opt_def = self
    if focus:
      opt_def = MultiOptimizer((focus, opt_def))
    state = opt_def.init_state(target)
    return Optimizer(opt_def, state, target)

  def state_dict(self, target, state):
    return serialization.to_state_dict({
        'target': serialization.to_state_dict(target),
        'state': serialization.to_state_dict(state)
    })

  def restore_state(self, opt_target, opt_state, state_dict):
    """Restore the optimizer target and state from the state dict.

    This function accepts the current optimizer target and state. This
    lets us know the exact structure of the optimizer target and state,
    as well as lets us add assertions that shapes and dtypes don't change.

    In practice, no values in `opt_target` and `opt_state` are actually
    used. Only the tree structure, shapes and types.

    Args:
      opt_target: the optimizer target.
      opt_state: the optimizer state.
      state_dict: the state dict containing the desired new state of the
                  optimizer.
    Returns:
      a tuple of the optimizer target and state with the restored values from
      the state dict.
    """

    opt_target = serialization.from_state_dict(opt_target, state_dict['target'])
    opt_state = serialization.from_state_dict(opt_state, state_dict['state'])
    return opt_target, opt_state


class _NoAux:
  """Placeholder used to indicate a lack of auxilairy outputs."""
  pass


class Optimizer(struct.PyTreeNode):
  """
  Flax optimizers are created using the :class:`OptimizerDef` class. That class
  specifies the initialization and gradient application logic. Creating an
  optimizer using the :meth:`OptimizerDef.create` method will result in an
  instance of the :class:`Optimizer` class, which encapsulates the optimization
  target and state. The optimizer is updated using the method
  :meth:`apply_gradient`.

  Example of constructing an optimizer for a model::

    from flax import optim
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    optimizer = optimizer_def.create(model)

  The optimizer is then used in a training step as follows::

    def train_step(optimizer, data):
      def loss_fn(model):
        y = model(data)
        loss = ... # compute the loss
        aux = ... # compute auxiliary outputs (eg. training metrics)
        return loss, aux
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, aux), grad = grad_fn(optimizer.target)
      new_optimizer = optimizer.apply_gradient(grad)
      return new_optimizer, loss, aux


  Distributed training only requires a few extra additions::

    from flax import optim
    optimizer_def = optim.GradientDescent(learning_rate=0.1)
    optimizer = optimizer_def.create(model)
    optimizer = jax_utils.replicate(optimizer)

    def train_step(optimizer, data):
      def loss_fn(model):
        y = model(data)
        loss = ... # compute the loss
        aux = ... # compute auxiliary outputs (eg. training metrics)
        return loss, aux
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, aux), grad = grad_fn(optimizer.target)
      grad = jax.lax.pmean(grad, 'batch')
      new_optimizer = optimizer.apply_gradient(grad)
      return new_optimizer, loss, aux

    distributed_train_step = jax.pmap(train_step, axis_name='batch')

  Attributes:
    optimizer_def: The optimizer definition.
    state: The initial state of the optimizer.
    target: The target to optimizer."""

  optimizer_def: OptimizerDef = struct.field(pytree_node=False)
  state: Any = struct.field(pytree_node=True)
  target: Any = struct.field(pytree_node=True)

  def apply_gradient(self, grads, **hyper_param_overrides):
    """Applies a pytree of gradients to the target.

    Args:
      grads: A pytree of gradients.
      **hyper_param_overrides: the hyper parameters passed to apply_gradient
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      A new optimizer with the updated target and state.
    """
    hyper_params = self.optimizer_def.update_hyper_params(
        **hyper_param_overrides)
    new_target, new_state = self.optimizer_def.apply_gradient(
        hyper_params, self.target, self.state, grads)
    return self.replace(target=new_target, state=new_state)

  def compute_gradient(self, loss_fn):
    """Computes gradient of loss_fn.

    DEPRECATION WARNING:
    compute_gradient() is deprecated.
    Use jax.grad() or jax.value_and_grad() instead.

    Args:
      loss_fn: a function that receives the target and returns a loss or a
        tuple of the loss and auxiliary outputs.
    Returns:
      A tuple consisting of the loss, auxiliary outputs if any,
        and a list of gradient.
    """
    warnings.warn('compute_gradient() will be removed soon.'
                  ' Use jax.grad() or jax.value_and_grad()'
                  'instead.',
                  DeprecationWarning)
    def loss_wrapper(target):
      loss_and_aux = loss_fn(target)
      if isinstance(loss_and_aux, jnp.ndarray):
        return loss_and_aux, _NoAux
      else:
        return loss_and_aux
    grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
    (loss, aux), grad = grad_fn(self.target)
    if aux is _NoAux:
      return loss, grad
    else:
      return loss, aux, grad
  compute_gradients = compute_gradient

  def optimize(self, loss_fn, **hyper_param_overrides):
    """Optimizes the target with respect to a loss function.

    DEPRECATION WARNING:
    optimize() is deprecated.
    Use jax.grad() or jax.value_and_grad() and apply_gradient() instead.

    Args:
      loss_fn:  function that receives the target and returns a loss or a
        tuple of the loss and auxiliary outputs.
      **hyper_param_overrides: the hyper parameters passed to apply_gradient
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      A tuple consisting of the new optimizer, the loss,
        and the auxiliary outputs if any.
    """
    warnings.warn('optimize() will be removed soon.'
                  ' Use jax.grad() or jax.value_and_grad()'
                  'and apply_gradient() instead.',
                  DeprecationWarning)

    output_and_grad = self.compute_gradient(loss_fn)
    grad = output_and_grad[-1]
    optimizer = self.apply_gradient(grad, **hyper_param_overrides)
    return (optimizer,) + output_and_grad[:-1]

  def replicate(self, devices=None, axis_name='batch'):
    """Replicates an optimizer for data parallel training.

    A replicated optimizer will automatically average the gradients across
    devices. For this to work correctly the optimize method should be called
    within the context of a `jax.pmap` call with the correct axis_name.

    DEPRECATION WARNING:
    replicate() is deprecated.
    Use jax_utils.replicate() instead.

    Args:
      devices: an optional list of devices defining which devices this optimizer
        is replicated to (default: all local devices).
      axis_name: the axis_name used for gradient averaging across devices.
    Returns:
      The replicated optimizer.
    """
    if devices is None:
      devices = jax.local_devices()
    optimizer_def = ReplicatedOptimizer(self.optimizer_def, devices, axis_name)
    optimizer = jax_utils.replicate(self, devices=devices)
    return optimizer.replace(optimizer_def=optimizer_def)

  def unreplicate(self):
    """Un-replicates an optimizer.

    This will create a new optimizer with the target and state of the first
    device this optimizer was replicated to. After this call the optimizer
    and the target can be used outside of a `jax.pmap` call.

    DEPRECATION WARNING:
    unreplicate() is deprecated.
    Use jax_utils.unreplicate() instead.

    Returns:
      The optimizer that is no longer replicated.
    """
    if not isinstance(self.optimizer_def, ReplicatedOptimizer):
      raise ValueError('Cannot unreplicate an optimizer '
                       'that is not replicated.')
    optimizer_def = self.optimizer_def.optimizer_def
    optimizer = jax_utils.unreplicate(self)
    return optimizer.replace(optimizer_def=optimizer_def)

  def state_dict(self):
    return self.optimizer_def.state_dict(self.target, self.state)

  def restore_state(self, state):
    target, state = self.optimizer_def.restore_state(
        self.target, self.state, state)
    return self.replace(target=target, state=state)


# Optimizer serialization is handled by the state_dict and restore_dict methods
# of the OptimizerDef. Currently, this is used to store only a single copy of
# a replicated optimizer.
serialization.register_serialization_state(
    Optimizer, Optimizer.state_dict, Optimizer.restore_state,
    override=True)


class ReplicatedOptimizer(OptimizerDef):
  """Data parallel optimizer.

  DEPRECATION WARNING:
  ReplicatedOptimizer will be removed soon.
  Use `jax_utils.replicate(optimizer)` and `lax.pmean(grad)` to explicitly
  control the replication of the the optimizer and the cross replica averaging
  over gradients, respectively.
  """

  def __init__(self, optimizer_def, devices=None, axis_name='batch'):
    super().__init__(optimizer_def.hyper_params)
    if devices is None:
      devices = jax.local_devices()
    self.optimizer_def = optimizer_def
    self.devices = devices
    self.axis_name = axis_name

  def init_state(self, params):
    return self.optimizer_def.init_state(params)

  def _cross_replica_mean(self, grad):
    axis_size = jax.lax.psum(1, axis_name=self.axis_name)
    return jax.lax.psum(grad, axis_name=self.axis_name) / axis_size

  def apply_gradient(self, hyper_params, params, state, grads):
    grads = jax.tree_util.tree_map(self._cross_replica_mean, grads)
    return self.optimizer_def.apply_gradient(hyper_params, params, state, grads)

  def update_hyper_params(self, **hyper_param_overrides):
    return self.optimizer_def.update_hyper_params(**hyper_param_overrides)

  def state_dict(self, target, state):
    state_dict = self.optimizer_def.state_dict(target, state)
    # only the first copy of the parameters and optimizer state are stored.
    state_dict = jax.tree_util.tree_map(lambda x: x[0], state_dict)
    return state_dict

  def restore_state(self, target, opt_state, state_dict):
    # replicate the parameters and state to all devices.
    state_dict = jax_utils.replicate(state_dict, devices=self.devices)
    return self.optimizer_def.restore_state(target, opt_state, state_dict)


@dataclasses.dataclass
class _ShapeDtype:
  shape: Any
  dtype: Any
  _value: Any
  _indices: List[int]

  @classmethod
  def create(cls, value):
    if not isinstance(value, jnp.ndarray):
      value = jnp.array(value)
    return cls(shape=value.shape, dtype=value.dtype, _value=value, _indices=[])


class MultiOptimizer(OptimizerDef):
  """
  A MultiOptimizer is subclass of :class:`OptimizerDef` and useful for applying
  separate optimizer algorithms to various subsets of the model parameters.

  The example below creates two optimizers using
  :class:`flax.traverse_util.ModelParamTraversal`:
  one to optimize ``kernel`` parameters and to optimize ``bias`` parameters.
  Note each optimizer is created with a different learning rate::

    kernels = traverse_util.ModelParamTraversal(lambda path, _: 'kernel' in path)
    biases = traverse_util.ModelParamTraversal(lambda path, _: 'bias' in path)
    kernel_opt = optim.Momentum(learning_rate=0.01)
    bias_opt = optim.Momentum(learning_rate=0.1)
    opt_def = MultiOptimizer((kernels, kernel_opt), (biases, bias_opt))
    optimizer = opt_def.create(model)

  In order to train only a subset of the parameters, you can simply use a single
  :class:`flax.traverse_util.ModelParamTraversal` instance.

  If you want to update the learning rates of both optimizers online with
  different learning rate schedules, you should update the learning rates when
  applying the gradient. In the following example, the second optimizer is not
  doing any optimization during the first 1000 steps::

    hparams = optimizer.optimizer_def.hyper_params
    new_optimizer = optimizer.apply_gradient(
        grads,
        hyper_params=[
          hparams[0].replace(learning_rate=0.2),
          hparams[1].replace(learning_rate=jnp.where(step < 1000, 0., lr)),
        ])
  """

  def __init__(
      self,
      *traversals_and_optimizers):
    """Create a new MultiOptimizer.

    See docstring of :class:`MultiOptimizer` for more details.

    Args:
      *traversals_and_optimizers: pairs of flax.traverse_util.Traversal and
      `flax.optim.OptimizerDef` instances.
    """
    traversals, sub_optimizers = zip(*traversals_and_optimizers)
    hyper_params = [opt.hyper_params for opt in sub_optimizers]
    super().__init__(hyper_params)
    self.traversals = traversals
    self.sub_optimizers = sub_optimizers

  def init_state(self, params):
    param_states = jax.tree_util.tree_map(_ShapeDtype.create, params)
    overlap = False
    for idx, (traversal,
              opt) in enumerate(zip(self.traversals, self.sub_optimizers)):

      for match in traversal.iterate(param_states):
        match._indices.append(idx)
        overlap |= len(match._indices) > 1

    if overlap:
      raise ValueError(
          'Multiple optimizers match the same leaves : ' +
          str(jax.tree_util.tree_map(lambda match: match._indices, param_states)))
    for traversal, opt in zip(self.traversals, self.sub_optimizers):
      param_states = traversal.update(lambda x: opt.init_param_state(x._value), param_states)
    # Use None as initial state for params that are not optimized by any sub optimizer.
    param_states = jax.tree_util.tree_map(lambda x: None if isinstance(x, _ShapeDtype) else x, param_states)

    return OptimizerState(jnp.asarray(0, dtype=jnp.int32), param_states)

  def apply_gradient(self, hyper_params, params, state, grads):
    new_params = params
    it = zip(self.traversals, self.sub_optimizers, hyper_params)
    new_param_states = jax.tree_util.tree_map(_ShapeDtype.create, params)
    for focus, opt, hp in it:
      ps = tuple(focus.iterate(params))
      gs = tuple(focus.iterate(grads))
      ss = tuple(focus.iterate(state.param_states))
      prev_ss = OptimizerState(state.step, ss)
      new_ps, new_ss = opt.apply_gradient(hp, ps, prev_ss, gs)
      new_params = focus.set(list(new_ps), new_params)
      new_param_states = focus.set(list(new_ss.param_states), new_param_states)
    # Update state to None when param is not optimized by any sub optimizer.
    new_param_states = jax.tree_util.tree_map(
        lambda x: None if isinstance(x, _ShapeDtype) else x, new_param_states)
    return new_params, OptimizerState(state.step + 1, new_param_states)

  def update_hyper_params(self, **hyper_param_overrides):
    """Updates the hyper parameters with a set of overrides.

    This method is called from :meth:`Optimizer.apply_gradient` to create the
    hyper parameters for a specific optimization step.
    MultiOptimizer will apply the overrides for each sub optimizer.

    Args:
      **hyper_param_overrides: the hyper parameters updates
        will override the defaults specified in the `OptimizerDef`.
        Pass `hyper_params=...` to replace all hyper parameters.
    Returns:
      The new hyper parameters.
    """
    hps = hyper_param_overrides.pop('hyper_params', self.hyper_params)
    if hyper_param_overrides:
      hps = [hp.replace(**hyper_param_overrides) for hp in hps]
    return hps
