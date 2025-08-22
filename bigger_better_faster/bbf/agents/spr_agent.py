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

"""Implementation of BiggerBetterFaster (BBF) and SR-SPR in JAX."""

import collections
import copy
import functools
import itertools
import time

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.rainbow import rainbow_agent as dopamine_rainbow_agent
from dopamine.replay_memory import prioritized_replay_buffer
from flax.core.frozen_dict import FrozenDict
from flax.training import dynamic_scale as dynamic_scale_lib
import gin
import jax
import jax.lib.xla_bridge as xb
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

from bigger_better_faster.bbf import spr_networks
from bigger_better_faster.bbf.replay_memory import subsequence_replay_buffer


def _pmap_device_order():
  """Gets JAX's default device assignments as used in pmap."""
  if jax.process_count() == 1:
    return [
        d
        for d in xb.get_backend().get_default_device_assignment(
            jax.device_count()
        )
        if d.process_index == jax.process_index()
    ]
  else:
    return jax.local_devices()


def prefetch_to_device(iterator, size, devices=None, device_axis=False):
  """Shard and prefetch batches on device.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.

  This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
  necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
  location that isn't free yet so they don't block. Instead those allocators
  OOM.

  Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
      dimension is sharded across devices.
    size: the size of the prefetch buffer.  If you're training on GPUs, 2 is
      generally the best choice because this guarantees that you can overlap a
      training step on GPU with a data prefetch step on CPU.
    devices: the list of devices to which the arrays should be prefetched.
      Defaults to the order of devices expected by `jax.pmap`.
    device_axis: Whether or not to have a device axis. False will only place the
      data on the first device.

  Yields:
    The original items from the iterator where each ndarray is now a sharded to
    the specified devices.
  """
  queue = collections.deque()
  devices = devices or _pmap_device_order()
  def map_select(i, data):
    return jax.tree_util.tree_map(lambda x: x[i], data)

  @jax.jit
  def _shard(data):
    list_data = []
    for i in range(len(devices)):
      list_data.append(map_select(i, data))
    return list_data

  if not device_axis:
    def enqueue(n):
      for data in itertools.islice(iterator, n):
        queue.append(jax.device_put(data, device=jax.local_devices()[0]))

  else:

    def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
      for data in itertools.islice(iterator, n):
        queue.append(jax.device_put_sharded(_shard(data), devices))

  enqueue(size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def copy_within_frozen_tree(old, new, prefix):
  new_entry = old[prefix].copy(add_or_replace=new)
  return old.copy(add_or_replace={prefix: new_entry})


def copy_params(source, target, keys=("encoder", "transition_model")):
  """Copies a set of keys from one set of params to another.

  Args:
    source: Set of parameters to take keys from.
    target: Set of parameters to overwrite keys in.
    keys: Set of keys to copy.

  Returns:
    A parameter dictionary of the same shape as target.
  """
  if (
      isinstance(source, dict)
      or isinstance(source, collections.OrderedDict)
      or isinstance(source, FrozenDict)
  ):
    fresh_dict = {}
    for k, v in source.items():
      if k in keys:
        fresh_dict[k] = v
      else:
        fresh_dict[k] = copy_params(source[k], target[k], keys)
    return fresh_dict
  else:
    return target


@functools.partial(jax.jit, static_argnames=("keys", "strip_params_layer"))
def interpolate_weights(
    old_params,
    new_params,
    keys,
    old_weight=0.5,
    new_weight=0.5,
    strip_params_layer=True,
):
  """Interpolates between two parameter dictionaries.

  Args:
    old_params: The first parameter dictionary.
    new_params: The second parameter dictionary, of same shape and structure.
    keys: Which keys in the parameter dictionaries to interpolate. If None,
      interpolates everything.
    old_weight: The weight to place on the old dictionary.
    new_weight: The weight to place on the new dictionary.
    strip_params_layer: Whether to strip an outer "params" layer, as is often
      present in e.g., Flax.

  Returns:
    A parameter dictionary of the same shape as the inputs.
  """
  if strip_params_layer:
    old_params = old_params["params"]
    new_params = new_params["params"]

  def combination(old_param, new_param):
    return old_param * old_weight + new_param * new_weight

  combined_params = {}
  if keys is None:
    keys = old_params.keys()
  for k in keys:
    combined_params[k] = jax.tree_util.tree_map(combination, old_params[k],
                                                new_params[k])
  for k, v in old_params.items():
    if k not in keys:
      combined_params[k] = v

  if strip_params_layer:
    combined_params = {"params": combined_params}
  return FrozenDict(combined_params)


@functools.partial(
    jax.jit,
    static_argnames=(
        "do_rollout",
        "state_shape",
        "keys_to_copy",
        "shrink_perturb_keys",
        "reset_target",
        "network_def",
        "optimizer",
    ),
)
def jit_reset(
    online_params,
    target_network_params,
    optimizer_state,
    network_def,
    optimizer,
    rng,
    state_shape,
    do_rollout,
    support,
    reset_target,
    shrink_perturb_keys,
    shrink_factor,
    perturb_factor,
    keys_to_copy,
):
  """A jittable function to reset network parameters.

  Args:
    online_params: Parameter dictionary for the online network.
    target_network_params: Parameter dictionary for the target network.
    optimizer_state: Optax optimizer state.
    network_def: Network definition.
    optimizer: Optax optimizer.
    rng: JAX PRNG key.
    state_shape: Shape of the network inputs.
    do_rollout: Whether to do a dynamics model rollout (e.g., if SPR is being
      used).
    support: Support of the categorical distribution if using distributional RL.
    reset_target: Whether to also reset the target network.
    shrink_perturb_keys: Parameter keys to apply shrink-and-perturb to.
    shrink_factor: Factor to rescale current weights by (1 keeps , 0 deletes).
    perturb_factor: Factor to scale random noise by in [0, 1].
    keys_to_copy: Keys to copy over without resetting.

  Returns:
  """
  online_rng, target_rng = jax.random.split(rng, 2)
  state = jnp.zeros(state_shape, dtype=jnp.float32)
  # Create some dummy actions of arbitrary length to initialize the transition
  # model, if the network has one.
  actions = jnp.zeros((5,))
  random_params = network_def.init(
      x=state,
      actions=actions,
      do_rollout=do_rollout,
      rngs={
          "params": online_rng,
          "dropout": rng
      },
      support=support,
  )
  target_random_params = network_def.init(
      x=state,
      actions=actions,
      do_rollout=do_rollout,
      rngs={
          "params": target_rng,
          "dropout": rng
      },
      support=support,
  )

  if shrink_perturb_keys:
    online_params = interpolate_weights(
        online_params,
        random_params,
        shrink_perturb_keys,
        old_weight=shrink_factor,
        new_weight=perturb_factor,
    )
  online_params = FrozenDict(
      copy_params(online_params, random_params, keys=keys_to_copy))

  updated_optim_state = []
  optim_state = optimizer.init(online_params)
  for i in range(len(optim_state)):
    optim_to_copy = copy_params(
        dict(optimizer_state[i]._asdict()),
        dict(optim_state[i]._asdict()),
        keys=keys_to_copy,
    )
    optim_to_copy = FrozenDict(optim_to_copy)
    updated_optim_state.append(optim_state[i]._replace(**optim_to_copy))
  optimizer_state = tuple(updated_optim_state)

  if reset_target:
    if shrink_perturb_keys:
      target_network_params = interpolate_weights(
          target_network_params,
          target_random_params,
          shrink_perturb_keys,
          old_weight=shrink_factor,
          new_weight=perturb_factor,
      )
    target_network_params = copy_params(
        target_network_params, target_random_params, keys=keys_to_copy)
    target_network_params = FrozenDict(target_network_params)

  return online_params, target_network_params, optimizer_state, random_params


@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


def exponential_decay_scheduler(
    decay_period, warmup_steps, initial_value, final_value, reverse=False
):
  """Instantiate a logarithmic schedule for a parameter.

  By default the extreme point to or from which values decay logarithmically
  is 0, while changes near 1 are fast. In cases where this may not
  be correct (e.g., lambda) pass reversed=True to get proper
  exponential scaling.

  Args:
      decay_period: float, the period over which the value is decayed.
      warmup_steps: int, the number of steps taken before decay starts.
      initial_value: float, the starting value for the parameter.
      final_value: float, the final value for the parameter.
      reverse: bool, whether to treat 1 as the asmpytote instead of 0.

  Returns:
      A decay function mapping step to parameter value.
  """
  if reverse:
    initial_value = 1 - initial_value
    final_value = 1 - final_value

  start = onp.log(initial_value)
  end = onp.log(final_value)

  if decay_period == 0:
    return lambda x: initial_value if x < warmup_steps else final_value

  def scheduler(step):
    steps_left = decay_period + warmup_steps - step
    bonus_frac = steps_left / decay_period
    bonus = onp.clip(bonus_frac, 0.0, 1.0)
    new_value = bonus * (start - end) + end

    new_value = onp.exp(new_value)
    if reverse:
      new_value = 1 - new_value
    return new_value

  return scheduler


def get_lambda_weights(l, horizon):
  weights = jnp.ones((horizon - 1,)) * l
  weights = jnp.cumprod(weights) * (1 - l) / (l)
  weights = jnp.concatenate([weights, jnp.ones((1,)) * (1 - jnp.sum(weights))])
  return weights


@jax.jit
def tree_norm(tree):
  return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


@functools.partial(jax.jit,
                   static_argnames="num")
def jit_split(rng, num=2):
  return jax.random.split(rng, num)


@functools.partial(
    jax.jit,
    static_argnums=(0, 4, 5, 6, 7, 8, 10, 11, 13)
)
def select_action(
    network_def,
    params,
    state,
    rng,
    num_actions,
    eval_mode,
    epsilon_eval,
    epsilon_train,
    epsilon_decay_period,
    training_steps,
    min_replay_history,
    epsilon_fn,
    support,
    no_noise,
):
  """Select an action from the set of available actions."""

  rng, rng1 = jax.random.split(rng)
  state = spr_networks.process_inputs(
      state, rng=rng1, data_augmentation=False, dtype=jnp.float32
  )

  epsilon = jnp.where(
      eval_mode,
      epsilon_eval,
      epsilon_fn(
          epsilon_decay_period,
          training_steps,
          min_replay_history,
          epsilon_train,
      ),
  )

  def q_online(state, key, actions=None, do_rollout=False):
    return network_def.apply(
        params,
        state,
        actions=actions,
        do_rollout=do_rollout,
        key=key,
        support=support,
        rngs={"dropout": key},
        mutable=["batch_stats"],
        eval_mode=no_noise,
    )

  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  p = jax.random.uniform(rng1, shape=(state.shape[0],))
  rng2 = jax.random.split(rng2, state.shape[0])
  q_values = get_q_values_no_actions(q_online, state, rng2)

  best_actions = jnp.argmax(q_values, axis=-1)
  new_actions = jnp.where(
      p <= epsilon,
      jax.random.randint(
          rng3,
          (state.shape[0],),
          0,
          num_actions,
      ),
      best_actions,
  )
  return rng, new_actions


@functools.partial(jax.vmap, in_axes=(None, 0, 0), axis_name="batch")
def get_q_values_no_actions(model, states, rng):
  results = model(states, actions=None, do_rollout=False, key=rng)[0]
  return results.q_values


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0), axis_name="batch")
def get_logits(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.logits, results.latent, results.representation


@functools.partial(jax.vmap, in_axes=(None, 0, 0, None, 0), axis_name="batch")
def get_q_values(model, states, actions, do_rollout, rng):
  results = model(states, actions=actions, do_rollout=do_rollout, key=rng)[0]
  return results.q_values, results.latent, results.representation


@functools.partial(jax.vmap, in_axes=(None, 0, 0), axis_name="batch")
@functools.partial(jax.vmap, in_axes=(None, 0, None), axis_name="time")
def get_spr_targets(model, states, key):
  results = model(states, key)
  return results


train_static_argnums = (
    0,
    3,
    14,
    15,
    17,
    19,
    20,
    21,
    22,
    26,
    27,
)
train_donate_argnums = (1, 2, 4)


def train(
    network_def,  # 0, static
    online_params,  # 1
    target_params,  # 2
    optimizer,  # 3, static
    optimizer_state,  # 4
    raw_states,  # 5
    actions,  # 6
    raw_next_states,  # 7
    rewards,  # 8
    terminals,  # 9
    same_traj_mask,  # 10
    loss_weights,  # 11
    support,  # 12
    cumulative_gamma,  # 13
    double_dqn,  # 14, static
    distributional,  # 15, static
    rng,  # 16
    spr_weight,  # 17, static (gates rollouts)
    dynamic_scale,  # 18
    data_augmentation,  # 19, static
    dtype,  # 20, static
    batch_size,  # 21, static
    use_target_backups,  # 22, static
    target_update_tau,  # 23
    target_update_every,  # 24
    step,  # 25
    match_online_target_rngs,  # 26, static
    target_eval_mode,  # 27, static
):
  """Run one or more training steps for BBF.

  Args:
    network_def: Network definition class.
    online_params: Online parameter dictionary.
    target_params: Target parameter dictionary.
    optimizer: Optax optimizer.
    optimizer_state: Optax optimizer state.
    raw_states: Raw state inputs (not preprocessed, uint8), (B, T, H, W, C).
    actions: Actions (int32), (B, T).
    raw_next_states: Raw inputs for states at s_{t+n}, (B, T, H, W, C).
    rewards: Rewards, (B, T).
    terminals: Terminal signals, (B, T).
    same_traj_mask: Mask denoting valid continuations of trajectories, (B, T).
    loss_weights: Loss weights from prioritized replay sampling, (B,).
    support: support for the categorical distribution in C51, if used.
      (num_atoms,) array.
    cumulative_gamma: Discount factors (B,), gamma^n for the current n, gamma.
    double_dqn: Bool, whether to use double DQN.
    distributional: Bool, whether to use C51.
    rng: JAX PRNG Key.
    spr_weight: SPR loss weight, float.
    dynamic_scale: Dynamic scale object, if mixed precision is used.
    data_augmentation: Bool, whether to apply data augmentation.
    dtype: Jax dtype for training (float32, float16, or bfloat16)
    batch_size: int, size of each batch to run. Must cleanly divide the leading
      axis of input arrays. If smaller, the function will chain together
      multiple batches.
    use_target_backups: Bool, use target network for backups.
    target_update_tau: Float in [0, 1], tau for target network updates. 1 is
      hard (online target), 0 is frozen.
    target_update_every: How often to do a target update (in gradient steps).
    step: The current gradient step.
    match_online_target_rngs: whether to use the same RNG for online and target
      networks, to sync dropout etc.
    target_eval_mode: Whether to run the target network in eval mode (disabling
      dropout).

  Returns:
    Updated online params, target params, optimizer state, dynamic scale,
    and dictionary of metrics.
  """

  @functools.partial(jax.jit,
                     donate_argnums=(0,),
                     )
  def train_one_batch(state, inputs):
    """Runs a training step."""
    # Unpack inputs from scan
    (
        online_params,
        target_params,
        optimizer_state,
        dynamic_scale,
        rng,
        step,
    ) = state
    (
        raw_states,
        actions,
        raw_next_states,
        rewards,
        terminals,
        same_traj_mask,
        loss_weights,
        cumulative_gamma,
    ) = inputs
    same_traj_mask = same_traj_mask[:, 1:]
    rewards = rewards[:, 0]
    terminals = terminals[:, 0]
    cumulative_gamma = cumulative_gamma[:, 0]

    rng, rng1, rng2 = jax.random.split(rng, num=3)
    states = spr_networks.process_inputs(
        raw_states, rng=rng1, data_augmentation=data_augmentation, dtype=dtype
    )
    next_states = spr_networks.process_inputs(
        raw_next_states[:, 0],
        rng=rng2,
        data_augmentation=data_augmentation,
        dtype=dtype,
    )
    current_state = states[:, 0]

    # Split the current rng to update the rng after this call
    rng, rng1, rng2 = jax.random.split(rng, num=3)

    batch_rngs = jax.random.split(rng, num=states.shape[0])
    if match_online_target_rngs:
      target_rng = batch_rngs
    else:
      target_rng = jax.random.split(rng1, num=states.shape[0])
    use_spr = spr_weight > 0

    def q_online(state, key, actions=None, do_rollout=False):
      return network_def.apply(
          online_params,
          state,
          actions=actions,
          do_rollout=do_rollout,
          key=key,
          rngs={"dropout": key},
          support=support,
          mutable=["batch_stats"],
      )

    def q_target(state, key):
      return network_def.apply(
          target_params,
          state,
          key=key,
          support=support,
          eval_mode=target_eval_mode,
          rngs={"dropout": key},
          mutable=["batch_stats"],
      )

    def encode_project(state, key):
      return network_def.apply(
          target_params,
          state,
          key=key,
          rngs={"dropout": key},
          eval_mode=True,
          method=network_def.encode_project,
      )

    def loss_fn(
        params,
        target,
        spr_targets,
        loss_multipliers,
    ):
      """Computes the distributional loss for C51 or huber loss for DQN."""

      def q_online(state, key, actions=None, do_rollout=False):
        return network_def.apply(
            params,
            state,
            actions=actions,
            do_rollout=do_rollout,
            key=key,
            rngs={"dropout": key},
            support=support,
            mutable=["batch_stats"],
        )

      if distributional:
        (logits, spr_predictions, _) = get_logits(
            q_online, current_state, actions[:, :-1], use_spr, batch_rngs
        )
        logits = jnp.squeeze(logits)
        # Fetch the logits for its selected action. We use vmap to perform this
        # indexing across the batch.
        chosen_action_logits = jax.vmap(lambda x, y: x[y])(
            logits, actions[:, 0]
        )
        dqn_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
            target, chosen_action_logits)
        td_error = dqn_loss + jnp.nan_to_num(target * jnp.log(target)).sum(-1)
      else:
        q_values, spr_predictions, _ = get_q_values(
            q_online, current_state, actions[:, :-1], use_spr, batch_rngs
        )
        q_values = jnp.squeeze(q_values)
        replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions[:, 0])
        dqn_loss = jax.vmap(losses.huber_loss)(target, replay_chosen_q)
        td_error = dqn_loss

      if use_spr:
        spr_predictions = spr_predictions.transpose(1, 0, 2)
        spr_predictions = spr_predictions / jnp.linalg.norm(
            spr_predictions, 2, -1, keepdims=True)
        spr_targets = spr_targets / jnp.linalg.norm(
            spr_targets, 2, -1, keepdims=True)
        spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
        spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)
      else:
        spr_loss = 0

      loss = dqn_loss + spr_weight * spr_loss
      loss = loss_multipliers * loss

      mean_loss = jnp.mean(loss)

      aux_losses = {
          "TotalLoss": jnp.mean(mean_loss),
          "DQNLoss": jnp.mean(dqn_loss),
          "TD Error": jnp.mean(td_error),
          "SPRLoss": jnp.mean(spr_loss),
      }

      return mean_loss, (aux_losses)

    # Use the weighted mean loss for gradient computation.
    target = target_output(
        q_online,
        q_target,
        next_states,
        rewards,
        terminals,
        support,
        cumulative_gamma,
        double_dqn,
        distributional,
        use_target_backups,
        target_rng,
    )
    target = jax.lax.stop_gradient(target)

    if use_spr:
      future_states = states[:, 1:]
      spr_targets = get_spr_targets(encode_project, future_states, target_rng)
      spr_targets = spr_targets.transpose(1, 0, 2)
    else:
      spr_targets = None

    # Get the unweighted loss without taking its mean for updating priorities.

    if dynamic_scale:
      grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
      (
          dynamic_scale,
          is_fin,
          (_, aux_losses),
          grad,
      ) = grad_fn(
          online_params,
          target,
          spr_targets,
          loss_weights,
      )
    else:
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, aux_losses), grad = grad_fn(
          online_params,
          target,
          spr_targets,
          loss_weights,
      )

    grad_norm = tree_norm(grad)
    aux_losses["GradNorm"] = grad_norm
    updates, new_optimizer_state = optimizer.update(
        grad, optimizer_state, params=online_params)
    new_online_params = optax.apply_updates(online_params, updates)

    if dynamic_scale:
      # if is_fin == False the gradients contain Inf/NaNs and state and
      # params should be restored (= skip this step).
      optimizer_state = jax.tree_util.tree_map(
          functools.partial(jnp.where, is_fin),
          new_optimizer_state,
          optimizer_state,
      )
      online_params = jax.tree_util.tree_map(
          functools.partial(jnp.where, is_fin), new_online_params,
          online_params)
    else:
      optimizer_state = new_optimizer_state
      online_params = new_online_params

    target_update_step = functools.partial(
        interpolate_weights,
        keys=None,
        old_weight=1 - target_update_tau,
        new_weight=target_update_tau,
    )
    target_params = jax.lax.cond(
        step % target_update_every == 0,
        target_update_step,
        lambda old, new: old,
        target_params,
        online_params,
    )

    return (
        (
            online_params,
            target_params,
            optimizer_state,
            dynamic_scale,
            rng2,
            step + 1,
        ),
        aux_losses,
    )

  init_state = (
      online_params,
      target_params,
      optimizer_state,
      dynamic_scale,
      rng,
      step,
  )
  assert raw_states.shape[0] % batch_size == 0
  num_batches = raw_states.shape[0] // batch_size

  inputs = (
      raw_states.reshape(num_batches, batch_size, *raw_states.shape[1:]),
      actions.reshape(num_batches, batch_size, *actions.shape[1:]),
      raw_next_states.reshape(
          num_batches, batch_size, *raw_next_states.shape[1:]
      ),
      rewards.reshape(num_batches, batch_size, *rewards.shape[1:]),
      terminals.reshape(num_batches, batch_size, *terminals.shape[1:]),
      same_traj_mask.reshape(
          num_batches, batch_size, *same_traj_mask.shape[1:]
      ),
      loss_weights.reshape(num_batches, batch_size, *loss_weights.shape[1:]),
      cumulative_gamma.reshape(
          num_batches, batch_size, *cumulative_gamma.shape[1:]
      ),
  )

  (
      (
          online_params,
          target_params,
          optimizer_state,
          dynamic_scale,
          rng,
          step,
      ),
      aux_losses,
  ) = jax.lax.scan(train_one_batch, init_state, inputs)

  return (
      online_params,
      target_params,
      optimizer_state,
      dynamic_scale,
      {k: jnp.reshape(v, (-1,)) for k, v in aux_losses.items()},
  )


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, None, 0, None, None, None, 0),
    axis_name="batch",
)
def target_output(
    model,
    target_network,
    next_states,
    rewards,
    terminals,
    support,
    cumulative_gamma,
    double_dqn,
    distributional,
    use_target_backups,
    rng,
):
  """Builds the C51 target distribution or DQN target Q-values."""
  is_terminal_multiplier = 1.0 - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier

  if use_target_backups or not double_dqn:
    target_dist, _ = target_network(next_states, key=rng)
  if not use_target_backups or double_dqn:
    online_dist, _ = model(next_states, key=rng)

  backup_dist = target_dist if use_target_backups else online_dist
  # Double DQN uses the current network for the action selection
  select_dist = online_dist if double_dqn else target_dist
  # Action selection using Q-values for next-state
  q_values = jnp.squeeze(select_dist.q_values)
  next_qt_argmax = jnp.argmax(q_values)

  if distributional:
    # Compute the target Q-value distribution
    probabilities = jnp.squeeze(backup_dist.probabilities)
    next_probabilities = probabilities[next_qt_argmax]
    target_support = rewards + gamma_with_terminal * support
    target = dopamine_rainbow_agent.project_distribution(
        target_support, next_probabilities, support)
  else:
    # Compute the target Q-value
    next_q_values = jnp.squeeze(backup_dist.q_values)
    replay_next_qt_max = next_q_values[next_qt_argmax]
    target = rewards + gamma_with_terminal * replay_next_qt_max

  return jax.lax.stop_gradient(target)


@gin.configurable
def create_scaling_optimizer(
    name="adam",
    learning_rate=6.25e-5,
    beta1=0.9,
    beta2=0.999,
    eps=1.5e-4,
    centered=False,
    warmup=0,
    weight_decay=0.0,
    decay_bias=False,
):
  """Create an optimizer for training.

  Currently, only the Adam and RMSProp optimizers are supported.

  Args:
    name: str, name of the optimizer to create.
    learning_rate: float, learning rate to use in the optimizer.
    beta1: float, beta1 parameter for the optimizer.
    beta2: float, beta2 parameter for the optimizer.
    eps: float, epsilon parameter for the optimizer.
    centered: bool, centered parameter for RMSProp.
    warmup: int, warmup steps for learning rate.
    weight_decay: float, weight decay parameter for AdamW.
    decay_bias: bool, also apply weight decay to bias (rank 1) parameters.

  Returns:
    A flax optimizer.
  """
  if name == "adam":
    logging.info(
        ("Creating AdamW optimizer with settings lr=%f, beta1=%f, "
         "beta2=%f, eps=%f, wd=%f"),
        learning_rate,
        beta1,
        beta2,
        eps,
        weight_decay,
    )
    if decay_bias:
      mask = lambda p: True
    else:
      mask = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    if warmup == 0:
      return optax.adamw(
          learning_rate,
          b1=beta1,
          b2=beta2,
          eps=eps,
          weight_decay=weight_decay,
          mask=mask,
      )
    schedule = optax.linear_schedule(0, learning_rate, warmup)
    return optax.inject_hyperparams(optax.adamw)(
        learning_rate=schedule,
        b1=beta1,
        b2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        mask=mask,
    )
  elif name == "rmsprop":
    logging.info(
        "Creating RMSProp optimizer with settings lr=%f, beta2=%f, eps=%f",
        learning_rate,
        beta2,
        eps,
    )
    if warmup == 0:
      return optax.rmsprop(
          learning_rate, decay=beta2, eps=eps, centered=centered)
    schedule = optax.linear_schedule(0, learning_rate, warmup)
    return optax.inject_hyperparams(optax.rmsprop)(
        learning_rate=schedule, decay=beta2, eps=eps, centered=centered)
  else:
    raise ValueError("Unsupported optimizer {}".format(name))


@gin.configurable
class BBFAgent(dqn_agent.JaxDQNAgent):
  """A compact implementation of the full Rainbow agent."""

  def __init__(
      self,
      num_actions,
      noisy=False,
      dueling=True,
      double_dqn=True,
      distributional=True,
      data_augmentation=False,
      num_updates_per_train_step=1,
      network=spr_networks.RainbowDQNNetwork,
      num_atoms=51,
      vmax=10.0,
      vmin=None,
      jumps=0,
      spr_weight=0,
      batch_size=32,
      replay_ratio=64,
      batches_to_group=1,
      update_horizon=10,
      max_update_horizon=None,
      min_gamma=None,
      epsilon_fn=dqn_agent.linearly_decaying_epsilon,
      replay_scheme="uniform",
      replay_type="deterministic",
      reset_every=-1,
      no_resets_after=-1,
      reset_offset=1,
      encoder_warmup=0,
      head_warmup=0,
      learning_rate=0.0001,
      encoder_learning_rate=0.0001,
      reset_target=True,
      reset_head=True,
      reset_projection=True,
      reset_encoder=False,
      reset_noise=True,
      reset_priorities=False,
      reset_interval_scaling=None,
      shrink_perturb_keys="",
      perturb_factor=0.2,  # original was 0.1
      shrink_factor=0.8,  # original was 0.4
      target_update_tau=1.0,
      max_target_update_tau=None,
      cycle_steps=0,
      target_update_period=1,
      target_action_selection=False,
      eval_noise=True,
      use_target_network=True,
      match_online_target_rngs=True,
      target_eval_mode=False,
      offline_update_frac=0,
      summary_writer=None,
      half_precision=False,
      log_churn=True,
      verbose=False,
      seed=None,
      log_every=100,
  ):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      noisy: bool, Whether to use noisy networks or not.
      dueling: bool, Whether to use dueling network architecture or not.
      double_dqn: bool, Whether to use Double DQN or not.
      distributional: bool, whether to use distributional RL or not.
      data_augmentation: bool, Whether to use data augmentation or not.
      num_updates_per_train_step: int, Number of gradient updates every training
        step. Defaults to 1.
      network: flax.linen Module, neural network used by the agent initialized
        by shape in _create_network below. See
        dopamine.jax.networks.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [vmin, vmax].
      vmin: float, the value distribution support is [vmin, vmax]. If vmin is
        None, it is set to -vmax.
      jumps: int, number of steps to predict in SPR.
      spr_weight: float, weight of the SPR loss (in the format of Schwarzer et
        al's code, not their paper, so 5.0 is default.)
      batch_size: number of examples per batch.
      replay_ratio: Average number of times an example is replayed during
        training. Divide by batch_size to get the 'replay ratio' definition
        based on gradient steps by D'Oro et al.
      batches_to_group: Number of batches to group together into a single jit.
      update_horizon: int, n-step return length.
      max_update_horizon: int, n-step start point for annealing.
      min_gamma: float, gamma start point for annealing.
      epsilon_fn: function expecting 4 parameters: (decay_period, step,
        warmup_steps, epsilon). This function should return the epsilon value
        used for exploration during training.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      replay_type: str, 'deterministic' or 'regular', specifies the type of
        replay buffer to create.
      reset_every: int, how many training steps between resets. 0 to disable.
      no_resets_after: int, training step to cease resets before.
      reset_offset: offset to initial reset.
      encoder_warmup: warmup steps for encoder optimizer.
      head_warmup: warmup steps for head optimizer.
      learning_rate: Learning rate for all non-encoder parameters.
      encoder_learning_rate: Learning rate for the encoder (if different).
      reset_target: bool, whether to reset target network on resets.
      reset_head: bool, whether to reset head on resets.
      reset_projection: bool, whether to reset penultimate layer on resets.
      reset_encoder: bool, whether to reset encoder on resets.
      reset_noise: bool, whether to reset noisy nets noise parameters (no effect
        if noisy nets are disabled).
      reset_priorities: bool, whether to reset priorities in replay buffer.
      reset_interval_scaling: Optional float, ratio by which to increase reset
        interval at each reset.
      shrink_perturb_keys: string of comma-separated keys, such as
        'encoder,transition_model', to which to apply shrink & perturb to.
      perturb_factor: float, weight of random noise in shrink & perturb.
      shrink_factor: float, weight of initial parameters in shrink & perturb.
      target_update_tau: float, update parameter for EMA target network.
      max_target_update_tau: float, highest value of tau for annealing cycles.
      cycle_steps: int, number of steps to anneal hyperparameters after reset.
      target_update_period: int, steps per target network update.
      target_action_selection: bool, act according to the target network.
      eval_noise: bool, use noisy nets in evaluation.
      use_target_network: bool, enable the target network in training. Subtly
        different from setting tau=1.0, as it allows action selection according
        to an EMA policy, allowing decoupled investigation.
      match_online_target_rngs: bool, use the same JAX prng key for both online
        and target networks during training. Guarantees that dropout is aligned
        between the two networks.
      target_eval_mode: bool, run target network in eval mode, disabling dropout
        etc.
      offline_update_frac: float, fraction of a reset interval to do offline
        after each reset to warm-start the new network. summary_writer=None,
      summary_writer: SummaryWriter object, for outputting training statistics.
      half_precision: bool, use fp16 in training. Doubles training throughput,
        but may reduce final performance.
      log_churn: bool, log policy churn metrics.
      verbose: bool, also print metrics to stdout during training.
      seed: int, a seed for Jax RNG and initialization.
      log_every: int, training steps between metric logging calls.
    """
    logging.info(
        "Creating %s agent with the following parameters:",
        self.__class__.__name__,
    )
    logging.info("\t double_dqn: %s", double_dqn)
    logging.info("\t noisy_networks: %s", noisy)
    logging.info("\t dueling_dqn: %s", dueling)
    logging.info("\t distributional: %s", distributional)
    logging.info("\t data_augmentation: %s", data_augmentation)
    logging.info("\t replay_scheme: %s", replay_scheme)
    logging.info("\t num_updates_per_train_step: %d",
                 num_updates_per_train_step)
    # We need casting because passing arguments can convert ints to floats
    vmax = float(vmax)
    self._num_atoms = int(num_atoms)
    vmin = float(vmin) if vmin else -vmax
    self._support = jnp.linspace(vmin, vmax, self._num_atoms)
    self._replay_scheme = replay_scheme
    self._replay_type = replay_type
    self._double_dqn = bool(double_dqn)
    self._noisy = bool(noisy)
    self._dueling = bool(dueling)
    self._distributional = bool(distributional)
    self._data_augmentation = bool(data_augmentation)
    self._replay_ratio = int(replay_ratio)
    self._batch_size = int(batch_size)
    self._batches_to_group = int(batches_to_group)
    self.update_horizon = int(update_horizon)
    self._jumps = int(jumps)
    self.spr_weight = spr_weight
    self.log_every = int(log_every)
    self.verbose = verbose
    self.log_churn = log_churn

    self.reset_every = int(reset_every)
    self.reset_target = reset_target
    self.reset_head = reset_head
    self.reset_projection = reset_projection
    self.reset_encoder = reset_encoder
    self.reset_noise = reset_noise
    self.reset_priorities = reset_priorities
    self.offline_update_frac = float(offline_update_frac)
    self.no_resets_after = int(no_resets_after)
    self.cumulative_resets = 0
    self.reset_interval_scaling = reset_interval_scaling
    self.reset_offset = int(reset_offset)
    self.next_reset = self.reset_every + self.reset_offset

    self.encoder_warmup = int(encoder_warmup)
    self.head_warmup = int(head_warmup)
    self.learning_rate = learning_rate
    self.encoder_learning_rate = encoder_learning_rate

    self.shrink_perturb_keys = [
        s for s in shrink_perturb_keys.lower().split(",") if s
    ]
    self.shrink_perturb_keys = tuple(self.shrink_perturb_keys)
    self.shrink_factor = shrink_factor
    self.perturb_factor = perturb_factor

    self.eval_noise = eval_noise
    self.target_action_selection = target_action_selection
    self.use_target_network = use_target_network
    self.match_online_target_rngs = match_online_target_rngs
    self.target_eval_mode = target_eval_mode

    self.grad_steps = 0
    self.cycle_grad_steps = 0
    self.target_update_period = int(target_update_period)
    self.target_update_tau = target_update_tau

    if max_update_horizon is None:
      self.max_update_horizon = self.update_horizon
      self.update_horizon_scheduler = lambda x: self.update_horizon
    else:
      self.max_update_horizon = int(max_update_horizon)
      n_schedule = exponential_decay_scheduler(
          cycle_steps, 0, 1, self.update_horizon / self.max_update_horizon
      )
      self.update_horizon_scheduler = lambda x: int(  # pylint: disable=g-long-lambda
          onp.round(n_schedule(x) * self.max_update_horizon)
      )

    if max_target_update_tau is None:
      self.max_target_update_tau = target_update_tau
      self.target_update_tau_scheduler = lambda x: self.target_update_tau
    else:
      self.max_target_update_tau = max_target_update_tau
      self.target_update_tau_scheduler = exponential_decay_scheduler(
          cycle_steps,
          0,
          self.max_target_update_tau,
          self.target_update_tau,
      )

    logging.info("\t Found following local devices: %s",
                 str(jax.local_devices()))

    platform = jax.local_devices()[0].platform
    if half_precision:
      if platform == "tpu":
        self.dtype = jnp.bfloat16
        self.dtype_str = "bfloat16"
      else:
        self.dtype = jnp.float16
        self.dtype_str = "float16"
    else:
      self.dtype = jnp.float32
      self.dtype_str = "float32"

    logging.info("\t Running with dtype %s", str(self.dtype))

    super().__init__(
        num_actions=num_actions,
        network=functools.partial(
            network,
            num_atoms=self._num_atoms,
            noisy=self._noisy,
            dueling=self._dueling,
            distributional=self._distributional,
            dtype=self.dtype,
        ),
        epsilon_fn=epsilon_fn,
        target_update_period=self.target_update_period,
        update_horizon=self.max_update_horizon,
        summary_writer=summary_writer,
        seed=seed,
    )

    self.set_replay_settings()

    if min_gamma is None or cycle_steps <= 1:
      self.min_gamma = self.gamma
      self.gamma_scheduler = lambda x: self.gamma
    else:
      self.min_gamma = min_gamma
      self.gamma_scheduler = exponential_decay_scheduler(
          cycle_steps, 0, self.min_gamma, self.gamma, reverse=True
      )

    self.cumulative_gamma = (onp.ones(
        (self.max_update_horizon,)) * self.gamma).cumprod()

    self.train_fn = jax.jit(train, static_argnums=train_static_argnums,
                            device=jax.local_devices()[0])

  def _build_networks_and_optimizer(self):
    self._rng, rng = jax.random.split(self._rng)
    self.state_shape = self.state.shape

    # Create some dummy actions of arbitrary length to initialize the transition
    # model, if the network has one.
    actions = jnp.zeros((5,))
    self.online_params = self.network_def.init(
        x=self.state.astype(self.dtype),
        actions=actions,
        do_rollout=self.spr_weight > 0,
        rngs={
            "params": rng,
            "dropout": rng
        },
        support=self._support,
    )
    optimizer = create_scaling_optimizer(
        self._optimizer_name,
        warmup=self.head_warmup,
        learning_rate=self.learning_rate,
    )
    encoder_optimizer = create_scaling_optimizer(
        self._optimizer_name,
        warmup=self.encoder_warmup,
        learning_rate=self.encoder_learning_rate,
    )

    encoder_keys = {"encoder", "transition_model"}
    self.encoder_mask = FrozenDict({
        "params": {k: k in encoder_keys for k in self.online_params["params"]}
    })
    self.head_mask = FrozenDict({
        "params": {
            k: k not in encoder_keys for k in self.online_params["params"]
        }
    })

    self.optimizer = optax.chain(
        optax.masked(encoder_optimizer, self.encoder_mask),
        optax.masked(optimizer, self.head_mask),
    )

    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = copy.deepcopy(self.online_params)
    self.random_params = copy.deepcopy(self.online_params)

    self.online_params = jax.device_put(
        self.online_params, jax.local_devices()[0]
    )
    self.target_params = jax.device_put(
        self.target_network_params, jax.local_devices()[0]
    )
    self.random_params = jax.device_put(
        self.random_params, jax.local_devices()[0]
    )
    self.optimizer_state = jax.device_put(
        self.optimizer_state, jax.local_devices()[0]
    )

    if self.dtype == jnp.float16:
      self.dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
      self.dynamic_scale = None

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ["uniform", "prioritized"]:
      raise ValueError("Invalid replay scheme: {}".format(self._replay_scheme))
    if self._replay_type not in ["deterministic"]:
      raise ValueError("Invalid replay type: {}".format(self._replay_type))
    if self._replay_scheme == "prioritized":
      buffer = subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.max_update_horizon,
          gamma=self.gamma,
          subseq_len=self._jumps + 1,
          batch_size=self._batch_size,
          observation_dtype=self.observation_dtype,
      )
    else:
      buffer = subsequence_replay_buffer.JaxSubsequenceParallelEnvReplayBuffer(
          observation_shape=self.observation_shape,
          stack_size=self.stack_size,
          update_horizon=self.max_update_horizon,
          gamma=self.gamma,
          subseq_len=self._jumps + 1,
          batch_size=self._batch_size,
          observation_dtype=self.observation_dtype,
      )

    self.n_envs = buffer._n_envs  # pylint: disable=protected-access
    self.start = time.time()
    return buffer

  def set_replay_settings(self):
    logging.info(
        "\t Operating with %s environments, batch size %s and replay ratio %s",
        self.n_envs, self._batch_size, self._replay_ratio)
    self._num_updates_per_train_step = max(
        1, self._replay_ratio * self.n_envs // self._batch_size)
    self.update_period = max(
        1, self._batch_size // self._replay_ratio * self.n_envs
    )
    logging.info(
        "\t Calculated %s updates per update phase",
        self._num_updates_per_train_step,
    )
    logging.info(
        "\t Calculated update frequency of %s step%s",
        self.update_period,
        "s" if self.update_period > 1 else "",
    )
    logging.info(
        "\t Setting min_replay_history to %s from %s",
        self.min_replay_history / self.n_envs,
        self.min_replay_history,
    )
    logging.info(
        "\t Setting epsilon_decay_period to %s from %s",
        self.epsilon_decay_period / self.n_envs,
        self.epsilon_decay_period,
    )
    self.min_replay_history = self.min_replay_history / self.n_envs
    self.epsilon_decay_period = self.epsilon_decay_period / self.n_envs
    self._batches_to_group = min(self._batches_to_group,
                                 self._num_updates_per_train_step)
    assert self._num_updates_per_train_step % self._batches_to_group == 0
    self._num_updates_per_train_step = int(
        max(1, self._num_updates_per_train_step / self._batches_to_group))

    logging.info(
        "\t Running %s groups of %s batch%s per %s env step%s",
        self._num_updates_per_train_step,
        self._batches_to_group,
        "es" if self._batches_to_group > 1 else "",
        self.update_period,
        "s" if self.update_period > 1 else "",
    )

  def _replay_sampler_generator(self):
    types = self._replay.get_transition_elements()
    while True:
      self._rng, rng = jit_split(self._rng)

      samples = self._replay.sample_transition_batch(
          rng,
          batch_size=self._batch_size * self._batches_to_group,
          update_horizon=self.update_horizon_scheduler(self.cycle_grad_steps),
          gamma=self.gamma_scheduler(self.cycle_grad_steps),
      )
      replay_elements = collections.OrderedDict()
      for element, element_type in zip(samples, types):
        replay_elements[element_type.name] = element
      yield replay_elements

  def sample_eval_batch(self, batch_size, subseq_len=1):
    self._rng, rng = jit_split(self._rng)
    samples = self._replay.sample_transition_batch(
        rng, batch_size=batch_size, subseq_len=subseq_len)
    types = self._replay.get_transition_elements()
    replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      replay_elements[element_type.name] = element
    # Add code for data augmentation.

    return replay_elements

  def initialize_prefetcher(self):
    self.prefetcher = prefetch_to_device(self._replay_sampler_generator(), 2)

  def _sample_from_replay_buffer(self):
    self.replay_elements = next(self.prefetcher)

  def reset_weights(self):
    self.cumulative_resets += 1
    if self.reset_interval_scaling is None or not self.reset_interval_scaling:
      interval = self.reset_every
    elif str(self.reset_interval_scaling).lower() == "linear":
      interval = self.reset_every * (1 + self.cumulative_resets)
    elif "epoch" in str(self.reset_interval_scaling):
      epochs = float(self.reset_interval_scaling.replace("epochs:", ""))
      steps = (
          epochs * self._replay.num_elements() /
          (self._batch_size * self._num_updates_per_train_step *
           self._batches_to_group))
      interval = int(steps) + self.reset_every
    elif isinstance(self.reset_interval_scaling, float) or "." in str(
        self.reset_interval_scaling):
      interval = self.reset_every * float(self.reset_interval_scaling)**(
          self.cumulative_resets)
    else:
      raise NotImplementedError()

    self.next_reset = int(interval) + self.training_steps
    if self.next_reset > self.no_resets_after + self.reset_offset:
      logging.info(
          "\t Not resetting at step %s, as need at least"
          " %s before %s to recover.",
          self.training_steps, interval, self.no_resets_after
      )
      return
    else:
      logging.info("\t Resetting weights at step %s.",
                   self.training_steps)

    self._rng, reset_rng = jax.random.split(self._rng, 2)

    # These are the parameter entries that will be copied over unchanged
    # from the current dictionary to the new (randomly-initialized) one
    keys_to_copy = []
    if not self.reset_projection:
      keys_to_copy.append("projection")
      if self.spr_weight > 0:
        keys_to_copy.append("predictor")
    if not self.reset_encoder:
      keys_to_copy.append("encoder")
      if self.spr_weight > 0:
        keys_to_copy.append("transition_model")
    if not self.reset_noise:
      keys_to_copy += ["kernell", "biass"]
    if not self.reset_head:
      keys_to_copy += ["head"]
    keys_to_copy = tuple(keys_to_copy)

    if self.reset_priorities:
      self._replay.reset_priorities()
    (
        self.online_params,
        self.target_network_params,
        self.optimizer_state,
        self.random_params,
    ) = jit_reset(
        self.online_params,
        self.target_network_params,
        self.optimizer_state,
        self.network_def,
        self.optimizer,
        reset_rng,
        self.state_shape,
        self.spr_weight > 0,
        self._support,
        self.reset_target,
        self.shrink_perturb_keys,
        self.shrink_factor,
        self.perturb_factor,
        keys_to_copy,
    )
    self.online_params = jax.device_put(
        self.online_params, jax.local_devices()[0]
    )
    self.target_params = jax.device_put(
        self.target_network_params, jax.local_devices()[0]
    )
    self.random_params = jax.device_put(
        self.random_params, jax.local_devices()[0]
    )
    self.optimizer_state = jax.device_put(
        self.optimizer_state, jax.local_devices()[0]
    )

    self.cycle_grad_steps = 0

    if self._replay.add_count > self.min_replay_history:
      offline_steps = int(interval * self.offline_update_frac *
                          self._num_updates_per_train_step)

      logging.info(
          "Running %s gradient steps after reset",
          offline_steps * self._batches_to_group,
      )
      for i in range(1, offline_steps + 1):
        self._training_step_update(i, offline=True)

  def _training_step_update(self, step_index, offline=False):
    """Gradient update during every training step."""
    should_log = (
        self.training_steps % self.log_every == 0 and not offline and
        step_index == 0)
    interbatch_time = time.time() - self.start
    self.start = time.time()
    train_start = time.time()

    if not hasattr(self, "replay_elements"):
      self._sample_from_replay_buffer()
    if self._replay_scheme == "prioritized":
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
      # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
      # suggested a fixed exponent actually performs better, except on Pong.
      probs = self.replay_elements["sampling_probabilities"]
      # Weight the loss by the inverse priorities.
      loss_weights = 1.0 / onp.sqrt(probs + 1e-10)
      loss_weights /= onp.max(loss_weights)
      indices = self.replay_elements["indices"]
    else:
      # Uniform weights if not using prioritized replay.
      loss_weights = onp.ones(self.replay_elements["state"].shape[0:2])

    if self.log_churn and should_log:
      eval_batch = self.sample_eval_batch(256)
      eval_states = eval_batch["state"].reshape(-1,
                                                *eval_batch["state"].shape[-3:])
      eval_actions = eval_batch["action"].reshape(-1,)
      self._rng, eval_rng = jax.random.split(self._rng, 2)
      og_actions = self.select_action(
          eval_states,
          self.online_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )
      og_target_actions = self.select_action(
          eval_states,
          self.target_network_params,
          eval_mode=True,
          force_zero_eps=True,
          rng=eval_rng,
          use_noise=False,
      )

    self._rng, train_rng = jit_split(self._rng, num=2)
    (
        new_online_params,
        new_target_params,
        new_optimizer_state,
        new_dynamic_scale,
        aux_losses,
    ) = self.train_fn(
        self.network_def,
        self.online_params,
        self.target_network_params,
        self.optimizer,
        self.optimizer_state,
        self.replay_elements["state"],
        self.replay_elements["action"],
        self.replay_elements["next_state"],
        self.replay_elements["return"],
        self.replay_elements["terminal"],
        self.replay_elements["same_trajectory"],
        loss_weights,
        self._support,
        self.replay_elements["discount"],
        self._double_dqn,
        self._distributional,
        train_rng,
        self.spr_weight,
        self.dynamic_scale,
        self._data_augmentation,
        self.dtype,
        self._batch_size,
        self.use_target_network,
        self.target_update_tau_scheduler(self.cycle_grad_steps),
        self.target_update_period,
        self.grad_steps,
        self.match_online_target_rngs,
        self.target_eval_mode,
    )
    self.grad_steps += self._batches_to_group
    self.cycle_grad_steps += self._batches_to_group

    # Sample asynchronously while we wait for training
    sample_start = time.time()
    self._sample_from_replay_buffer()
    sample_time = time.time() - sample_start

    prio_set_start = time.time()
    if self._replay_scheme == "prioritized":
      # Rainbow and prioritized replay are parametrized by an exponent
      # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
      # leave it as is here, using the more direct sqrt(). Taking the square
      # root "makes sense", as we are dealing with a squared loss.  Add a
      # small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will
      # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      indices = onp.reshape(onp.asarray(indices), (-1,))
      dqn_loss = onp.reshape(onp.asarray(aux_losses["DQNLoss"]), (-1))
      priorities = onp.sqrt(dqn_loss + 1e-10)
      self._replay.set_priority(indices, priorities)
    prio_set_time = time.time() - prio_set_start

    training_time = time.time() - train_start
    if (self.training_steps % self.log_every == 0 and not offline and
        step_index == 0):
      metrics = {
          **{k: onp.mean(v) for k, v in aux_losses.items()},
          "PNorm": float(tree_norm(new_online_params)),
          "Inter-batch time": float(interbatch_time) / self._batches_to_group,
          "Training time": float(training_time) / self._batches_to_group,
          "Sampling time": float(sample_time) / self._batches_to_group,
          "Set priority time": float(prio_set_time) / self._batches_to_group,
      }

      if self.log_churn:
        new_actions = self.select_action(
            eval_states,
            new_online_params,
            eval_mode=True,
            force_zero_eps=True,
            rng=eval_rng,
            use_noise=False,
        )
        new_target_actions = self.select_action(
            eval_states,
            new_target_params,
            eval_mode=True,
            force_zero_eps=True,
            rng=eval_rng,
            use_noise=False,
        )
        online_churn = onp.mean(new_actions != og_actions)
        target_churn = onp.mean(new_target_actions != og_target_actions)
        online_off_policy_frac = onp.mean(new_actions != eval_actions)
        target_off_policy_frac = onp.mean(new_target_actions != eval_actions)
        online_target_agreement = onp.mean(new_actions == new_target_actions)
        churn_metrics = {
            "Online Churn": online_churn,
            "Target Churn": target_churn,
            "Online-Target Agreement": online_target_agreement,
            "Online Off-Policy Rate": online_off_policy_frac,
            "Target Off-Policy Rate": target_off_policy_frac,
        }
        metrics.update(**churn_metrics)

      if self.dynamic_scale:
        metrics["Dynamic Scale"] = self.dynamic_scale.scale

      if self.summary_writer is not None:
        with self.summary_writer.as_default():
          for k, v in metrics.items():
            tf.summary.scalar(k, v, step=self.training_steps)
      if self.verbose:
        logging.info(str(metrics))

    self.target_network_params = new_target_params
    self.online_params = new_online_params
    self.optimizer_state = new_optimizer_state
    self.dynamic_scale = new_dynamic_scale

  def _store_transition(
      self,
      last_observation,
      action,
      reward,
      is_terminal,
      *args,
      priority=None,
      episode_end=False,
  ):
    """Stores a transition when in training mode."""
    is_prioritized = isinstance(
        self._replay,
        prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer,
    ) or isinstance(
        self._replay,
        subsequence_replay_buffer.PrioritizedJaxSubsequenceParallelEnvReplayBuffer,
    )
    if is_prioritized and priority is None:
      priority = onp.ones((last_observation.shape[0]))
      if self._replay_scheme == "uniform":
        pass  # Already 1, doesn't matter
      else:
        priority.fill(self._replay.sum_tree.max_recorded_priority)

    if not self.eval_mode:
      self._replay.add(
          last_observation,
          action,
          reward,
          is_terminal,
          *args,
          priority=priority,
          episode_end=episode_end,
      )

  def _sync_weights(self, tau):
    if tau >= 1 or tau < 0:
      self.target_network_params = self.online_params
    else:
      self.target_network_params = interpolate_weights(
          self.target_network_params,
          self.online_params,
          keys=None,  # all keys
          old_weight=1 - tau,
          new_weight=tau,
          strip_params_layer=True,
      )

  def _train_step(self):
    """Runs a single training step.

    Runs training if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online_network_params to target_network_params if
    training steps is a multiple of target update period.
    """
    if self._replay.add_count == self.min_replay_history:
      self.initialize_prefetcher()

    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for i in range(self._num_updates_per_train_step):
          self._training_step_update(i, offline=False)
    if self.reset_every > 0 and self.training_steps > self.next_reset:
      self.reset_weights()

    self.training_steps += 1

  def _reset_state(self, n_envs):
    """Resets the agent state by filling it with zeros."""
    self.state = onp.zeros(n_envs, *self.state_shape)

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    observation = observation.squeeze(-1)
    if len(observation.shape) == len(self.observation_shape):
      self._observation = onp.reshape(observation, self.observation_shape)
    else:
      self._observation = onp.reshape(
          observation, (observation.shape[0], *self.observation_shape))
    # Swap out the oldest frame with the current frame.
    self.state = onp.roll(self.state, -1, axis=-1)
    self.state[Ellipsis, -1] = self._observation

  def reset_all(self, new_obs):
    """Resets the agent state by filling it with zeros."""
    n_envs = new_obs.shape[0]
    self.state = onp.zeros((n_envs, *self.state_shape))
    self._record_observation(new_obs)

  def reset_one(self, env_id):
    self.state[env_id].fill(0)

  def delete_one(self, env_id):
    self.state = onp.concatenate([self.state[:env_id], self.state[env_id + 1:]],
                                 0)

  def cache_train_state(self):
    self.training_state = (
        copy.deepcopy(self.state),
        copy.deepcopy(self._last_observation),
        copy.deepcopy(self._observation),
    )

  def restore_train_state(self):
    (self.state, self._last_observation, self._observation) = (
        self.training_state)

  def log_transition(self, observation, action, reward, terminal, episode_end):
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(
          self._last_observation,
          action,
          reward,
          terminal,
          episode_end=episode_end,
      )

  def select_action(
      self,
      state,
      select_params,
      eval_mode=False,
      use_noise=True,
      force_zero_eps=False,
      rng=None,
  ):
    force_rng = rng is not None
    if not force_rng:
      rng = self._rng
    new_rng, action = select_action(
        self.network_def,
        select_params,
        state,
        rng,
        self.num_actions,
        eval_mode or force_zero_eps,
        self.epsilon_eval if not force_zero_eps else 0.0,
        self.epsilon_train,
        self.epsilon_decay_period,
        self.training_steps,
        self.min_replay_history,
        self.epsilon_fn,
        self._support,
        not use_noise,
    )
    if not force_rng:
      self._rng = new_rng
    return action

  def step(self):
    """Records the most recent transition, returns the agent's next action, and trains if appropriate.
    """
    if not self.eval_mode:
      self._train_step()
    state = self.state

    use_target = self.target_action_selection
    select_params = (
        self.target_network_params if use_target else self.online_params)
    use_noise = self.eval_noise or not self.eval_mode

    action = self.select_action(
        state,
        select_params,
        eval_mode=self.eval_mode,
        use_noise=use_noise,
    )
    self.action = onp.asarray(action)
    return self.action
