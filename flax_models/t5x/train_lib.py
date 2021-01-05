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

"""Train utility functions for T5X."""
import os
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
import dataclasses
from flax import jax_utils
from flax.optim.base import Optimizer
from flax.training import common_utils
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from t5x import decode
from t5x import models
from tensorflow.io import gfile

# pylint:disable=invalid-name

Array = Any
ConfigDict = ml_collections.ConfigDict
PyTreeDef = type(jax.tree_structure(None))
TransformerConfig = models.TransformerConfig
TpuMesh = Tuple[int, int, int, int]
OtherMesh = Tuple[int, int]
Mesh = Union[TpuMesh, OtherMesh]


@dataclasses.dataclass(frozen=True)
class Topology:
  """Info about the overall topology and the current host's position in it.

  TODO(danielandor): Split into overall and current host information.
  """
  num_replicas: int
  num_replica_sets: int
  per_replica_mesh: Mesh
  per_replica_set_mesh: Mesh
  per_replica_set_num_replicas: int
  per_host_num_partitions: int
  replica_set_id: int
  per_replica_set_host_id: int
  device_assignment: List[jax.lib.xla_client.Device]
  this_host_device_assignment: List[jax.lib.xla_client.Device]


# -----------------------------------------------------------------------------
# Jax utility functions
# -----------------------------------------------------------------------------


def _unbroadcast(x):
  """Assuming `x` is replicated along its leading axis, remove that axis."""
  # Unbroadcast is a hack to take the output of a pmap with out_axes=0 and turn
  # it into the input of a pmap with in_axes=None. This is necessary because we
  # don't have out_axes=None in pmap, so the output arrays of the training step
  # function all still end up with an extra leading logical axis of size
  # `num_local_devices`.
  sharding_spec = x.sharding_spec
  # The leading logical axis should be sharded like the result of a pmap with
  # out_axes=0.
  assert sharding_spec.sharding[0] == jax.pxla.Unstacked(x.shape[0])
  # Remove that leading logical axis and its corresponding sharding.
  aval = jax.abstract_arrays.ShapedArray(x.shape[1:], x.dtype)
  sharding = sharding_spec.sharding[1:]

  # Replace the mesh mapping entry that pointed to that axis with Replicated,
  # and decrement the other entries.
  def replace_mesh_mapping(mm):
    if isinstance(mm, jax.pxla.ShardedAxis):
      if mm.axis == 0:
        return jax.pxla.Replicated(x.shape[0])
      return jax.pxla.ShardedAxis(mm.axis - 1)
    return mm

  mesh_mapping = map(replace_mesh_mapping, sharding_spec.mesh_mapping)
  sharding_spec = jax.pxla.ShardingSpec(sharding, mesh_mapping)
  return jax.pxla.ShardedDeviceArray(aval, sharding_spec, x.device_buffers)


def unbroadcast(tree):
  """Assuming `tree` is replicated along its leading axis, remove that axis."""
  return jax.tree_map(_unbroadcast, tree)


def broadcast(tree,
              num_replicas,
              num_partitions,
              devices=None):
  """Broadcast `tree` according to `num_replicas` and `num_partitions`.

  Replications are duplicates of `tree` along the leading axis. Partitions are
  further replications of `tree` using `replication_factors` in the
  `ShardingSpec` of the returned arrays.

  Args:
    tree: pytree of arrays
    num_replicas: number of replicas (i.e. pmap dimension size).
    num_partitions: number of partitions
    devices: flattened device assignment (defaults to jax.local_devices())

  Returns:
    A tree of ShardedDeviceArrays with leading sharded axis of size
    `num_replicas`, each of which contains a copy of the tree element, and is
    further replicated `num_partitions` times. This is suitable for passing to
    pmap(sharded_jit) if the data should be replicated on every device.
  """
  assert num_replicas * num_partitions == jax.local_device_count()
  # Replicate across all devices.
  replicated = jax_utils.replicate(tree, devices=devices)

  # Rewrite the sharding specs to include replicated partitioning.
  def redo_sharding_spec(x):
    assert isinstance(x, jax.pxla.ShardedDeviceArray)
    sharding_spec = x.sharding_spec
    # We replicated `tree` across all devices, but we only want a leading axis
    # of size `num_replicas`.
    aval = jax.abstract_arrays.ShapedArray((num_replicas,) + x.shape[1:],
                                           x.dtype)
    # Fix the size of the corresponding sharding.
    sharding = (jax.pxla.Unstacked(num_replicas),) + sharding_spec.sharding[1:]
    # Add replication over the remaining axis of the mesh.
    mesh_mapping = sharding_spec.mesh_mapping + (
        jax.pxla.Replicated(num_partitions),)
    sharding_spec = jax.pxla.ShardingSpec(sharding, mesh_mapping)
    return jax.pxla.ShardedDeviceArray(aval, sharding_spec, x.device_buffers)

  if num_partitions > 1:
    return jax.tree_map(redo_sharding_spec, replicated)
  else:
    return replicated


def compute_multihost_topology(num_partitions):
  """Logic to handle the multi-host data+model parallel topology.

  We need to relate three things:
  - the physical topology of devices and their interconnect
  - the logical topology of replicas + partitions (data + model parallelism)
  - the topology of which devices are connected to which hosts
  Since model parallelism involves more communication, partitions are
  assumed to be local. Both hosts and replicas enclose rectangular subgroups
  of devices that tile the overall physical mesh.

  Variables referring to tilings of the physical mesh are (x, y, z, core).
  Most such mesh variables are in units of devices, although variables called
  X_mesh and X_coords are shapes of and positions within meshes in units of X,
  and variables called per_X_Y are shapes of, or counts/indices within, a
  particular instance of X.

  Args:
    num_partitions: Requested number of partitions.

  Returns:
    a Topology object containing the device assignments etc.
  """
  num_replicas = max(1, jax.device_count() // num_partitions)
  logging.info('num_replicas: %d; num_partitions: %d', num_replicas,
               num_partitions)

  def bounds_from_last_device(device):
    # Must be passed the device at the highest-coordinate corner of the
    # relevant mesh, which is a requirement we know is satisfied by the last
    # device in jax.devices()
    if hasattr(device, 'coords'):
      x, y, z = device.coords
      return x + 1, y + 1, z + 1, device.id % 2 + 1
    else:
      # On non-TPU platforms, the "mesh" is hosts x devices per host in order
      # to take advantage of faster within-host interconnect
      return jax.host_count(), jax.local_device_count()

  global_mesh = bounds_from_last_device(jax.devices()[-1])
  logging.info('global_mesh: %s', global_mesh)
  if jax.local_devices()[0].platform == 'tpu':
    # TODO(jekbradbury): potentially move per_replica_mesh to config
    if num_partitions == 1:
      per_replica_mesh = (1, 1, 1, 1)
    elif num_partitions == 2:
      per_replica_mesh = (1, 1, 1, 2)
    elif num_partitions == 4:
      per_replica_mesh = (1, 2, 1, 2)
    elif num_partitions == global_mesh[1] * 2:
      # The y-axis is more likely to have the wraparound torus links, e.g. on
      # 16x32 or multipod topologies
      per_replica_mesh = (1, num_partitions // 2, 1, 2)
    elif num_partitions == 8:
      per_replica_mesh = (2, 2, 1, 2)
    elif num_partitions == 16:
      per_replica_mesh = (4, 2, 1, 2)
    else:
      raise NotImplementedError()
  else:
    per_replica_mesh = (max(1, num_partitions // jax.local_device_count()),
                        min(num_partitions, jax.local_device_count()))
  logging.info('per_replica_mesh: %s', per_replica_mesh)
  per_host_mesh = bounds_from_last_device(jax.local_devices(0)[-1])
  for per_replica, per_host in zip(per_replica_mesh, per_host_mesh):
    assert per_replica % per_host == 0 or per_host % per_replica == 0
  per_host_partition_mesh = tuple(
      min(pr, ph) for pr, ph in zip(per_replica_mesh, per_host_mesh))
  per_host_num_partitions = np.prod(per_host_partition_mesh)
  # Hosts and replicas are both tilings of the physical device mesh, and they're
  # not aligned with each other. But they both affect the data pipeline: all
  # devices within a host get fed data together, while all devices within a
  # replica need to be fed the same data, whether or not they're attached to the
  # same host.
  # A "replica set" is the least common multiple of hosts and replicas, or the
  # minimal group of devices that contains both an integer number of replicas
  # and an integer number of hosts. Each replica set will correspond to a unique
  # instance of the data pipeline, and the data coming out of the data pipeline
  # on each replica set will be the same on each host in that replica set and
  # will be split among the constituent replicas.
  per_replica_set_mesh = tuple(
      max(pr, ph) for pr, ph in zip(per_replica_mesh, per_host_mesh))
  num_replica_sets = max(1, jax.device_count() // np.prod(per_replica_set_mesh))
  per_replica_set_num_replicas = num_replicas // num_replica_sets

  # Here we begin to compute values that are specific to this host.
  first_local_device = jax.local_devices()[0]

  def get_coords(device):
    if hasattr(device, 'coords'):
      return (*device.coords, device.id % 2)
    return (device.host_id, device.id % jax.local_device_count())

  # The device coordinates of this host are those of its "first" device
  device_coords = get_coords(first_local_device)
  replica_set_coords = tuple(
      dc // prsm for dc, prsm in zip(device_coords, per_replica_set_mesh))
  # An X_id is a linear index of a particular X within the mesh of Xs (a value
  # in 0 <= X_id < num_Xs). The order of enumeration is arbitrary but must be
  # computed consistently between hosts.
  replica_set_id = 0
  for gm, prsm, rsc in zip(global_mesh, per_replica_set_mesh,
                           replica_set_coords):
    replica_set_id = replica_set_id * gm // prsm + rsc
  per_replica_set_host_coords = tuple(dc % prsm // phm for dc, prsm, phm in zip(
      device_coords, per_replica_set_mesh, per_host_mesh))
  per_replica_set_host_id = 0
  for prshc, prsm, phm in zip(per_replica_set_host_coords, per_replica_set_mesh,
                              per_host_mesh):
    per_replica_set_host_id = per_replica_set_host_id * prsm // phm + prshc
  logging.info(
      'host_id: %d, replica_set_id: %d/%d, '
      'per_replica_set_host_id: %d/%d', jax.host_id(), replica_set_id,
      num_replica_sets, per_replica_set_host_id,
      jax.host_count() // num_replica_sets)

  # The device assignment relates the logical mesh of replicas and partitions
  # to the physical mesh of devices by assigning replica and partition IDs to
  # each device. We construct it by iterating through the global devices and
  # computing the corresponding replica and partition from the mesh information,
  # but JAX/XLA expects it as the reverse mapping (from replicas and partitions
  # to devices) and in a flattened replica-major form.
  device_assignment = [[None
                        for partition in range(num_partitions)]
                       for replica in range(num_replicas)]
  for device in jax.devices():
    _device_coords = get_coords(device)
    _replica_id = 0
    _partition_id = 0
    for gm, prm, dc in zip(global_mesh, per_replica_mesh, _device_coords):
      _partition_id = _partition_id * prm + dc % prm
      _replica_id = _replica_id * gm // prm + dc // prm
    device_assignment[_replica_id][_partition_id] = device
  logging.info('device_assignment: %s', device_assignment)
  device_assignment = [d for ds in device_assignment for d in ds]  # pylint:disable=g-complex-comprehension
  this_host_device_assignment = [
      d for d in device_assignment if d.host_id == jax.host_id()
  ]
  return Topology(
      num_replicas=num_replicas,
      num_replica_sets=num_replica_sets,
      per_replica_mesh=per_replica_mesh,
      per_replica_set_mesh=per_replica_set_mesh,
      per_replica_set_num_replicas=per_replica_set_num_replicas,
      per_host_num_partitions=per_host_num_partitions,
      replica_set_id=replica_set_id,
      per_replica_set_host_id=per_replica_set_host_id,
      device_assignment=device_assignment,
      this_host_device_assignment=this_host_device_assignment)


# -----------------------------------------------------------------------------
# Training utility functions.
# -----------------------------------------------------------------------------


def create_learning_rate_scheduler(
    factors = 'constant * linear_warmup * rsqrt_decay',
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_factor = 0.5,
    steps_per_decay = 20000,
    steps_per_cycle = 100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


@jax.custom_vjp
def cross_entropy_with_logits(logits,
                              targets,
                              z_loss = 0.0):
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

  Args:
    logits: [batch * length, num_classes] float array.
    targets: categorical one-hot targets [batch * length, num_classes] float
      array.
    z_loss: coefficient for auxilliary z-loss loss term.

  Returns:
    scalar cross-entropy loss
  """
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  loss += z_loss * lax.square(log_z)
  return loss


def _cross_entropy_with_logits_fwd(
    logits,
    targets,
    z_loss = 0.0
):
  """Cross entropy loss forward pass."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  loss += z_loss * lax.square(log_z)
  return loss, (logits, targets, z_loss, exp_shifted, sum_exp, log_softmax,
                log_z)


def _cross_entropy_with_logits_bwd(
    res,
    g):
  """Cross entropy loss backward pass."""
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = (
      jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp -
      targets)
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (jnp.asarray(g_logits,
                      logits.dtype), jnp.asarray(g_targets, targets.dtype),
          jnp.array(0.0))  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd,
                                 _cross_entropy_with_logits_bwd)


def compute_weighted_cross_entropy(
    logits,
    targets,
    weights = None,
    label_smoothing = 0.0,
    z_loss = 0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical one-hot targets [batch, length, category] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
    z_loss: coefficient for auxilliary z-loss loss term.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  targets = targets.reshape((-1))
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)
  loss = cross_entropy_with_logits(logits, soft_targets, z_loss=z_loss)
  loss = loss - normalizing_constant

  normalizing_factor = np.prod(targets.shape)
  if weights is not None:
    weights = weights.reshape((-1))
    loss = loss * weights
    normalizing_factor = jnp.sum(weights)

  # HACK T5's "loss_denominator" correction for batchsize 2048 * 114 targetlen..
  # normalizing_factor = 233472.0

  return jnp.sum(loss), normalizing_factor


def compute_weighted_accuracy(
    logits,
    targets,
    weights = None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical one-hot targets [batch, length, category] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  targets = targets.reshape((-1))
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    weights = weights.reshape((-1))
    accuracy = accuracy * weights
    normalizing_factor = jnp.sum(weights)

  return jnp.sum(accuracy), normalizing_factor


def compute_metrics(
    logits,
    labels,
    weights,
    label_smoothing = 0.0):
  """Compute summary metrics."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights,
                                                    label_smoothing)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  return metrics


def _sync_devices(x):
  return jax.lax.psum(x, 'i')


def sync_devices():
  """Creates a barrier across all hosts/devices."""
  jax.pmap(_sync_devices,
           'i')(np.ones(jax.local_device_count())).block_until_ready()


def checkpoint_exists(ckpt_dir, prefix = 'checkpoint_'):
  """Detects if a checkpoint has already been saved."""
  glob_path = os.path.join(ckpt_dir, f'{prefix}*')
  return bool(gfile.glob(glob_path))


def train_step(
    optimizer,
    batch,
    prev_metrics,
    dropout_rng,
    config,
    learning_rate_fn,
    num_microbatches = None,
    label_smoothing = 0.0,
    z_loss = 0.0,
    use_bfloat16 = False
):
  """Perform a training step.

  There are two different types of training step implemented: a normal, direct
  single-step version; and a microbatched, gradient-accumulating train step
  useful for running larger batchsizes on smaller hardware.  To ease readability
  for the majority of users using the simpler, direct train step, we keep their
  implementations separate and dispatch to either here.

  Args:
    optimizer: flax optimizer for model parameters
    batch: input batch consisting of either - simply-padded batched features
      'inputs', 'targets' - packed, batched features 'inputs', 'targets',
      'inputs_position', 'targets_position', 'inputs_segmentation',
      'targets_segmentation'
    prev_metrics: previous step's metric stats accumulated, dict of floats with
      key `loss', 'accuracy', 'learning_rate', 'denominator'
    dropout_rng: jax PRNGKey for dropout.
    config: a TransformerConfig configuration dataclass with model options.
    learning_rate_fn: learning rate schedule function.
    num_microbatches: if non-None, we invoke the microbatched training step,
      otherwise ignored.
    label_smoothing: smoothing factor for loss function targets.
    z_loss: coefficient for auxilliary z-loss loss term.
    use_bfloat16: if true, use bfloat16.

  Returns:
    Updated optimizer & parameters, updated metrics, and new dropout PRNGKey.
  """
  if num_microbatches:
    return microbatched_train_step(
        optimizer,
        batch,
        prev_metrics,
        dropout_rng,
        config,
        learning_rate_fn,
        num_microbatches=num_microbatches,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        use_bfloat16=use_bfloat16)
  else:
    return direct_train_step(
        optimizer,
        batch,
        prev_metrics,
        dropout_rng,
        config,
        learning_rate_fn,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        use_bfloat16=use_bfloat16)


def direct_train_step(
    optimizer,
    batch,
    prev_metrics,
    dropout_rng,
    config,
    learning_rate_fn,
    label_smoothing = 0.0,
    z_loss = 0.0,
    use_bfloat16 = False
):
  """Perform a single, normal training step."""
  # X_position and X_segmentation are needed only when using 'packed examples'
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is treated
  # like a normal, unpacked sequence example.
  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in train_keys]
  logging.info('using direct training step.')

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  # We handle PRNG splitting inside the top pmap to improve efficiency.
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        {'params': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={'dropout': dropout_rng})

    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights,
                                                      label_smoothing, z_loss)
    mean_loss = loss / weight_sum
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  if use_bfloat16:
    grad = jax.tree_map(lambda x: x.astype(jnp.bfloat16), grad)
  grad = jax.lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  metrics = compute_metrics(logits, targets, weights)
  metrics['learning_rate'] = 1.0 * lr * metrics['denominator']
  metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)

  return new_optimizer, metrics, new_dropout_rng


def microbatched_train_step(
    optimizer,
    batch,
    prev_metrics,
    dropout_rng,
    config,
    learning_rate_fn,
    num_microbatches = 1,
    label_smoothing = 0.0,
    z_loss = 0.0,
    use_bfloat16 = False
):
  """Perform a single training step with microbatched gradient accumulation."""
  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  assert batch['inputs'].shape[0] % num_microbatches == 0, (
      "Batch size isn't divided evenly by num_microbatches.")
  microbatch_size = batch['inputs'].shape[0] // num_microbatches
  logging.info('using microbatches: %d microbatches, %d size', num_microbatches,
               microbatch_size)

  def get_microbatch(batch,
                     idx):
    """Fetch microbatch slice from possibly-packed input data."""
    out = []
    for k in train_keys:
      if k in batch:
        microbatch_idx = idx * microbatch_size
        out.append(
            lax.dynamic_slice(batch[k], (microbatch_idx, 0),
                              (microbatch_size, batch[k].shape[1])))
      else:
        out.append(None)
    return tuple(out)

  def loss_fn(
      params, batch, dropout_rng
  ):
    """loss function used for training."""
    (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
     targets_segmentation) = batch
    logits = models.Transformer(config).apply(
        {'params': params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={'dropout': dropout_rng})
    weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
    loss, weight_sum = compute_weighted_cross_entropy(logits, targets, weights,
                                                      label_smoothing, z_loss)
    # mean_loss = loss / weight_sum
    metrics = compute_metrics(logits, targets, weights)
    return loss, (weight_sum, metrics)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  def per_microbatch_train_step(
      loop_cnt, state
  ):
    (dropout_rng, grad_accum, weight_accum, prev_metrics) = state
    dropout_rng, sub_dropout_rng = random.split(dropout_rng)
    mbatch = get_microbatch(batch, loop_cnt)
    (_, (weight_sum, metrics)), grad = grad_fn(optimizer.target, mbatch,
                                               sub_dropout_rng)
    if use_bfloat16:
      grad = jax.tree_map(lambda x: x.astype(jnp.bfloat16), grad)
    grad_accum = jax.tree_multimap(jnp.add, grad_accum, grad)
    weight_accum += weight_sum
    metrics = jax.tree_multimap(jnp.add, metrics, prev_metrics)
    return dropout_rng, grad_accum, weight_accum, metrics

  # Initialize gradient accumulation loop state.
  grad_dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
  grad_accum_init = jax.tree_map(lambda x: jnp.zeros(x.shape, grad_dtype),
                                 optimizer.target)
  weight_accum_init = jnp.array(0.0)
  metrics_init = {'loss': 0.0, 'accuracy': 0.0, 'denominator': 0.0}
  loop_init = (dropout_rng, grad_accum_init, weight_accum_init, metrics_init)
  # Run gradient accumulation loop.
  new_dropout_rng, grad_accum, weight_accum, metrics = lax.fori_loop(
      0, num_microbatches, per_microbatch_train_step, loop_init)
  # Normalize gradient by loss denominator and average over devices.
  grad_accum = jax.tree_map(lambda x: x / weight_accum, grad_accum)
  grad_accum = jax.lax.pmean(grad_accum, 'batch')

  # Update optimizer using accumulated gradient.
  step = optimizer.state.step
  lr = learning_rate_fn(step)
  new_optimizer = optimizer.apply_gradient(grad_accum, learning_rate=lr)
  metrics['learning_rate'] = 1.0 * lr * metrics['denominator']
  metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)

  return new_optimizer, metrics, new_dropout_rng


def eval_step(params,
              batch,
              config,
              label_smoothing = 0.0):
  """Calculate evaluation metrics on a batch."""
  inputs, targets = batch['inputs'], batch['targets']
  weights = jnp.where(targets > 0, 1.0, 0.0)
  logits = models.Transformer(config).apply({'params': params}, inputs, targets)

  return compute_metrics(logits, targets, weights, label_smoothing)


def predict_step(inputs,
                 params,
                 eos_id,
                 max_decode_len,
                 config,
                 beam_size = 4,
                 return_entire_beam = False):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare zeroed-out autoregressive cache.
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  cache = models.Transformer(config).init(
      jax.random.PRNGKey(0), jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))['cache']
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = decode.flat_batch_beam_expand(
      models.Transformer(config).apply({'params': params},
                                       inputs,
                                       method=models.Transformer.encode),
      beam_size)
  raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

  def tokens_ids_to_logits(flat_ids,
                           flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, vocab]
    flat_logits, new_vars = models.Transformer(config).apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=['cache'],
        method=models.Transformer.decode)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_id=eos_id,
      max_decode_len=max_decode_len)

  # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
  # sorted in increasing order of log-probability.
  # Return the highest scoring beam sequence, drop first dummy 0 token.
  if return_entire_beam:
    return beam_seqs[:, :, 1:]
  else:
    return beam_seqs[:, -1, 1:]


# -----------------------------------------------------------------------------
# Utility functions for prediction and evaluation.
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
  """Expand batch to desired size by repeating last slice."""
  batch_pad = desired_batch_size - x.shape[0]
  return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def pad_batch_to_size(batch,
                      full_batch_size):
  """Handles ragged last batch by adding dummy examples to fill it out."""
  batch_size = batch.shape[0]
  if batch_size % full_batch_size:
    padded_size = int(np.ceil(batch_size / full_batch_size) * full_batch_size)
    batch = jax.tree_map(lambda x: pad_examples(x, padded_size), batch)
  return batch, batch_size


# NB: This needs to be top-level for the jax compilation cache.
def host_allgather_psum(x):
  """Host psum for host_allgather."""
  return jax.lax.psum(x, 'hosts')


def host_allgather(in_tree, num_replica_sets,
                   replica_set_id,
                   is_first_host_in_replica_set):
  """Gather data from across hosts/replica sets.

  Args:
    in_tree: pytree of arrays - each array _must_ have the same shape across the
      hosts.
    num_replica_sets: int denoting the number of replica sets (least common
      multiples of hosts and replicas) in the computation.
    replica_set_id: int denoting which replica set the current host belongs to.
    is_first_host_in_replica_set: bool denoting whether the current host is the
      first one in its replica set. Only that first host will contribute the
      data for the all-gather from its replica set.

  Returns:
    A pytree matching in_tree where each leaf array has a new leading
    dimension of size num_replica_sets, carrying the data copied from all hosts.
  """
  num_local_devices = jax.local_device_count()
  # We collect data per-host by creating two new axes: a pmap outer axis, and
  # an inner 'host' axis.  The latter is filled based on host_id, and the outer
  # only has this single nonzero entry.  Thus after a psum, we collect the
  # first member of the outer axis and have a new 'host' dimension such that
  # the returned leaves contain the data gathered from other hosts.
  host_psum = jax.pmap(host_allgather_psum, axis_name='hosts')

  def pre_pmap(x):
    y = np.zeros((num_local_devices, num_replica_sets, *x.shape), dtype=x.dtype)
    if is_first_host_in_replica_set:
      y[0, replica_set_id] = x
    return y

  def post_pmap(x):
    return jax.device_get(x)[0]

  return jax.tree_map(post_pmap, host_psum(jax.tree_map(pre_pmap, in_tree)))
