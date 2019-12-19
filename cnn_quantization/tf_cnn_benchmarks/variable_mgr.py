# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Defines VariableMgr and subclasses used to manage variables.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import re

import tensorflow.compat.v1 as tf

from cnn_quantization.tf_cnn_benchmarks import allreduce
from cnn_quantization.tf_cnn_benchmarks import batch_allreduce
from cnn_quantization.tf_cnn_benchmarks import variable_mgr_util
from tensorflow.contrib import training as contrib_training


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCNN to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """

  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []
    self.use_resource_vars = benchmark_cnn.params.use_resource_vars

    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None

    self._reuse_vars = False

  def each_tower_has_variables(self):
    """Returns True if each GPU tower of the model has separate variables."""
    assert False, 'Must be implemented in subclass'

  def supports_staged_vars(self):
    """Whether staged variable management is supported."""
    return False

  def create_outer_variable_scope(self, device_num):
    """Create the tf.variable_scope around all model graph operations."""
    del device_num  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def preprocess_device_grads(self, device_grads):
    """Preprocess the device gradients prior to applying them.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns: a tuple of (apply_gradients_devices, gradient_state).
      gradient_state is an opaque structure that should be passed to
      get_gradients_to_apply() and append_apply_gradients_ops() (in that order).
      apply_gradients_devices is a list of devices where the gradients will be
      applied with get_gradients_to_apply() and append_apply_gradients_ops().
    """
    del device_grads  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_gradients_to_apply(self, device_num, gradient_state):
    """Returns the [(gradient, variable)] list to apply for device_num.

    Args:
      device_num: indexes into apply_gradients_devices, which was returned by an
        earlier call to preprocess_device_grads.
      gradient_state: from previous call to apply_gradients_devices.
    """
    del device_num, gradient_state  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    """Adds training ops for grads to 'training_ops'.



    Args:
      gradient_state: from previous call to apply_gradients_devices.
      opt: the underlying optimizer
      grads: [(grad, var)] to apply
      training_ops: list to which to add ops
      loss_scale_params: parameters for loss scaling.
    """
    del gradient_state  # unused by this implementation

    def get_apply_gradients_ops_func():
      """Returns the apply_gradients op."""
      return [opt.apply_gradients(grads)]

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    """Return the set of trainable variables on device.

    Args:
      rel_device_num: local worker device index.
      abs_device_num: global graph device index.
      writable: whether to get a reference to the underlying variable.

    Returns:
      The set of trainable variables on the specified device.
    """
    del rel_device_num, writable
    if self.each_tower_has_variables():
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('v%s/' % abs_device_num)
      ]
    else:
      params = tf.trainable_variables()
    return params

  @contextlib.contextmanager
  def reuse_variables(self):
    """Context manager that causes variables requested to be reused.

    Variables requested under this context manager must already exist, and will
    be reused instead of being created again. This should be used if the
    evaluation model is being built after the training model has already been
    built. This is because the evaluation model should reuse variables from the
    training model.

    Yields:
      Nothing.
    """
    old_reuse_vars = self._reuse_vars
    try:
      self._reuse_vars = True
      yield
    finally:
      self._reuse_vars = old_reuse_vars


class VariableMgrIndependent(VariableMgr):
  """VariableMgr that implements the --independent mode for local jobs.

     Each GPU has its own copy of the variables, and gradients are
     not shared between towers. This can be used to check
     performance when no data is moved between GPUs.
  """

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num, reuse=self._reuse_vars,
                             use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    return (self.benchmark_cnn.devices, device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    tower_grad = device_grads[device_num]

    if self.benchmark_cnn.enable_auto_loss_scale and device_num == 0:
      # Since we don't aggregate variables in --independent mode, we cannot tell
      # if there are NaNs on all GPUs. So we arbitrarily choose to only check
      # NaNs on the first GPU.
      has_inf_nan_list = []
      for grad, _ in tower_grad:
        has_inf_nan_list.append(tf.reduce_all(tf.is_finite(grad)))
      self.grad_has_inf_nan = tf.logical_not(tf.reduce_all(has_inf_nan_list))

    return tower_grad

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrLocalFetchFromPS(VariableMgr):
  """VariableMgr that implements the --parameter_server mode for local jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v', reuse=bool(device_num) or self._reuse_vars,
                             use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    device_grads = gradient_state
    agg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.
        aggregate_gradients_using_copy_with_variable_colocation(
            device_grads,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    return agg_grads

  def get_devices(self):
    raw_devices = self.benchmark_cnn.raw_devices
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      return [
          variable_mgr_util.ParamServerDeviceSetter(d, raw_devices)
          for d in raw_devices
      ]
    else:
      return [
          tf.train.replica_device_setter(
              worker_device=d,
              ps_device=self.benchmark_cnn.param_server_device,
              ps_tasks=1) for d in raw_devices
      ]


class VariableMgrLocalFetchFromStagedPS(VariableMgrLocalFetchFromPS):
  """Implements fetching a local variable through staging buffers.
  """

  def __init__(self, benchmark_cnn):
    super(VariableMgrLocalFetchFromStagedPS, self).__init__(benchmark_cnn)
    # A data structure to track where the variables are used on each device.
    # Indexed by device_num and var_name, each entry stores the "put" and "get"
    # ops used for that variable on that device:
    #   staging_vars_on_devices[device_num][var_name] == (put_op, get_op)
    self.staging_vars_on_devices = [
        dict() for _ in self.benchmark_cnn.raw_devices
    ]

  def supports_staged_vars(self):
    return True

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = variable_mgr_util.StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices, None, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num) or self._reuse_vars,
        custom_getter=self._custom_getter, use_resource=self.use_resource_vars)

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    return self._custom_getter.trainable_variables_on_device(
        rel_device_num, abs_device_num, writable=writable)


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either a local all-reduce algorithm is applied or a regular
     cross-device aggregation is used to replicate the combined
     gradients to all towers.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec,
               agg_small_grads_max_bytes, agg_small_grads_max_group,
               allreduce_merge_scope):
    super(VariableMgrLocalReplicated, self).__init__(benchmark_cnn)
    if all_reduce_spec:
      spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
      if len(spec) != 1:
        raise ValueError(
            'replicated mode does not support hybrid all-reduce strategies')
      self._all_reduce_spec = spec[0]
    else:
      self._all_reduce_spec = None
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    self._warmup_ops = []
    self._allreduce_merge_scope = allreduce_merge_scope
    self._gradient_put_ops = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope('v%s' % device_num, reuse=self._reuse_vars,
                             use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    compact_grads = (self.benchmark_cnn.params.use_fp16 and
                     self.benchmark_cnn.params.compact_gradient_transfer)
    defer_grads = (self.benchmark_cnn.params.variable_consistency == 'relaxed')

    grads_to_reduce = [[g for g, _ in grad_vars] for grad_vars in device_grads]
    algorithm = batch_allreduce.algorithm_from_params(self.benchmark_cnn.params)
    reduced_grads, self._warmup_ops = algorithm.batch_all_reduce(
        grads_to_reduce, self.benchmark_cnn.params.gradient_repacking,
        compact_grads, defer_grads, self.benchmark_cnn.params.xla_compile)
    if self.benchmark_cnn.enable_auto_loss_scale:
      # Check for infs or nans
      is_finite_list = []
      with tf.name_scope('check_for_inf_and_nan'):
        for tower_grads in reduced_grads:
          with tf.colocate_with(tower_grads[0]):
            # TODO(tanmingxing): Create fused op that takes in a list of tensors
            # as input and returns scalar boolean True if there are any
            # infs/nans.
            is_finite_list.append(tf.reduce_all(
                [tf.reduce_all(tf.is_finite(g)) for g in tower_grads]))
        self.grad_has_inf_nan = tf.logical_not(tf.reduce_all(is_finite_list))
    reduced_device_grads = [[
        (g, v) for g, (_, v) in zip(grads, grad_vars)
    ] for grads, grad_vars in zip(reduced_grads, device_grads)]
    return self.benchmark_cnn.devices, reduced_device_grads

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    return device_grads[device_num]

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    post_init_ops += self._warmup_ops
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrDistributedAllReduce(VariableMgr):
  """VariableMgr that implements the --distributed_all_reduce mode.

     Each GPU has its own copy of the variables. To apply gradients,
     the specified all-reduce algorithm is used to reduce the gradients
     and replicate the final value to all GPUs.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec, job_name,
               num_workers, agg_small_grads_max_bytes,
               agg_small_grads_max_group, allreduce_merge_scope):
    super(VariableMgrDistributedAllReduce, self).__init__(benchmark_cnn)
    if not all_reduce_spec:
      raise ValueError(
          'distributed_all_reduce requires a non-empty all_reduce_spec')
    self._all_reduce_spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
    self._all_reduce_device_prefixes = (
        allreduce.build_all_reduce_device_prefixes(job_name, num_workers))
    self._num_workers = num_workers
    self._agg_small_grads_max_bytes = agg_small_grads_max_bytes
    self._agg_small_grads_max_group = agg_small_grads_max_group
    self._allreduce_merge_scope = allreduce_merge_scope
    if not self._all_reduce_spec:
      raise ValueError('all_reduce_spec must be specified')
    self._single_session = True

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    """Create a scope for the named device.

    Args:
      device_num: index of device for variable scope. (Note that
        device_num spans all processes in cluster since a single global
        graph is used.)

    Returns:
      the requested variable_scope
    """
    return tf.variable_scope('v%s' % device_num, reuse=self._reuse_vars,
                             use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    remaining_grads = device_grads
    aggregated_grads = []
    for spec_tuple in self._all_reduce_spec:
      if spec_tuple.limit < 0:
        this_grads = remaining_grads
        remaining_grads = []
      else:
        (this_grads, remaining_grads) = allreduce.split_grads_by_size(
            spec_tuple.limit, remaining_grads)
      if this_grads:
        range_agg_grads = allreduce.sum_gradients_all_reduce(
            self._single_session,
            self._all_reduce_device_prefixes,
            this_grads,
            self._num_workers,
            spec_tuple.alg,
            spec_tuple.shards,
            self.benchmark_cnn.gpu_indices,
            agg_small_grads_max_bytes=self._agg_small_grads_max_bytes,
            agg_small_grads_max_group=self._agg_small_grads_max_group,
            allreduce_merge_scope=self._allreduce_merge_scope)
        if not aggregated_grads:
          aggregated_grads = range_agg_grads
        else:
          assert len(aggregated_grads) == len(range_agg_grads)
          for i in range(len(aggregated_grads)):
            aggregated_grads[i] += range_agg_grads[i]
    assert not remaining_grads
    full_device_set = []
    for grads in device_grads:
      g, v = grads[0]
      del v
      full_device_set.append(g.device)
    return (full_device_set, aggregated_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    if device_num >= len(device_grads):
      raise ValueError('device_num %d exceeds length of device_grads (%d)' %
                       (device_num, len(device_grads)))
    return device_grads[device_num]

  def get_post_init_ops(self):
    """Copy initialized values for variables to other devices."""
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      split_name = v.name.split('/')
      # TODO(b/62630508): use more specific prefix than v or v0.
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        continue
      split_name[0] = 'v0'
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      split_name = v.name.split('/')
      if split_name[0] == 'v0' or not v.name.startswith('v'):
        params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


# TODO(tucker): Merge this mode with DistributedAllReduce.
class VariableMgrCollectiveAllReduce(VariableMgr):
  """VariableMgr that implements the --collective_all_reduce mode.

     Each GPU has its own copy of the variables. To apply gradients
     the TF native collective all-reduce op is used to reduce the gradients
     and replicate the final value to all GPUs.
  """

  def __init__(self, benchmark_cnn, all_reduce_spec,
               num_workers, num_gpus, task_id, allreduce_merge_scope):
    super(VariableMgrCollectiveAllReduce, self).__init__(benchmark_cnn)
    if not all_reduce_spec:
      raise ValueError(
          'collective_all_reduce requires a non-empty all_reduce_spec: %s'
          % all_reduce_spec)
    parsed_spec = allreduce.parse_all_reduce_spec(all_reduce_spec)
    # So far we only support a length-1 all_reduce_spec
    if len(parsed_spec) > 1 or parsed_spec[0].limit > 0:
      raise ValueError(
          'collective_all_reduce requires one single-range all_reduce_spec %s'
          % parsed_spec)
    self._all_reduce_spec = parsed_spec[0]
    if self._all_reduce_spec.alg != 'collective':
      raise ValueError(
          'VariableMgrCollectiveAllReduce initialized with non-collective '
          'all_reduce_spec %s' % self.all_reduce_spec)
    self._num_workers = num_workers
    self._num_gpus = num_gpus
    self._task_id = task_id
    self._allreduce_merge_scope = allreduce_merge_scope
    self._instance_key_counter = 10000
    self._instance_key_table = dict()
    self._single_session = False
    # List of prefixes for generating PS devices, unused here.
    self._all_reduce_device_prefixes = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    """Create a scope for the named device.

    Args:
      device_num: index of device for variable scope.

    Returns:
      the requested variable_scope
    """
    return tf.variable_scope('v%s' % device_num, reuse=self._reuse_vars)

  def preprocess_device_grads(self, device_grads):
    reduced_grads = allreduce.sum_gradients_all_reduce(
        self._single_session,
        self._all_reduce_device_prefixes,
        device_grads,
        self._num_workers,
        'collective',
        self._all_reduce_spec.shards,
        self.benchmark_cnn.gpu_indices,
        allreduce_merge_scope=self._allreduce_merge_scope)
    assert len(reduced_grads) == len(device_grads)
    full_device_set = []
    for grads in device_grads:
      g, _ = grads[0]
      full_device_set.append(g.device)
    return (full_device_set, reduced_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    if device_num >= len(device_grads):
      raise ValueError('device_num %d exceeds length of device_grads (%d)' %
                       (device_num, len(device_grads)))
    return device_grads[device_num]

  def _get_instance_key(self, name):
    if name not in self._instance_key_table.keys():
      self._instance_key_counter += 1
      self._instance_key_table[name] = self._instance_key_counter
    return self._instance_key_table[name]

  def get_post_init_ops(self):
    """Broadcast initialized values of variables to other devices.

    Returns:
      At task 0 device 0, broadcast_send.
      At all other devices and tasks, broadcast_recv.
    """
    global_vars = tf.global_variables()
    group_size = self._num_workers * self._num_gpus
    post_init_ops = []
    # Gather variables into same-var-different-device groups.
    vars_by_suffix = dict()
    for v in global_vars:
      split_name = v.name.split('/')
      mo = re.match(r'v(\d+)$', split_name[0])
      if mo:
        device_id = int(mo.group(1))
        suffix = '/'.join(split_name[1:])
        if suffix in vars_by_suffix.keys():
          vars_by_suffix[suffix].append(v)
        else:
          vars_by_suffix[suffix] = [v]
    # Generate broadcast ops for each such group.
    for suffix in sorted(vars_by_suffix):
      vlist = vars_by_suffix[suffix]
      assert self._num_gpus == len(vlist)
      devices = [v.device for v in vlist]
      # NOTE: this key should generate the same value for all tasks
      group_key = allreduce.collective_group_key(devices)
      group_size = self._num_workers * len(devices)
      instance_key = self._get_instance_key(suffix)
      for v in vlist:
        split_name = v.name.split('/')
        mo = re.match(r'v(\d+)$', split_name[0])
        if mo:
          device_id = int(mo.group(1))
          if (self._task_id == 0 and device_id == 0):
            with tf.device(v.device):
              bcast_send = allreduce.broadcast_send(
                  v, v.shape, v.dtype, group_size, group_key, instance_key)
              post_init_ops.append(v.assign(bcast_send))
          else:
            with tf.device(v.device):
              bcast_recv = allreduce.broadcast_recv(
                  v.shape, v.dtype, group_size, group_key, instance_key)
              post_init_ops.append(v.assign(bcast_recv))
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    if self._task_id == 0:
      for v in tf.global_variables():
        split_name = v.name.split('/')
        if split_name[0] == 'v0' or not v.name.startswith('v'):
          params.append(v)
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices


class VariableMgrDistributedFetchFromPS(VariableMgr):
  """Implements --variable_update=parameter_server mode for distributed jobs.

     Variables are stored on a parameter server.  For each step, each tower gets
     a copy of the variables from the parameter server, and sends its gradients
     to the param server.
  """

  def each_tower_has_variables(self):
    return False

  def create_outer_variable_scope(self, device_num):
    if self.benchmark_cnn.local_parameter_device_flag == 'gpu':
      caching_devices = self.benchmark_cnn.raw_devices
    else:
      caching_devices = [self.benchmark_cnn.cpu_device]
    custom_getter = variable_mgr_util.OverrideCachingDevice(
        caching_devices, self.benchmark_cnn.cpu_device, 1024 * 64)
    return tf.variable_scope(
        'v', reuse=bool(device_num) or self._reuse_vars,
        custom_getter=custom_getter, use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    # Returns (gradient_devices, gradient_state)
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    assert device_num == 0
    agg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.aggregate_gradients_using_copy(
            gradient_state,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))
    return agg_grads

  def get_devices(self):
    ps_strategy = contrib_training.GreedyLoadBalancingStrategy(
        self.benchmark_cnn.num_ps, contrib_training.byte_size_load_fn)
    return [
        tf.train.replica_device_setter(
            worker_device=d,
            cluster=self.benchmark_cnn.cluster_manager.get_cluster_spec(),
            ps_strategy=ps_strategy) for d in self.benchmark_cnn.raw_devices
    ]


class VariableMgrDistributedFetchFromStagedPS(
    VariableMgrDistributedFetchFromPS):
  """Extends VariableMgrDistributedFetchFromPS for --staged_vars."""

  def __init__(self, benchmark_cnn):
    super(VariableMgrDistributedFetchFromStagedPS, self).__init__(benchmark_cnn)
    self.staging_vars_on_devices = [
        dict() for _ in self.benchmark_cnn.raw_devices
    ]
    self.staged_vars_on_cpu = {}

  def create_outer_variable_scope(self, device_num):
    self._custom_getter = variable_mgr_util.StagedVariableGetter(
        device_num, self.benchmark_cnn.raw_devices,
        self.benchmark_cnn.cpu_device, self)
    return tf.variable_scope(
        'v', reuse=bool(device_num) or self._reuse_vars,
        custom_getter=self._custom_getter, use_resource=self.use_resource_vars)

  def supports_staged_vars(self):
    return True

  def trainable_variables_on_device(self,
                                    rel_device_num,
                                    abs_device_num,
                                    writable=False):
    return self._custom_getter.trainable_variables_on_device(
        rel_device_num, abs_device_num, writable=writable)


class VariableMgrDistributedReplicated(VariableMgr):
  """VariableMgr that implements the --distributed_replicated mode.

     Each GPU has a copy of the variables, and updates its copy after the
     parameter servers are all updated with the gradients from all servers. Only
     works with cross_replica_sync=true. Unlike 'replicated', does not use nccl
     all-reduce for replicating within a server.
  """

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope(
        'v%s' % device_num, reuse=self._reuse_vars,
        custom_getter=variable_mgr_util.OverrideToLocalVariableIfNotPsVar(),
        use_resource=self.use_resource_vars)

  def preprocess_device_grads(self, device_grads):
    return ([self.benchmark_cnn.param_server_device], device_grads)

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    avg_grads, self.grad_has_inf_nan = (
        variable_mgr_util.aggregate_gradients_using_copy_with_device_selection(
            self.benchmark_cnn,
            device_grads,
            use_mean=True,
            check_inf_nan=self.benchmark_cnn.enable_auto_loss_scale))

    # Make shadow variable on a parameter server for each original trainable
    # variable.
    for i, (g, v) in enumerate(avg_grads):
      my_name = variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/' + v.name
      if my_name.endswith(':0'):
        my_name = my_name[:-2]
      new_v = tf.get_variable(
          my_name,
          dtype=v.dtype.base_dtype,
          initializer=v.initial_value,
          trainable=True)
      avg_grads[i] = (g, new_v)
    return avg_grads

  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    device_grads = gradient_state  # From 2nd result of preprocess_device_grads.

    def get_apply_gradients_ops_func():
      """Returns a list of ops for updating gradients."""
      apply_gradients_ops = []
      # For each variable, apply the combined gradients for this server on
      # the parameter server, and then wait for all other servers to do this.
      for i, (g, v) in enumerate(grads):
        apply_gradient_op = opt.apply_gradients([(g, v)])
        barrier = self.benchmark_cnn.add_sync_queues_and_barrier(
            'replicate_variable_%s' % i, [apply_gradient_op])
        with tf.control_dependencies([barrier]):
          with tf.device(self.benchmark_cnn.cpu_device):
            updated_value = v.read_value()
            for my_d in range(len(self.benchmark_cnn.devices)):
              apply_gradients_ops.append(
                  device_grads[my_d][i][1].assign(updated_value))
      return apply_gradients_ops

    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)

  def _strip_port(self, s):
    if s.endswith(':0'):
      return s[:-2]
    return s

  def get_post_init_ops(self):
    # Copy initialized variables for variables on the parameter server
    # to the local copy of the variable.

    local_vars = tf.local_variables()
    local_var_by_name = dict(
        [(self._strip_port(v.name), v) for v in local_vars])
    post_init_ops = []
    for v in tf.global_variables():
      if v.name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0/'):
        prefix = self._strip_port(
            v.name[len(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0'):])
        for i in range(self.benchmark_cnn.num_gpus):
          name = 'v%s%s' % (i, prefix)
          if name in local_var_by_name:
            copy_to = local_var_by_name[name]
            post_init_ops.append(copy_to.assign(v.read_value()))
    return post_init_ops

  def _remove_shadow_var_prefix_if_present(self, var_name):
    if var_name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/'):
      return var_name[len(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/'):]
    else:
      return var_name

  def var_dict_name(self, v):
    return self._strip_port(self._remove_shadow_var_prefix_if_present(v.name))

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    params = {}
    for v in tf.global_variables():
      assert (v.name.startswith(variable_mgr_util.PS_SHADOW_VAR_PREFIX + '/v0/')
              or v.name in ('global_step:0', 'loss_scale:0',
                            'loss_scale_normal_steps:0')), (
                                'Invalid global variable: %s' % v)
      # We store variables in the checkpoint with the shadow variable prefix
      # removed so we can evaluate checkpoints in non-distributed replicated
      # mode. The checkpoints can also be loaded for training in
      # distributed_replicated mode.
      name = self._strip_port(self._remove_shadow_var_prefix_if_present(v.name))
      params[name] = v
    for v in tf.local_variables():
      # Non-trainable variables, such as batch norm moving averages, do not have
      # corresponding global shadow variables, so we add them here. Trainable
      # local variables have corresponding global shadow variables, which were
      # added in the global variable loop above.
      if v.name.startswith('v0/') and v not in tf.trainable_variables():
        params[self._strip_port(v.name)] = v
    return params

  def get_devices(self):
    return self.benchmark_cnn.raw_devices
