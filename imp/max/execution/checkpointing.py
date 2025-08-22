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

"""Utils for reading/writing checkpoints."""

import asyncio
import dataclasses
import functools
import os
import time
from typing import Any, Sequence

from absl import logging
from flax import serialization
from flax.training import checkpoints as flax_ckpt
from flax.training import train_state as flax_train_state
import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
import numpy as np
from t5x import checkpoint_importer as t5x_checkpointing
import tensorflow as tf
import tensorstore as ts
import tree

from imp.max.execution import partitioning
from imp.max.utils import typing


# most of the modules are directly borrowed from the T5X pipeline at
# https://github.com/google-research/t5x/blob/main/t5x/checkpoints.py
# here, we just adapt them to our specific needs (esp. the train_state module)

PyTree = Any
DUMMY_TS_SPEC = ts.Spec({'driver': 'zarr', 'kvstore': {'driver': 'memory'}})
_DESIRED_CHUNK_SIZE_BYTES = 64 * 1024 * 1024

# import checkpointing module from T5X
LazyArray = t5x_checkpointing.LazyArray
LazyThreadPoolArray = t5x_checkpointing.LazyThreadPoolArray
LazyAwaitableArray = t5x_checkpointing.LazyAwaitableArray


@dataclasses.dataclass
class ParameterInfo:
  """Information needed to read/write and slice a partitioned parameter."""
  # The unique parameter name.
  name: str
  # The shape of the parameter.
  shape: tuple[int, Ellipsis] | None
  # The TensoreStore Spec containing the minimal information for read/write.
  ts_spec: ts.Spec
  # The LocalChunkInfo for the part of the parameter local to this host.
  local_chunk_info: partitioning.LocalChunkInfo | None


class BytesConditionVariable(object):
  """Wraps a condition variable to control concurrency based on bytes."""

  def __init__(self, num_bytes):
    self._max_bytes = num_bytes
    self._num_bytes = num_bytes
    self._cv = asyncio.Condition(lock=asyncio.Lock())

  async def wait_for_bytes(self, n_bytes):
    async with self._cv:
      await self._cv.wait_for(lambda: self._num_bytes > n_bytes)
      self._num_bytes -= n_bytes
      if self._num_bytes < 0:
        raise ValueError(
            f'The requested number of bytes `{n_bytes}` exceeds the maximum '
            f'allocated number of bytes: {self._max_bytes}'
            )

  async def return_bytes(self, n_bytes):
    async with self._cv:
      self._num_bytes += n_bytes
      if self._num_bytes > self._max_bytes:
        raise ValueError(
            f'The requested number of bytes `{n_bytes}` exceeds the maximum '
            f'allocated number of bytes: {self._max_bytes}'
            )

      self._cv.notify_all()


# Register functions with flax.serialization to handle `ts.Spec`.
serialization.register_serialization_state(
    ts.Spec,
    ty_to_state_dict=lambda t: {'ts_spec': t.to_json()},
    ty_from_state_dict=lambda t, s: correct_possible_ts_specs(s),
    override=True,
    )


def cast(target, dtype):
  """Cast arrays in target to dtype."""

  def maybe_cast(x):
    # TODO(hassanak): add support for jnp int
    if isinstance(x, (int, str)):
      # Ignore common non-array types that shouldn't be cast.
      return x
    elif x.dtype == dtype:
      return x
    elif isinstance(x, jax.ShapeDtypeStruct):
      return jax.ShapeDtypeStruct(x.shape, dtype)
    else:
      return x.astype(dtype)

  return jax.tree.map(maybe_cast, target)


def get_async(target):
  return jax.tree.map(lambda x: x.get_async(), target)


def run_future_tree(future_tree):
  """Block until all futures are resolved on this host."""
  future_leaves, treedef = jax.tree_util.tree_flatten(future_tree)

  # TODO(hassanak): Use asyncio.run in py3.7+.
  loop = asyncio.get_event_loop()
  leaves = loop.run_until_complete(asyncio.gather(*future_leaves))
  return jax.tree_util.tree_unflatten(treedef, leaves)


def state_tree_map(tree_map_fn,
                   state,
                   **kwargs):
  """tree_map support for Flax TrainState class."""

  step_args = dict([
      (key, value.step) for key, value in kwargs.items()])  # pytype: disable=attribute-error
  params_args = dict([
      (key, value.params) for key, value in kwargs.items()])  # pytype: disable=attribute-error
  opt_state_args = dict([
      (key, value.opt_state) for key, value in kwargs.items()])  # pytype: disable=attribute-error

  return state.replace(apply_fn=None,
                       step=tree_map_fn(state.step, **step_args),
                       params=tree_map_fn(state.params, **params_args),
                       opt_state=tree_map_fn(state.opt_state, **opt_state_args))


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """

  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else np.inf
  while True:
    checkpoint_path = flax_ckpt.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if time.time() + seconds_to_sleep > stop_time:
        return None
      else:
        time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint at %s', checkpoint_path)
      return checkpoint_path


def checkpoints_iterator(checkpoint_dir,
                         min_interval_secs=180,
                         timeout=5 * 24 * 60 * 60,  # 5d
                         timeout_fn=None):
  """Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  """

  checkpoint_path = None
  while True:
    new_checkpoint_path = wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_path, timeout=timeout)
    if new_checkpoint_path is None:
      if timeout_fn is None:
        # timed out
        logging.info('Timed-out waiting for a checkpoint.')
        return
      if timeout_fn():
        # The timeout_fn indicated that we are truly done.
        return
      else:
        # The timeout_fn indicated that more checkpoints may come.
        continue
    start = time.time()
    checkpoint_path = new_checkpoint_path
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


def choose_chunk_shape(write_shape,
                       target_elements):
  """Chooses a chunk shape that evenly divides write_shape.

  The chunk shape is chosen such that the total number of elements is less than
  or equal to `target_elements`, but is otherwise as large as possible.

  This uses a greedy algorithm that attempts to split the largest dimensions
  first.

  Args:
    write_shape: Write shape for which to choose a chunk shape.
    target_elements: Desired number of elements in chosen chunk shape.  Must be
      >= 1.

  Returns:
    List of length `len(write_shape)` specifying the chosen chunk shape.
  """

  assert target_elements >= 1
  rank = len(write_shape)

  # `dim_factors[i]` is the list of divisors of `write_shape[i]`
  dim_factors = [
      [i for i in range(1, size + 1) if size % i == 0] for size in write_shape
  ]

  # The current chunk shape is:
  # [dim_factors[i][-1] for i in range(rank)]

  def get_total_elements():
    """Returns the number of elements in the current chunk shape."""
    total_elements = 1
    for i in range(rank):
      total_elements *= dim_factors[i][-1]
    return total_elements

  # Reduce the current chunk shape until the desired number of elements is
  # reached.
  while get_total_elements() > target_elements:
    # Greedily reduce the largest dimension.  This is not guaranteed to bring us
    # the closest to `target_elements`, but is simple to implement and should
    # work well enough.
    dim_to_reduce = -1
    dim_to_reduce_size = 1
    for i in range(rank):
      size = dim_factors[i][-1]
      if size > dim_to_reduce_size:
        dim_to_reduce_size = size
        dim_to_reduce = i
    # Can only fail to choose `dim_to_reduce` if all dimensions have size of 1.
    # But that cannot happen since `target_elements >= 1`.
    assert dim_to_reduce_size > 1
    dim_factors[dim_to_reduce].pop()
  return [dim_factors[i][-1] for i in range(rank)]


def get_param_info(name,
                   array,
                   global_shape,
                   pspec,
                   partitioner):
  """Gets the TensorStore spec for a given parameter."""

  if array is None:
    return ParameterInfo(
        name=name,
        shape=None,
        ts_spec=DUMMY_TS_SPEC,
        local_chunk_info=None,
        )

  elif pspec is None or not any(tuple(pspec)):
    # the PartitionSpec is None OR all of the PartitionSpec axes are None
    return ParameterInfo(
        name=name,
        shape=global_shape,
        ts_spec=DUMMY_TS_SPEC,
        local_chunk_info=None,
        )

  local_chunk_info = partitioner.get_local_chunk_info(global_shape, pspec)
  write_shape = np.array([
      si if sl == slice(None) else sl.stop - sl.start
      for si, sl in zip(global_shape, local_chunk_info.slice)
  ])

  chunk_shape = choose_chunk_shape(
      write_shape,
      target_elements=_DESIRED_CHUNK_SIZE_BYTES / array.dtype.itemsize)

  dtype = np.dtype(array.dtype).str.replace('<V2', 'bfloat16')

  spec = {
      'driver': 'zarr',
      'kvstore': {'driver': 'gfile',
                  'path': 'tensorstore/' + name.replace('/', '.')},
      'metadata': {'compressor': {'id': 'gzip'},
                   'shape': global_shape,
                   'chunks': chunk_shape,
                   'dtype': dtype},
  }

  return ParameterInfo(
      name=name,
      shape=global_shape,
      ts_spec=ts.Spec(spec),
      local_chunk_info=local_chunk_info,
      )


def get_state_names(state):
  def _name_getter(path, _):
    path = [str(p) for p in path]
    return '.'.join(path)

  params_names = tree.map_structure_with_path(_name_getter, state.params)
  opt_state_names = tree.map_structure_with_path(_name_getter, state.opt_state)

  return state.replace(step='step',
                       apply_fn=None,
                       params=params_names,
                       opt_state=opt_state_names)


def get_ts_spec(state,
                partitioner):
  """Fetches TensorSpec for all leaves in the state tree."""

  state_names = get_state_names(state)
  state_specs = partitioner.get_state_specs().replace(apply_fn=None)
  state_shapes = partitioner.get_state_shapes().replace(apply_fn=None)
  param_info_getter = functools.partial(get_param_info, partitioner=partitioner)

  params_ts_specs = jax.tree.map(param_info_getter,
                                 state_names.params,
                                 state.params,
                                 state_shapes.params,
                                 state_specs.params)

  opt_state_ts_specs = jax.tree.map(param_info_getter,
                                    state_names.opt_state,
                                    state.opt_state,
                                    state_shapes.opt_state,
                                    state_specs.opt_state)

  step_ts_specs = param_info_getter(name='step',
                                    array=state.step,
                                    global_shape=state_shapes.step,
                                    pspec=None)

  return state.replace(apply_fn=None,
                       step=step_ts_specs,
                       params=params_ts_specs,
                       opt_state=opt_state_ts_specs)


def get_lazy_state(target,
                   lazy_load = True):
  """Gets the state and casts targets to the save dtype."""

  def _lazy_load_device_array(arr):
    if isinstance(arr, jax.Array):
      return LazyThreadPoolArray(arr.shape, arr.dtype, lambda: np.array(arr))
    return arr

  if lazy_load:
    target = jax.tree.map(_lazy_load_device_array, target)

  return target


def get_array_or_store_ts(
    store_dir,
    state,
    ts_specs,
    lazy,
    dtype = None,
    concurrent_bytes = 128 * 10**9,
):
  """Writes extracted state from train state to Tensorstore."""

  if lazy:
    bytes_cv = BytesConditionVariable(concurrent_bytes)

  async def _get_array_or_store_ts(array_or_object,
                                   param_info):
    """Maybe write to TensorStore, or return array/spec to write to msgpack."""

    if param_info.ts_spec is DUMMY_TS_SPEC:
      # Write to the msgpack file on host 0.
      if isinstance(array_or_object, LazyArray):
        return await array_or_object.get_async()
      return array_or_object

    # Only write each chunk of a parameter from one host
    if (
        param_info.local_chunk_info is not None
        and param_info.local_chunk_info.replica_id == 0
    ):
      arr = array_or_object

      if lazy:
        # Wait until memory is available.
        n_bytes = arr.nbytes
        if n_bytes > concurrent_bytes:
          logging.warning(
              'Temporarily increasing the concurrency limits from %d bytes to '
              '%d bytes to fit %s.', concurrent_bytes, n_bytes, param_info.name)
          n_bytes = concurrent_bytes
        await bytes_cv.wait_for_bytes(n_bytes)  # pylint: disable=undefined-variable

      if isinstance(array_or_object, LazyArray):
        arr = await arr.get_async()
      elif not isinstance(arr, np.ndarray):
        # Cast jax.DeviceArray to np.ndarray.
        arr = np.array(array_or_object, dtype=array_or_object.dtype)

      tmp_ts_spec_dict = param_info.ts_spec.to_json()

      # Path is updated in-place.
      tmp_ts_spec_dict['kvstore']['path'] = os.path.join(
          store_dir, tmp_ts_spec_dict['kvstore']['path'])
      if tmp_ts_spec_dict['metadata']['dtype'] != np.dtype(arr.dtype):
        raise ValueError(
            'The requested store dtype does not match dtype of the'
            f' array. {tmp_ts_spec_dict["metadata"]["dtype"]} vs. '
            f'{np.dtype(arr.dtype)}'
        )

      t = await ts.open(
          tmp_ts_spec_dict,
          create=True,
          open=True,
          context=ts.Context({'file_io_concurrency': {'limit': 128}}),
          )
      await t[param_info.local_chunk_info.slice].write(arr)

      if lazy:
        await bytes_cv.return_bytes(n_bytes)  # pylint: disable=undefined-variable

    return param_info.ts_spec

  def write_or_get_array(target, ts_specs):
    return jax.tree.map(_get_array_or_store_ts, target, ts_specs)

  lazy_state = state_tree_map(tree_map_fn=get_lazy_state, state=state)

  if dtype is not None:
    cast_fn = functools.partial(cast, dtype=dtype)
    lazy_state = state_tree_map(tree_map_fn=cast_fn, state=lazy_state)

  future_array_or_tsspec = state_tree_map(tree_map_fn=write_or_get_array,
                                          state=lazy_state,
                                          ts_specs=ts_specs)

  # Block until complete on this host.
  array_or_tsspec_state = run_future_tree(future_array_or_tsspec)

  # Recover the apply_fn
  array_or_tsspec_state = array_or_tsspec_state.replace(apply_fn=state.apply_fn)

  # Block until complete on all hosts.
  multihost_utils.sync_global_devices(
      f'checkpointer:ts_write_complete:{store_dir}')

  return array_or_tsspec_state


def get_array_or_restore_ts(
    restore_dir,
    restored_contents,
    target_param_infos,
    lazy,
    dtype = None,
):
  """Gets a mix of array and ts_spec and restores those with ts_spec."""

  async def _get_array_or_restore_ts(array_or_tspec,
                                     param_info):
    """Maybe restore from TensorStore, or return array."""

    # If saved as a numpy array, but a partitioned read is requested, return a
    # slice of the array for that host. Otherwise, return the entire array.
    if isinstance(array_or_tspec, np.ndarray):
      if param_info.local_chunk_info:
        arr = array_or_tspec
        return arr[param_info.local_chunk_info.slice]
      else:
        return array_or_tspec

    elif not isinstance(array_or_tspec, ts.Spec):
      return array_or_tspec

    tmp_ts_spec_dict = array_or_tspec.to_json()
    # Remove non-required params so that we can open Tensorstore
    # that was created with a different set of params.
    del tmp_ts_spec_dict['metadata']['chunks']
    del tmp_ts_spec_dict['metadata']['compressor']

    # Path is updated in-place.
    tmp_ts_spec_dict['kvstore']['path'] = os.path.join(
        restore_dir, tmp_ts_spec_dict['kvstore']['path'])

    if param_info.shape is not None:
      ts_spec_arr_shape = tuple(tmp_ts_spec_dict['metadata']['shape'])
      # Check if the shapes of the array on disk match the expected shape based
      # on the optimizer that is being restored.
      if ts_spec_arr_shape != param_info.shape:
        raise ValueError(f'Shape of `{param_info.name}` in checkpoint '
                         f'{ts_spec_arr_shape} does not match expected '
                         f'{param_info.shape}.')
    # Read the array.
    t = await ts.open(tmp_ts_spec_dict, open=True)
    if param_info.local_chunk_info is not None:
      # Just read the subsection we care about.
      t = t[param_info.local_chunk_info.slice]
    arr = await t.read()

    # TODO(hassanak): check if this is necessary
    if arr.dtype == np.uint16:
      arr = arr.view(jnp.bfloat16)
    return arr

  def get_lazy_restored_state(arrays_or_tspecs, param_infos):
    def _lazy_restore_fn(array_or_tspec, param_info):
      return LazyAwaitableArray.from_tensor_store_spec_or_array(
          maybe_ts_spec=array_or_tspec,
          get_fn=functools.partial(_get_array_or_restore_ts,
                                   array_or_tspec=array_or_tspec,
                                   param_info=param_info),
          )

    return jax.tree.map(_lazy_restore_fn, arrays_or_tspecs, param_infos)

  restored_state = state_tree_map(tree_map_fn=get_lazy_restored_state,
                                  state=restored_contents,
                                  param_infos=target_param_infos)

  if not lazy:
    future_restored_state = state_tree_map(tree_map_fn=get_async,
                                           state=restored_state)
    restored_state = run_future_tree(future_restored_state)

  if dtype is not None:
    restored_state = state_tree_map(tree_map_fn=cast,
                                    state=restored_state,
                                    dtype=dtype)

  # Recover the apply_fn
  restored_state = restored_state.replace(apply_fn=restored_contents.apply_fn)

  return restored_state


def correct_possible_ts_specs(ckpt_contents):
  """Convert the JSON dict of ts.Spec to a real ts.Spec."""

  def is_leaf(s):
    if isinstance(s, dict) and 'ts_spec' in s:
      return True

    if isinstance(s, np.ndarray):
      return True

    return False

  def ts_specfy(s):
    if isinstance(s, dict):
      return ts.Spec(s['ts_spec'])
    else:
      return s

  return jax.tree.map(ts_specfy, ckpt_contents, is_leaf=is_leaf)


class CheckpointManager(object):
  """A checkpointer with partitioned weights support."""

  def __init__(self,
               workdir,
               partitioner,
               state = None,
               keep = 1,
               checkpoint_data = False,
               lazy_save = False,
               lazy_restore = False,
               parallel_restore = True,
               prefix = 'checkpoint_',
               save_dtype = None,
               restore_dtype = None,
               concurrent_store_bytes = 128 * 10**9):
    """Checkpointer constructor."""

    self.state = state
    self.partitioner = partitioner
    self.data_layout = partitioner.get_data_layout()
    self.workdir = workdir
    self.keep = keep
    self.checkpoint_data = checkpoint_data
    self.lazy_save = lazy_save
    self.lazy_restore = lazy_restore
    self.parallel_restore = parallel_restore
    self.prefix = prefix
    self.save_dtype = save_dtype
    self.restore_dtype = restore_dtype
    self._concurrent_store_bytes = concurrent_store_bytes
    self._dataset_checkpointer = None
    self._dataset_ckpt_name = os.path.join(
        'dataset',
        f'{self.data_layout.shard_id:03}-of-{self.data_layout.num_shards:03}')

    asyncio.set_event_loop(asyncio.new_event_loop())

  def _get_ts_specs(self):
    return get_ts_spec(self.state, self.partitioner)

  def save(self,
           state,
           data):
    """Stores the train state and dataloader.

    This function stores the train state in a combination of Tensorstore and
    MSGPack. Tensorstore only holds information about partitioned
    weights, while MSGPack stores replicated weights and/or other meta data.
    If self.checkpoint_data == True and 'data' is provided, this function
    also stores the dataloader's iteration state (to resume training later
    accurately). This information is stored using tf.train.Checkpoint which
    produces similar files to the normal TF checkpoints.

    Args:
      state: A standard Flax train state containing the model parameters,
        optimizer states, and model call signature.
      data: An optional dataloader collection (see max.data.datasets.dataloader)
        which contains iterators and other metadata. All dataloader states
        are stored. If this function is called for the first time, the
        checkpointer saves the structure of 'data' and expects the future
        calls to get a 'data' module with the exact same structure and meta
        data. The only state that can be changed across calls is the iterator's
        state.
    """
    # Transfer state to local hosts
    pspecs = self.partitioner.get_state_specs()
    # Partitioner yields None for replicated variables; which is necessary for
    # certain internal functions, however we need to remove all None leaves here
    # before calling the jax util function below.
    pspecs = jax.tree.map(
        lambda x: jax.sharding.PartitionSpec() if x is None else x,
        pspecs,
        is_leaf=(
            lambda x: x is None or isinstance(x, jax.sharding.PartitionSpec)),
    )
    state = multihost_utils.global_array_to_host_local_array(
        global_inputs=state,
        global_mesh=self.partitioner.mesh,
        pspecs=pspecs,
    )
    step = state.step
    step = step.get() if isinstance(step, LazyArray) else step

    # Share a timestamp across devices.
    timestamp = multihost_utils.broadcast_one_to_all(np.int32(time.time()))

    final_dir = os.path.join(self.workdir, self.prefix + str(step))
    tmp_dir = os.path.join(self.workdir, f'tmp_ckpt_{step}-{timestamp}')

    if jax.process_index() == 0:
      tf.io.gfile.makedirs(tmp_dir)
    # Block all hosts until directory is ready.
    multihost_utils.sync_global_devices(f'checkpointer:make_dir:{tmp_dir}')

    ts_specs = get_ts_spec(state, self.partitioner)
    array_or_tsspec_state = get_array_or_store_ts(
        store_dir=tmp_dir,
        state=state,
        ts_specs=ts_specs,
        lazy=self.lazy_save,
        dtype=self.save_dtype,
        concurrent_bytes=self._concurrent_store_bytes,
        )

    # Write dataset iterators
    if (data is not None and self.checkpoint_data
        and self.data_layout.is_first_host_in_replica_set):
      if self._dataset_checkpointer is None:
        self._dataset_checkpointer = tf.train.Checkpoint(ds=data)

      try:
        dataset_write_path = os.path.join(tmp_dir, self._dataset_ckpt_name)
        self._dataset_checkpointer.write(dataset_write_path)

      except tf.errors.FailedPreconditionError as e:
        logging.error(
            'Data pipeline must be stateless in order to checkpoint. Cache '
            'stateful steps offline or disable iterator checkpointing.')
        raise e

    # Block until complete on all hosts.
    multihost_utils.sync_global_devices(
        f'checkpointer:write_complete:{tmp_dir}')
    if jax.process_index() != 0:
      return

    ##### Host 0 only. #####
    # Write msgpack file in host 0 only
    msgpack_bytes = serialization.to_bytes(array_or_tsspec_state)
    with tf.io.gfile.GFile(os.path.join(tmp_dir, 'msgpack'), 'wb') as fp:
      fp.write(msgpack_bytes)

    # Finalize checkpoint directory.
    tf.io.gfile.rename(tmp_dir, final_dir)
    logging.info('Saved checkpoint for step %d to %s', step, final_dir)

    # Remove old checkpoints, if necessary.
    all_checkpoints = tf.io.gfile.glob(
        os.path.join(self.workdir, self.prefix + '*'))
    all_checkpoints = flax_ckpt.natural_sort(all_checkpoints)

    if len(all_checkpoints) > self.keep:
      old_ckpts = all_checkpoints[:-self.keep]
      # Note: old_ckpts is sorted from oldest to newest.
      for path in old_ckpts:
        logging.info('Removing checkpoint at %s', path)
        tf.io.gfile.rmtree(path)

    # Remove temporary checkpoints, if necessary.
    tmp_checkpoints = tf.io.gfile.glob(os.path.join(self.workdir, 'tmp_ckpt_*'))
    for path in tmp_checkpoints:
      logging.info('Removing checkpoint at %s', path)
      tf.io.gfile.rmtree(path)

  def restore(self,
              path = None,
              target = None,
              data = None,
              put_on_devices = False):
    """Reads the checkpoint metadata and restores the variables and states.

    Args:
      path: A string indicating the path to where the checkpoint is stored.
        This path could be directly pointing to a specific checkpoint or a
        place where a collection of checkpoints are stored. In the latter,
        this function looks for the latest checkpoint (assuming that they all
        follow this pattern f'{self.prefix}_{x}' where 'x' is a positive
        integer. e.g. checkpoint_3).
      target: A structure which is expected in the checkpoint to be restored.
        Considering the dependency of the Tensorstore meta data on the Flax
        train state, this target SHOULD currently be the full 'state'. If
        None is passed, a NotImplementedError is raised.
      data: An optional dataloader collection (see max.data.datasets.dataloader)
        which contains iterators and other metadata. If provided, the iterator
        in the 'data' module is updated with the state stored in the checkpoint.
        It is expected that the given 'data' module have the exact same state
        and structure as the one in the checkpoint, except the iterator, which
        could have a different iteration state.
      put_on_devices: A bool indicating whether to put the variables on JAX's
        DeviceArray.
    Returns:
      A state with the same structure as 'target'.
    """

    ckpt_dir = path or self.workdir
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    ckpt_dir = flax_ckpt.safe_normpath(ckpt_dir)

    if not tf.io.gfile.exists(ckpt_dir):
      raise ValueError(f'Found no valid file or directory at {ckpt_dir}')

    msgpack_path = os.path.join(ckpt_dir, 'msgpack')
    restore_dir = ckpt_dir

    if not tf.io.gfile.exists(msgpack_path):
      # probably a generic workdir
      maybe_ckpt_dir = flax_ckpt.latest_checkpoint(ckpt_dir=ckpt_dir,
                                                   prefix=self.prefix)
      if maybe_ckpt_dir:
        msgpack_path = os.path.join(maybe_ckpt_dir, 'msgpack')
        restore_dir = maybe_ckpt_dir
      else:
        raise ValueError(f'Found no checkpoint at {ckpt_dir}')

    if not tf.io.gfile.exists(msgpack_path):
      raise ValueError(f'Found no checkpoint at {ckpt_dir}')

    target = target or self.state
    # TODO(hassanak): find a solution for free-form dict loading
    if target is None:
      raise NotImplementedError

    target_param_infos = get_ts_spec(target, self.partitioner)
    restored_contents = flax_ckpt.restore_checkpoint(
        ckpt_dir=msgpack_path,
        target=target,
        step=None,
        prefix=self.prefix,
        parallel=self.parallel_restore,
        )

    restored_contents = correct_possible_ts_specs(restored_contents)
    restored_state = get_array_or_restore_ts(
        restore_dir=restore_dir,
        restored_contents=restored_contents,
        target_param_infos=target_param_infos,
        lazy=self.lazy_restore,
        dtype=self.restore_dtype,
        )

    if data is not None and self.checkpoint_data:
      if self._dataset_checkpointer is None:
        self._dataset_checkpointer = tf.train.Checkpoint(ds=data)
      dataset_read_path = os.path.join(restore_dir, self._dataset_ckpt_name)

      try:
        logging.info(
            'Restoring dataloader checkpoint from %s', dataset_read_path)
        self._dataset_checkpointer.read(dataset_read_path)

      except tf.errors.FailedPreconditionError as e:
        logging.error(
            'Data pipeline must be stateless in order to checkpoint. Cache '
            'stateful steps offline or disable iterator checkpointing.')
        raise e

    if put_on_devices:
      # distribute state across processes
      pspecs = self.partitioner.get_state_specs()
      # remove all None leaves
      pspecs = jax.tree.map(
          lambda x: jax.sharding.PartitionSpec() if x is None else x,
          pspecs,
          is_leaf=(
              lambda x: x is None or isinstance(x, jax.sharding.PartitionSpec)),
      )
      restored_state = multihost_utils.host_local_array_to_global_array(
          local_inputs=restored_state,
          global_mesh=self.partitioner.mesh,
          pspecs=pspecs,
      )

    return restored_state
