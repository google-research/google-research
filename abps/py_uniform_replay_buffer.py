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

"""Uniform replay buffer in Python.

The base class provides all the functionalities of a uniform replay buffer:
  - add samples in a First In First Out way.
  - read samples uniformly.

PyHashedReplayBuffer is a flavor of the base class which
compresses the observations when the observations have some partial overlap
(e.g. when using frame stacking).
"""

import threading
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.replay_buffers import replay_buffer
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import numpy_storage


class PyUniformReplayBuffer(replay_buffer.ReplayBuffer):
  """A Python-based replay buffer that supports uniform sampling.

  Writing and reading to this replay buffer is thread safe.

  This replay buffer can be subclassed to change the encoding used for the
  underlying storage by overriding _encoded_data_spec, _encode, _decode, and
  _on_delete.
  """

  def __init__(self, data_spec, capacity):
    """Creates a PyUniformReplayBuffer.

    Args:
      data_spec: An ArraySpec or a list/tuple/nest of ArraySpecs describing a
        single item that can be stored in this buffer.
      capacity: The maximum number of items that can be stored in the buffer.
    """
    super(PyUniformReplayBuffer, self).__init__(data_spec, capacity)

    self._storage = numpy_storage.NumpyStorage(self._encoded_data_spec(),
                                               capacity)
    self._lock = threading.Lock()
    self._np_state = numpy_storage.NumpyState()

    # Adding elements to the replay buffer is done in a circular way.
    # Keeps track of the actual size of the replay buffer and the location
    # where to add new elements.
    self._np_state.size = np.int64(0)
    self._np_state.cur_id = np.int64(0)

    # Total number of items that went through the replay buffer.
    self._np_state.item_count = np.int64(0)

  def _encoded_data_spec(self):
    """Spec of data items after encoding using _encode."""
    return self._data_spec

  def _encode(self, item):
    """Encodes an item (before adding it to the buffer)."""
    return item

  def _decode(self, item):
    """Decodes an item."""
    return item

  def _on_delete(self, encoded_item):
    """Do any necessary cleanup."""
    pass

  @property
  def size(self):
    return self._np_state.size

  def _add_batch(self, items):
    outer_shape = nest_utils.get_outer_array_shape(items, self._data_spec)
    if outer_shape[0] != 1:
      raise NotImplementedError('PyUniformReplayBuffer only supports a batch '
                                'size of 1, but received `items` with batch '
                                'size {}.'.format(outer_shape[0]))

    item = nest_utils.unbatch_nested_array(items)
    with self._lock:
      if self._np_state.size == self._capacity:
        # If we are at capacity, we are deleting element cur_id.
        self._on_delete(self._storage.get(self._np_state.cur_id))
      self._storage.set(self._np_state.cur_id, self._encode(item))
      self._np_state.size = np.minimum(self._np_state.size + 1, self._capacity)
      self._np_state.cur_id = (self._np_state.cur_id + 1) % self._capacity
      self._np_state.item_count += 1

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    num_steps_value = num_steps if num_steps is not None else 1

    def get_single():
      """Gets a single item from the replay buffer."""
      with self._lock:
        if self._np_state.size <= 0:

          def empty_item(spec):
            return np.empty(spec.shape, dtype=spec.dtype)

          if num_steps is not None:
            item = [
                tf.nest.map_structure(empty_item, self.data_spec)
                for n in range(num_steps)
            ]
            if time_stacked:
              item = nest_utils.stack_nested_arrays(item)
          else:
            item = tf.nest.map_structure(empty_item, self.data_spec)
          return item
        idx = np.random.randint(self._np_state.size - num_steps_value + 1)
        if self._np_state.size == self._capacity:
          # If the buffer is full, add cur_id (head of circular buffer) so that
          # we sample from the range [cur_id, cur_id + size - num_steps_value].
          # We will modulo the size below.
          idx += self._np_state.cur_id

        if num_steps is not None:
          # TODO(b/120242830): Try getting data from numpy in one shot rather
          # than num_steps_value.
          item = [
              self._decode(self._storage.get((idx + n) % self._capacity))
              for n in range(num_steps)
          ]
        else:
          item = self._decode(self._storage.get(idx % self._capacity))

      if num_steps is not None and time_stacked:
        item = nest_utils.stack_nested_arrays(item)
      return item

    if sample_batch_size is None:
      return get_single()
    else:
      samples = [get_single() for _ in range(sample_batch_size)]
      return nest_utils.stack_nested_arrays(samples)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    if num_parallel_calls is not None:
      raise NotImplementedError('PyUniformReplayBuffer does not support '
                                'num_parallel_calls (must be None).')

    data_spec = self._data_spec
    if sample_batch_size is not None:
      data_spec = array_spec.add_outer_dims_nest(data_spec,
                                                 (sample_batch_size,))
    if num_steps is not None:
      data_spec = (data_spec,) * num_steps
    shapes = tuple(s.shape for s in tf.nest.flatten(data_spec))
    dtypes = tuple(s.dtype for s in tf.nest.flatten(data_spec))

    def generator_fn():
      """Generator function."""
      while True:
        if sample_batch_size is not None:
          batch = [
              self._get_next(num_steps=num_steps, time_stacked=False)
              for _ in range(sample_batch_size)
          ]
          item = nest_utils.stack_nested_arrays(batch)
        else:
          item = self._get_next(num_steps=num_steps, time_stacked=False)
        yield tuple(tf.nest.flatten(item))

    def time_stack(*structures):
      time_axis = 0 if sample_batch_size is None else 1
      return tf.nest.map_structure(
          lambda *elements: tf.stack(elements, axis=time_axis), *structures)

    ds = tf.data.Dataset.from_generator(
        generator_fn, dtypes,
        shapes).map(lambda *items: tf.nest.pack_sequence_as(data_spec, items))
    if num_steps is not None:
      return ds.map(time_stack)
    else:
      return ds

  def _gather_all(self):
    data = [
        self._decode(self._storage.get(idx))
        for idx in range(self._capacity)
        if self._storage.get(idx).observation[0]
    ]
    # stacked = nest_utils.stack_nested_arrays(data)
    # batched = tf.nest.map_structure(lambda t: np.expand_dims(t, 0), stacked)
    return data

  def _clear(self):
    self._np_state.size = np.int64(0)
    self._np_state.cur_id = np.int64(0)
