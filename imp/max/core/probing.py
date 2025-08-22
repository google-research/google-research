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

"""Probing utils used across modeling and execution."""

from typing import Callable, Mapping, TypeVar

from flax import struct
from flax import traverse_util
import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import seqio
import tensorflow as tf

from imp.max.core import constants
from imp.max.data import tokenizers
from imp.max.utils import typing

FlaxCollection = constants.FlaxCollection
ProbeType = constants.ProbeType
ProbeDataRank = constants.ProbeDataRank
T = TypeVar('T')
K = TypeVar('K')
_PTYPE_TO_PRANK = {
    ProbeType.IMAGE: ProbeDataRank.IMAGE,
    ProbeType.WAVEFORM: ProbeDataRank.WAVEFORM,
    ProbeType.TEXT: ProbeDataRank.TEXT,
    ProbeType.SCALAR: ProbeDataRank.SCALAR,
    ProbeType.HISTOGRAM: ProbeDataRank.HISTOGRAM,
}

tuple_init = lambda: ()
scalar_init = lambda: 0.
drop_reduce = lambda prev, curr: curr
sum_reduce = lambda prev, curr: prev + curr
tuple_reduce = lambda prev, curr: prev + (curr,)


@struct.dataclass
class ProbeData:
  """The module representing probed data."""
  data: jax.Array
  ptype: str = struct.field(pytree_node=False)

  def __add__(self, other):
    if self.ptype != other.ptype:
      raise ValueError(
          'Two `ProbeData` with different `ptype` cannot be summed together.')
    return ProbeData(data=self.data + other.data, ptype=self.ptype)

  def assert_data_rank(self):
    """Asserts if the data has the expected rank corresponding to its ptype."""
    data_rank = self.data.ndim
    expected_rank = _PTYPE_TO_PRANK[self.ptype]
    if data_rank != expected_rank:
      raise ValueError(
          f'The probe data with rank={data_rank} does not match the expected '
          f'rank={expected_rank} for probe type `{self.ptype}`.')
    return self

  def unroll_scanned_data(self, scan_axis):
    """Unrolls the scanned data given the scan axis."""
    if self.data.ndim != 1 + _PTYPE_TO_PRANK[self.ptype]:
      if self.data.ndim == _PTYPE_TO_PRANK[self.ptype]:
        raise ValueError(
            f'The probed data with prope_type={self.ptype} has a standard '
            f'rank of {self.data.ndim}, which may indicate that the data is '
            'not scanned. Please make sure the probed data is scanned before '
            'calling this method.')
      else:
        raise ValueError('The probed data has a wrong rank. Please make sure '
                         'that probing is done properly.')

    if scan_axis > _PTYPE_TO_PRANK[self.ptype]:
      raise ValueError('Wrong scan axis!')

    num_splits = self.data.shape[scan_axis]
    unrolled_data = jnp.split(self.data, num_splits, scan_axis)
    unrolled_data = [
        jnp.squeeze(data, axis=scan_axis) for data in unrolled_data
    ]
    try:
      unrolled_probe = tuple([
          ProbeData(data=data, ptype=self.ptype).assert_data_rank()
          for data in unrolled_data
      ])
    except ValueError as exc:
      raise ValueError(
          f'Unrolling the probe with the {scan_axis=} is not possible. Please '
          'make sure you pass the correct `scan_axis`.') from exc
    return unrolled_probe


def add_probe(data,
              name,
              ptype,
              collection = FlaxCollection.PROBES,
              init_fn = scalar_init,
              reduce_fn = drop_reduce,
              module = None):
  """Generic function to probe values based on the Flax module.sow.

  Args:
    data: The array to be probed and put in a Flax collection.
    name: Name of the data for which we are probing.
    ptype: The probing type, which could hold any of the types in 'ProbeType'.
    collection: The flax collection where the probe falls in. The default value
      is 'probes'.
    init_fn: The function which would be used to set the initial value of the
      probe. The default value is 0.
    reduce_fn: The function which would be called to aggregate the previously
      stored values. The default behavior is drop the previous probes.
    module: The module under which we probe the values. If not provided, the
      the last module in the module stack under the flax context is fetched.
  """
  if module is None:
    module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
    assert module is not None

  module.sow(col=collection,
             name=name,
             value=ProbeData(data=data, ptype=ptype),
             init_fn=init_fn,
             reduce_fn=reduce_fn)


def _add_multimedia_probe(data,
                          name,
                          ptype,
                          expected_data_rank):
  """Generic helper function to add multimedia data to the probes collection."""
  data_rank = data.ndim
  if data_rank != expected_data_rank:
    raise ValueError(
        f'The data passed for the {ptype} probing should have a '
        f'rank of {expected_data_rank}. Instead received {data_rank=}.')
  add_probe(data=data,
            ptype=ptype,
            name=name,
            reduce_fn=drop_reduce)


def add_image_probe(data, name):
  """Adds image data to the probes collection."""
  _add_multimedia_probe(data=data,
                        name=name,
                        ptype=ProbeType.IMAGE,
                        expected_data_rank=constants.ProbeDataRank.IMAGE)


def add_waveform_probe(data, name):
  """Adds waveform data to the probes collection."""
  _add_multimedia_probe(data=data,
                        name=name,
                        ptype=ProbeType.WAVEFORM,
                        expected_data_rank=constants.ProbeDataRank.WAVEFORM)


def add_text_probe(data, name):
  """Adds text data to the probes collection."""
  _add_multimedia_probe(data=data,
                        name=name,
                        ptype=ProbeType.TEXT,
                        expected_data_rank=constants.ProbeDataRank.TEXT)


def add_scalar_probe(data, name):
  """Adds scalar data to the probes collection."""
  ptype = ProbeType.SCALAR
  data = jnp.asarray(data).reshape(-1)
  if data.shape[0] != 1:
    raise ValueError(
        f'The data passed for the {ptype} probing should contain '
        f'only one element. Instead received an array with {data.shape[0]} '
        'elements.')
  add_probe(data=data.reshape(()),
            ptype=ptype,
            name=name,
            reduce_fn=drop_reduce)


def add_histogram_probe(data, name):
  """Adds histogram data to the probes collection."""
  data = data.reshape(-1)
  add_probe(data=data,
            ptype=ProbeType.HISTOGRAM,
            name=name,
            reduce_fn=drop_reduce)


def add_aux_loss(data,
                 name,
                 module = None):
  """Adds auxiliary loss to the module's collection."""
  data = data.reshape(-1)
  if data.shape[0] != 1:
    raise ValueError(
        f'The data passed for the auxiliary loss probing should contain only '
        f'one element. Instead received an array with {data.shape[0]} elements.'
        )
  if module is None:
    module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
    assert module is not None
  module.sow(col=FlaxCollection.AUX_LOSS,
             name=name,
             value=data.reshape(()),
             init_fn=scalar_init,
             reduce_fn=sum_reduce)


class SummaryWriter():
  """Tensorflow SummaryWriter to write probes to be visualized in TensorBoard."""

  def __init__(self,
               path,
               max_probe_instances = 10,
               waveform_sampling_rate = 16000,
               text_vocabulary = tokenizers
               .get_default_vocabulary(),
               histogram_buckets = 50):
    """Inits SummaryWriter with paths.

    Arguments:
      path: the summary directory name.
      max_probe_instances: At most this many instances of the probes will be
        emitted at each step. When more than `max_probe_instances` many probe
        instances are provided, the first `max_probe_instances` many probe
        instances will be used and the rest silently discarded. This will only
        apply on images and waveform probe types.
      waveform_sampling_rate: The playback sampling rate of the audio (in Hz).
      text_vocabulary: The SeqIO vocabulary to decode the tokenized text probes.
      histogram_buckets: The histogram visualization will have this many
        buckets, except in two edge cases. If there is no data, then there are
        no buckets. If there is data but all points have the same value, then
        all buckets' left and right endpoints are the same and only the last
        bucket has nonzero count.
    """
    self.writer = tf.summary.create_file_writer(path)
    self.max_probe_instances = max_probe_instances
    self.waveform_sampling_rate = waveform_sampling_rate
    self.text_vocabulary = text_vocabulary
    self.histogram_buckets = histogram_buckets

  def __call__(self,
               probes,
               metrics,
               step):
    """Writes probes and metrics to the experiment event.

    Args:
      probes: A nested dictionary of probes. The leaves of this should be an
        instance of `ProbeData`.
      metrics: A nested dictionary of metrics values.
      step: integer. The training step.
    """
    if not isinstance(probes, Mapping):
      raise ValueError(
          'Probes passed to the summary writer should form a dictionary.')
    if not isinstance(metrics, Mapping):
      raise ValueError(
          'Metrics passed to the summary writer should form a dictionary.')

    def _flatten_nested_dict(nested_dict):
      return traverse_util.flatten_dict(nested_dict, sep='/')

    probes = _flatten_nested_dict(probes)
    metrics = _flatten_nested_dict(metrics)
    with self.writer.as_default():
      for name, probe in probes.items():
        name = f'{FlaxCollection.PROBES}/' + name
        if probe.ptype == ProbeType.SCALAR:
          tf.summary.scalar(name=name,
                            data=np.asarray(probe.data, dtype='float32'),
                            step=step)

        elif probe.ptype == ProbeType.IMAGE:
          tf.summary.image(
              name=name,
              data=np.asarray(probe.data, dtype='float32'),
              step=step,
              max_outputs=self.max_probe_instances)

        elif probe.ptype == ProbeType.WAVEFORM:
          tf.summary.audio(
              name=name,
              data=np.asarray(probe.data, dtype='float32'),
              step=step,
              sample_rate=self.waveform_sampling_rate,
              max_outputs=self.max_probe_instances)

        elif probe.ptype == ProbeType.TEXT:
          decoded = self.text_vocabulary.decode_tf(probe.data)
          tf.summary.text(name=name, data=decoded, step=step)

        elif probe.ptype == ProbeType.HISTOGRAM:
          tf.summary.histogram(
              name=name,
              data=np.asarray(probe.data, dtype='float32'),
              step=step,
              buckets=self.histogram_buckets)

      for name, metric_value in metrics.items():
        tf.summary.scalar(name=name, data=metric_value, step=step)

      self.writer.flush()
