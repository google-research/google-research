# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Read dataset and do some preprocessing."""

import time

from absl import logging
import jax
import tensorflow as tf

from global_metnet import geo_tensor


def cast(xs, dtype):
  """Cast a structure of tf.Tensor/GeoTensor.

  Currently only casts from float32 to bfloat16.

  Args:
    xs: A structure of tf.Tensor/GeoTensor.
    dtype: Dtype as string.

  Returns:
    A structure with the values cast.
  """

  def _cast(x):
    if dtype == 'bfloat16' and x.dtype == tf.float32:
      if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.bfloat16)
      elif isinstance(x, geo_tensor.GeoTensor):
        return x.astype(tf.bfloat16)
      else:
        raise ValueError(f'{x=}')
    else:
      return x

  return tf.nest.map_structure(_cast, xs)


def global_preprocess(
    hps,
    dataset_fn,
    split,
):
  """Preprocessor for metnetv5.

  This preprocessor has the assumption that the output of the dataset_reader
  is of the format {
      'inputs': {key: GeoTensor for key in config.inputs},
      'targets': {key: GeoTensor for key in config.heads},
  }

  Args:
    hps: The configuration used to extract data on the fly and for training.
    dataset_fn: The dataset reader used to extract data.
    split: The train/test/val being used for training.

  Returns:
    A tf.data.Dataset after preprocessing on the data.
  """
  is_eval = split != 'train'

  dataset = dataset_fn(None)

  # Preprocess inputs.
  def preprocess_inputs_and_targets(v):
    start = time.time()
    results = {'inputs': {}, 'target': {}}

    for input_key, input_ in hps.inputs.items():
      input_ = input_.preprocess(
          hps.context_size_km,
          hps.input_resolution_km,
          is_eval,
          **{k: v['inputs'][k] for k in input_.dataset_keys},
      )
      results['inputs'][input_key] = cast(input_, hps.dtype)

    for key, head in hps.heads.items():
      timedeltas = hps.get('target_tds')
      targets = head.preprocessor(is_eval).preprocess(
          size_km=hps.target_size_km,
          resolution_km=head.resolution_km,
          is_eval=is_eval,
          **{
              k: v['targets'][k]
              for k in head.preprocessor(is_eval).dataset_keys
          },
      )

      if head.temporal_preprocessor:
        targets, _ = head.temporal_preprocessor.preprocess(targets, timedeltas)
      results['target'][key] = targets
    elapsed_time = time.time() - start
    logging.info(
        'preprocess_inputs_and_targets took %0.1f seconds', elapsed_time
    )
    return results

  dataset = dataset.map(preprocess_inputs_and_targets, num_parallel_calls=2)

  # Sample lead times.
  if is_eval:
    possible_times = hps.eval_tds
  else:
    possible_times = hps.target_tds
  logging.info('Split: %s Possible times: %s', split, possible_times)

  splits_dataset = lambda: tf.data.Dataset.from_tensor_slices(possible_times)

  dataset = dataset.flat_map(
      lambda val: tf.data.Dataset.zip(
          (splits_dataset(), tf.data.Dataset.from_tensors(val).repeat())
      )
  )

  # Preprocess targets and masks.
  def interpolate_targets_and_compute_masks(t, v):
    start = time.time()
    results = {k: v[k] for k in ['inputs', 'target']}

    # Target index.
    results['inputs']['lead_times'] = {
        k: t + head.target_offsets for k, head in hps.heads.items()
    }
    target_index = tf.squeeze(tf.searchsorted(hps.target_tds, [t]), axis=0)
    tf.debugging.assert_equal(tf.gather(hps.target_tds, target_index), t)
    results['inputs']['target_index'] = target_index
    results['mask'] = {}
    elapsed_time = time.time() - start
    logging.info(
        'interpolate_targets_and_compute_masks took %0.1f seconds', elapsed_time
    )
    return results

  dataset = dataset.map(
      interpolate_targets_and_compute_masks, num_parallel_calls=2
  )

  def remove_metadata(d):
    return tf.nest.map_structure(
        lambda v: v.data if isinstance(v, geo_tensor.GeoTensor) else v, d
    )

  dataset = dataset.map(remove_metadata)
  return dataset


class DatasetPreprocessor(object):
  """Read dataset, preprocess and do bunch of optimizations like prefetching."""

  def __init__(self, hps):
    self._hps = hps

  def get_dataset(
      self,
      split,
  ):
    """Returns a TF dataset for a given split."""

    def dataset_fn():
      raise NotImplementedError(
          'The dataset reader function is not implemented yet.'
      )

    return self._preprocess(
        dataset_fn,
        split,
    )

  def get_iterator_from_dataset(self, dataset):
    it = iter(dataset)

    def as_numpy(x):
      return jax.tree.map(lambda x: x.numpy(), x)

    return map(as_numpy, it)

  def _preprocess(
      self,
      dataset_fn,
      split,
  ):
    """Preprocess dataset."""
    def new_dataset_fn(include_keys):
      dataset = dataset_fn(include_keys)

      def key_as_feature(k, v):
        v['key'] = k
        return v

      return dataset.map(
          key_as_feature,
          num_parallel_calls=tf.data.AUTOTUNE,
      )

    dataset = global_preprocess(
        self._hps,
        new_dataset_fn,
        split,
    )

    def prepare_inputs(s):
      start = time.time()
      inputs = s['inputs']
      if self._hps.get('assert_finite_inputs', False):
        for k, v in inputs.items():
          if isinstance(v, tf.Tensor) and v.dtype.is_floating:
            tf.debugging.assert_all_finite(
                v, 'Non-finite value in input {}.'.format(k)
            )

      # Filter unused inputs.
      s.update(
          inputs={
              k: inputs.get(k, tf.zeros([], self._hps.dtype))
              for k in self._hps.inputs.keys()
          }
      )
      elapsed_time = time.time() - start
      logging.info('Prepare_inputs took %0.1f seconds', elapsed_time)
      return s

    dataset = dataset.map(
        prepare_inputs,
        num_parallel_calls=self._hps.get('dataset_num_parallel_calls', 1),
    )

    def discretize(s):
      """Discretize."""
      start = time.time()
      if 'target' not in s:
        return s

      target = s['target']

      for key, val in target.items():
        target[key] = self._hps.heads[key].preprocess_target(val, split=split)
      elapsed_time = time.time() - start
      logging.info('Discretize took %0.1f seconds', elapsed_time)
      return s

    dataset = dataset.map(
        discretize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(max(self._hps.batch_size, 1), drop_remainder=True)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
