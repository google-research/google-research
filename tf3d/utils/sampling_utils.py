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

"""Utility functions for sampling."""
import tensorflow as tf
import tensorflow_probability as tfp


def get_instance_id_count(instance_ids, valid_mask=None, max_instance_id=None):
  """Computes the count of each instance id.

  Args:
    instance_ids: A tf.int32 tensor of size [batch_size, n].
    valid_mask: A tf.bool tensor of size [batch_size, n] that is True when an
      element is valid and False if it needs to be ignored. By default the value
      is None which means it is not applied.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.

  Returns:
    A tf.float32 tensor of size [batch_size, n] where each row will contain the
    count of the instance id it belongs to.
  """
  if valid_mask is None:
    valid_mask = tf.cast(tf.ones_like(instance_ids), dtype=tf.bool)
  if max_instance_id is None:
    max_instance_id = tf.reduce_max(instance_ids)
  else:
    valid_mask = tf.logical_and(valid_mask,
                                tf.less_equal(instance_ids, max_instance_id))
    instance_ids = tf.where(
        tf.less_equal(instance_ids, max_instance_id), instance_ids,
        tf.zeros_like(instance_ids))
  instance_ids = tf.where(valid_mask, instance_ids,
                          tf.ones_like(instance_ids) * (max_instance_id + 1))
  instance_ids_one_hot = tf.one_hot(
      indices=instance_ids, depth=(max_instance_id + 2), dtype=tf.float32)
  instance_ids_one_hot_count = tf.reduce_sum(instance_ids_one_hot, axis=1)
  return tf.where(
      tf.less_equal(instance_ids, max_instance_id),
      tf.gather(instance_ids_one_hot_count, instance_ids, batch_dims=1),
      tf.zeros_like(instance_ids, dtype=instance_ids_one_hot_count.dtype))


def get_balanced_sampling_probability(instance_ids,
                                      valid_mask=None,
                                      max_instance_id=None):
  """Returns sampling probabilities by balancing based on instance ids.

  Args:
    instance_ids: A tf.int32 tensor of size [batch_size, n].
    valid_mask: A tf.bool tensor of size [batch_size, n] that is True when an
      element is valid and False if it needs to be ignored. By default the value
      is None which means it is not applied.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.
  """
  instance_id_counts = get_instance_id_count(
      instance_ids=instance_ids,
      valid_mask=valid_mask,
      max_instance_id=max_instance_id)
  inverse_counts = 1.0 / tf.maximum(instance_id_counts, 1.0)
  if valid_mask is not None:
    inverse_counts *= tf.cast(valid_mask, dtype=tf.float32)
  return inverse_counts / tf.math.reduce_sum(
      inverse_counts, axis=1, keepdims=True)


def balanced_sample(features,
                    instance_ids,
                    num_samples,
                    valid_mask=None,
                    max_instance_id=None):
  """Samples features by encouraging a balanced selections based on instance id.

  Args:
    features: A tf.float32 tensor of size [batch_size, n, f].
    instance_ids: A tf.int32 tensor of size [batch_size, n].
    num_samples: An int determinig the number of samples.
    valid_mask: A tf.bool tensor of size [batch_size, n] that is True when an
      element is valid and False if it needs to be ignored. By default the value
      is None which means it is not applied.
    max_instance_id: If set, instance ids larger than that value will be
      ignored. If not set, it will be computed from instance_ids tensor.

  Returns:
    A tf.float32 tensor of size [batch_size, num_samples, f] containing sampled
      features.
    A tf.int32 tensor of size [batch_size, num_samples] containing sampled
      instance ids
    A tf.int32 tensor of size [batch_size, num_samples] containing sampled
      indices.
  """
  sampling_probs = get_balanced_sampling_probability(
      instance_ids=instance_ids,
      valid_mask=valid_mask,
      max_instance_id=max_instance_id)
  dist = tfp.distributions.OneHotCategorical(probs=sampling_probs)
  sample_obj = tfp.distributions.Sample(dist, sample_shape=num_samples)
  samples = sample_obj.sample()
  sampled_indices = tf.math.argmax(samples, axis=2)
  sampled_features = tf.gather(features, sampled_indices, batch_dims=1)
  sampled_instance_ids = tf.gather(instance_ids, sampled_indices, batch_dims=1)
  return sampled_features, sampled_instance_ids, sampled_indices
