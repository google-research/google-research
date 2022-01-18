# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Functions for sampling from different distributions.

Sampling functions for YOTO. Also includes functions to transform the samples,
for instance via softmax.
"""

import ast
import enum
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd


@gin.constants_from_enum
class DistributionType(enum.Enum):
  UNIFORM = 0
  LOG_UNIFORM = 1


@gin.constants_from_enum
class TransformType(enum.Enum):
  IDENTITY = 0
  LOG = 1


@gin.configurable("DistributionSpec")
class DistributionSpec(object):
  """Spec of a distribution for YOTO training or evaluation."""
  # NOTE(adosovitskiy) Tried to do it with namedtuple, but failed to make
  # it work with gin

  def __init__(self, distribution_type, params, transform):
    self.distribution_type = distribution_type
    self.params = params
    self.transform = transform


# TODO(adosovitskiy): have one signature with distributionspec and one without
def get_samples_as_dicts(distribution_spec, num_samples=1,
                         names=None, seed=None):
  """Sample weight dictionaries for multi-loss problems.

  Supports many different distribution specifications, including random
  distributions given via DistributionSpec or fixed sets of weights given by
  dictionaries or lists of dictionaries. The function first parses the different
  options and then actually computes the weights to be returned.

  Args:
    distribution_spec: One of the following:
      * An instance of DistributionSpec
      * DistributionSpec class
      * A dictionary mapping loss names to their values
      * A list of such dictionaries
      * A string representing one of the above
    num_samples: how many samples to return (only if given DistributionSpec)
    names: names of losses (only if given DistributionSpec)
    seed: random seed to use for sampling (only if given DistributionSpec)

  Returns:
    samples_dicts: list of dictionaries with the samples weights
  """

  # If given a string, first eval it
  if isinstance(distribution_spec, str):
    distribution_spec = ast.literal_eval(distribution_spec)

  # Now convert to a list of dictionaries or an instance of DistributionSpec
  if isinstance(distribution_spec, dict):
    given_keys = distribution_spec.keys()
    if not (names is None or set(names) == set(given_keys)):
      raise ValueError(
          "Provided names {} do not match with the keys of the provided "
          "dictionary {}".format(names, given_keys))
    distribution_spec = [distribution_spec]
  elif isinstance(distribution_spec, list):
    if not (distribution_spec and
            isinstance(distribution_spec[0], dict)):
      raise ValueError(
          "If distribution_spec is a list, it should be non-empty and "
          "consist of dictionaries.")
    given_keys = distribution_spec[0].keys()
    if not (names is None or set(names) == set(given_keys)):
      raise ValueError(
          "Provided names {} do not match with the keys of the provided "
          "dictionary {}".format(names, given_keys))
  elif isinstance(distribution_spec, type):
    distribution_spec = distribution_spec()
  else:
    raise TypeError(
        "The distribution_spec should be a dictionary ot a list of dictionaries"
        " or an instance of DistributionSpec or class DistributionSpec")

  assert (isinstance(distribution_spec, DistributionSpec) or
          isinstance(distribution_spec, list)), \
          "By now distribution_spec should be a DistributionSpec or a list"

  # Finally, actually make the samples
  if isinstance(distribution_spec, DistributionSpec):
    # Sample and convert to a list of dictionaries
    samples = get_sample((num_samples, len(names)), distribution_spec,
                         seed=seed, return_numpy=True)
    samples_dicts = []
    for k in range(num_samples):
      samples_dicts.append(
          {name: samples[k, n] for n, name in enumerate(names)})
  elif isinstance(distribution_spec, list):
    samples_dicts = distribution_spec

  return samples_dicts


def get_sample_untransformed(shape, distribution_type, distribution_params,
                             seed):
  """Get a distribution based on specification and parameters.

  Parameters can be a list, in which case each of the list members is used to
  generate one row (or column?) of the resulting sample matrix. Otherwise, the
  same parameters are used for the whole matrix.

  Args:
    shape: Tuple/List representing the shape of the output
    distribution_type: DistributionType object
    distribution_params: Dict of distributon parameters
    seed: random seed to be used

  Returns:
    sample: TF Tensor with a sample from the distribution
  """
  if isinstance(distribution_params, list):
    if len(shape) != 2 or len(distribution_params) != shape[1]:
      raise ValueError("If distribution_params is a list, the desired 'shape' "
                       "should be 2-dimensional and number of elements in the "
                       "list should match 'shape[1]'")
    all_samples = []
    for curr_params in distribution_params:
      curr_samples = get_one_sample_untransformed([shape[0], 1],
                                                  distribution_type,
                                                  curr_params, seed)
      all_samples.append(curr_samples)
    return tf.concat(all_samples, axis=1)
  else:
    return get_one_sample_untransformed(shape, distribution_type,
                                        distribution_params, seed)


def get_one_sample_untransformed(shape, distribution_type, distribution_params,
                                 seed):
  """Get one untransoformed sample."""
  if distribution_type == DistributionType.UNIFORM:
    low, high = distribution_params["low"], distribution_params["high"]
    distribution = tfd.Uniform(low=tf.constant(low, shape=shape[1:]),
                               high=tf.constant(high, shape=shape[1:],))
    sample = distribution.sample(shape[0], seed=seed)
  elif distribution_type == DistributionType.LOG_UNIFORM:
    low, high = distribution_params["low"], distribution_params["high"]
    distribution = tfd.Uniform(
        low=tf.constant(np.log(low), shape=shape[1:], dtype=tf.float32),
        high=tf.constant(np.log(high), shape=shape[1:], dtype=tf.float32))
    sample = tf.exp(distribution.sample(shape[0], seed=seed))
  else:
    raise ValueError("Unknown distribution type {}".format(distribution_type))
  return sample


def get_sample(shape, distribution_spec, seed=None, return_numpy=False):
  """Sample a tensor of random numbers.

  Args:
    shape: shape of the resulting tensor
    distribution_spec: DistributionSpec
    seed: random seed to use for sampling
    return_numpy: if True, returns a fixed numpy array, otherwise - a TF op
      that allows sampling repeatedly

  Returns:
    samples: numpy array or TF op representing the random numbers
  """
  distribution_type = distribution_spec.distribution_type  # pytype: disable=attribute-error
  distribution_params = distribution_spec.params  # pytype: disable=attribute-error
  transform_type = distribution_spec.transform  # pytype: disable=attribute-error

  sample_tf = get_sample_untransformed(shape, distribution_type,
                                       distribution_params, seed)

  if transform_type is not None:
    transform = get_transform(transform_type)
    sample_tf = transform(sample_tf)

  if return_numpy:
    with tf.Session() as sess:
      sample_np = sess.run([sample_tf])[0]
    return sample_np
  else:
    return sample_tf


def get_transform(transform_type):
  """Get transforms for converting raw samples to weights and back."""
  if transform_type == TransformType.IDENTITY:
    transform = lambda x: x
  elif transform_type == TransformType.LOG:
    transform = tf.log
  else:
    raise ValueError("Unknown transform type {}".format(transform_type))
  return transform
