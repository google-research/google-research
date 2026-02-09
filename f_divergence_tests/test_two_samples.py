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

"""Single permutation two-sample test with fixed kernel and bandwidth."""

from collections.abc import Mapping
import json
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging

from f_divergence_tests import distributions
from f_divergence_tests.divergence import kernel as kernel_module
from f_divergence_tests.divergence import likelihood_ratio
from f_divergence_tests.hypothesis_test import fuse


_EXPERIMENT_NAME = flags.DEFINE_string(
    name='experiment_name',
    default='',
    help='Name to append to the results path of the experiment.',
)

_DIVERGENCE_CLASS = flags.DEFINE_enum(
    name='divergence_class',
    default=None,
    enum_values=[
        'drmmd',
        'kl_divergence',
        'reverse_kl',
        'jensen_shannon',
        'total_variation',
        'pearson_chi_squared',
        'squared_hellinger',
        'hockey_stick',
    ],
    help='Divergence to use as a statistic for two sample test.',
    required=True,
)

_DISTRIBUTION = flags.DEFINE_enum(
    name='distribution',
    default=None,
    enum_values=[
        'perturbed_uniform',
        'expo1d',
        'gaussian',
        'laplace',
    ],
    help='Distribution to use for the samples.',
    required=True,
)

_DISTRIBUTION_PARAMS = flags.DEFINE_string(
    name='distribution_params',
    default='{}',
    help='Distribution parameters as a JSON string (e.g., \'{"alpha": 0.5}\')',
)

_DIVERGENCE_PARAMS = flags.DEFINE_string(
    name='divergence_params',
    default='{}',
    help='Divergence parameters as a JSON string (e.g., \'{"alpha": 0.5}\')',
)

_DIVERGENCE_SWEEP_PARAMS = flags.DEFINE_string(
    name='divergence_sweep_params',
    default='{}',
    help=(
        'Divergence parameters to sweep as a JSON string (e.g.,'
        ' \'{"alpha": [0.2, 0.5, 1.0]}\')'
    ),
)

_KERNEL_TYPE = flags.DEFINE_enum(
    name='kernel_type',
    default=None,
    enum_values=[k.kernel_type for k in kernel_module.Kernel],
    help=(
        'Kernel type to use. Available options:'
        f' {[k.kernel_type for k in kernel_module.Kernel]}'
    ),
)

_SIGNIFICANCE = flags.DEFINE_float(
    name='significance',
    default=0.05,
    help='Significance level for the test.',
)


_NUM_PERMUTATIONS = flags.DEFINE_integer(
    name='num_permutations',
    default=500,
    help='Number of permutations to use for the test.',
)
_NUM_BANDWIDTHS = flags.DEFINE_integer(
    name='num_bandwidths',
    default=5,
    help='Number of bandwidths to use for a fuse test.',
)

_MIN_MEMORY_KERNEL = flags.DEFINE_bool(
    name='min_memory_kernel',
    default=True,
    help='Reduce memory usage when computing the kernel.',
)

_MIN_MEMORY_PERMUTATIONS = flags.DEFINE_bool(
    name='min_memory_permutations',
    default=True,
    help='Reduce memory usage when computing the permutations.',
)

_BATCH_SIZE_PERMUTATIONS = flags.DEFINE_integer(
    name='batch_size_permutations',
    default=500,
    help=(
        'Batch size for the permutations when using min_memory_permutations.'
        ' This method uses jax.lax.scan.'
    ),
)

_SEED_SAMPLES = flags.DEFINE_integer(
    name='seed_samples',
    default=1234,
    help='Seed for the samples.',
)

_SEED_TEST = flags.DEFINE_integer(
    name='seed_test',
    default=4321,
    help='Seed for the test.',
)

_LOG_JSON = flags.DEFINE_bool(
    name='log_json',
    default=False,
    help='Log the JSON results or only the test result.',
)

_NUM_TESTS = flags.DEFINE_integer(
    name='num_tests',
    default=1,
    help='Number of tests to run.',
)
_DEBUG_LOGGING = flags.DEFINE_bool(
    name='debug_logging',
    default=False,
    help='Log the all statistics.',
)

DIVERGENCE_MAP = {
    'hockey_stick': likelihood_ratio.SymmetricHockeyStick,
    'kl_divergence': likelihood_ratio.SymmetricKLDivergence,
    'reverse_kl': likelihood_ratio.SymmetricReverseKL,
    'jensen_shannon': likelihood_ratio.SymmetricJensenShannon,
    'total_variation': likelihood_ratio.SymmetricTotalVariation,
    'pearson_chi_squared': likelihood_ratio.SymmetricPearsonChiSquared,
    'squared_hellinger': likelihood_ratio.SymmetricSquaredHellinger,
}


def _sanitize_dict(d):
  """Recursively sanitizes a dictionary for serialization.

  This function creates a new dictionary where all values are converted to
  primitive types (int, float, bool, str, None).  Non-primitive values
  are converted to strings.

  Args:
    d: The dictionary to sanitize.

  Returns:
    A new dictionary with all values converted to primitive types.
  """
  new_d = {}
  for k, v in d.items():
    if isinstance(v, Mapping):
      new_d[k] = _sanitize_dict(v)
    elif isinstance(v, (int, float, bool, str, type(None))):
      new_d[k] = v
    else:
      new_d[k] = str(v)
  return new_d


def get_kernel():
  for k in kernel_module.Kernel:
    if k.kernel_type == _KERNEL_TYPE.value:
      return k


def get_divergence():
  """Returns the divergence class based on the flag."""
  if _DIVERGENCE_CLASS.value == 'drmmd':
    return likelihood_ratio.DrMMDSymmetric
  else:
    try:
      return DIVERGENCE_MAP[_DIVERGENCE_CLASS.value]
    except KeyError as e:
      available = ', '.join(DIVERGENCE_MAP.keys())
      raise ValueError(
          f"Unknown divergence: '{_DIVERGENCE_CLASS.value}'. "
          f'Available options are: [{available}]'
      ) from e


def get_params_dict_from_string(json_str):
  try:
    return json.loads(json_str)
  except json.JSONDecodeError:
    print('Error: Invalid JSON string for divergence parameters.')
    return


def parse_bool_string(value):
  """Returns parsed string as a boolean."""
  if isinstance(value, str):
    lower_value = value.lower()
    if lower_value == 'true':
      return True
    elif lower_value == 'false':
      return False
  return None


def parse_string_params(params_str):
  """Parses a JSON string and converts to native types.

  Receives a dictionary with string values and parses them to their
  corresponding int, float, or boolean types. If the conversion fails, the
  value is kept as a string.

  Args:
    params_str: A JSON string with string values.

  Returns:
    A dictionary with native types.
  """
  params = get_params_dict_from_string(params_str) or {}
  parsed_params = {}
  for key, value in params.items():
    if isinstance(value, str):
      # Try boolean conversion first
      bool_value = parse_bool_string(value)
      if bool_value is not None:
        parsed_params[key] = bool_value
        continue  # Move to the next item if it was a boolean
      # Try parsing as a list of floats
      if value.startswith('[') and value.endswith(']'):
        logging.info('Parsing as a list of floats')
        try:
          # Remove brackets and split by comma
          elements_str = value[1:-1].split(',')
          # Convert each element to float, stripping whitespace
          float_list = [float(e.strip()) for e in elements_str if e.strip()]
          parsed_params[key] = float_list
          continue
        except ValueError:
          logging.info('Failed to parse as a list of floats')
          pass  # If it fails, try other conversions below

      # Then try integer conversion
      try:
        parsed_params[key] = int(value)
      except ValueError:
        # Then try float conversion
        try:
          parsed_params[key] = float(value)
        except ValueError:
          # Keep as string if no conversion worked
          parsed_params[key] = value
    else:
      parsed_params[key] = value  # Keep original if not a string
  return parsed_params


def get_samples(key_seed):
  """Returns the samples based on the flags."""
  params = parse_string_params(_DISTRIBUTION_PARAMS.value)
  params.pop('key_seed', None)
  if _DISTRIBUTION.value == 'gaussian':
    return distributions.ParametricDistributionSamples(
        name='gaussian', key_seed=key_seed, **params
    )
  elif _DISTRIBUTION.value == 'laplace':
    return distributions.ParametricDistributionSamples(
        name='laplace', key_seed=key_seed, **params
    )
  elif _DISTRIBUTION.value == 'perturbed_uniform':
    return distributions.PerturbedUniformSamples(
        n_samples=params['n_samples'],
        key_seed=key_seed,
        num_perturbations=params['num_perturbations'],
        scale=params['scale'],
        dimension=params['dimension'],
        smoothness=params['smoothness'],
    )
  elif _DISTRIBUTION.value == 'expo1d':
    return distributions.Expo1dSamples(
        location=params['location'],
        scale=params['scale'],
        multiplier=params['multiplier'],
        n_samples=params['n_samples'],
        key_seed=key_seed,
    )
  else:
    raise ValueError(f'Unknown distribution: {_DISTRIBUTION.value}')


def setup_test():
  """Runs two-sample test."""
  sweep_params = parse_string_params(_DIVERGENCE_SWEEP_PARAMS.value)
  test = fuse.FuseTest(
      divergence_class=get_divergence(),
      divergence_params=parse_string_params(_DIVERGENCE_PARAMS.value),
      kernels=get_kernel(),
      significance=_SIGNIFICANCE.value,
      number_bandwidths=_NUM_BANDWIDTHS.value,
      num_permutations=_NUM_PERMUTATIONS.value,
      min_memory_kernel=_MIN_MEMORY_KERNEL.value,
      min_memory_permutations=_MIN_MEMORY_PERMUTATIONS.value,
      batch_size_permutations=_BATCH_SIZE_PERMUTATIONS.value,
      divergence_sweep_hyperparams=sweep_params,
  )
  return test


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.set_verbosity(logging.INFO)
  logging.info('Starting test.')

  test = setup_test()
  logging.info('Test successfully setup.')

  for i in range(_NUM_TESTS.value):
    samples = get_samples(
        key_seed=(_NUM_TESTS.value * _SEED_SAMPLES.value) + i,
    )
    logging.info('Samples successfully generated.')
    results = test.run(
        samples_x=samples.samples_x,
        samples_y=samples.samples_y,
        key_seed=47 + (_NUM_TESTS.value * _SEED_SAMPLES.value) + i,
    )

    logging.info(
        'Test successfully run. Rejects: %s, p_val: %s',
        results.result_test,
        results.p_val,
    )


if __name__ == '__main__':
  app.run(main)
