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

"""Contains functions to define flags and params.

Calling a DEFINE_* function will add a ParamSpec namedtuple to the param_spec
dict. The DEFINE_* arguments match those in absl. Calling define_flags() creates
a command-line flag for every ParamSpec defined by a DEFINE_* functions.

The reason we don't use absl flags directly is that we want to be able to use
tf_cnn_benchmarks as a library. When using it as a library, we don't want to
define any flags, but instead pass parameters to the BenchmarkCNN constructor.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from absl import flags as absl_flags
import six


FLAGS = absl_flags.FLAGS


# ParamSpec describes one of benchmark_cnn.BenchmarkCNN's parameters.
ParamSpec = namedtuple('_ParamSpec',
                       ['flag_type', 'default_value', 'description',
                        'kwargs'])


# Maps from parameter name to its ParamSpec.
param_specs = {}


def DEFINE_string(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('string', default, help, {})


def DEFINE_boolean(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('boolean', default, help, {})


def DEFINE_integer(name, default, help, lower_bound=None, upper_bound=None):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
  param_specs[name] = ParamSpec('integer', default, help, kwargs)


def DEFINE_float(name, default, help, lower_bound=None, upper_bound=None):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
  param_specs[name] = ParamSpec('float', default, help, kwargs)


def DEFINE_enum(name, default, enum_values, help):  # pylint: disable=invalid-name,redefined-builtin
  kwargs = {'enum_values': enum_values}
  param_specs[name] = ParamSpec('enum', default, help, kwargs)


def DEFINE_list(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  param_specs[name] = ParamSpec('list', default, help, {})


def define_flags(specs=None):
  """Define a command line flag for each ParamSpec in flags.param_specs."""
  specs = specs or param_specs
  define_flag = {
      'boolean': absl_flags.DEFINE_boolean,
      'float': absl_flags.DEFINE_float,
      'integer': absl_flags.DEFINE_integer,
      'string': absl_flags.DEFINE_string,
      'enum': absl_flags.DEFINE_enum,
      'list': absl_flags.DEFINE_list
  }
  for name, param_spec in six.iteritems(specs):
    if param_spec.flag_type not in define_flag:
      raise ValueError('Unknown flag_type %s' % param_spec.flag_type)
    else:
      define_flag[param_spec.flag_type](name, param_spec.default_value,
                                        help=param_spec.description,
                                        **param_spec.kwargs)
