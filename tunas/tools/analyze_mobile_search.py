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

# Lint as: python2, python3
"""Tool that prints the most likely architecture from a mobile model search."""


from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os
from typing import Text

from absl import app
from absl import flags
import numpy as np
import six
from six.moves import map
import tensorflow.compat.v1 as tf

# The mobile_classifier_factory import might look like it's unused, but
# importing it will register some namedtuples that are needed for model_spec
# deserialization.
from tunas import analyze_mobile_search_lib
from tunas import mobile_classifier_factory  # pylint:disable=unused-import
from tunas import mobile_cost_model
from tunas import schema
from tunas import schema_io


_OUTPUT_FORMAT_LINES = 'lines'
_OUTPUT_FORMAT_CSV = 'csv'
_OUTPUT_FORMATS = (_OUTPUT_FORMAT_LINES, _OUTPUT_FORMAT_CSV)


flags.DEFINE_string(
    'dirname', None,
    'Directory containing the logs to read. Can also be a glob, in which '
    'case we will separately process each directory that matches the pattern.')
flags.DEFINE_enum(
    'output_format', 'lines', _OUTPUT_FORMATS,
    'Format to use for the printed output.')

FLAGS = flags.FLAGS


def _scan_directory(directory,
                    output_format,
                    ssd):
  """Scan a directory for log files and write the final model to stdout."""
  if output_format == _OUTPUT_FORMAT_LINES:
    print('directory =', directory)

  model_spec_filename = os.path.join(directory, 'model_spec.json')
  if not tf.io.gfile.exists(model_spec_filename):
    print('file {} not found; skipping'.format(model_spec_filename))
    if output_format == _OUTPUT_FORMAT_LINES:
      print()
    return

  with tf.io.gfile.GFile(model_spec_filename, 'r') as handle:
    model_spec = schema_io.deserialize(handle.read())

  paths = []
  oneofs = dict()
  def populate_oneofs(path, oneof):
    paths.append(path)
    oneofs[path] = oneof
  schema.map_oneofs_with_paths(populate_oneofs, model_spec)

  all_path_logits = analyze_mobile_search_lib.read_path_logits(directory)
  if not all_path_logits:
    print('event data missing from directory {}; skipping'.format(directory))
    if output_format == _OUTPUT_FORMAT_LINES:
      print()
    return

  global_step = max(all_path_logits)
  if output_format == _OUTPUT_FORMAT_LINES:
    print('global_step = {:d}'.format(global_step))

  all_path_logit_keys = six.viewkeys(all_path_logits[global_step])
  oneof_keys = six.viewkeys(oneofs)
  if all_path_logit_keys != oneof_keys:
    raise ValueError(
        'OneOf key mismatch. Present in event files but not in model_spec: {}. '
        'Present in model_spec but not in event files: {}'.format(
            all_path_logit_keys - oneof_keys,
            oneof_keys - all_path_logit_keys))

  indices = []
  for path in paths:
    index = np.argmax(all_path_logits[global_step][path])
    indices.append(index)

  indices_str = ':'.join(map(str, indices))
  if output_format == _OUTPUT_FORMAT_LINES:
    print('indices = {:s}'.format(indices_str))

  cost_model_time = mobile_cost_model.estimate_cost(indices, ssd)
  if output_format == _OUTPUT_FORMAT_LINES:
    print('cost_model = {:f}'.format(cost_model_time))

  if output_format == _OUTPUT_FORMAT_LINES:
    print()
  elif output_format == _OUTPUT_FORMAT_CSV:
    fields = [indices_str, global_step, directory, cost_model_time]
    print(','.join(map(str, fields)))


def _get_ssd(dirname):
  with tf.io.gfile.GFile(os.path.join(dirname, 'params.json')) as handle:
    params = json.load(handle)
  return params['ssd']


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  directories = tf.io.gfile.glob(FLAGS.dirname)
  if not directories:
    print('No matches found:', FLAGS.dirname)
    return

  if FLAGS.output_format == _OUTPUT_FORMAT_CSV:
    columns = ['indices', 'global_step', 'directory', 'cost_model']
    print(','.join(columns))

  for dirname in directories:
    _scan_directory(
        directory=dirname,
        output_format=FLAGS.output_format,
        ssd=_get_ssd(dirname))


if __name__ == '__main__':
  flags.mark_flag_as_required('dirname')
  tf.logging.set_verbosity(tf.logging.ERROR)
  tf.disable_v2_behavior()
  app.run(main)
