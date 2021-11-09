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

"""Utils for end-to-end beam pipeline."""

import collections
import os
from typing import List

from absl import flags
from absl import logging

import tensorflow as tf
from non_semantic_speech_benchmark.export_model import model_export_utils

Metadata = collections.namedtuple('Metadata', [
    'xid',
    'model_num',
    'experiment_dir',
    'output_filename',
    'params',
    'conversion_type',
    'experiment_name',
])
# Valid conversion types.
TFLITE_ = 'tflite'
SAVEDMODEL_ = 'savedmodel'


def get_pipeline_metadata(base_experiment_dir, xids,
                          output_dir,
                          conversion_types):
  """Get metadata for entire pipeline."""
  for conversion_type in conversion_types:
    if conversion_type not in [TFLITE_, SAVEDMODEL_]:
      raise ValueError(f'Conversion type not recognized: {conversion_type}')

  metadata = []
  for xid in xids:
    cur_experiment_dir = os.path.join(base_experiment_dir, xid)
    # Get experiment dirs names, params, and output location.
    exp_names = model_export_utils.get_experiment_dirs(cur_experiment_dir)
    for i, exp_name in enumerate(exp_names):
      output_filename = os.path.join(output_dir, f'frillsson_{xid}', exp_name)
      for conversion_type in conversion_types:
        suffix = '.tflite' if conversion_type == TFLITE_ else '_savedmodel'
        cur_metadata = Metadata(
            xid=xid,
            model_num=i,
            experiment_dir=os.path.join(cur_experiment_dir, exp_name),
            output_filename=output_filename + suffix,
            params=model_export_utils.get_params(exp_name),
            conversion_type=conversion_type,
            experiment_name=exp_name,
        )
        metadata.append(cur_metadata)
  return metadata


def sanity_check_output_filename(output_filename):
  """Check that models don't already exist, create directories if necessary."""
  if tf.io.gfile.exists(output_filename):
    raise ValueError(f'Models cant already exist: {output_filename}')
  else:
    tf.io.gfile.makedirs(os.path.dirname(output_filename))


def convert_and_write_model(m, include_frontend,
                            sanity_check):
  """Convert model and write to disk for data prep."""
  logging.info('Working on experiment dir: %s', m.experiment_dir)

  tflite_friendly = m.conversion_type == TFLITE_

  model = model_export_utils.get_model(
      checkpoint_folder_path=m.experiment_dir,
      params=m.params,
      tflite_friendly=tflite_friendly,
      checkpoint_number=None,
      include_frontend=include_frontend)
  if not tf.io.gfile.exists(os.path.dirname(m.output_filename)):
    raise ValueError(
        f'Existing dir didn\'t exist: {os.path.dirname(m.output_filename)}')
  if tflite_friendly:
    model_export_utils.convert_tflite_model(
        model,
        quantize=False,
        model_path=m.output_filename)
  else:
    assert m.conversion_type == SAVEDMODEL_
    tf.keras.models.save_model(model, m.output_filename)
  if not tf.io.gfile.exists(m.output_filename):
    raise ValueError(f'Not written: {m.output_filename}')

  if sanity_check:
    logging.info('Sanity checking...')
    def _p_or_flag(k):
      return m.params[k] if k in m.params else getattr(flags.FLAGS, k)
    model_export_utils.sanity_check(
        include_frontend=include_frontend,
        model_path=m.output_filename,
        embedding_dim=1024,
        tflite=m.conversion_type == TFLITE_,
        n_required=_p_or_flag('n_required'),
        frame_width=_p_or_flag('frame_width'),
        num_mel_bins=_p_or_flag('num_mel_bins'))
