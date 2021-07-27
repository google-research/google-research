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

"""Tests for model_conversion_beam_utils."""

import os

from absl.testing import absltest
import tensorflow as tf

from non_semantic_speech_benchmark.export_model import model_conversion_beam_utils


class ModelConversionBeamUtilsTest(absltest.TestCase):

  def setUp(self):
    super(ModelConversionBeamUtilsTest, self).setUp()
    self.xids = ['12321', '23123', '32123']
    self.folders = [
        '1-al=8.0,ap=False,bd=1024,cop=False,lr=0.0001,ms=large,n_required=80000,qat=False,tbs=8',
        '2-al=8.0,ap=False,bd=1024,cop=False,lr=0.0001,ms=large,n_required=80000,qat=False,tbs=16',
        '3-al=8.0,ap=False,bd=1024,cop=False,lr=0.0001,ms=large,n_required=80000,qat=False,tbs=32',
    ]
    self.base_experiment_dir = os.path.join(
        absltest.get_default_test_tmpdir(), 'base_dir')
    self.output_dir = os.path.join(absltest.get_default_test_tmpdir(), 'out')
    self.conversion_types = ['tflite', 'savedmodel']

    # Make files.
    for xid in self.xids:
      for folder in self.folders:
        tf.io.gfile.makedirs(
            os.path.join(self.base_experiment_dir, xid, folder))

  def test_metadata_correctness(self):
    # Get metadata.
    metadata = model_conversion_beam_utils.get_pipeline_metadata(
        self.base_experiment_dir, self.xids, self.output_dir,
        self.conversion_types)

    # Check correctness.
    self.assertLen(
        metadata,
        len(self.xids) * len(self.folders) * len(self.conversion_types))
    xid_counts = {k: 0 for k in self.xids}
    model_num_counts = {k: 0 for k in range(len(self.folders))}
    conversion_type_counts = {k: 0 for k in self.conversion_types}
    for m in metadata:
      xid_counts[m.xid] += 1
      model_num_counts[m.model_num] += 1
      conversion_type_counts[m.conversion_type] += 1

    for xid in self.xids:
      self.assertEqual(xid_counts[xid], 6)
    for k in range(3):
      self.assertEqual(model_num_counts[k], 6)
    for ctype in conversion_type_counts:
      self.assertEqual(conversion_type_counts[ctype], 9)

    # Check output uniqueness.
    all_output_filenames = [m.output_filename for m in metadata]
    self.assertEqual(len(set(all_output_filenames)), len(all_output_filenames))

  def test_metadata_sanity(self):
    # Get metadata.
    metadata = model_conversion_beam_utils.get_pipeline_metadata(
        self.base_experiment_dir, self.xids, self.output_dir,
        self.conversion_types)

    for m in metadata:
      model_conversion_beam_utils.sanity_check_output_filename(
          m.output_filename)

if __name__ == '__main__':
  absltest.main()
