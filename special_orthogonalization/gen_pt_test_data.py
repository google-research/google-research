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

"""Generate modified point cloud test data with fixed random rotations."""
import glob
import os
import pathlib

from . import utils
from absl import app
from absl import flags
import numpy as np
from scipy.stats import special_ortho_group

FLAGS = flags.FLAGS
flags.DEFINE_string('input_test_files', '',
                    'Regular expression for the original input test files.')
flags.DEFINE_string('output_directory', '',
                    'Output directory where new test files will be stored.')
flags.DEFINE_integer('num_rotations_per_file', 100,
                     'Number of random rotation augmentations per test point '
                     'cloud.')
flags.DEFINE_boolean('random_rotation_axang', True,
                     'If true, samples random rotations using the method '
                     'from the original benchmark code. Otherwise samples '
                     'by Haar measure.')


def gen_test_data():
  """Generate the new (modified) test data."""
  # Create output directory.
  os.makedirs(FLAGS.output_directory, exist_ok=True)

  # Get all test point cloud files in the original dataset.
  input_test_files = glob.glob(FLAGS.input_test_files)

  for in_file in input_test_files:
    out_file_prefix = pathlib.Path(in_file).stem
    pts = np.loadtxt(in_file)  # N x 3
    num_pts_to_keep = pts.shape[0] // 2
    pts = pts[:num_pts_to_keep, :]  # N//2 x 3.

    for k in range(FLAGS.num_rotations_per_file):
      if FLAGS.random_rotation_axang:
        r = utils.random_rotation_benchmark_np(1)
        r = r[0]
      else:
        r = special_ortho_group.rvs(3)
      joined = np.float32(np.concatenate((r, pts), axis=0))  # (N//2+3) x 3.
      out_file = os.path.join(
          FLAGS.output_directory, '%s_r%03d.pts'%(out_file_prefix, k))
      np.savetxt(out_file, joined)


def main(unused_argv):
  gen_test_data()

if __name__ == '__main__':
  app.run(main)
