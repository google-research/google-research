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
"""Checks LLFF pose sampling with only the NumPy parts of the dataset module.

The full RegNeRF dataset module imports legacy JAX/OpenCV dependencies that
are unrelated to pose sampling. Extracting its actual method and pure-NumPy
helpers permits an isolated deterministic regression test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


def _load_llff_pose_generator():
  filename = Path(__file__).with_name('datasets.py')
  program = ast.parse(filename.read_text(encoding='utf-8'))
  functions = [
      node for node in program.body
      if isinstance(node, ast.FunctionDef)
      and node.name in ('normalize', 'viewmatrix', 'poses_avg')
  ]
  llff = next(
      node for node in program.body
      if isinstance(node, ast.ClassDef) and node.name == 'LLFF')
  functions.extend(
      node for node in llff.body
      if isinstance(node, ast.FunctionDef)
      and node.name == '_generate_random_poses')
  namespace = {'np': np}
  exec(compile(ast.Module(body=functions, type_ignores=[]),
               str(filename), 'exec'), namespace)
  return namespace['_generate_random_poses']


class LLFFRandomPoseTest(unittest.TestCase):

  def _dataset(self):
    # Three forward-facing cameras with nonzero variation in each axis.
    base_pose = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    poses = np.stack([base_pose.copy() for _ in range(3)])
    poses[:, :3, 3] = [
        [0.4, 0., 0.],
        [0., 0.6, 0.],
        [0.2, 0., 0.8],
    ]
    bounds = np.array([[1., 4.], [1.1, 3.5], [0.9, 4.2]])
    return SimpleNamespace(camtoworlds_all=poses, bounds=bounds)

  def test_generates_requested_number_of_random_camera_poses(self):
    self.addCleanup(np.random.set_state, np.random.get_state())
    np.random.seed(32)
    dataset = self._dataset()
    _load_llff_pose_generator()(
        dataset, SimpleNamespace(n_random_poses=6))
    self.assertEqual(dataset.random_poses.shape, (6, 3, 4))
    origins = dataset.random_poses[:, :, 3]
    self.assertEqual(np.unique(origins, axis=0).shape[0], 6)
    self.assertTrue(np.isfinite(dataset.random_poses).all())

  def test_one_requested_pose_preserves_single_pose_shape(self):
    self.addCleanup(np.random.set_state, np.random.get_state())
    np.random.seed(32)
    dataset = self._dataset()
    _load_llff_pose_generator()(
        dataset, SimpleNamespace(n_random_poses=1))
    self.assertEqual(dataset.random_poses.shape, (1, 3, 4))


if __name__ == '__main__':
  unittest.main()
