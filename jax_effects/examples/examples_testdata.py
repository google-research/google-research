# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Test data for examples_test."""

import numpy as np

qlearning_expected_qtable = np.asarray(
    [
        [
            [-1.1415762, -20.0, -1.1415762, -1.2587268],
            [-0.7763184, -0.907632, -0.97172475, -0.84326243],
            [-0.5816, -0.69048, -0.67242825, -0.66168],
            [-0.58808, -0.524, -0.639344, -0.58808],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.6671072, -0.6552, -20.0648, -0.61112],
            [-0.60464, -0.524, -0.61112, -0.4248],
            [-0.396, -0.524, -0.42479998, -0.396],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.436464, -0.524, -20.105856, -0.42479998],
            [-0.396, -0.524, -0.396, -0.396],
            [-0.396, -0.524, -0.396, -0.396],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.4248, -0.36, -20.17084, -0.23599999],
            [-0.4248, -0.36, -0.396, -0.2],
            [-0.396, -0.36, -0.2, -0.2],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.2, -0.2, -20.172943, -0.23599999],
            [-0.396, -0.36, -0.2, -0.2],
            [-0.2, -0.2, -0.396, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [-0.2, 0.0, -0.2, 0.0],
            [-0.2, -0.2, -0.2, 0.0],
            [-0.2, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=np.float32,
)
