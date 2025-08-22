# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Single small fully-connected model experiment."""
from typing import Sequence

from absl import app

from fast_gradient_clipping.src import fc_experiment_tools


def main(_):
  p = 2
  q = 1
  r = 4
  m = 2
  batch_size = 1
  param = [p, q, r, m, batch_size]
  runtimes, memories = fc_experiment_tools.get_fully_connected_compute_profile(
      params=[param], repeats=1
  )
  print(runtimes)
  print(memories)


if __name__ == '__main__':
  app.run(main)
