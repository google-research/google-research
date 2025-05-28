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

"""Read experiments results."""

import json
import os

from absl import app


def main(argv):
  del argv
  base_name = 'experiment'
  base_dir = '/tmp/model_dir'
  for name in ['mice', 'mnist', 'fashion', 'isolet', 'coil', 'activity']:
    print(name)
    for seed in [1, 2, 3, 4, 5]:
      model_dir = os.path.join(base_dir, name, f'{base_name}_seed_{seed}')
      fit_dir = os.path.join(model_dir, 'fit', 'results.json')
      with open(fit_dir, 'r') as fp:
        results = json.load(fp)
      print(results)

if __name__ == '__main__':
  app.run(main)
