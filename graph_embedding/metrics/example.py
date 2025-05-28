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

"""An example program for computing unsupervised embedding quality metrics."""
from collections.abc import Sequence

from absl import app
from absl import flags
import numpy as np

from graph_embedding.metrics import metrics

_N_POINTS = flags.DEFINE_integer(
    'n_points', 1024, 'Number of clusters.', lower_bound=0
)
_N_DIMS = flags.DEFINE_integer(
    'n_dims', 128, 'Number of dimensions.', lower_bound=0
)


def report_case(case_name, inputs):
  for name, value in sorted(metrics.report_all_metrics(inputs).items()):
    print(f'{case_name} {name}: {value:.4f}')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  cases = {
      'zeros': np.zeros((_N_POINTS.value, _N_DIMS.value)),
      'ones': np.ones((_N_POINTS.value, _N_DIMS.value)),
      'random': np.random.randn(_N_POINTS.value, _N_DIMS.value),
  }
  for case_name, inputs in cases.items():
    report_case(case_name, inputs)
    print('*' * 40)


if __name__ == '__main__':
  app.run(main)
