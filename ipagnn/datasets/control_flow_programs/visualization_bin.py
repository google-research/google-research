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

"""Visualize the control flow graphs of the generated programs."""

import os

from absl import app
from absl import logging  # pylint: disable=unused-import

from python_graphs import control_flow_graphviz

from ipagnn.datasets.control_flow_programs import python_programs
from ipagnn.datasets.control_flow_programs.program_generators import arithmetic_repeats_config
from ipagnn.datasets.control_flow_programs.program_generators import program_generators


def main(argv):
  del argv  # Unused.

  config = arithmetic_repeats_config.ArithmeticRepeatsConfig(
      base=10,
      length=30,
      max_repeat_statements=10,
      max_repetitions=9,
      max_repeat_block_size=20,
      repeat_probability=0.2,
      permit_nested_repeats=True,
  )
  python_source = program_generators.generate_python_source(
      config.length, config)
  cfg = python_programs.to_cfg(python_source)
  num_graphs = len(os.listdir('/tmp/control_flow_graphs/'))
  path = '/tmp/control_flow_graphs/cfg{:03d}.png'.format(num_graphs)
  control_flow_graphviz.render(cfg, include_src=python_source, path=path)


if __name__ == '__main__':
  app.run(main)
