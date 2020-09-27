# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Extracts edges by TAZ from a file."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import sumolib

from simulation_research.traffic import file_util

FLAGS = flags.FLAGS
flags.DEFINE_string('taz_file', '', 'TAZ file.')
flags.DEFINE_string('output_dir', '', 'Path to which to write outputs.')


def write_edges_by_taz(taz_file):
  for taz in sumolib.xml.parse_fast(taz_file, 'taz', ['id', 'edges']):
    output_file_path = os.path.join(FLAGS.output_dir, taz.id)
    logging.info('Writing edges to path: %s', output_file_path)
    with file_util.f_open(output_file_path, 'w') as f:
      f.write('\n'.join(taz.edges.split(' ')))


def main(_):
  write_edges_by_taz(FLAGS.taz_file)


if __name__ == '__main__':
  app.run(main)
