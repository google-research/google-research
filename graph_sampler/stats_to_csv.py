# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

#!/usr/bin/python
r"""Combine the stats dictionaries from a bunch of graphml files into a csv.

Usage:
./stats_to_csv.py --output=weighted/stats.csv weighted/*.graphml
"""

from absl import app
from absl import flags
from graph_sampler import graph_io
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('output', None, 'Path to output csv file.')


def main(argv):
  filenames = argv[1:]
  stats_list = [graph_io.get_stats(filename) for filename in filenames]
  df = pd.DataFrame(stats_list, index=pd.Index(filenames, name='filename'))
  df.to_csv(FLAGS.output)


if __name__ == '__main__':
  flags.mark_flag_as_required('output')
  app.run(main)
