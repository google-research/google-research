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
r"""Reject graphs based on importance to produce a uniform sample set.

Usage:
prefix=3_COFH
./reject_to_uniform.py \
    --in_file=weighted/${prefix}.graphml \
    --out_file=uniform/${prefix}.graphml
"""

from absl import app
from absl import flags

from graph_sampler import graph_io
from graph_sampler import molecule_sampler

FLAGS = flags.FLAGS

flags.DEFINE_string('in_file', None, 'Input file path.')
flags.DEFINE_string('out_file', None, 'Output file path.')
flags.DEFINE_string('seed', None, 'Seed used for random number generation.')


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(f'Unexpected arguments: {argv[1:]}')

  input_stats = graph_io.get_stats(FLAGS.in_file)
  max_importance = input_stats['max_final_importance']
  with open(FLAGS.in_file) as input_file:
    rejector = molecule_sampler.RejectToUniform(
        base_iter=graph_io.graph_reader(input_file),
        max_importance=max_importance,
        rng_seed=FLAGS.seed)
    with open(FLAGS.out_file, 'w') as output_file:
      for graph in rejector:
        graph_io.write_graph(graph, output_file)
        if rejector.num_accepted % 10000 == 0:
          acc = rejector.num_accepted
          proc = rejector.num_processed
          print(f'Accepted {acc}/{proc}: {acc / proc * 100:.2f}%')

      output_stats = dict(
          num_samples=rejector.num_accepted,
          estimated_num_graphs=input_stats['estimated_num_graphs'],
          rng_seed=rejector.rng_seed)
      graph_io.write_stats(output_stats, output_file)

  acc = rejector.num_accepted
  proc = rejector.num_processed
  print(f'Done rejecting to uniform! Accepted {acc}/{proc}: '
        f'{acc / proc * 100:.2f}%')


if __name__ == '__main__':
  flags.mark_flags_as_required(['in_file', 'out_file'])
  app.run(main)
