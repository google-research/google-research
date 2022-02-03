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
r"""Program to sample molecules from a given stoichiometry.

Example usage:
prefix=3_COFH
./sample_molecules.py --min_samples=3000 \
    --stoich_file=stoichs/${prefix}.stoich \
    --out_file=weighted/${prefix}.graphml
"""

import sys
import timeit

from absl import app
from absl import flags

from graph_sampler import graph_io
from graph_sampler import molecule_sampler
from graph_sampler import stoichiometry

FLAGS = flags.FLAGS

flags.DEFINE_string('stoich_file', None, 'Csv file with desired stoichiometry.')
flags.DEFINE_integer('min_samples', 10000, 'Minimum number of samples.')
flags.DEFINE_float(
    'relative_precision', 0.01,
    'Keep sampling until (std_err / estimate) is less than this number.')
flags.DEFINE_float(
    'min_uniform_proportion', None,
    'Keep sampling until this this set of samples can be rejected down to a '
    'uniform sample containing at least this proportion of the estimated '
    'number of graphs.')
flags.DEFINE_string('out_file', None, 'Output file path.')
flags.DEFINE_string('seed', None, 'Seed used for random number generation.')


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(f'Unexpected arguments: {argv[1:]}')

  print(f'Reading stoich from: {FLAGS.stoich_file}')
  with open(FLAGS.stoich_file) as f:
    stoich = stoichiometry.read(f)

  mol_sampler = molecule_sampler.MoleculeSampler(
      stoich,
      min_samples=FLAGS.min_samples,
      min_uniform_proportion=FLAGS.min_uniform_proportion,
      relative_precision=FLAGS.relative_precision,
      rng_seed=FLAGS.seed)
  start_time = timeit.default_timer()
  num = 0

  def print_progress():
    stats = mol_sampler.stats()
    std_err_frac = stats['num_graphs_std_err'] / stats['estimated_num_graphs']
    est_proportion = (
        stats['num_after_rejection'] / stats['estimated_num_graphs'])
    print(f'Sampled {stats["num_samples"]} ({num} valid), '
          f'{timeit.default_timer() - start_time:.03f} sec, '
          f'{stats["estimated_num_graphs"]:.3E} graphs '
          f'(std err={100 * std_err_frac:.3f}%), '
          f'proportion after rejection={est_proportion:.3E}')
    sys.stdout.flush()

  with open(FLAGS.out_file, 'w') as out:
    for graph in mol_sampler:
      graph_io.write_graph(graph, out)
      num += 1
      if num % 10000 == 0:
        print_progress()

    stats = mol_sampler.stats()
    stats['elapsed time'] = timeit.default_timer() - start_time
    graph_io.write_stats(stats, out)

  print('Done generating molecules!')
  if num % 10000 != 0:
    print_progress()


if __name__ == '__main__':
  flags.mark_flags_as_required(['stoich_file', 'out_file'])
  app.run(main)
