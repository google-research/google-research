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
r"""Aggregates a bunch of uniform sample sets into one uniform sample set.

Example usage:
./aggregate_uniform_samples.py \
    --output=merged_uniform_samples.graphml \
    uniform/*.graphml
"""

from absl import app
from absl import flags
from graph_sampler import graph_io
from graph_sampler import molecule_sampler

FLAGS = flags.FLAGS
flags.DEFINE_float('target_samples', None,
                   'Desired number of samples in the output.')
flags.DEFINE_string('output', None, 'Path for output graphml file.')
flags.DEFINE_string('seed', None, 'Seed used for random number generation.')


def main(argv):
  graph_fnames = argv[1:]

  bucket_sizes, sample_sizes, = [], []
  for graph_fname in graph_fnames:
    stats = graph_io.get_stats(graph_fname)
    bucket_sizes.append(stats['estimated_num_graphs'])
    sample_sizes.append(stats['num_samples'])

  def graph_iter(graph_fname):
    with open(graph_fname) as graph_file:
      for graph in graph_io.graph_reader(graph_file):
        yield graph

  base_iters = (graph_iter(graph_fname) for graph_fname in graph_fnames)
  aggregator = molecule_sampler.AggregateUniformSamples(
      bucket_sizes=bucket_sizes,
      sample_sizes=sample_sizes,
      base_iters=base_iters,
      target_num_samples=FLAGS.target_samples,
      rng_seed=FLAGS.seed)

  with open(FLAGS.output, 'w') as output_file:
    for graph in aggregator:
      graph_io.write_graph(graph, output_file)
      if aggregator.num_accepted % 10000 == 0:
        print(f'Working on file {aggregator.num_iters_started}/'
              f'{len(graph_fnames)}. Accepted {aggregator.num_accepted}/'
              f'{aggregator.num_proccessed} so far.')

    stats = dict(
        target_num_samples=aggregator.target_num_samples,
        num_samples=aggregator.num_accepted,
        rng_seed=aggregator.rng_seed,
        estimated_total_num_graphs=sum(bucket_sizes))
    graph_io.write_stats(stats, output_file)

  acc = aggregator.num_accepted
  proc = aggregator.num_proccessed
  print(f'Done aggregating uniform samples! Accepted {acc}/{proc}: '
        f'{acc / proc * 100:.2f}%')


if __name__ == '__main__':
  app.run(main)
