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

r"""Search bang-bang protocols for QAOA.

This is a demo code running on single machine. For the result in the paper, we
run the search on a distributed system.

"""

import functools

from absl import app
from absl import flags
from absl import logging

from bangbang_qaoa import circuit_lib
from bangbang_qaoa import stochastic_descent_lib
from bangbang_qaoa.two_sat import dnf_circuit_lib
from bangbang_qaoa.two_sat import dnf_lib

flags.DEFINE_integer('num_literals', 10,
                     'The number of literals in the disjunctive normal form.',
                     lower_bound=2)
flags.DEFINE_integer('num_clauses', 10, 'The number of random clauses to add.',
                     lower_bound=1)
flags.DEFINE_enum('initialization_method', 'random',
                  ['random', 'adiabatic', 'anti_adiabatic'],
                  'Initialization method of starting protocol of the '
                  'stochastic descent.')
flags.DEFINE_integer('max_num_flips', 1,
                     'The maximum number of flips for stochastic descent.',
                     lower_bound=1)
flags.DEFINE_integer('num_chunks', 100,
                     'The number of chunks in the bang-bang protocol.',
                     lower_bound=1)
flags.DEFINE_float('total_time', 3., 'The total time of the protocol.',
                   lower_bound=0)
flags.DEFINE_integer('num_samples', 1,
                     'The total number of samples to get for each timestep.',
                     lower_bound=1)
flags.DEFINE_bool('skip_search', False,
                  'Whether to skip the stochastic descent search. '
                  'If True, only the initial protocol are evaluated. '
                  'This is used as a baseline.')


FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Generate a random DNF')
  dnf = dnf_lib.get_random_dnf(FLAGS.num_literals, FLAGS.num_clauses)
  logging.info(dnf)

  logging.info('initialization_method: %s', FLAGS.initialization_method)
  if FLAGS.initialization_method == 'random':
    get_initial_protocol = stochastic_descent_lib.get_random_protocol
  elif FLAGS.initialization_method == 'adiabatic':
    get_initial_protocol = functools.partial(
        stochastic_descent_lib.get_random_adiabatic_protocol, ascending=True)
  elif FLAGS.initialization_method == 'anti_adiabatic':
    get_initial_protocol = functools.partial(
        stochastic_descent_lib.get_random_adiabatic_protocol, ascending=False)

  logging.info('Stochastic descent')
  for i in range(FLAGS.num_samples):
    logging.info('Trial index: %d', i)
    bangbang_protocol, protocol_eval, num_epoch = (
        stochastic_descent_lib.stochastic_descent(
            circuit=dnf_circuit_lib.BangBangProtocolCircuit(
                chunk_time=FLAGS.total_time / FLAGS.num_chunks, dnf=dnf),
            max_num_flips=FLAGS.max_num_flips,
            initial_protocol=get_initial_protocol(FLAGS.num_chunks),
            minimize=False,
            skip_search=FLAGS.skip_search))
    logging.info(
        'Optimal protocol: %s',
        circuit_lib.protocol_to_string(bangbang_protocol))
    logging.info('Protocol eval: %f', protocol_eval)
    logging.info('Number of epoch: %d', num_epoch)


if __name__ == '__main__':
  app.run(main)
