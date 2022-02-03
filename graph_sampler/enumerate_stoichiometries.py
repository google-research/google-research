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
r"""Generates every neutral stoichiometry with a given number of heavy atoms.

Example usage:
mkdir stoichs
enumerate_stoichiometries.py --output_prefix=stoichs/ \
    --num_heavy=3 --heavy_elements=C,N,O,S
"""

from absl import app
from absl import flags

from graph_sampler import stoichiometry

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_heavy', None, 'Number of non-hydrogen atoms.')
flags.DEFINE_list('heavy_elements', ['C', 'N', 'N+', 'O', 'O-', 'F'],
                  'Which heavy elements to use.')
flags.DEFINE_string('output_prefix', '', 'Prefix for output files.')
flags.DEFINE_list(
    'valences', [],
    'Valences of atom types (only required for atom types whose valence cannot '
    'be inferred by rdkit, (e.g. "X=7,R=3" if you\'re using "synthetic atoms" '
    'with valences 7 and 3).')
flags.DEFINE_list(
    'charges', [],
    'Charges of atom types (only required for atom types whose charge cannot '
    'be inferred by rdkit, (e.g. "X=0,R=-1" if you\'re using "synthetic atoms" '
    'with valences 0 and -1).')


def main(argv):
  if len(argv) > 1:
    raise RuntimeError(f'Unexpected arguments: {argv[1:]}')
  FLAGS.valences = stoichiometry.parse_dict_flag(FLAGS.valences)
  FLAGS.charges = stoichiometry.parse_dict_flag(FLAGS.charges)

  count = 0
  for stoich in stoichiometry.enumerate_stoichiometries(FLAGS.num_heavy,
                                                        FLAGS.heavy_elements,
                                                        FLAGS.valences,
                                                        FLAGS.charges):
    element_str = ''.join(stoich.to_element_list())
    fn = '%s%d_%s.stoich' % (FLAGS.output_prefix, FLAGS.num_heavy, element_str)
    print(element_str)
    with open(fn, 'w') as f:
      stoich.write(f)
    count += 1

  print(f'{count} files written!')


if __name__ == '__main__':
  flags.mark_flag_as_required('num_heavy')
  app.run(main)
