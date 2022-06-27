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
r"""Convert files of graphml objects to SMILES strings (one per line).

Usage:
./graphs_to_smiles.py merged_uniform.graphml > merged_uniform.smi
"""

from absl import app
from absl import flags
from graph_sampler import graph_io
from graph_sampler import molecule_sampler

FLAGS = flags.FLAGS
flags.DEFINE_list(
    'symbol_dict', [],
    'Optional comma-separated lookup table for converting your symbols into '
    'rdkit-digestible symbols. E.g. if you used "On" to represent a negative '
    'oxygen ion, include "On=O-" in this list so rdkit understands what you '
    'meant. If you used a symbol with an "=" in it, woe is you.')
flags.DEFINE_integer(
    'n', -1, 'Maximum number of graphs to read, or -1 for no limit (default).')


def main(argv):
  symbol_dict = dict(x.split('=') for x in FLAGS.symbol_dict)
  num_remaining = FLAGS.n

  for input_file in argv[1:]:
    if num_remaining == 0:
      break
    with open(input_file) as f:
      for graph in graph_io.graph_reader(f):
        if num_remaining == 0:
          break
        mol = molecule_sampler.to_mol(graph, symbol_dict)
        print(molecule_sampler.to_smiles(mol))
        num_remaining -= 1


if __name__ == '__main__':
  app.run(main)
