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

"""Merges and converts bond_topology.log files.

The bond_topology_?.log files are from a fortran program and the enumeration
order determines our bond_topology id numbers. The program merges the available
files, puts them in an easier to parse format, and assigns ids and SMILES.
"""

import csv
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from tensorflow.io import gfile

from smu.parser import smu_utils_lib

flags.DEFINE_string(
    'input_glob',
    '/namespace/gas/primary/smu/enumeration/bond_topology_[2-7].log',
    'Glob of bond_topology log files')
flags.DEFINE_string(
    'output_csv',
    '/namespace/gas/primary/smu/enumeration/merged_bond_topology2-7.csv',
    'Output file path')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with gfile.GFile(FLAGS.output_csv, 'w') as outfile:
    writer = csv.writer(outfile, dialect='unix', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        'id', 'num_atoms', 'atoms_str', 'connectivity_matrix', 'hydrogens',
        'smiles'
    ])
    infiles = sorted(gfile.Glob(FLAGS.input_glob))
    bt_id = 1
    for infn in infiles:
      logging.info('Opening %s at id %d', infn, bt_id)
      with gfile.GFile(infn) as infile:
        for line in infile:
          num_atoms, atoms, connectivity, hydrogens = (
              smu_utils_lib.parse_bond_topology_line(line))
          # The atoms strings looks like 'C N N+O O-' where every atom has a
          # space, +, or - after it. create_bond_topology doesn't want the
          # charge markings (just a string like 'CNNOO') so the [::2] skips
          # those.
          bond_topology = smu_utils_lib.create_bond_topology(
              atoms[::2], connectivity, hydrogens)
          smiles = smu_utils_lib.compute_smiles_for_bond_topology(
              bond_topology, include_hs=False)
          writer.writerow(
              [bt_id, num_atoms, atoms, connectivity, hydrogens, smiles])
          bt_id += 1

    # Add the special cases for SMU 1
    for _, bt_id, atom, valence in smu_utils_lib.SPECIAL_ID_CASES:
      # Note that the SMILES is just the atom. Convenient
      writer.writerow([bt_id, 1, atom, '', valence, atom])

if __name__ == '__main__':
  app.run(main)
