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

"""Beam pipeline identifying non-canonical SMILES.

We want to rely on RDKit's canonical smiles computation to lookup graphs, but
what if different smiles are generated for different orders or atoms?
This pipeline tries to identify bond topologies where ordering or atoms
produces different SMILES strings.

"""

import copy
import itertools
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
from rdkit import Chem
from tensorflow.io import gfile

from smu import dataset_pb2
from smu.parser import smu_utils_lib

flags.DEFINE_string(
    'input_bond_topology_csv',
    '/namespace/gas/primary/smu/enumeration/merged_bond_topology2-7.csv',
    'CSV file of bond topologies (see merge_bond_topologies)')
flags.DEFINE_string(
    'output_csv',
    None,
    'Output file, will have 3 columns (bond topology id, smiles0, smiles1) '
    'with two different smiles produced by that bond topology id')

FLAGS = flags.FLAGS


def generate_bond_topology_permutations(original_bt):
  num_heavy = np.sum(
      [a != dataset_pb2.BondTopology.ATOM_H for a in original_bt.atoms])
  for perm in itertools.permutations(range(num_heavy)):
    bt = copy.deepcopy(original_bt)
    for atom_idx in range(num_heavy):
      bt.atoms[perm[atom_idx]] = original_bt.atoms[atom_idx]
    for bond in bt.bonds:
      if bond.atom_a < num_heavy:
        bond.atom_a = perm[bond.atom_a]
      if bond.atom_b < num_heavy:
        bond.atom_b = perm[bond.atom_b]
    yield bt


def check_smiles_permutation_invariance(original_bt):
  logging.info('Checking %d', original_bt.bond_topology_id)
  smiles = None
  variance_found = False
  for bt in generate_bond_topology_permutations(original_bt):
    mol = smu_utils_lib.bond_topology_to_molecule(bt)
    this_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(mol, sanitize=False),
        kekuleSmiles=True,
        isomericSmiles=False)
    # my little testing hack that includes the atom type of the first atom in
    # the smiles
    # this_smiles += str(bt.atoms[0])
    if smiles is None:
      smiles = this_smiles
    else:
      if this_smiles != smiles:
        variance_found = True
        yield (original_bt.bond_topology_id, smiles, this_smiles)

  if variance_found:
    beam.metrics.Metrics.counter('smu', 'bt_variant').inc()
  else:
    beam.metrics.Metrics.counter('smu', 'bt_invariant').inc()


def pipeline(root):
  """Beam pipeline.

  Args:
    root: the root of the pipeline.
  """
  _ = (
      root
      | 'CreateTopologies' >> beam.Create(
          smu_utils_lib.generate_bond_topologies_from_csv(
              gfile.GFile(FLAGS.input_bond_topology_csv, 'r')))
      | 'Reshuffle1' >> beam.Reshuffle()
      | 'CheckInvariance' >> beam.FlatMap(check_smiles_permutation_invariance)
      | 'Reshuffle2' >> beam.Reshuffle()
      | 'CSVFormat' >> beam.Map(lambda vals: ','.join(str(x) for x in vals))
      | 'WriteOutput' >> beam.io.WriteToText(
          FLAGS.output_csv, header='bt_id,smiles0,smiles1', num_shards=1))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Pipeline Starts.')
  # If you have custom beam options, add them here.
  beam_options = None
  with beam.Pipeline(beam_options) as root:
    pipeline(root)


if __name__ == '__main__':
  app.run(main)
