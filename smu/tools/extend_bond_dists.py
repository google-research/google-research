# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Examines extending bond lengths from our initial topology detection."""

import csv
import itertools

from absl import app
from absl import logging

from smu import dataset_pb2

from smu import smu_sqlite
from smu.geometry import bond_length_distribution
from smu.geometry import topology_from_geom
from smu.geometry import topology_molecule


def get_modified_bond_lengths(epsilon):
  """Get modified bond lengths.

  Args:
    epsilon:

  Returns:
    Bond lengths.
  """
  orig_bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
  orig_bond_lengths.add_from_sparse_dataframe_file(
      '20220128_bond_lengths.csv',
      bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
      bond_length_distribution.STANDARD_SIG_DIGITS)

  bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
  for atom_a, atom_b in itertools.combinations_with_replacement([
      dataset_pb2.BondTopology.ATOM_C, dataset_pb2.BondTopology.ATOM_N,
      dataset_pb2.BondTopology.ATOM_O, dataset_pb2.BondTopology.ATOM_F
  ], 2):
    for bond in [
        dataset_pb2.BondTopology.BOND_UNDEFINED,
        dataset_pb2.BondTopology.BOND_SINGLE,
        dataset_pb2.BondTopology.BOND_DOUBLE,
        dataset_pb2.BondTopology.BOND_TRIPLE
    ]:
      if not bond_length_distribution.is_valid_bond(atom_a, atom_b, bond):
        continue
      try:
        dist = orig_bond_lengths[(atom_a, atom_b)][bond]
        if bond == dataset_pb2.BondTopology.BOND_UNDEFINED:
          bond_lengths.add(
              atom_a, atom_b, bond,
              bond_length_distribution.FixedWindow(dist.min() - epsilon, 2.0,
                                                   dist.right_tail_mass))
        else:
          bond_lengths.add(
              atom_a, atom_b, bond,
              bond_length_distribution.FixedWindow(dist.min() - epsilon,
                                                   dist.max() + epsilon, None))
      except KeyError:
        # We have a few missing cases from our original empirical dists
        # For this exercise, we will just copy the CC bond dists for this order
        bond_lengths.add(
            atom_a, atom_b, bond,
            orig_bond_lengths[(dataset_pb2.BondTopology.ATOM_C,
                               dataset_pb2.BondTopology.ATOM_C)][bond])

  return bond_lengths


def main(unused_argv):
  db = smu_sqlite.SMUSQLite('20220128_complete_v2.sqlite')

  buffers = [0, 0.01, 0.025, 0.05]
  bond_lengths = {buf: get_modified_bond_lengths(buf) for buf in buffers}
  for dists in bond_lengths.values():
    bond_length_distribution.add_itc_h_lengths(dists)
  smiles_id_dict = db.get_smiles_id_dict()
  matching_parameters = topology_molecule.MatchingParameters()
  matching_parameters.check_hydrogen_dists = True

  count_processed = 0
  count_matched = 0

  with open('extend_bond_dists.csv', 'w') as outf:
    fields = ['mol_id']
    for buf in buffers:
      fields.append(f'is_matched_{buf}')
      fields.append(f'num_matched_{buf}')
    writer = csv.DictWriter(outf, fields)
    writer.writeheader()

    for molecule in db:
      # for molecule in [db.find_by_mol_id(375986006)]:
      if molecule.prop.calc.fate != dataset_pb2.Properties.FATE_FAILURE_TOPOLOGY_CHECK:
        continue

      count_processed += 1
      if count_processed % 25000 == 0:
        logging.info('Processed %d, matched %d', count_processed, count_matched)

      row = {'mol_id': molecule.mol_id}
      any_match = False

      for buf in buffers:

        matches = topology_from_geom.bond_topologies_from_geom(
            molecule,
            bond_lengths=bond_lengths[buf],
            matching_parameters=matching_parameters)

        matching_bt = [
            smiles_id_dict[bt.smiles] for bt in matches.bond_topology
        ]
        is_matched = (molecule.bond_topo[0].topo_id in matching_bt)

        # if matches.bond_topology:
        #   logging.info('For %d, bt %d, got %s',
        #                molecule.mol_id,
        #                molecule.bond_topo[0].topo_id,
        #                str(matching_bt))

        row[f'is_matched_{buf}'] = is_matched
        row[f'num_matched_{buf}'] = len(matching_bt)
        any_match = any_match or is_matched

      if any_match:
        writer.writerow(row)
        count_matched += 1
        # if count_matched > 1000:
        #   break

  print(f'Final stats: {count_matched} / {count_processed}')


if __name__ == '__main__':
  app.run(main)
