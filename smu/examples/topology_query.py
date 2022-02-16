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

"""Use the query by topology functionality."""

from smu import smu_sqlite
from smu.geometry import bond_length_distribution

db = smu_sqlite.SMUSQLite('20220128_standard_v2.sqlite')
smiles = '[O-]N=N[NH+]=[N+]([O-])F'

print('Conformers can have multiple bond topologies that match the geometry')
print('(see multiple_bond_topology.py for some illustration)')

print('This is what the "find_by_smiles" method is using and is very efficient')

original_conformers = sorted(db.find_by_smiles_list([smiles]),
                             key=lambda c: c.conformer_id)
print('find_by_smiles on', smiles, 'finds these conformer ids')
print([c.conformer_id for c in original_conformers])

print()
print('But you can modify the allowed distances for each type of bond')
print('and find all conformers which match a given topology with these modifications')

print('While this does not have the read the whole database, it is a much less efficient operation than querying by smiles, so only use it if you modify the allowed distances')

print()
print('First you have to load the default bond lengths')
bond_lengths = bond_length_distribution.AllAtomPairLengthDistributions()
bond_lengths.add_from_sparse_dataframe_file(
  '20220128_bond_lengths.csv',
  bond_length_distribution.STANDARD_UNBONDED_RIGHT_TAIL_MASS,
  bond_length_distribution.STANDARD_SIG_DIGITS)

print()
print('You then provide the desired topology as a SMILES string')
print('The topology query without modifying bond lengths, finds the same result')
unmodified_conformers = sorted(
  list(db.find_by_topology(smiles, bond_lengths)),
  key=lambda c: c.conformer_id)
print('Unmodified find_by_topology finds these conformer ids')
print([c.conformer_id for c in unmodified_conformers])

print()
print('We now modify the bond lengths by allowing N to N bonds of any order')
print('to be between 1A and 2A with the string "N~N:1.0-2.0"')
bond_lengths.add_from_string_spec('N~N:1.0-2.0')
modified_conformers = sorted(
  list(db.find_by_topology(smiles, bond_lengths)),
  key=lambda c: c.conformer_id)
print('We now find these conformer ids')
print([c.conformer_id for c in modified_conformers])

print()
print('Also note that the bond_topologies field of all conformers returned')
print('from a topology query have been modified with the new allowed bond distances')

#-----------------------------------------------------------------------------
# This is a utility function we will use below to print summary information
# about the bnd topologies found
#-----------------------------------------------------------------------------
def print_conformers_and_bond_topology_id(conformers):
  for conf in conformers:
    print('   ', conf.conformer_id, 'has bond topologies:',
          [bt.bond_topology_id for bt in conf.bond_topologies])

print()
print('Compare the bond topologies from the orignal query:')
print_conformers_and_bond_topology_id(original_conformers)
print()
print('With the topologies from the query with modified bond distances:')
print_conformers_and_bond_topology_id(modified_conformers)
