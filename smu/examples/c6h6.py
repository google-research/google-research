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
"""More complex example processing all isomers of C6H6.

This example will use more python features and idioms than the other examples.
If you are not comfortable with python and just want to get some data out
of SMU, the other examples are easier to work from.

Purpose:
  Select molecules that
    * match a given stoichiometry
    * fulfil certain minimum requirements for the status variable
    * are the lowest in enthalpy among all molecules of a given topology
"""

import collections
import csv
import math
import sys

from smu import smu_sqlite

#-----------------------------------------------------------------------------
# We'll use the complete database because we'll be a little more permissive
# on the status variable.
#-----------------------------------------------------------------------------

db = smu_sqlite.SMUSQLite('20220621_complete_v4.sqlite')

print('# Output from c6h6.py, an example bringing together many concepts')
print('# Please see that file for documentation')

#-----------------------------------------------------------------------------
# Here we select molecules by stoichiometry
# See indices.py for other way to locate a limited set of molecules
#  (by molecule or bond topology ID, by expanded stoichiometry)
#-----------------------------------------------------------------------------

molecules = list(db.find_by_stoichiometry('c6h6'))
print(f'# Found a total of {len(molecules)} molecules')

#-----------------------------------------------------------------------------
# We are going to process each bond topology separately, so we will group the
# molecules by bond topology. We'll use the SMILES string because it is
# unique and canonical per bond topology.
# For collections.defaultdict, see
# https://docs.python.org/3/library/collections.html
#-----------------------------------------------------------------------------

smiles_to_molecules = collections.defaultdict(list)
for mol in molecules:
  #---------------------------------------------------------------------------
  # We select only molecules for which the status variable is 4 or lower
  # and that are not simple duplicates of others.
  # See special_cases.py for a description of the duplicate_of field
  #---------------------------------------------------------------------------

  if (mol.prop.calc.status >= 4 or mol.duplicate_of != 0):
    continue

  #---------------------------------------------------------------------------
  # It's good hygiene to make sure the molecule has the field we want
  # to process. In this case, all the molecules will have this field
  # so this check won't actually do anything but we have it here as a
  # reminder/example. See missing_fields.py for details.
  #---------------------------------------------------------------------------

  if not mol.prop.HasField('elec_dip_pbe0_augpc1'):
    continue

  #---------------------------------------------------------------------------
  # See multiple_bond_topology.py for details on multiple bond topologies
  # per molecule
  #
  # Here we consider all molecules (i.e. optimized geometries) that match
  # a given bond topology (expressed as SMILES).
  #
  # ALTERNATIVE
  # You can just consider the bond topology that was used to produce
  # this geometry. Notably, thermochemical analyses depend on the
  # chosen bond topology and those reported in the database are
  # (strictly) valid only for these "starting" bond topologies, even
  # though dependence on the chosen bond topology is usually fairly
  # small.
  #
  # If you want this, jsut uncomment the if statement below
  #---------------------------------------------------------------------------

  for bt in mol.bond_topo:
    # if not bt.info & dataset_pb2.BondTopology.SOURCE_STARTING:
    #   continue
    smiles_to_molecules[bt.smiles].append(mol)

#-----------------------------------------------------------------------------
# Now process each bond topology separately and write the output.
# We will use the csv writer, see to_csv.py for more details
#
# You can also create a pandas dataframe which gives more flexibility for
# output formats. See dataframe.py for how to do this.
#-----------------------------------------------------------------------------
writer = csv.writer(sys.stdout)
writer.writerow(
    ['mol_id', 'smiles', 'molecule_count', 'dip_x', 'dip_y', 'dip_z', 'dip'])

for smiles in smiles_to_molecules:
  energies = [
      mol.prop.at2_std_b5_hf298.val for mol in smiles_to_molecules[smiles]
  ]
  #---------------------------------------------------------------------------
  # While this line may look mysterious, it's doing exactly what the words say:
  # it finds the index of the minimum value of the energies
  #---------------------------------------------------------------------------
  min_energy_mol_idx = energies.index(min(energies))
  mol = smiles_to_molecules[smiles][min_energy_mol_idx]

  #---------------------------------------------------------------------------
  # See field_access.py for details on the formats of many different kinds of
  # fields
  #---------------------------------------------------------------------------

  dipole = mol.prop.elec_dip_pbe0_augpc1
  writer.writerow([
      mol.mol_id, smiles,
      len(smiles_to_molecules[smiles]), dipole.x, dipole.y, dipole.z,
      '{:.8f}'.format(math.sqrt(dipole.x**2 + dipole.y**2 + dipole.z**2))
  ])
