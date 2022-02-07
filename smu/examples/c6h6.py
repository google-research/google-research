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

"""More complex example processing all isomers of C6H6.

This example will use more python features and idioms than the other examples.
If you are not comfortable with python and just want to get some data out
of SMU, the other examples are easier to work from.

This will produce information on the dipole moment for the lowest energy
(enthalpy_of_formation_298k_atomic_b5) conformer for each possible bond
topology.
"""

import collections
import csv
import math
import sys

from smu import smu_sqlite

# We'll use the complete database because we'll be a little more permissive
# on the status variable.
db = smu_sqlite.SMUSQLite('20220128_complete.sqlite')

print('# Output from c6h6.py, an example bringing together many concepts')
print('# Please see that file for documentation')

# See indices.py for other way to locate a limited set of conformers
conformers = list(db.find_by_stoichiometry('c6h6'))
print(f'# Found a total of {len(conformers)} conformers')

# We are going to process each bond topology separately, so we will group the
# conformers by bond topology. We'll use the SMILES string because it is
# unique and canonical per bond topology.
# For collections.defaultdict, see
# https://docs.python.org/3/library/collections.html
smiles_to_conformers = collections.defaultdict(list)
for conf in conformers:
  # See special_cases.py for a description of the duplicated_by field
  if (conf.properties.errors.status >= 4 or
      conf.duplicated_by != 0):
    continue

  # It's good hygiene to make sure the conformer has the field we want
  # to process. In this case, all the conformers will have this field
  # so this check won't actually do anything but we have it here as a
  # reminder/example. See missing_fields.py for details.
  if not conf.properties.HasField('dipole_moment_pbe0_aug_pc_1'):
    continue

  # See multiple_bond_topology.py for details on multiple bond topologies
  # per conformer
  for bt in conf.bond_topologies:
    smiles_to_conformers[bt.smiles].append(conf)


# Now process each bond topology separately and write the output.

# We will use the csv writer, see to_csv.py for more details
writer = csv.writer(sys.stdout)
writer.writerow(['conformer_id',
                 'smiles',
                 'conformer_count',
                 'dip_x',
                 'dip_y',
                 'dip_z',
                 'dip'])

for smiles in smiles_to_conformers:
  energies = [conf.properties.enthalpy_of_formation_298k_atomic_b5.value
              for conf in smiles_to_conformers[smiles]]
  min_energy_conformer_idx = energies.index(min(energies))
  conf = smiles_to_conformers[smiles][min_energy_conformer_idx]
  # See field_access.py for details on the formats of many different kinds of
  # fields
  dipole = conf.properties.dipole_moment_pbe0_aug_pc_1
  writer.writerow([conf.conformer_id,
                   smiles,
                   len(smiles_to_conformers[smiles]),
                   dipole.x,
                   dipole.y,
                   dipole.z,
                   math.sqrt(dipole.x**2 + dipole.y**2 + dipole.z**2)])
