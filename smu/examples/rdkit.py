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

# This example shows how to convert conformers to RDKit molecules.

import sys

from rdkit import Chem

from smu import smu_sqlite
from smu.parser import smu_utils_lib

db = smu_sqlite.SMUSQLite('20220104_standard.sqlite', 'r')

# We will look at one conformer that illustrates the complexities of
# converting a conformer to molecules(s). Not all conformers will have
# this level of complexity.

conformer = db.find_by_conformer_id(8240001)

# This RDKit object will be used to write SDF files to stdout to illustrate
# the example.
writer = Chem.SDWriter(sys.stdout)

# We'll start with the simplest case that always generates a single molcule.
case0_mols = list(smu_utils_lib.conformer_to_molecules(
    conformer,
    include_initial_geometries=False,
    include_optimized_geometry=True,
    include_all_bond_topologies=False))
assert len(case0_mols) == 1

# Note the three "include" arguments above
# * include_initial_geometries: means to generate molecules for all the
#   initial geometries. There will always be at least one initial
#   geometry, but there can be many.
# * include_optimized_geometry: means to include output for the
#   (single) optimized geometry.
# * include_all_bond_topologies: False means to use only the first
#   (i.e. best matching) bond topology that fits this geometry. True
#   means to generate separate rdkit molcules for each topology.

print('A single molecule comes from asking for only the optimized geometry and only',
      'the first bond topology')
print('Note that the title line start with "SMU" and the conformer id')
print('Note the "geom=opt" indicating this is the optimized geometry')
print('Note the bt=8240(1/2) indicating this is the first of 2 bond topologies')
print('In this case, we only generated output for a single topology')
writer.write(case0_mols[0])
writer.flush()

case1_mols = list(smu_utils_lib.conformer_to_molecules(
    conformer,
    include_initial_geometries=True,
    include_optimized_geometry=False,
    include_all_bond_topologies=False))
assert len(case1_mols) == 4

print()
print('When we ask for the initial gemetries only, we get 1 or more molecules')
print('In this case we get 4 molecules because there are 4 initial_geometries')
print('Note the "geom=init(X/4)" inidicating which geometry this is')
for mol in case1_mols:
    writer.write(mol)
writer.flush()

case2_mols = list(smu_utils_lib.conformer_to_molecules(
    conformer,
    include_initial_geometries=False,
    include_optimized_geometry=True,
    include_all_bond_topologies=True))
assert len(case2_mols) == 2

print()
print('When we ask for all bond topologies, we get 1 or more molecules')
print('In this case, there are 2 bond topologies to describe this molecule')
print('Note that the atoms are the same but the connection table is different')
print('Note the "bt=8240(1/2)" and "bt=8237(2/2)" for the bond topology ids.')
print('All the available bond topologies are listed in bond_topology.csv')
for mol in case2_mols:
    writer.write(mol)
writer.flush()

case3_mols = list(smu_utils_lib.conformer_to_molecules(
    conformer,
    include_initial_geometries=True,
    include_optimized_geometry=True,
    include_all_bond_topologies=True))
assert len(case3_mols) == 10

print()
print('If we ask for everything, we get 10 molecules')
print('5 possible geometries (4 initial and 1 optimized)')
print('times')
print('2 possible bond topologies')
print('Number of molecules generated:', len(case3_mols))


print()
print('See the comments in rdkit.py for API details.')
