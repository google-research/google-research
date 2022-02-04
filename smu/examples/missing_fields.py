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

"""Shows how to check for missing fields."""

from smu import smu_sqlite

# Note that we are loading the *complete* database
db = smu_sqlite.SMUSQLite('20220128_complete.sqlite')

# We'll grab a couple of conformers with different amount of information
# stored.
PARTIAL_CONFORMER_ID = 35004068
MINIMAL_CONFORMER_ID = 35553043
partial_conformer = db.find_by_conformer_id(PARTIAL_CONFORMER_ID)
minimal_conformer = db.find_by_conformer_id(MINIMAL_CONFORMER_ID)

print('When you process the *complete* database, you have to be careful to',
      'check what data is available')

print('We will examine conformers', PARTIAL_CONFORMER_ID, 'and',
      MINIMAL_CONFORMER_ID)

print(
    'If you ask for the optimized_geometry_energy for both you get sensible values'
)
print(PARTIAL_CONFORMER_ID,
      partial_conformer.properties.optimized_geometry_energy.value)
print(MINIMAL_CONFORMER_ID,
      minimal_conformer.properties.optimized_geometry_energy.value)

print()
print('But if you ask for zpe_unscaled, the second gives a 0')
print(PARTIAL_CONFORMER_ID, partial_conformer.properties.zpe_unscaled.value)
print(MINIMAL_CONFORMER_ID, minimal_conformer.properties.zpe_unscaled.value)

print()
print('And if you ask for enthalpy_of_formation_298k_atomic_b6, both give 0')
print(PARTIAL_CONFORMER_ID,
      partial_conformer.properties.enthalpy_of_formation_298k_atomic_b6.value)
print(MINIMAL_CONFORMER_ID,
      minimal_conformer.properties.enthalpy_of_formation_298k_atomic_b6.value)

print()
print('These are cases of missing values.')
print(
    'If you request a value which is actually missing, you will silently get a default value (0.0 for floats)'
)
print('You can check whether a Conformer has a value with the HasField method')
print('Calling HasField for optimized_geometry_energy:')
print(PARTIAL_CONFORMER_ID,
      partial_conformer.properties.HasField('optimized_geometry_energy'))
print(MINIMAL_CONFORMER_ID,
      minimal_conformer.properties.HasField('optimized_geometry_energy'))
print('Calling HasField for zpe_unscaled:')
print(PARTIAL_CONFORMER_ID,
      partial_conformer.properties.HasField('zpe_unscaled'))
print(MINIMAL_CONFORMER_ID,
      minimal_conformer.properties.HasField('zpe_unscaled'))
print('Calling HasField for enthalpy_of_formation_298k_atomic_b6:')
print(
    PARTIAL_CONFORMER_ID,
    partial_conformer.properties.HasField(
        'enthalpy_of_formation_298k_atomic_b6'))
print(
    MINIMAL_CONFORMER_ID,
    minimal_conformer.properties.HasField(
        'enthalpy_of_formation_298k_atomic_b6'))

print()
print('The one field that is different is normal_modes')
print(
    'Since normal_modes is a list of composite values, missing just means the list is length 0'
)
print('You cannot call HasField on normal_modes')
print('The length of normal_modes in our two conformers are:')
print(PARTIAL_CONFORMER_ID, len(partial_conformer.properties.normal_modes))
print(MINIMAL_CONFORMER_ID, len(minimal_conformer.properties.normal_modes))

print()
print('The properties.errors.status variable can shed some light on why',
      'fields are missing')
print('However, the exact rules of what fields are missing when are complex')
print(
    'Therefore, whenever accessing properties fields in the complete database,',
    'you should check HasField first')
