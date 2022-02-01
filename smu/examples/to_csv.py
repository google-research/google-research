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

"""Generates a CSV file with a few fields."""
import csv
import sys

from smu import smu_sqlite

db = smu_sqlite.SMUSQLite('20220128_standard.sqlite')

print('# This is example csv output from smu/examples/to_csv.py')
print('# Please see the comments in that file for explanation')

# For this example, we will write to standard out.
# For a real application, you would want to use
# open('path_to_my_file.csv', 'w')
# instead of sys.stdout
writer = csv.writer(sys.stdout)

# This is where you write the header of the csv with whatever columns names
# you like.
writer.writerow(
    ['conformer_id', 'energy', 'homo', 'lumo', 'first important frequency'])

count = 0
# This iteration will go through all conformers in the database.
for conformer in db:

  # This is kind of a silly filter, but this shows how to filter
  # for some conformers and not just the first couple.
  if conformer.optimized_geometry.atom_positions[0].x > -3:
    continue

  # This is where you would choose the fields to print.
  # See field_access.py for more examples of accessing fields.
  writer.writerow([
      conformer.conformer_id,
      conformer.properties.single_point_energy_atomic_b5.value,
      conformer.properties.homo_pbe0_6_311gd.value,
      conformer.properties.lumo_pbe0_6_311gd.value,
      conformer.properties.harmonic_frequencies.value[6],
  ])

  # This breaks out of the loop after a couple of records just so this
  # examples runs quickly. If you want process the whole dataset,
  # remove this.
  count += 1
  if count == 5:
    break
