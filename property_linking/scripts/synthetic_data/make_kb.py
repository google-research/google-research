# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Make kb based on Pokemon tables."""
import csv

kb = []


def update(row_list, csv_file, name, rel_a, rel_b,
           test_func, prefix_a, prefix_b, fixed_b=None):
  """Update list of rows with newfound relations."""
  # prefix is ["p", "m", "t", "b", "a"]
  # inverse is always true
  with open(csv_file, "r") as c:
    file_reader = csv.reader(c)
    next(file_reader)
    for row in file_reader:
      b_value = row[rel_b] if fixed_b is None else fixed_b
      if test_func(row) and int(row[0]) < 10000:
        row_list.append(
            ["r." + name, prefix_a + row[rel_a], prefix_b + b_value])
        row_list.append(
            ["r." + name + ".inv", prefix_b + b_value, prefix_a + row[rel_a]])

# most recent version
# pokemon
update(kb, "data/pokemon_moves.csv", "lea", 0, 2,
       lambda x: x[1] == "18", "p", "m")
update(kb, "data/pokemon_abilities.csv", "abi", 0, 1,
       lambda x: True, "p", "a")
update(kb, "data/pokemon_form_generations.csv", "gen", 0, 1,
       lambda x: True, "p", "g")
update(kb, "data/pokemon_types.csv", "typ", 0, 1,
       lambda x: True, "p", "t")
update(kb, "data/pokemon_types.csv", "isa", 0, None,
       lambda x: True, "p", "p", fixed_b="")

# moves
update(kb, "data/moves.csv", "gen", 0, 2, lambda x: True, "m", "g")
update(kb, "data/moves.csv", "typ", 0, 3, lambda x: True, "m", "t")
update(kb, "data/moves.csv", "isa", 0, None,
       lambda x: True, "m", "m", fixed_b="")

# ability
update(kb, "data/pokemon_abilities.csv", "isa", 1, None,
       lambda x: True, "a", "a", fixed_b="")

# generation
update(kb, "data/pokemon_form_generations.csv", "isa", 1, None,
       lambda x: True, "g", "g", fixed_b="")

# type
update(kb, "data/pokemon_types.csv", "isa", 1, None,
       lambda x: True, "t", "t", fixed_b="")

deduped = set([tuple(kb_row) for kb_row in kb])
kb = sorted(list(deduped))  # dedupe?

with open("pokemon_kb.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for kb_row in kb:
    file_writer.writerow(kb_row)
