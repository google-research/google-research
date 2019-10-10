# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Make a names.tsv file based on given table files."""
import csv
import sys

pokemon_file = sys.argv[1]
moves_file = sys.argv[2]
types_file = sys.argv[3]
abilities_file = sys.argv[4]
gen_file = sys.argv[5]

names = []


def update(name_list, csv_file, name_col, test_func, prefix):
  """Update name_list with information from column if test_func holds."""
  with open(csv_file, "r") as c:
    file_reader = csv.reader(c)
    next(file_reader)
    for row in file_reader:
      if test_func(row):
        name_list.append(["name", prefix + row[0],
                          row[name_col].replace("-", " ")])

update(names, pokemon_file, 1, lambda x: int(x[0]) < 10000, "p")
# 9 is english, >10K are fake Pokemon
lambda_real_and_english = lambda x: (x[1] == "9" and int(x[0]) < 10000)
update(names, moves_file, 2, lambda_real_and_english, "m")
update(names, types_file, 2, lambda_real_and_english, "t")
update(names, abilities_file, 2, lambda_real_and_english, "a")
update(names, gen_file, 2, lambda x: x[1] == "9", "g")

# relations
names.append(["name", "r.lea", "learns"])
names.append(["name", "r.abi", "has ability"])
names.append(["name", "r.gen", "from generation"])
names.append(["name", "r.typ", "has type"])
names.append(["name", "r.isa", "is a"])
names.append(["name", "r.lea.inv", "learned by"])
names.append(["name", "r.abi.inv", "with ability"])
names.append(["name", "r.gen.inv", "generation"])
names.append(["name", "r.typ.inv", "with type"])
names.append(["name", "r.isa.inv", "is a"])
names.append(["name", "p", "Pokemon"])
names.append(["name", "m", "Move"])
names.append(["name", "a", "Ability"])
names.append(["name", "t", "Type"])
names.append(["name", "g", "Generation"])

with open("pokemon_names.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for name_row in names:
    file_writer.writerow(name_row)
