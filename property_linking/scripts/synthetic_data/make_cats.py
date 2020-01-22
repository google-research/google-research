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

"""Sample and make templatically generated categories given a pokemon KB."""
from __future__ import print_function
from collections import defaultdict  # pylint: disable=g-importing-member
import csv
import numpy as np

np.random.seed(0)


def make_value_dict(name_file):
  """Make a dict from value to name."""
  value_dict = defaultdict(list)
  with open(name_file, "r") as c:
    file_reader = csv.reader(c, delimiter="\t")
    for row in file_reader:
      value_dict[row[1][0]].append((row[1], row[2]))
  return value_dict


def format_to_category(query):
  """Use template to generate category based on properties."""
  entity, attributes, values = query
  prefix = []
  midfix = []
  suffix = []
  double_move = (len(attributes) == 2 and
                 attributes[0] == attributes[1] and
                 attributes[0] == "r.lea")
  double_type = (len(attributes) == 2 and
                 attributes[0] == attributes[1] and
                 attributes[0] == "r.typ")
  double_pokemon = (len(attributes) == 2 and
                    attributes[0] == attributes[1] and
                    attributes[0] == "r.lea.inv")
  for i, attribute in enumerate(attributes):
    value = values[i][1]
    if attribute == "r.typ":
      if double_type and i > 0:
        prefix.append("and")
      prefix.append(value)
    elif attribute == "r.abi":
      suffix.append("with {}".format(value))
    elif attribute == "r.lea.inv":
      if double_pokemon and i > 0:
        suffix.append("and")
        suffix.append(value)
      else:
        suffix.append("learned by {}".format(value))
    elif attribute == "r.gen":
      if entity == "Pokemon":
        suffix.append("in {}".format(value))
      else:
        midfix.append("{}".format(value))
    elif attribute == "r.lea":
      if double_move and i > 0:
        suffix.append("and")
        suffix.append(value)
      else:
        suffix.append("that learn {}".format(value))
    else:
      raise ValueError(attribute)

  return " ".join(prefix + midfix + [entity] + suffix)


def compute_query(query, kb_dict):
  """Compute the entities satisfying the properties."""
  entity, _, values = query
  full_set = defaultdict(set)
  if entity == "Pokemon":
    for row in kb_dict["p"]:
      full_set[row[1]].add(row[0])
  elif entity == "Moves":
    for row in kb_dict["m"]:
      full_set[row[1]].add(row[0])
  else:
    raise ValueError(entity)
  original = set().union(*full_set.values())
  for value in values:
    original &= full_set[value[0]]
  return "|".join(list(original))

targets = ["Pokemon", "Moves"]

trait_choices = {
    "Pokemon": ["r.lea", "r.lea", "r.abi", "r.typ", "r.typ", "r.gen", ""],
    "Moves": ["r.lea.inv", "r.lea.inv", "r.typ", "r.gen", ""]
}

remap_trait = {
    "r.lea": "m",
    "r.lea.inv": "p",
    "r.abi": "a",
    "r.typ": "t",
    "r.gen": "g",
}

all_values = make_value_dict("pokemon_names.tsv")
kb = make_value_dict("pokemon_kb.tsv")

categories = []

total_sampled = 500000
sampled_values = set()

# could sample the bits in batch to be a bit faster, but this is okay
sample = 0
for sample in range(total_sampled):
  if len(categories) >= 31000:
    break
  if sample % 100 == 0:
    print (sample, len(categories))
  sampled_entity = np.random.choice(targets, 1, [0.5, 0.5])[0]
  sampled_attributes = list(np.random.choice(trait_choices[sampled_entity],
                                             2,
                                             replace=False))
  sampled_attributes = [attr for attr in sampled_attributes if attr]
  final_values = [all_values[remap_trait[attr]][np.random.choice(
      len(all_values[remap_trait[attr]]), 1)[0]]
                  for attr in sampled_attributes if attr]
  hashable_tuple = (sampled_entity,
                    tuple(sampled_attributes),
                    tuple(final_values))
  if hashable_tuple in sampled_values:
    continue
  sampled_values.add(hashable_tuple)
  query_tuple = (sampled_entity, sampled_attributes, final_values)
  query_result = compute_query(query_tuple, kb)
  if not query_result:
    continue
  categories.append([str(sample),
                     format_to_category(query_tuple),
                     query_result,
                     "|".join([",".join([x[0], x[1][0], x[1][1]])
                               for x in zip(query_tuple[1], query_tuple[2])])])
print ("done sampling")

print ("{}/[{}|{}] kept".format(len(categories), sample, total_sampled))
with open("pokemon_cats_unshuffled.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for cat_row in categories:
    file_writer.writerow(cat_row)

np.random.shuffle(categories)
final_categories = categories
with open("pokemon_cats.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for cat_row in final_categories[:10000]:
    file_writer.writerow(cat_row)
  for cat_row in final_categories[11000:]:
    file_writer.writerow(cat_row)

with open("pokemon_cats_dev.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for cat_row in final_categories[10000:10500]:
    file_writer.writerow(cat_row)

with open("pokemon_cats_test.tsv", "w+") as csvout:
  file_writer = csv.writer(csvout, delimiter="\t")
  for cat_row in final_categories[10500:11000]:
    file_writer.writerow(cat_row)
