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

"""Iteratively prune large KG by removing low-degree nodes."""
from __future__ import print_function
from collections import Counter  # pylint: disable=g-importing-member
import csv
import sys


def open_prune(tmp_cat_blacklist, tmp_ent_blacklist, data_lines):
  """Count head and tail entities that are not in a tmp blacklist."""
  new_lines = []
  cat_freq = Counter()
  ent_freq = Counter()
  for data_line in data_lines:
    if (data_line[2] not in tmp_cat_blacklist and
        data_line[1] not in tmp_ent_blacklist):
      ent_freq[data_line[1]] -= 1
      cat_freq[data_line[2]] -= 1
      new_lines.append(data_line)
  return (new_lines, (cat_freq, ent_freq))

PATH = sys.argv[1]  # e.g. ~/property_linking/

cat_blacklist = {}
ent_blacklist = {}

cat_rels = open(PATH + "entity_0_kb.tsv", "r")
cat_lines = list(csv.reader(cat_rels, delimiter="\t"))

for i in range(10):
  print (len(cat_lines))
  cat_lines, (cat_dict, ent_dict) = open_prune(cat_blacklist,
                                               ent_blacklist,
                                               cat_lines)
  # prune 10% of each list
  total_cats = len(cat_dict)
  total_ents = len(ent_dict)
  print ("iter: {}, cats: {}, ents: {}".format(i, total_cats, total_ents))
  cat_blacklist.update(dict(cat_dict.most_common(int(0.1 * total_cats))))
  ent_blacklist.update(dict(ent_dict.most_common(int(0.2 * total_ents))))
  print ("cat sample: {}".format(
      cat_dict.most_common()[::int(0.1 * total_cats)]))
  print ("ent sample: {}".format(
      ent_dict.most_common()[::int(0.1 * total_ents)]))
  print ("cat_blacklist: {}, ent_blacklist:{}".format(len(cat_blacklist),
                                                      len(ent_blacklist)))

# revise lists
old_kb = open(PATH + "entity_kb.tsv", "r")
old_cats = open(PATH + "entity_cats.tsv", "r")
old_names = open(PATH + "entity_names.tsv", "r")

new_kb = open(PATH + "entity_7_kb.tsv", "w+")
new_cats = open(PATH + "entity_7_cats.tsv", "w+")
new_names = open(PATH + "entity_7_names.tsv", "w+")

csv.field_size_limit(sys.maxsize)
kb_lines = csv.reader(old_kb, delimiter="\t")
kb_writer = csv.writer(new_kb, delimiter="\t")

whitelist = set()
for line in kb_lines:
  if (line[2] not in cat_blacklist and line[1] not in ent_blacklist):
    kb_writer.writerow(line)
    whitelist.add(line[2])
    whitelist.add(line[1])
    whitelist.add(line[0])

cats_lines = csv.reader(old_cats, delimiter="\t")
cats_writer = csv.writer(new_cats, delimiter="\t")


def update(data_line, bl):
  data_line[2] = "|".join([ent for ent in data_line[2].split("|")
                           if ent not in bl])
  if not data_line[2]:
    return None
  return data_line

for line in cats_lines:
  if line[0] not in cat_blacklist:
    outline = update(line, ent_blacklist)
    if outline is not None:
      cats_writer.writerow(outline)
      whitelist.add(line[0])

names_lines = csv.reader(old_names, delimiter="\t")
names_writer = csv.writer(new_names, delimiter="\t")
for line in names_lines:
  if line[1] in whitelist:
    names_writer.writerow(line)
