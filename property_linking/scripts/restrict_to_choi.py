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

"""Restrict a larger kb/cats/names file to just those in yago."""

from __future__ import print_function


restrict_cats = open("../yago_choi/category")
restrict_ents = open("../yago_choi/entity")
restrict_rels = open("../yago_choi/yagoTypes.clean")

input_cats = open("entity_cats.tsv", "r")
input_names = open("entity_names.tsv", "r")
input_kb = open("entity_kb.tsv", "r")

# restrict names first
ents_list = list([x.strip() for x in restrict_ents])
print (ents_list[:5])
ents = set(ents_list)
output_lines = []
output_ents = []
restricted_ent_ids = []
all_touched_ids = []
print (len(ents))
input_names = list(input_names)
name_dict = {line.split("\t")[1].strip(): line.split("\t")[2].strip()
             for line in input_names}
for line in input_names:
  value = line.strip().split("\t")[2].strip()
  if value in ents:
    output_lines.append(line)
    output_ents.append(value)
    all_touched_ids.append(line.strip().split("\t")[1].strip())
    restricted_ent_ids.append(line.strip().split("\t")[1].strip())

output_ents = set(output_ents)
restricted_ent_ids = set(restricted_ent_ids)
print (len(output_lines))
print (len(output_ents))
print (len(restricted_ent_ids))

filtered_cats = []
cats_list = list([x.strip() for x in restrict_cats])
print (cats_list[:5])
cats = set(cats_list)

cat_output_lines = []
cat_output_ids = []
pairs = []
cat_output_names = []
for line in input_cats:
  values = line.strip().split("\t")
  cat_name = values[1]
  cat_ents = values[2].split("|")
  if cat_name in cats:
    new_ents = [v for v in cat_ents if v in restricted_ent_ids]
    if not new_ents:
      continue
    cat_output_lines.append("{}\t{}\t{}".format(
        values[0],
        values[1],
        "|".join(new_ents)
        ))
    cat_output_names.append(cat_name)
    all_touched_ids.append(values[0])
    for v in cat_ents:
      if v in restricted_ent_ids:
        pairs.append((cat_name, v))
    cat_output_ids.append(values[0])
cat_output_names = set(cat_output_names)
print (len(cat_output_ids))
print (len(cat_output_lines))
print (len(pairs))
print (pairs[:5])
print ("ents per cat: {}".format(float(len(pairs))/len(cat_output_ids)))
print ("average cat length: {}".format(sum([len(x.split())
                                            for x in list(cat_output_names)]) /
                                       float(len(cat_output_names))))

kb_output_lines = []
input_kb = list(input_kb)


def check_prop(p):
  p = name_dict[p]
  return (not ("ID" in p or
               p.endswith(" code") or
               p.lower().endswith(" url") or
               p.lower().endswith(" id") or
               "Commons" in p or
               p == "isa" or
               p == "lang" or
               p == "id" or
               "/w/" in p or
               p == "image" or
               p.endswith(" number")))

all_rels = set([line.split("\t")[0] for line in input_kb])
print (len(all_rels))
valid_rels = {p for p in all_rels if check_prop(p)}
print (len(valid_rels))
print (list(valid_rels)[:100])
for line in input_kb:
  values = line.strip().split("\t")
  if (values[1] in restricted_ent_ids and
      values[0] in valid_rels):
    kb_output_lines.append(line)
    all_touched_ids.append(values[2])
    all_touched_ids.append(values[0])

print (len(kb_output_lines))
print (kb_output_lines[:5])

all_touched_ids = set([x.strip() for x in all_touched_ids])
print ("i/Q53265" in all_touched_ids)
name_output_lines = []
for line in input_names:
  values = line.split("\t")[1]
  if values.strip() in all_touched_ids:
    name_output_lines.append(line.replace("Category:", ""))
print (name_output_lines[:10])

output_cats = open("yago_s_cats.tsv", "w+")
output_names = open("yago_s_names.tsv", "w+")
output_kb = open("yago_s_kb.tsv", "w+")
for line in name_output_lines:
  output_names.write(line)
for line in cat_output_lines:
  output_cats.write(line + "\n")
for line in kb_output_lines:
  output_kb.write(line)

print ("Done")
