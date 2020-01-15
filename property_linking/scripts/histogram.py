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

"""Generic stats aggregation for 3-column tsv file."""

import collections
import sys

name_file = open(sys.argv[2], "r")
name_dict = {}
for line in name_file:
  values = line.strip().split("\t")
  name_dict[values[1]] = values[2]

with open(sys.argv[1], "r") as kb_file:
  kb_lines = kb_file.readlines()
  all_rels = [line.split("\t")[0].strip() for line in kb_lines]
  all_items = [line.split("\t")[1].strip() for line in kb_lines]
  all_values = [line.split("\t")[2].strip() for line in kb_lines]
  ranked_rels = sorted(collections.Counter(all_rels).items(),
                       key=lambda x: -x[1])
  ranked_items = sorted(collections.Counter(all_items).items(),
                        key=lambda x: -x[1])
  ranked_values = sorted(collections.Counter(all_values).items(),
                         key=lambda x: -x[1])
  print (len(set(all_rels)), len(set(all_items)), len(set(all_values)))
  print ([(name_dict[r[0]], r[1]) for r in ranked_rels[:600]])
  print ([(name_dict[r[0]], r[1]) for r in ranked_items[:600]])
  print ([(name_dict[r[0]], r[1]) for r in ranked_values[:600]])
