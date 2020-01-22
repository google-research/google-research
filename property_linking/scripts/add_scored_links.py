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

"""[WIP] Merges output of soft linker with existing categories (example) files."""

from collections import defaultdict  # pylint: disable=g-importing-member
import csv
import sys

csv.field_size_limit(sys.maxsize)


soft_match = csv.reader(open(sys.argv[1], "r"), delimiter="\t")
cat_dict = defaultdict(list)
eng_dict = defaultdict(list)
for line in soft_match:
  if float(line[0]) > 0.6 and len(cat_dict[line[4]]) < 5:
    cat_dict[line[4]].append((line[0], line[3]))
    eng_dict[line[2]].append((line[0], line[1]))

print (len(eng_dict), eng_dict.items()[:20])

cats_file = csv.reader(open(sys.argv[2], "r"), delimiter="\t")
out_file = csv.writer(open(sys.argv[3], "w+"), delimiter="\t")
for i, line in enumerate(cats_file):
  line.append("|".join([pair[1] for pair in cat_dict[line[0]]]))
  out_file.writerow(line)
