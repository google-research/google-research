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

"""Convert and export pokemon synthetic data to property linking directory."""
import csv
import sys


in_dir = sys.argv[1]
out_dir = sys.argv[2]
domain = sys.argv[3]

cat_file = in_dir + domain + "_cats.tsv"
cat_dev_file = in_dir + domain + "_cats_dev.tsv"
name_file = in_dir + domain + "_names.tsv"
kb_file = in_dir + domain + "_kb.tsv"

cats = []
cats_dev = []
names = []
kb = []

with open(name_file, "r") as n:
  file_reader = csv.reader(n, delimiter="\t")
  for line in file_reader:
    names.append([line[0], "i/"+ line[1], line[2]])

with open(cat_file, "r") as c:
  file_reader = csv.reader(c, delimiter="\t")
  for line in file_reader:
    cats.append(["i/" + line[0], line[1], line[2]])
    names.append(["name", "i/" + line[0], line[1]])

with open(cat_dev_file, "r") as c:
  file_reader = csv.reader(c, delimiter="\t")
  for line in file_reader:
    cats_dev.append(["i/" + line[0], line[1], line[2]])
    names.append(["name", "i/" + line[0], line[1]])

with open(kb_file, "r") as k:
  file_reader = csv.reader(k, delimiter="\t")
  for line in file_reader:
    kb.append(["i/" + line[0], "i/" + line[1], "i/" + line[2]])

out_cat_file = out_dir + domain + "_v2_cats.tsv"
out_cat_dev_file = out_dir + domain + "_v2_cats_dev.tsv"
out_name_file = out_dir + domain + "_v2_names.tsv"
out_kb_file = out_dir + domain + "_v2_kb.tsv"


def write_out(out_path, row_list):
  with open(out_path, "w+") as csvout:
    file_writer = csv.writer(csvout, delimiter="\t")
    for row in row_list:
      file_writer.writerow(row)

write_out(out_cat_file, cats)
write_out(out_cat_dev_file, cats_dev)
write_out(out_name_file, names)
write_out(out_kb_file, kb)
