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

"""Replace target script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "data/train.tsv", "Input tsv file.")

flags.DEFINE_string(
    "mapping_dict", None,
    "File containing a mapping dictionary from new targets to old targets.")

flags.DEFINE_string("target_file", "data/targets.txt",
                    "File containing list of old targets.")

flags.DEFINE_string("output_target_file", "data/new_targets.txt",
                    "Output file for list of new targets.")

flags.DEFINE_string("output_data", "data/new_train.tsv",
                    "Output file new data.")


def replace_labels(labels, idx2target, mapping_dict, target2idx):
  """Replace old targets with new targets.

  Args:
      labels: comma-separated list of ids (for old targets)
      idx2target: dictionary, mapping old target ids to old target names
      mapping_dict: dictionary, with new target (str) : new targets (list of
        strings) key: value pairs
      target2idx: dictionary, mapping new target names to new target ids

  Returns:
      comma-separated list of ids for new targets
  """
  split = labels.split(",")
  new_labels = []
  for label_idx in split:
    old_target = idx2target[int(label_idx)]
    found = False
    for new_target, v in mapping_dict.items():
      if old_target in v:
        new_labels.append(str(target2idx[new_target]))
        found = True
        break
    if not found:
      new_labels.append(str(target2idx[old_target]))
  assert new_labels
  return ",".join(new_labels)


def main(_):

  data = pd.read_csv(
      FLAGS.input, sep="\t", header=None, names=["text", "labels", "id"])
  targets = open(FLAGS.target_file).read().splitlines() + ["neutral"]
  idx2target = {i: t for i, t in enumerate(targets)}

  with open(FLAGS.mapping_dict) as f:
    mapping_dict = json.loads(f.read())

  new_targets = list(mapping_dict.keys())

  # Find those targets that are not in the mapping dictionary
  not_found = []
  for t in targets:
    found = False
    for _, v in mapping_dict.items():
      if t in v:
        found = True
        break
    if not found:
      print("%s is not found" % t)
      not_found.append(t)

  print("New targets:")
  new_targets = sorted(new_targets + not_found)
  print(new_targets)
  target2idx = {t: i for i, t in enumerate(new_targets)}

  data["labels"] = data["labels"].apply(
      replace_labels, args=(idx2target, mapping_dict, target2idx))

  with open(FLAGS.output_target_file, "w") as f:
    f.write("\n".join(new_targets))

  data.to_csv(
      FLAGS.output_data, sep="\t", encoding="utf-8", header=False, index=False)


if __name__ == "__main__":
  app.run(main)
