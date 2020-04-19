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

"""Replace emotion labels (necessary when grouping emotions into higher-level categories)."""

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
    "File containing a mapping dictionary from new emotions to old emotions.")

flags.DEFINE_string("emotion_file", "data/emotions.txt",
                    "File containing list of old emotions.")

flags.DEFINE_string("output_emotion_file", "data/new_emotions.txt",
                    "Output file for list of new emotions.")

flags.DEFINE_string("output_data", "data/new_train.tsv",
                    "Output file new data.")


def replace_labels(labels, idx2emotion, mapping_dict, emotion2idx):
  """Replace old emotions with new emotions.

  Args:
      labels: comma-separated list of ids (for old emotions)
      idx2emotion: dictionary, mapping old emotion ids to old emotion names
      mapping_dict: dictionary, with new emotion (str) : new emotions (list of
        strings) key: value pairs
      emotion2idx: dictionary, mapping new emotion names to new emotion ids

  Returns:
      comma-separated list of ids for new emotions
  """
  split = labels.split(",")
  new_labels = []
  for label_idx in split:
    old_emotion = idx2emotion[int(label_idx)]
    found = False
    for new_emotion, v in mapping_dict.items():
      if old_emotion in v:
        new_labels.append(str(emotion2idx[new_emotion]))
        found = True
        break
    if not found:
      new_labels.append(str(emotion2idx[old_emotion]))
  assert new_labels
  return ",".join(new_labels)


def main(_):

  data = pd.read_csv(
      FLAGS.input, sep="\t", header=None, names=["text", "labels", "id"])
  emotions = open(FLAGS.emotion_file).read().splitlines() + ["neutral"]
  idx2emotion = {i: t for i, t in enumerate(emotions)}

  with open(FLAGS.mapping_dict) as f:
    mapping_dict = json.loads(f.read())

  new_emotions = list(mapping_dict.keys())

  # Find those emotions that are not in the mapping dictionary
  not_found = []
  for t in emotions:
    found = False
    for _, v in mapping_dict.items():
      if t in v:
        found = True
        break
    if not found:
      print("%s is not found" % t)
      not_found.append(t)

  print("New emotions:")
  new_emotions = sorted(new_emotions + not_found)
  print(new_emotions)
  emotion2idx = {t: i for i, t in enumerate(new_emotions)}

  data["labels"] = data["labels"].apply(
      replace_labels, args=(idx2emotion, mapping_dict, emotion2idx))

  with open(FLAGS.output_emotion_file, "w") as f:
    f.write("\n".join(new_emotions))

  data.to_csv(
      FLAGS.output_data, sep="\t", encoding="utf-8", header=False, index=False)


if __name__ == "__main__":
  app.run(main)
