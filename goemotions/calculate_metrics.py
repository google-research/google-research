# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Lint as: python3
"""Calculate evaluation metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


FLAGS = flags.FLAGS

flags.DEFINE_string("test_data", None, "Test tsv file with true labels.")

flags.DEFINE_string("predictions", None, "Predictions tsv file.")

flags.DEFINE_string("output", "results.json", "Output json file.")

flags.DEFINE_string("emotion_file", "data/emotions.txt",
                    "File containing list of emotions.")

flags.DEFINE_boolean("add_neutral", True, "Whether to add neutral as emotion.")

flags.DEFINE_float("threshold", 0.3, "Threshold for binarizing predictions.")


def main(_):
  preds = pd.read_csv(FLAGS.predictions, sep="\t")
  true = pd.read_csv(
      FLAGS.test_data, sep="\t", header=None, names=["text", "labels", "id"])
  emotions = open(FLAGS.emotion_file).read().splitlines()
  if FLAGS.add_neutral:
    emotions.append("neutral")
  num_emotions = len(emotions)

  idx2emotion = {i: e for i, e in enumerate(emotions)}

  preds_mat = np.zeros((len(preds), num_emotions))
  true_mat = np.zeros((len(preds), num_emotions))
  for i in range(len(preds)):
    true_labels = [int(idx) for idx in true.loc[i, "labels"].split(",")]
    for j in range(num_emotions):
      preds_mat[i, j] = preds.loc[i, idx2emotion[j]]
      true_mat[i, j] = 1 if j in true_labels else 0

  threshold = FLAGS.threshold
  pred_ind = preds_mat.copy()
  pred_ind[pred_ind > threshold] = 1
  pred_ind[pred_ind <= threshold] = 0
  results = {}
  results["accuracy"] = accuracy_score(true_mat, pred_ind)
  results["macro_precision"], results["macro_recall"], results[
      "macro_f1"], _ = precision_recall_fscore_support(
          true_mat, pred_ind, average="macro")
  results["micro_precision"], results["micro_recall"], results[
      "micro_f1"], _ = precision_recall_fscore_support(
          true_mat, pred_ind, average="micro")
  results["weighted_precision"], results["weighted_recall"], results[
      "weighted_f1"], _ = precision_recall_fscore_support(
          true_mat, pred_ind, average="weighted")
  for i in range(num_emotions):
    emotion = idx2emotion[i]
    emotion_true = true_mat[:, i]
    emotion_pred = pred_ind[:, i]
    results[emotion + "_accuracy"] = accuracy_score(emotion_true, emotion_pred)
    results[emotion + "_precision"], results[emotion + "_recall"], results[
        emotion + "_f1"], _ = precision_recall_fscore_support(
            emotion_true, emotion_pred, average="binary")

  with open(FLAGS.output, "w") as f:
    f.write(json.dumps(results))


if __name__ == "__main__":
  app.run(main)
