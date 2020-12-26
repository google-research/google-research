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
"""Script for getting the top words associated with each emotion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter  # pylint: disable=g-importing-member
from collections import defaultdict  # pylint: disable=g-importing-member
import math
import operator
import os
import re
import string

from absl import app
from absl import flags
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string("data", "data/full_dataset",
                    "Directory containing full dataset.")

flags.DEFINE_string("output", "tables/emotion_words.csv",
                    "Output csv file for the emotion words.")

flags.DEFINE_string("emotion_file", "data/emotions.txt",
                    "File containing list of emotions.")

punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“"
}) - set(["#"]))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))


def CheckAgreement(ex, min_agreement, all_emotions, max_agreement=100):
  """Return the labels that at least min_agreement raters agree on."""
  sum_ratings = ex[all_emotions].sum(axis=0)
  agreement = ((sum_ratings >= min_agreement) & (sum_ratings <= max_agreement))
  return ",".join(sum_ratings.index[agreement].tolist())


def CleanText(text):
  """Clean text."""
  if isinstance(text, float):
    return []
  # lower case
  text = text.lower()
  # eliminate urls
  text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)
  # eliminate @mentions
  text = re.sub(r"\s@\S+", " ", text)
  # substitute all other punctuation with whitespace
  text = replace.sub(" ", text)
  # replace all whitespace with a single space
  text = re.sub(r"\s+", " ", text)
  # strip off spaces on either end
  text = text.strip()
  words = text.split()
  return [w for w in words if len(w) > 2]


def LogOdds(counts1, counts2, prior, zscore=True):
  """Calculates log odds ratio.

  Source: Dan Jurafsky.

  Args:
    counts1: dict of word counts for group 1
    counts2: dict of word counts for group 2
    prior: dict of prior word counts
    zscore: whether to z-score the log odds ratio

  Returns:
    delta: dict of delta values for each word.
  """

  sigmasquared = defaultdict(float)
  sigma = defaultdict(float)
  delta = defaultdict(float)

  n1 = sum(counts1.values())
  n2 = sum(counts2.values())

  nprior = sum(prior.values())
  for word in prior.keys():
    if prior[word] == 0:
      delta[word] = 0
      continue
    l1 = float(counts1[word] + prior[word]) / ((n1 + nprior) -
                                               (counts1[word] + prior[word]))
    l2 = float(counts2[word] + prior[word]) / ((n2 + nprior) -
                                               (counts2[word] + prior[word]))
    sigmasquared[word] = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
        float(counts2[word]) + float(prior[word]))
    sigma[word] = math.sqrt(sigmasquared[word])
    delta[word] = (math.log(l1) - math.log(l2))
    if zscore:
      delta[word] /= sigma[word]
  return delta


def GetCounts(df):
  words = []
  for t in df["text"]:
    words.extend(t)
  return Counter(words)


def main(_):
  print("Loading data...")
  dfs = []
  for filename in os.listdir(FLAGS.data):
    if filename.endswith(".csv"):
      dfs.append(
          pd.read_csv(os.path.join(FLAGS.data, filename), encoding="utf-8"))
  data = pd.concat(dfs)
  print("%d Examples" % (len(set(data["id"]))))
  print("%d Annotations" % len(data))

  with open(FLAGS.emotion_file, "r") as f:
    all_emotions = f.read().splitlines()
  print("%d emotion Categories" % len(all_emotions))

  print("Processing data...")
  data["text"] = data["text"].apply(CleanText)
  agree_dict = data.groupby("id").apply(CheckAgreement, 2,
                                        all_emotions).to_dict()
  data["agreement"] = data["id"].map(agree_dict)

  data = data[~data["agreement"].isnull()]
  dicts = []
  for e in all_emotions:
    print(e)
    contains = data["agreement"].str.contains(e)
    emotion_words = GetCounts(data[contains])
    other_words = GetCounts(data[~contains])
    prior = Counter()
    prior.update(dict(emotion_words))
    prior.update(dict(other_words))
    emotion_words_total = sum(emotion_words.values())
    delta = LogOdds(emotion_words, other_words, prior, True)
    c = 0
    for k, v in sorted(delta.items(), key=operator.itemgetter(1), reverse=True):
      if v < 3:
        continue
      dicts.append({
          "emotion": e,
          "word": k,
          "odds": "%.2f" % v,
          "freq": "%.3f" % (emotion_words[k] / emotion_words_total)
      })
      c += 1
      if c < 11:
        print("%s (%.2f)" % (k, v))
    print("--------")
    
  if not os.path.isdir(FLAGS.output):
    os.makedirs(FLAGS.output)

  emotion_words_df = pd.DataFrame(dicts)
  emotion_words_df.to_csv(FLAGS.output, index=False, encoding="utf-8")


if __name__ == "__main__":
  app.run(main)
