# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Preprocesses exisiting datasets to be used with the multi_annotator model.
"""
import collections
import json
import os

from absl import app
from absl import flags
from datasets import load_dataset
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", "GabHateCorpus_annotations.tsv",
    "The path to the annotations file.")

flags.DEFINE_enum(
    "corpus", "GHC", ["emotions", "GHC", "mftc"],
    "the corpus name.")


def main(argv):
  if not os.path.exists(os.path.join(os.getcwd(), "data")):
    os.mkdir(os.path.join(os.getcwd(), "data"))

  del argv
  if FLAGS.corpus == "GHC":
    preprocess_ghc()
  elif FLAGS.corpus == "emotions":
    emotions = ["joy", "sadness", "anger", "disgust", "surprise", "fear"]
    preprocess_emotions(emotions)
    get_train_test_emotions(emotions)
  elif FLAGS.corpus == "mftc":
    preprocess_mftc()


def get_long_annotations(annotations):
  """reformats an annotation df to long annotation dataframe.

  Args:
    annotations: a pandas dataframe for annotations, each row represents one
    annotations 'label' from an annotator 'annotator_id' for a textual item
    'text' and the 'text_id'
  Returns:
    a long pandas dataframe with each row representing an item 'text' and
    annotations from all annotators
  """
  new_data = collections.defaultdict(list)
  annotators = set(annotations["annotator_id"].tolist())

  for _, group in annotations.groupby("text_id"):
    new_data["text"].append(group["text"].tolist()[0])
    for _, row in group.iterrows():
      new_data[row["annotator_id"]].append(int(row["label"]))

    for anno in annotators:
      if anno not in group["annotator_id"].tolist():
        new_data[anno].append("")

  return pd.DataFrame.from_dict(new_data)


def preprocess_ghc():
  """Preprocesses the GHC dataset.
  """
  ## Renaming GHC columns to match with other files
  annotations = pd.read_csv(FLAGS.data_path, delimiter="\t")
  annotations = annotations[["ID", "Annotator", "Text", "Hate"]]

  annotations = annotations.rename(columns={
      "ID": "text_id",
      "Annotator": "annotator_id",
      "Text": "text",
      "Hate": "label"
  })
  try:
    os.mkdir(os.path.join(os.getcwd(), "data", "GHC"))
  except FileExistsError:
    pass

  long_annotations = get_long_annotations(annotations)
  long_annotations.to_csv(
      os.path.join(os.getcwd(), "data", "GHC", "hate_multi.csv"), index=False)


def preprocess_emotions(emotions):
  """Download the GoEmotions dataset and preprocesses.

  Args:
    emotions: list of the 6 emotions.
  """
  emotions_data = load_dataset("go_emotions", "raw")
  df = pd.DataFrame(emotions_data["train"])

  new_data = {emo: collections.defaultdict(list) for emo in emotions}
  annotators = set(df["rater_id"].tolist())

  for name, group in df.groupby("id"):
    for emo in emotions:
      new_data[emo]["id"].append(name)
      new_data[emo]["text"].append(group["text"].tolist()[0])
      for _, row in group.iterrows():
        new_data[emo][row["rater_id"]].append(row[emo])

      for anno in annotators:
        if anno not in group["rater_id"].tolist():
          new_data[emo][anno].append(np.nan)
  try:
    os.mkdir(os.path.join(os.getcwd(), "data", "emotions"))
  except FileExistsError:
    pass

  for emo in emotions:
    pd.DataFrame.from_dict(new_data[emo]).to_csv(
        os.path.join(os.getcwd(), "data",
                     "emotions", emo + "_multi.csv"), index=False)


def get_train_test_emotions(emotions):
  """Preprocesses the train, val, and test portions of GoEmotions dataset.

  Args:
    emotions: list of the 6 emotions.
  """
  all_emotions = [
      "admiration", "amusement", "anger", "annoyance", "approval", "caring",
      "confusion", "curiosity", "desire", "disappointment", "disapproval",
      "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
      "joy", "love", "nervousness", "optimism", "pride", "realization",
      "relief", "remorse", "sadness", "surprise", "neutral"
  ]
  emotions_data_simple = load_dataset("go_emotions", "simplified")
  train_df = pd.DataFrame(emotions_data_simple["train"])
  val_df = pd.DataFrame(emotions_data_simple["validation"])
  test_df = pd.DataFrame(emotions_data_simple["test"])

  for x_df in [train_df, test_df, val_df]:
    new_cols = collections.defaultdict(list)
    for _, row in x_df.iterrows():
      mentioned = [all_emotions[ment] for ment in row["labels"]]
      for emo in emotions:
        new_cols[emo].append(1 if emo in mentioned else 0)

    for col, values in new_cols.items():
      x_df[col] = pd.Series(values)

  train_df.to_csv(os.path.join(
      os.getcwd(), "data", "emotions", "train_emotions.csv"), index=False)
  val_df.to_csv(os.path.join(
      os.getcwd(), "data", "emotions", "val_emotions.csv"), index=False)
  test_df.to_csv(os.path.join(
      os.getcwd(), "data", "emotions", "test_emotions.csv"), index=False)


def preprocess_mftc():
  """Preprocesses the MFTC and prepares it for model training.
  """
  annotations_file = json.load(open(FLAGS.data_path, "r"))
  moral_labels = ["care", "fairness", "authority", "purity", "loyalty"]
  annotations_dict = {
      label: collections.defaultdict(list) for label in moral_labels
  }

  for corpus in annotations_file:
    for tweet in corpus["Tweets"]:
      for annotation in tweet["annotations"]:
        for m_label in moral_labels:
          annotations_dict[m_label]["text"].append(tweet["tweet_text"])
          annotations_dict[m_label]["text_id"].append(tweet["tweet_id"])
          annotations_dict[m_label]["annotator_id"].append(
              annotation["annotator"])
          annotations_dict[m_label]["label"].append(
              1 if m_label in annotation["annotation"].split(",") else 0)

  annotations_df = {m_label: pd.DataFrame.from_dict(
      annotations_dict[m_label]) for m_label in moral_labels}

  annotators = set(annotations_df["care"]["annotator_id"])

  for label, annotations in annotations_df.items():
    annotations = annotations.replace(
        {"annotator_id": {an: i for i, an in enumerate(annotators)}})
    annotations = annotations.drop_duplicates(
        subset=["text_id", "annotator_id"])
    long_annotations = get_long_annotations(annotations)
    long_annotations.to_csv(os.path.join(
        os.getcwd(), "data", "mftc", label + "_multi.csv"), index=False)


if __name__ == "__main__":
  app.run(main)
