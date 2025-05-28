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

"""Includes the main code for running the single model experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import re

from absl import app
from absl import flags
import multi_model
import pandas as pd
import pytz
import utils


FLAGS = flags.FLAGS
flags.DEFINE_enum("model", "single", ["multi_task", "single"],
                  "the model name, can be single, or multi_task")

flags.DEFINE_string(
    "source_dir", os.getcwd(),
    "The directory with a 'data' and 'result' folder for the task understudy")

flags.DEFINE_enum(
    "corpus", "emotions", ["emotions", "GHC", "mftc"],
    "the corpus name, also the name of the folder under "
    "source_dir that includes the data files. Select 'emotions' or 'GHC'.")

flags.DEFINE_string(
    "label", "joy",
    "a label/task in the corpus that the model is learning to predict,"
    "for GHC, the label should be hate, for emotions, label can be "
    "anger, disgust, fear, joy, sadness, or surprise.")

flags.DEFINE_integer("batch_size", 16, "the batch size for training")

flags.DEFINE_float("lr", 5e-5, "Learning rate for training the model.")

flags.DEFINE_integer(
    "max_l", 64, "maximum length of the sentences, "
    "the rest of the sentence would be discarded")

flags.DEFINE_integer("n_epoch", 10,
                     "maximum number of epochs during the early stopping")

flags.DEFINE_integer(
    "random_state", 9999,
    "random state for dividing the dataset to train, test and validation sets")

flags.DEFINE_integer("n_folds", 5, "number of folds in the cross validation")

flags.DEFINE_boolean(
    "mc_dropout", False, "whether predictions are generated with Monte Carlo"
    "Dropout approach")

flags.DEFINE_integer(
    "mc_passes", 10, "number of iterations for calculating "
    "the Monte Carlo dropout uncertainty for predictions")

flags.DEFINE_string("bert_path",
                    "bert-base-cased/",
                    "the folder where the model and vocab is located")

flags.DEFINE_integer(
    "top_n_annotators", 1000,
    "size of the subset of annotators with maximum number "
    "of annotations to be considered in the model training")

flags.DEFINE_float("min_epoch_change", 1e-4,
                   "difference of two epochs that causes the early stopping")

flags.DEFINE_boolean(
    "use_majority_weight", False,
    "whether or not each classifier head should use the"
    "weight of the majority label or its own weight")

flags.DEFINE_integer(
    "early_stopping_check", 2,
    "number of epochs for which the loss difference should "
    "be less that the min_epoch_change to enable early stopping")

flags.DEFINE_string(
    "annotator", "", "a single annotator to train a single "
    "model for training an testing")

flags.DEFINE_string(
    "iteration_label", "",
    "Name of the train/test or cross validation experiment"
    "it will be added to the result file")

flags.DEFINE_float(
    "val_ratio", 0.25, "percentage of the train data"
    "that is to be used as the validation set"
    "to enable early stopping during training")


def main(argv):
  del argv

  # Reading the annotation dataset which includes a column for each annotator
  # and a text column that include each instance. Each annotation is
  # 0, 1 or NaN
  df = pd.read_csv(
      os.path.join(FLAGS.source_dir, "data", FLAGS.corpus,
                   FLAGS.label + "_multi.csv"))

  # The annotator names are specified by a regular expression, the names
  # should be a subset of the dataset columns
  # in both GHC and emotions dataset, the annotator names are numeric values
  annotators = [
      col for col in df.columns if re.fullmatch(re.compile(r"^[0-9]+"), col)
  ]

  model = multi_model.MultiAnnotator(
      df, annotators=annotators, params=FLAGS)

  #############################################################################
  ############################## Experiments ##################################
  #############################################################################

  # For the emotions dataset, the experiment is performed on a train, test, and
  # validation subset of the dataset that has been published as a part of the
  # GoEmotions dataset
  if FLAGS.corpus == "emotions":
    df_subsets = dict()

    for sub in ["train", "val", "test"]:
      df_subsets[sub] = pd.read_csv(
          os.path.join(FLAGS.source_dir, "data", FLAGS.corpus,
                       sub + "_emotions.csv"))

    score, val_results, test_results = model.train_val_test(
        df_subsets["train"], df_subsets["val"], df_subsets["test"],
        utils.multi_task_loss)

  # experiments on GHC and mftc datasets are performed as cross validations
  # elif FLAGS.corpus == "GHC" or FLAGS.corpus == "mftc":
  else:
    score, val_results, test_results = model.run_cv(df, utils.multi_task_loss)

  #############################################################################
  ################################ Results ####################################
  #############################################################################

  # saving the results along with the hyper parameters of the model
  score["model"] = FLAGS.model

  # saving the time
  pacific = pytz.timezone("US/Pacific")
  sa_time = datetime.datetime.now(pacific)
  name_time = sa_time.strftime("%m%d%y-%H:%M")
  name_time = name_time + "_" + FLAGS.iteration_label
  score["time"] = name_time
  score["label"] = FLAGS.label

  # the results of each test is added as a row to the classification.csv file
  try:
    os.makedirs(
        os.path.join(FLAGS.source_dir, "results", FLAGS.corpus, FLAGS.label))
  except FileExistsError:
    pass

  score_dir = os.path.join(FLAGS.source_dir, "results", FLAGS.corpus,
                           FLAGS.label, "classification.csv")

  # the predictions on the test and validation datasets are saved
  val_results_dir = os.path.join(FLAGS.source_dir, "results", FLAGS.corpus,
                                 FLAGS.label,
                                 name_time + "_" + score["model"] + "_val.csv")
  test_results_dir = os.path.join(
      FLAGS.source_dir, "results", FLAGS.corpus, FLAGS.label,
      name_time + "_" + score["model"] + "_test.csv")

  score_df = pd.DataFrame.from_records([score])
  score_df.to_csv(
      score_dir, header=False, index=False, mode="a")

  pd.DataFrame.from_dict(val_results).to_csv(
      val_results_dir, index=False)
  pd.DataFrame.from_dict(test_results).to_csv(
      test_results_dir, index=False)


if __name__ == "__main__":
  app.run(main)
