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

"""Processes a dataset of annotations to train a multi-annotator model."""

import math
import os
import tempfile

from absl import logging
import classifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.compat.v1 as tf
import transformers
import utils


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")
tf.compat.v1.disable_eager_execution()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

logging.set_verbosity(logging.ERROR)
logging.warning("should not print")

tf.get_logger().setLevel("ERROR")


class MultiAnnotator():
  """Creates a Classifier instance for training single and multi-task models."""

  def __init__(self, data, annotators,
               params):
    """Instantiates the MultiAnnotator model for training classifiers.

    Args:
      data: a data file which includes a text column and a column for each
        annotator. Annotator columns can include 0, 1, or ""
      annotators: a list of annotators which is a subset of the data file's
        columns
      params: a Params instance which includes the hyperparameters of the model
    """
    self.params = params
    self.data = data

    self.weights = list()
    self.cache_dir = tempfile.gettempdir()
    self.tokenizer = transformers.BertTokenizer.from_pretrained(
        os.path.join(self.params.bert_path, "vocab.txt"),
        cache_dir=self.cache_dir)

    # All the processing and modeling will
    # be applied to the set of top annotators
    self.annotators = self.get_top_annotators(annotators)

    # Calculating the majority vote of the (top) annotators
    self.data["majority"] = utils.get_majority_vote(self.data, self.annotators)

    # Creating the uncertainty column based on the variance of annotations
    # self.data["uncertainty"] = uncertainty(self.data, self.annotators)

    # The model type (single, or multi_task) is set
    self.set_model_type()
    # The classification tasks are set based on the model type
    self.set_task_labels()
    # instantiating a classifier based on the task
    self.classifier = classifier.Classifier(self.params, self.task_labels)

  def get_top_annotators(self, annotators):
    """Finds the top N annotators with highest number of annotations.

    Args:
      annotators: the list of all annotators.

    Returns:
      the list of top N annotators
    """
    # if only a subset of annotators with the highest number of annotations
    # are to be considered in the modeling
    return [
        anno for anno, count in self.data[annotators].count(axis=0).sort_values(
            ascending=False).items()
    ][:min(self.params.top_n_annotators, len(annotators))]

  def set_model_type(self):
    """The type of the model to be trained is set based on self.params."""
    self.multi_task = self.single = False
    setattr(self, self.params.model, True)

  def set_task_labels(self):
    """Sets the task_labels as the list of classification tasks."""
    if self.single:
      # if the model is a single task, the task is to predict the majority
      self.task_labels = ["majority"]
    else:
      # if the model is multi_task, each task is to predict an annotator's
      # predictions
      self.task_labels = self.annotators

  def run_cv(
      self, data, loss_function
  ):
    """Runs a cross validation on the input data.

    Args:
      data: the dataset to run the CV on. Should include majority columns along
        with annotator columns
      loss_function: the loss function to be used in training

    Returns:
      the validation scores and the predicted labels for the whole dataset
      the predictions for the validation set
      the predictions for the test set
    """
    test_kfold = StratifiedKFold(
        n_splits=self.params.n_folds,
        shuffle=True,
        random_state=self.params.random_state)
    val_kfold = StratifiedKFold(
        n_splits=int(math.floor(1 / self.params.val_ratio)),
        shuffle=True,
        random_state=self.params.random_state)

    results = pd.DataFrame()
    i = 1
    for train_idx, test_idx in test_kfold.split(
        np.zeros(data.shape[0]), utils.encode_annotators(data,
                                                         self.annotators)):

      # Splitting the data to train, test and validation
      train = data.loc[train_idx].reset_index()
      test = data.loc[test_idx].reset_index()
      train_idx, val_idx = next(
          iter(
              val_kfold.split(
                  np.zeros(train.shape[0]),
                  utils.encode_annotators(train, self.annotators))))

      val = train.loc[val_idx].reset_index()
      train = train.loc[train_idx].reset_index()

      train_batches, val_batches, test_batches = self.get_subset_batches(
          train, val, test)

      self.create_loss_weights(train)
      _, val_result, fold_result = self.run_model(train_batches, val_batches,
                                                  test_batches, loss_function)
      fold_result["fold"] = pd.Series([i for id in test_idx])

      if self.params.mc_dropout and self.single:
        # if the monte carlo dropout is to be calculated
        # the trained model will be used for predicting uncertainty

        certainty_results = self.classifier.mc_predict(test_batches)
        fold_result = fold_result.join(certainty_results)
      else:
        # Reporting the performance of the model on each fold
        logging.info("Test:")
        logging.info(utils.report_results(fold_result, self.task_labels))

      i += 1
      # the results includes the predictions for each instance, because
      # each instance is used exactly once for testing the model
      results = results.append(fold_result)

    # the final score is calculated over the predictions
    # for all instances over all folds
    scores = utils.report_results(results, self.task_labels)

    return scores, val_result, results

  def train_val_test(self, train_subset, val_subset,
                     test_subset, loss_function):
    """Trains, evaluates and tests a model.

    Args:
      train_subset: the indices of self.data instances that are to be used as
        the train set
      val_subset: the indices of self.data instances that are to be used as the
        val set
      test_subset: the indices of self.data instances that are to be used as the
        test set
      loss_function: the loss function to be used in training

    Returns:
      the scores of testing the model with predictions
    """
    train = self.data.loc[self.data["id"].isin(
        train_subset["id"].tolist())].reset_index()

    val = self.data.loc[self.data["id"].isin(
        val_subset["id"].tolist())].reset_index()

    test = self.data.loc[self.data["id"].isin(
        test_subset["id"].tolist())].reset_index()

    train_batches, val_batches, test_batches = self.get_subset_batches(
        train, val, test)

    self.create_loss_weights(train)
    epochs, val_result, test_result = self.run_model(train_batches, val_batches,
                                                     test_batches,
                                                     loss_function)

    logging.info("Test:")
    scores = utils.report_results(test_result, task_labels=self.task_labels)
    scores["epochs"] = epochs
    logging.info(scores)

    if self.params.mc_dropout and self.single:
      certainty_results = self.classifier.mc_predict(test_batches)
      test_result = test_result.join(certainty_results)

    return scores, val_result, test_result

  def get_subset_batches(
      self, train, val, test
  ):
    """Prepared the train, validation and test dataframes for modeling.

    Args:
      train: the train dataframe
      val: the validation dataframe
      test: the test dataframe

    Returns:
      three dictionaries representing tokenized inputs and labels
    """
    train_batches = utils.prepare_data(train, self.task_labels, self.tokenizer,
                                       self.params.max_l)
    val_batches = utils.prepare_data(val, self.task_labels, self.tokenizer,
                                     self.params.max_l)
    test_batches = utils.prepare_data(test, self.task_labels, self.tokenizer,
                                      self.params.max_l)
    return train_batches, val_batches, test_batches

  def run_model(
      self, train_batches, val_batches,
      test_batches,
      loss_function):
    """Trains a model and returns the predictions on val, and test data.

    Args:
      train_batches: the subset of self.data that is used for training the model
      val_batches: the subset of self.data that is used for evaluating the model
      test_batches: the subset of self.data that is used for testing the model
      loss_function: a function for calculating the loss value during training

    Returns:
      epochs: the number of training epoch before early stopping happens
      val_result: a dataset of predicted labels and actual labels for each
        instance in the validation dataset and each task.
      test_result: a dataset of predicted labels and actual labels for each
        instance in the test dataset and each task.
    """
    self.new_model()

    epochs = self.classifier.train_model(train_batches, val_batches,
                                         loss_function, self.weights)
    val_result = self.classifier.predict(val_batches)
    test_result = self.classifier.predict(test_batches)

    return epochs, val_result, test_result

  def new_model(self):
    """Creates and returns classifier model."""
    self.classifier.create()

  def create_loss_weights(self, data):
    """Creates the weights for each task_label based on the sparsity of data.

    Args:
      data: the dataset, based on which the weights are to be calculated
    """
    self.weights = dict()

    for task_label in self.task_labels:
      if self.params.use_majority_weight:
        # all classifier heads are using the label weight of the
        # majority label
        labels = data["majority"].dropna().values.astype(int)
      else:
        # each label weight is calculated separately
        labels = data[task_label].dropna().values.astype(int)
      weight = compute_class_weight(
          class_weight="balanced", classes=np.unique(labels), y=labels)
      if len(weight) == 1:
        # if a label does not appear in the data
        weight = [0.5, 0.5]

      self.weights[task_label] = weight
