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

"""Generic functions that are used for multi_annotator modeling."""
import collections

from absl import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import tensorflow.compat.v1 as tf


def get_majority_vote(data, columns):
  """Gets the binary row-wise majority vote from several columns.

  In the absence of a majority vote in binary annotations, majority
  is set to 1.

  Args:
    data: a pandas dataframe
    columns: a list of columns that exist in data, values should be binary or
      np.nan

  Returns:
    a pandas Series which includes the majority votes mapped to binary labels
    with the shape of [data.shape[0], 1]
  Raises:
    KeyError: if any of the columns specified in columns argument
    is missing from the data columns
  """
  if check_columns(data, columns):
    majority = (data[columns].sum(axis=1) / data[columns].count(axis=1) >=
                0.5).astype(int)
    return majority


def calculate_uncertainty(data, columns):
  """Gets the variance of several columns as their uncertainty.

  Args:
    data: a pandas dataframe.
    columns: a list of columns that exist in data, with 0, 1, or np.nan values.

  Returns:
    a pandas Series which includes the variance of binary labels
    with the shape of [data.shape[0], 1]
  Raises:
    KeyError: in any of the columns specified in columns argument
    is missing from the data columns
  """
  if check_columns(data, columns):
    uncertainty_col = data[columns].var(axis=1)
    return uncertainty_col


def encode_annotators(data, columns):
  """Creates an encoded string that shows non-empty columns for each row.

  Args:
    data: a pandas dataframe with 0, 1, or np.nan values.
    columns: a list of columns that exist in data

  Returns:
    a numpy array which maps each row of data to a label. Each label represents
    a pattern of 0 and 1s respectively representing the missing and non-missing
    column values for a row.
  Raises:
    KeyError: in any of the columns specified in columns argument
    is missing from the data columns
  """
  if check_columns(data, columns):
    # Mapping missing values to 0 and available data to 1
    data = data[columns].replace(0, 1).replace(np.nan, 0)
    new_labels = LabelEncoder().fit_transform(
        ["".join(str(label) for label in row) for _, row in data.iterrows()])
    return new_labels


def check_columns(data, columns):
  """Checks if all columns exist in the data and all values are binary.

  Args:
      data: a pandas dataframe.
      columns: a list of columns

  Returns:
    true, if all columns are in the dataset and all values are binary or nan
  Raises:
    KeyError: in any of the columns specified in columns argument
    is missing from the data columns
  """
  if not set(columns).issubset(set(data.columns.tolist())):
    missing_columns = [col for col in columns if col not in data]
    raise KeyError(
        "The following columns do not exist in the dataset: {}".format(
            missing_columns))
  elif ((data[columns].values == 0) | (data[columns].values == 1)
        | np.isnan(data[columns].values)).all():
    return True
  else:
    raise ValueError("The specified columns include non-binary "
                     "values and cannot be used by the method")


def prepare_data(data, task_labels, tokenizer, max_len):
  """Creates batches for model training and testing.

  Args:
    data: a dataset of text, and labels. The "majority" column must already been
      set.
    task_labels: column names that are going to be used as target labels
    tokenizer: a BertTokenizer instance.
    max_len: maximum length of the text after tokenization

  Returns:
    a dictionary:
    {
    "majority": includes tha majority label for each text.
    <task_label>: labels for each task if the model is multi_task.
    "text_id": the index of the text instance.
    "inputs": the tokenized and padded text.
    "attentions": a binary list that shows whether each token is a word or [PAD]
    }
  """

  data_info = tokenize_inputs(data["text"].tolist(), tokenizer, max_len)

  for col in ["majority", "text_id"]:
    if col in data:
      data_info[col] = data[col]

  for task_label in task_labels:
    if task_label in data:
      data_info[task_label] = data[task_label]

  return data_info


def tokenize_inputs(inputs, tokenizer, max_len):
  """Tokenizes and encodes the inputs to create a batch.

  Args:
    inputs: a list of strings
    tokenizer: a BertTokenizer instance.
    max_len: maximum length of the text after tokenization

  Returns:
    a dictionary that includes encoded data ("input") and attentions
  """
  tokens = [tokenizer.tokenize(b) for b in inputs]
  token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]

  batch_info = collections.defaultdict(list)

  for seq in token_ids:
    new_seq = tokenizer.convert_tokens_to_ids(["[CLS]"])
    new_seq.extend(seq if len(seq) < (max_len - 2) else seq[:(max_len) - 3])
    new_seq.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
    attn = [1 for tok in new_seq]
    pads = tokenizer.convert_tokens_to_ids(["[PAD]"] *
                                           max(max_len - len(new_seq), 0))
    attn.extend([0 for tok in pads])

    new_seq.extend(pads)

    batch_info["inputs"].append(np.array(new_seq))
    batch_info["attentions"].append(np.array(attn))

  batch_info["inputs"] = np.array(batch_info["inputs"])
  batch_info["attentions"] = np.array(batch_info["attentions"])

  return batch_info


def report_results(results, task_labels, continuous=False):
  """Calculates the precision, recall and f1 for predicting majority labels.

  Args:
    results: a pandas dataframe that includes <task>_pred and <task>_label
      columns with regard to each target label.
    task_labels: the list of target labels, the majority voting of which is
      going to be evaluated against the final label
    continuous: if true, the labels are continuous and R2 is calculated.

  Returns:
    a dictionary of accuracy, precision, recall, and f1 score.
  """
  if continuous:
    label_col = task_labels[0] + "_label"
    pred_col = task_labels[0] + "_pred"
    r2 = r2_score(results[label_col], results[pred_col])
    scores = {"r2": round(r2, 4)}
    return scores

  elif len(task_labels) > 1:
    # Accuracy is to be evaluated for several classification tasks
    logging.info("Accuracy of the majority vote (using all annotator heads):")

    pred_cols = [col + "_pred" for col in task_labels]
    majority_pred = get_majority_vote(results, pred_cols)

  else:
    # Accuracy is to be evaluated for a single classification task
    logging.info("Accuracy of single label")

    majority_pred = results[task_labels[0] + "_pred"]

  a = accuracy_score(results["majority"], majority_pred)
  p, r, f, _ = precision_recall_fscore_support(
      results["majority"], majority_pred, average="binary")

  scores = {
      "A": round(a, 4),
      "P": round(p, 4),
      "R": round(r, 4),
      "F1": round(f, 4)
  }
  return scores


def multi_task_loss(y_true, y_pred):
  """Masks the missing (-1) annotation when calculating loss.

  Args:
    y_true: true labels
    y_pred: logit predictions

  Returns:
    the masked loss
  """
  mask = tf.cast(tf.math.not_equal(y_true, -1), tf.float32)
  return tf.BinaryCrossentropy(from_logits=True)(y_true * mask, y_pred * mask)


def to_binary(labels):
  """Converts outputs of sigmoid to binary labels.

  Args:
    labels: Vector of any size with values between 0 and 1.

  Returns:
    The binary transform of the labels.
  """
  return (labels.flatten() >= 0.5).astype(int)


def reduce_learning_rate(epoch, lr):
  """Reduces the learning rate during training.

  Args:
    epoch: current epoch, used if learning decay depends on the epoch
    lr: value of learning rate

  Returns:
    reduced value of the learning rate (~0.9 of original)
  """
  del epoch
  return lr * tf.math.exp(-0.1)
