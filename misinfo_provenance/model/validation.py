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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Retrieval Evaluation Functions Module.

Implements Precision@k, Recall@k, uAP, mAP
"""

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf


# Create dataset from TFRecords.
def read_labeled_tfrecord_photoshop(record):
  """Read ps-battles tfrecords.

  Args:
    record:

  Returns:

  """

  name_to_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'bands': tf.io.FixedLenFeature([], tf.int64),
      'image_name': tf.io.FixedLenFeature([], tf.string),
      'relevant': tf.io.FixedLenFeature([], tf.string),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  record = tf.io.parse_single_example(record, name_to_features)
  image = tf.io.decode_raw(
      record['image_raw'],
      out_type=tf.uint8,
      little_endian=True,
      fixed_length=None,
      name=None)
  relevant = record['relevant']
  image_name = record['image_name']
  height, width, bands = record['height'], record['width'], record['bands']
  image = tf.reshape(image, (height, width, bands))
  image = tf.image.resize(image, (256, 256))
  image = image / 255.0
  return (image, relevant, image_name)


def read_labeled_tfrecord_copydays10k(record):
  """Read copydays10k tfrecords.

  Args:
    record:

  Returns:

  """
  name_to_features = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'bands': tf.io.FixedLenFeature([], tf.int64),
      'image_name': tf.io.FixedLenFeature([], tf.string),
      'm_type': tf.io.FixedLenFeature([], tf.string),
      'relevant': tf.io.FixedLenFeature([], tf.string),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }
  record = tf.io.parse_single_example(record, name_to_features)
  image = tf.io.decode_raw(
      record['image_raw'],
      out_type=tf.uint8,
      little_endian=True,
      fixed_length=None,
      name=None)
  relevant = record['relevant']
  image_name = record['image_name']
  height, width, bands = record['height'], record['width'], record['bands']
  image = tf.reshape(image, (height, width, bands))
  image = tf.image.resize(image, (256, 256))
  image = image / 255.0
  return (image, relevant, image_name)


def read_dataset(file_pattern, dataset_name='PIR', batch_size=128):
  """Read a tfRecord dataset from cns for validation purpose.

  The implemented datasets are 'PIR', 'copydays10k', and
  'copydays10k-strong'.
  Args:
    file_pattern: TfRecords of the dataset
    dataset_name: Validation dataset name.
    batch_size: Batch size of the generated dataset.

  Returns:
    Dataset

  Raises:
    NotImplementedError: if dataset not implemented for validation.
  """

  auto = tf.data.experimental.AUTOTUNE
  ignore_order = tf.data.Options()
  dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.prefetch(auto)
  dataset.with_options(ignore_order)
  if dataset_name == 'PIR':
    dataset = dataset.map(
        read_labeled_tfrecord_photoshop, num_parallel_calls=auto)
  elif dataset_name == 'copydays10k' or dataset_name == 'copydays10k-strong':
    dataset = dataset.map(
        read_labeled_tfrecord_copydays10k, num_parallel_calls=auto)
  else:
    raise NotImplementedError(
        f'Evaluation for dataset {dataset_name} not implemented')
  dataset = dataset.batch(batch_size=batch_size)
  dataset_iter = iter(dataset)
  return dataset_iter


def get_df_emb(model, dataset, dataset_name='PIR'):
  """Extracts embeddings from datasets.

  Args:
    model: Model used to extract the embeddings.
    dataset: Dataset <tf.datasets>.
    dataset_name: Validation dataset name.

  Returns:
    df: <pandas DataFrame> Dataframe containing all info related to each dataset
        item.
    embeddings: <Tf.Tensor> Extracted Embeddings from the datasets items.
  Raises:
    NotImplemtedError: If dataset name not implemented for validation.
  """

  df = pd.DataFrame()
  embeddings = []

  index = 0
  for batch_index in dataset:
    images, relevant, image_name = batch_index
    # _, embs = model(images) # classification
    embs = model(images)  # ssl
    for emb, r, img_name in zip(embs, relevant, image_name):
      embeddings.append(list(emb.numpy()))
      df.loc[index, 'relevant'] = r.numpy().decode()
      img_name = img_name.numpy().decode()
      df.loc[index, 'path_name'] = img_name
      if dataset_name == 'PIR':
        df.loc[index, 'image_name'] = img_name
      elif dataset_name == 'copydays10k' or dataset_name == 'copydays10k-strong':
        df.loc[
            index,
            'image_name'] = img_name if '_' not in img_name else img_name.split(
                '_')[1]
      else:
        raise NotImplementedError(
            f'Evaluation for dataset {dataset_name} not implemented')
      index += 1
  embeddings = np.array(embeddings)
  return df, embeddings


def argsort(seq):
  # from https://stackoverflow.com/a/3382369/3853462
  return sorted(range(len(seq)), key=seq.__getitem__)


def precision_recall(y_true,
                     probas_pred,
                     num_positives):
  """Calculate precision and recall.

    Compute precisions, recalls and thresholds.
  Args:
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a
        distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.
  Returns:
    precisions:
    recalls:
    probas_pred:
  """
  probas_pred = probas_pred.flatten()
  y_true = y_true.flatten()
  # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
  # eg,the final order will be (0.5, False), (0.5, False), (0.5, True),
  # (0.4, False), ...
  # This allows to have the worst possible AP.
  # It prevents participants from putting the same score for all predictions
  # to get a good AP.
  order = argsort(list(zip(probas_pred, ~y_true)))
  order = order[::-1]  # sort by decreasing score
  probas_pred = probas_pred[order]
  y_true = y_true[order]

  ntp = np.cumsum(y_true)  # number of true positives <= threshold
  nres = np.arange(len(y_true)) + 1  # number of results

  precisions = ntp / nres
  recalls = ntp / num_positives
  return precisions, recalls, probas_pred


def average_precision(recalls, precisions):
  """Calulcate the average precision.

  Args:
    recalls:
    precisions:

  Returns:
    Average Precision

  Raises:
    Exception: Order of recall not increasing.
  """

  # Check that it's ordered by increasing recall
  if not np.all(recalls[:-1] <= recalls[1:]):
    raise Exception('recalls array must be sorted before passing in')

  return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()


def perform_eval(model, k=100, dataset_name='PIR', batch_size=256):
  """Calculate the uAP, mAP, and recal @[1,10,100] of the model on the.

  dataset_name.
  We are considering only one relevant per query.
  Args:
    model: Tensorflow model.
    k: rank size.
    dataset_name: <str> Implemented for PIR and copydays10k
    batch_size: Batch size to process the dataset

  Returns:
    uAP: micro-average precision
    mAP: macro-average precision
    r@1 : recall @ 1
    r@10 : recall @ 10
    r@100 : recall @ 100
  """
  with tf.device('/gpu:0'):

    logging.info('=== Reading Datasets ===')
    if dataset_name == 'PIR':
      file_pattern_query = 'Insert-Path-To-Query-Dataset'
      file_pattern_index = 'Insert-Path-To-Index-Dataset'
    elif dataset_name == 'copydays10k':
      file_pattern_query = 'Insert-Path-To-Query-Dataset'
      file_pattern_index = 'Insert-Path-To-Index-Dataset'
    elif dataset_name == 'copydays10k-strong':
      file_pattern_query = 'Insert-Path-To-Query-Dataset'
      file_pattern_index = 'Insert-Path-To-Index-Dataset'

    query_df, query_embeddings = get_df_emb(
        model,
        read_dataset(
            file_pattern_query,
            dataset_name=dataset_name,
            batch_size=batch_size),
        dataset_name=dataset_name)
    index_df, index_embeddings = get_df_emb(
        model,
        read_dataset(
            file_pattern_index,
            dataset_name=dataset_name,
            batch_size=batch_size),
        dataset_name=dataset_name)

    logging.info('=== Reading Datasets DONE ===')
    # Performn exact knn search
    logging.info('=== KNN Search ===')
    knn_search = query_embeddings @ index_embeddings.T
    retrieved_rank, similarities = knn_search.argsort()[:, -k:], np.sort(
        knn_search)[:, -k:]
    # sorting on desceding order, so index 0 is the nearsted item from the query
    retrieved_rank = np.flip(retrieved_rank, axis=1)
    similarities = np.flip(similarities, axis=1)

    def rank_relevance(qid, retrieved_rank):
      """Create Rank Relevenace."""
      rank = index_df.iloc[retrieved_rank]
      relevant_ids = rank[rank['image_name'] == query_df.iloc[qid]
                          ['relevant']].values

      gnd = np.zeros(len(rank), dtype='int')
      for index, tp in enumerate(rank.image_name):
        if tp in relevant_ids:
          gnd[index] = 1
      return gnd

    # Identify the relevant items from the retrieved rank
    y_true = [
        rank_relevance(qid, retrieved_rank[qid])
        for qid in range(len(query_df))
    ]
    y_true = np.array(y_true)

    # Calculate the uAP
    precision, recall, _ = precision_recall(y_true, similarities, len(y_true))
    # pylint: disable=invalid-name
    uAP = average_precision(recall, precision)

    # Calculate the mAP
    mAP = 0
    for rank, rename_me in enumerate(y_true):
      rank_true = rename_me
      rank_sim = similarities[rank]
      p, r, _ = precision_recall(rank_true, rank_sim, 1)
      mAP += average_precision(r, p)

    mAP /= len(y_true)
    # pylint: enable=invalid-name

    # Calculate recalls
    y_cumsum_true = y_true.cumsum(axis=1)
    r_1 = y_true[:, 0].sum() / len(y_true)
    r_10 = y_cumsum_true[:, 9].sum() / len(y_true)
    r_100 = y_cumsum_true[:, 99].sum() / len(y_true)

  return uAP, mAP, r_1, r_10, r_100
