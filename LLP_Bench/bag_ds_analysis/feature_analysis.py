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

"""Performs feature analysis (Analysis of BagSep) on datasets."""

from collections.abc import Sequence
import functools
import json

from absl import app
from absl import flags
from absl import logging
import analysis_constants
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


_C1 = flags.DEFINE_integer('c1', 0, 'What is column 1 to aggregate on?')
_C2 = flags.DEFINE_integer('c2', 1, 'What is column 2 to aggregate on?')
_GRP_KEY_SIZE_ONE = flags.DEFINE_bool(
    'grp_key_size_one', False, 'Is the size of the group key one?'
)
_WHICH_DATASET = flags.DEFINE_enum(
    'which_dataset',
    'criteo_ctr',
    ['criteo_ctr', 'criteo_sscl'],
    'Which dataset to preprocess.',
)


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  if _WHICH_DATASET.value == 'criteo_ctr':
    sparse_cols = [analysis_constants.C + str(i) for i in range(1, 27)]
    dense_cols = [analysis_constants.I + str(i) for i in range(1, 14)]
    all_cols = dense_cols + sparse_cols
    criteo_df = pd.read_csv(
        '../data/preprocessed_dataset/preprocessed_criteo.csv', usecols=all_cols
    )
  else:
    sparse_cols = [analysis_constants.C + str(i) for i in range(1, 18)]
    dense_cols = [analysis_constants.N + str(i) for i in range(1, 4)]
    all_cols = dense_cols + sparse_cols
    criteo_df = pd.read_csv(
        '../data/preprocessed_dataset/preprocessed_criteo_sscl.csv',
    )
  logging.info('DataFrame Loaded')

  if _GRP_KEY_SIZE_ONE.value:
    c1 = analysis_constants.C + str(_C1.value)
    c2 = '-'
    criteo_df[c1 + '_copy'] = criteo_df[c1]
    grouped_df = (
        criteo_df.groupby([c1 + '_copy'])
        .filter(lambda x: ((len(x) >= 50) and (len(x) <= 2500)))
        .groupby([c1])
    )
  else:
    c1 = analysis_constants.C + str(_C1.value)
    c2 = analysis_constants.C + str(_C2.value)
    criteo_df[c1 + '_copy'] = criteo_df[c1]
    criteo_df[c2 + '_copy'] = criteo_df[c2]
    grouped_df = (
        criteo_df.groupby([c1 + '_copy', c2 + '_copy'])
        .filter(lambda x: ((len(x) >= 50) and (len(x) <= 2500)))
        .groupby([c1 + '_copy', c2 + '_copy'])
    )
  logging.info('Grouping Done')

  if _WHICH_DATASET.value == 'criteo_ctr':
    embeddings = {}
    for col in all_cols:
      embeddings[col] = np.load(
          '../results/autoint_embeddings/' + col + '_embeddings.npy'
      )
    for col in dense_cols:
      embeddings[col] = embeddings[col].flatten()
  else:
    embeddings = {}
    for col in sparse_cols:
      embeddings[col] = np.load(
          '../results/autoint_embeddings/mse_' + col + '.npy'
      )[0]
  logging.info('Embeddings Loaded')

  def generate_embedding(row):
    """Puts a row into the embedding space."""
    embedding = []
    for col in dense_cols:
      embedding.append(row[col] * embeddings[col])
    for col in sparse_cols:
      embedding.append(embeddings[col][row[col]])
    return np.concatenate(embedding)

  def generate_embedding_sscl(row):
    """Puts a row into the embedding space."""
    embedding = [row[dense_cols]]
    for col in sparse_cols:
      embedding.append(embeddings[col][int(row[col])])
    return np.concatenate(embedding)

  def compute_mean_and_avg_sq_norm(which_dataset, df):
    """Computes mean and average squared norm of all vectors in df."""
    if which_dataset == 'criteo_ctr':
      embedding_matrix = np.array(df.apply(generate_embedding, axis=1).tolist())
    else:
      embedding_matrix = np.array(
          df.apply(generate_embedding_sscl, axis=1).tolist()
      )
    mean_embedding_vector = np.mean(embedding_matrix, axis=0)
    avg_sq_norm = np.mean(np.linalg.norm(embedding_matrix, axis=1) ** 2)
    sq_norm_of_avg = np.linalg.norm(mean_embedding_vector) ** 2
    mean_embedding_vector_with_norm = np.append(
        mean_embedding_vector, avg_sq_norm
    )
    return (2 * (avg_sq_norm - sq_norm_of_avg), mean_embedding_vector_with_norm)

  aggregate_df = pd.DataFrame(
      grouped_df.apply(
          functools.partial(compute_mean_and_avg_sq_norm, _WHICH_DATASET.value)
      ).tolist(),
      columns=[
          'Mean_Intra_Bag_Distances',
          'Mean_vectors_of_each_bags_with_avg_squared_norm',
      ],
  )
  # pylint: disable=invalid-name
  Intra = np.array(aggregate_df['Mean_Intra_Bag_Distances'])
  logging.info(
      'Intra Bag Distances and Mean vectors and Sum of Squared Norms Computed'
  )

  def bag_sq_euclidean_distance(b1, b2):
    """Computes BagSep between b1 and b2."""
    b1_avg_sq_norm = b1[-1]
    b2_avg_sq_norm = b2[-1]
    b1 = b1[:-1]
    b2 = b2[:-1]
    return b1_avg_sq_norm + b2_avg_sq_norm - 2 * np.dot(b1, b2)

  inter_bag_dist_matrix = squareform(
      pdist(
          np.array(
              aggregate_df[
                  'Mean_vectors_of_each_bags_with_avg_squared_norm'
              ].tolist()
          ),
          bag_sq_euclidean_distance,
      )
  )
  Inter = np.sum(inter_bag_dist_matrix, axis=1) / (
      len(inter_bag_dist_matrix) - 1
  )
  logging.info('Inter Bag Distances Computed')

  data = {
      'c1': c1,
      'c2': c2,
      'min_inter_distance': np.min(Inter),
      'max_inter_distance': np.max(Inter),
      'mean_inter_distance': np.mean(Inter),
      'std_inter_distance': np.std(Inter),
      'min_intra_distance': np.min(Intra),
      'max_intra_distance': np.max(Intra),
      'mean_intra_distance': np.mean(Intra),
      'std_intra_distance': np.std(Intra),
      'min_ratio': np.min(Inter / Intra),
      'max_ratio': np.max(Inter / Intra),
      'mean_ratio': np.mean(Inter / Intra),
      'std_ratio': np.std(Inter / Intra),
      'ratio_of_means': np.mean(Inter) / np.mean(Intra),
  }
  logging.info('Metrics computed')

  if _WHICH_DATASET.value == 'criteo_ctr':
    saving_dir = '../results/dist_dicts/'
  else:
    saving_dir = '../results/dist_dicts/sscl_'
  if _GRP_KEY_SIZE_ONE.value:
    saving_file = saving_dir + c1 + '.json'
  else:
    saving_file = saving_dir + c1 + '_' + c2 + '.json'
  with open(saving_file, 'w') as fp:
    json.dump(data, fp)
  logging.info('Program completed')


if __name__ == '__main__':
  app.run(main)
