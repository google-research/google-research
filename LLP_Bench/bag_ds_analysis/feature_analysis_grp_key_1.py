# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Performs feature analysis (Analysis of BagSep) on datasets with grp key size 1."""
from collections.abc import Sequence
import pickle

from absl import app
from absl import flags
from absl import logging
import analysis_constants
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


_C = flags.DEFINE_integer('c', 0, 'What is column to aggregate on?')


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.info('Program Started')
  sparse_cols = [analysis_constants.C + str(i) for i in range(1, 27)]
  dense_cols = [analysis_constants.I + str(i) for i in range(1, 14)]
  all_cols = dense_cols + sparse_cols
  criteo_df = pd.read_csv(
      '../data/preprocessed_dataset/preprocessed_criteo.csv', usecols=all_cols
  )
  logging.info('DataFrame Loaded')

  c = analysis_constants.C + str(_C.value)
  criteo_df[c + '_copy'] = criteo_df[c]
  grouped_df = (
      criteo_df.groupby([c + '_copy'])
      .filter(lambda x: ((len(x) >= 50) and (len(x) <= 2500)))
      .groupby([c])
  )
  logging.info('Grouping Done')

  embeddings = {}
  for col in all_cols:
    embeddings[col] = np.load(
        '../results/autoint_embeddings/' + col + '_embeddings.npy'
    )
  for col in dense_cols:
    embeddings[col] = embeddings[col].flatten()
  logging.info('Embeddings Loaded')

  def generate_embedding(row):
    """Puts a row in the embedding space."""
    embedding = []
    for col in dense_cols:
      embedding.append(row[col] * embeddings[col])
    for col in sparse_cols:
      embedding.append(embeddings[col][row[col]])
    return np.concatenate(embedding)

  def compute_mean_and_avg_sq_norm(df):
    """Computed Mean and Avg Squared Norm of all vectors in df."""
    embedding_matrix = np.array(df.apply(generate_embedding, axis=1).tolist())
    mean_embedding_vector = np.mean(embedding_matrix, axis=0)
    avg_sq_norm = np.mean(np.linalg.norm(embedding_matrix, axis=1) ** 2)
    sq_norm_of_avg = np.linalg.norm(mean_embedding_vector) ** 2
    mean_embedding_vector_with_norm = np.append(
        mean_embedding_vector, avg_sq_norm
    )
    return (2 * (avg_sq_norm - sq_norm_of_avg), mean_embedding_vector_with_norm)

  aggregate_df = pd.DataFrame(
      grouped_df.apply(compute_mean_and_avg_sq_norm).tolist(),
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
      'c1': c,
      'c2': '-',
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

  saving_dir = '../results/dist_dicts/'
  pickle.dump(data, saving_dir + c + '.pkl')
  logging.info('Program completed')


if __name__ == '__main__':
  app.run(main)
