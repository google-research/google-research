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

"""Evaluation script for the R methods run on GCP.

This script takes in an embedding of the cells in Loom format, as well as the
parameters used to generate it, runs basic metrics, and writes both the
parameters and the scores in a CSV.
"""

import collections
import csv
import itertools

from absl import app
from absl import flags
import anndata
import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csvs', None,
                    'Base name for the csvs containing the cells.')
flags.DEFINE_string('input_loom', None,
                    'Path to the local Loom file containing the cells.')
flags.DEFINE_string('output_csv', None,
                    'Path to the local csv file where we write the metrics.')
flags.DEFINE_string('output_h5ad', None,
                    'Path to the local h5ad file where we write the complete '
                    'data.')
flags.DEFINE_string('reduced_dim', None,
                    'Name of obsm field containing the embedding.')
flags.DEFINE_string('tissue', None, 'Name of the dataset the cells come from.')
flags.DEFINE_string('source', None, 'Method used to generate the embedding.')

################
# Seurat FLAGS #
################

flags.DEFINE_enum('seurat_norm', None, ['LogNormalize', 'CLR'],
                  'Seurat normalization method.')
flags.DEFINE_enum('seurat_find_variable', None, ['vst', 'mvp', 'disp'],
                  'Seurat variable selection method.')
flags.DEFINE_integer('seurat_n_features', None,
                     'Number of genes kept after filtering.')
flags.DEFINE_integer('seurat_n_pcs', None, 'Dimension of the latent space.')

###############
# Scran FLAGS #
###############

flags.DEFINE_integer('scran_sum_factor', None,
                     'Scran use of sum factors for normalization.')
flags.DEFINE_integer('scran_ercc', None,
                     'Scran use of ERCC sum factor for normalization.')
flags.DEFINE_integer('scran_n_pcs', None, 'Dimension of the latent space.')
flags.DEFINE_integer('scran_n_tops', None,
                     'Number of genes kept after filtering.')
flags.DEFINE_enum('scran_assay', None, ['counts', 'logcounts'],
                  'Scran assay used for the dimension reduction.')

##################
# ZinbWAVE FLAGS #
##################

flags.DEFINE_integer('zinbwave_dims', None, 'Dimension of the latent space.')
flags.DEFINE_integer('zinbwave_epsilon', None,
                     'ZinbWave epsilon regularization term.')
flags.DEFINE_integer('zinbwave_keep_variance', None,
                     'Number of genes kept after filtering.')
flags.DEFINE_integer('zinbwave_gene_covariate', None,
                     'ZinbWave use of gene covariates in the reduction.')


_SOURCE_TO_FLAGS = {
    'seurat':
        frozenset([
            'seurat_norm', 'seurat_find_variable', 'seurat_n_features',
            'seurat_n_pcs'
        ]),
    'scran':
        frozenset([
            'scran_sum_factor', 'scran_ercc', 'scran_assay', 'scran_n_pcs',
            'scran_n_tops'
        ]),
    'zinbwave':
        frozenset([
            'zinbwave_dims', 'zinbwave_epsilon', 'zinbwave_keep_variance',
            'zinbwave_gene_covariate'
        ]),
}

Metrics = collections.namedtuple(
    'Metrics', ['silhouette', 'kmeans_silhouette', 'ami', 'ari'])


def check_flags_combination(flags_dict):
  actual_flags = frozenset(f for f, v in flags_dict.items()
                           if f != 'source' and v is not None)
  return actual_flags == _SOURCE_TO_FLAGS[flags_dict['source']]


def evaluate_method(adata, dimension, n_clusters):
  """Runs Kmeans-clustering on the latent representation and evaluates it."""
  silhouette = sklearn.metrics.silhouette_score(adata.obsm[dimension],
                                                adata.obs['label'])

  kmeans = sklearn.cluster.KMeans(
      n_clusters=n_clusters, random_state=0).fit(adata.obsm[dimension])
  adata.obs['predicted_clusters'] = kmeans.labels_

  # If all kmeans clusters end up together (failure to converge), the silhouette
  # computation will crash.
  if len(np.unique(adata.obs['predicted_clusters'])) < 2:
    kmeans_silhouette = float('nan')
  else:
    kmeans_silhouette = sklearn.metrics.silhouette_score(
        adata.obsm[dimension], adata.obs['predicted_clusters'])

  ari = sklearn.metrics.adjusted_rand_score(adata.obs['label'],
                                            adata.obs['predicted_clusters'])
  ami = sklearn.metrics.adjusted_mutual_info_score(
      adata.obs['label'], adata.obs['predicted_clusters'])

  return Metrics(
      silhouette=silhouette,
      kmeans_silhouette=kmeans_silhouette,
      ami=ami,
      ari=ari)


def main(unused_argv):
  # TODO(fraimundo): update all R scripts to use CSVS instead of loom.
  if FLAGS.input_loom:
    adata = anndata.read_loom(FLAGS.input_loom)
  if FLAGS.input_csvs:
    count = pd.read_csv(f'{FLAGS.input_csvs}.counts.csv', index_col=0)
    metadata = pd.read_csv(f'{FLAGS.input_csvs}.metadata.csv', index_col=0)
    featuredata = pd.read_csv(f'{FLAGS.input_csvs}.featuredata.csv',
                              index_col=0)
    dimred = pd.read_csv(f'{FLAGS.input_csvs}.dimred.csv', index_col=0)
    adata = adata = anndata.AnnData(X=count.transpose(),
                                    obs=metadata,
                                    var=featuredata,
                                    obsm={FLAGS.reduced_dim: dimred.to_numpy()})

  n_clusters = adata.obs['label'].nunique()
  metrics = evaluate_method(adata, FLAGS.reduced_dim, n_clusters)

  with open(FLAGS.output_csv, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')

    if FLAGS.source == 'seurat':
      csv_writer.writerow([
          'method',
          'seurat_norm',
          'seurat_find_variable',
          'seurat_n_features',
          'seurat_n_pcs',
          'silhouette',
          'kmeans_ami',
          'kmeans_ari',
          'kmeans_silhouette',
          'n_cells',
          'tissue',
          'n_clusters',
      ])
      csv_writer.writerow([
          'seurat',
          FLAGS.seurat_norm,
          FLAGS.seurat_find_variable,
          FLAGS.seurat_n_features,
          FLAGS.seurat_n_pcs,
          metrics.silhouette,
          metrics.ami,
          metrics.ari,
          metrics.kmeans_silhouette,
          adata.n_obs,
          FLAGS.tissue,
          n_clusters,
      ])

    elif FLAGS.source == 'scran':
      csv_writer.writerow([
          'method',
          'scran_sum_factor',
          'scran_ercc',
          'scran_assay',
          'scran_n_pcs',
          'scran_n_tops',
          'silhouette',
          'kmeans_ami',
          'kmeans_ari',
          'kmeans_silhouette',
          'n_cells',
          'tissue',
          'n_clusters',
      ])
      csv_writer.writerow([
          'scran',
          FLAGS.scran_sum_factor,
          FLAGS.scran_ercc,
          FLAGS.scran_assay,
          FLAGS.scran_n_pcs,
          FLAGS.scran_n_tops,
          metrics.silhouette,
          metrics.ami,
          metrics.ari,
          metrics.kmeans_silhouette,
          adata.n_obs,
          FLAGS.tissue,
          n_clusters,
      ])

    elif FLAGS.source == 'zinbwave':
      csv_writer.writerow([
          'method',
          'zinbwave_dims',
          'zinbwave_epsilon',
          'zinbwave_keep_variance',
          'zinbwave_gene_covariate',
          'silhouette',
          'kmeans_ami',
          'kmeans_ari',
          'kmeans_silhouette',
          'n_cells',
          'tissue',
          'n_clusters',
      ])
      csv_writer.writerow([
          'zinbwave',
          FLAGS.zinbwave_dims,
          FLAGS.zinbwave_epsilon,
          FLAGS.zinbwave_keep_variance,
          FLAGS.zinbwave_gene_covariate,
          metrics.silhouette,
          metrics.ami,
          metrics.ari,
          metrics.kmeans_silhouette,
          adata.n_obs,
          FLAGS.tissue,
          n_clusters,
      ])

    if FLAGS.output_h5ad:
      adata.write(FLAGS.output_h5ad)


if __name__ == '__main__':
  flags.mark_flags_as_mutual_exclusive(['input_loom', 'input_csvs'])
  flags.mark_flag_as_required('output_csv')
  flags.mark_flag_as_required('reduced_dim')
  flags.mark_flag_as_required('tissue')
  flags.mark_flag_as_required('source')
  flags.register_multi_flags_validator(
      flag_names=(
          ['source'] +
          list(itertools.chain.from_iterable(_SOURCE_TO_FLAGS.values()))),
      multi_flags_checker=check_flags_combination,
      message='Source and other flags are not compatible.')
  app.run(main)
