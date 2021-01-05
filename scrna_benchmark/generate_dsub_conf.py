# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Generates the dsub configuration for running scran, Seurat and ZinbWave.

This script is used to generate the hyperparameter grid for scran, Seurat and
ZinbWave, and feed it to dsub as a task file (tsv) according to
https://github.com/DataBiosphere/dsub#submitting-a-batch-job
"""

import csv
import itertools
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('cloud_input', None, 'GFS folder containing the datasets')
flags.DEFINE_string('cloud_output', None,
                    'GFS folder where the output is written')
flags.DEFINE_string('conf_path', None,
                    'Folder where the TSV dsub configs will be written.')

TISSUES = [
    'Zhengmix4eq',
    'Zhengmix4uneq',
    'Zhengmix5eq',
    'Zhengmix8eq',
    'Zhengmix8uneq',
    'tm_tissue_mix.12k',
    'sc_10x',
    'sc_10x_5cl',
    'sc_celseq2',
    'sc_celseq2_5cl',
]

# Shared parameters across the three methods.
DIMS = [2, 8, 10, 16, 32, 50, 64, 128]
N_FEATURES = [100, 300, 500, 1000, 2000, 3000]

#################
# Seurat params #
#################

SEURAT_NORM = ['LogNormalize', 'CLR']
SEURAT_FIND_VARIABLE = ['vst', 'mvp', 'disp']

#################
# scran params  #
#################

SCRAN_SUM_FACTORS = [0, 1]
SCRAN_ERCC = [0, 1]
SCRAN_ASSAY = ['counts', 'logcounts']

####################
# ZinbWave params  #
####################

ZINB_EPSILON = [200, 500, 1000, 2000]
ZINB_GENE_COVARIATE = [1, 0]


def generate_seurat_conf(conf_path, input_path, output_path):
  """Generates the hyperparameter grid for Seurat."""
  with open(os.path.join(conf_path, 'seurat_conf.tsv'), 'w') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow([
        '--env NORM',
        '--env VARIABLE',
        '--env FEATURES',
        '--env NPCS',
        '--env TISSUE',
        '--input SCE',
        '--output LOOM',
        '--output CSV',
    ])
    for tissue in TISSUES:
      for norm, selection, features, dims in itertools.product(
          SEURAT_NORM, SEURAT_FIND_VARIABLE, N_FEATURES, DIMS):
        run_file = (f'{tissue}.normalization_method={norm}.'
                    f'variable_features={selection}.n_features={features}.'
                    f'n_pcs={dims}')
        tsv_writer.writerow([
            norm,
            selection,
            features,
            dims,
            tissue,
            os.path.join(input_path, f'{tissue}.rds'),
            os.path.join(output_path, 'seurat', 'loom', f'{run_file}.loom'),
            os.path.join(output_path, 'seurat', 'csv', f'{run_file}.csv'),
        ])


def generate_scran_conf(conf_path, input_path, output_path):
  """Generates the hyperparameter grid for scran."""
  with open(os.path.join(conf_path, 'scran_conf.tsv'), 'w') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow([
        '--env SUM_FACTOR',
        '--env ERCC',
        '--env ASSAY',
        '--env NPCS',
        '--env NTOPS',
        '--env TISSUE',
        '--input SCE',
        '--output LOOM',
        '--output CSV',
    ])
    for tissue in TISSUES:
      for sum_factor, ercc, assay, dims, features in itertools.product(
          SCRAN_SUM_FACTORS, SCRAN_ERCC, SCRAN_ASSAY, DIMS, N_FEATURES):
        run_file = (f'{tissue}.sum_factor={sum_factor}.ercc={ercc}.'
                    f'assay={assay}.n_pcs={dims}.n_tops={features}')
        tsv_writer.writerow([
            sum_factor,
            ercc,
            assay,
            dims,
            features,
            tissue,
            os.path.join(input_path, f'{tissue}.rds'),
            os.path.join(output_path, 'scran', 'loom', f'{run_file}.loom'),
            os.path.join(output_path, 'scran', 'csv', f'{run_file}.csv'),
        ])


def generate_zinbwave_conf(conf_path, input_path, output_path):
  """Generates the hyperparameter grid for ZinbWave."""
  with open(os.path.join(conf_path, 'zinbwave_conf.tsv'), 'w') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow([
        '--env DIMS',
        '--env EPSILON',
        '--env KEEP_FEATURES',
        '--env GENE_COVARIATE',
        '--env TISSUE',
        '--input SCE',
        '--output LOOM',
        '--output CSV',
    ])
    for tissue in TISSUES:
      for dims, epsilon, features, gene_covariate in itertools.product(
          DIMS, ZINB_EPSILON, N_FEATURES, ZINB_GENE_COVARIATE):
        run_file = (f'{tissue}.dims={dims}.epsilon={epsilon}.'
                    f'features={features}.gene_covariate={gene_covariate}')
        tsv_writer.writerow([
            dims,
            epsilon,
            features,
            gene_covariate,
            tissue,
            os.path.join(input_path, f'{tissue}.rds'),
            os.path.join(output_path, 'zinbwave', 'loom', f'{run_file}.loom'),
            os.path.join(output_path, 'zinbwave', 'csv', f'{run_file}.csv'),
        ])


def main(unused_argv):
  generate_seurat_conf(FLAGS.conf_path, FLAGS.cloud_input, FLAGS.cloud_output)
  generate_scran_conf(FLAGS.conf_path, FLAGS.cloud_input, FLAGS.cloud_output)
  generate_zinbwave_conf(FLAGS.conf_path, FLAGS.cloud_input, FLAGS.cloud_output)


if __name__ == '__main__':
  flags.mark_flag_as_required('cloud_input')
  flags.mark_flag_as_required('cloud_output')
  flags.mark_flag_as_required('conf_path')
  app.run(main)
