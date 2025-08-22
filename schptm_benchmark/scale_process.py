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

"""Script for running SCALE.

See https://www.nature.com/articles/s41467-019-12630-7
"""
import os
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import anndata
import numpy as np
import pandas as pd
from scale import SCALE
import scale.dataset
import scale.utils
import scipy.io
import scipy.sparse
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

CHUNK_SIZE = 20000

FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', None, 'Path to the 10x formatted folder.')
flags.DEFINE_string('output_path', None, 'Path to the output directory.')
flags.DEFINE_float('lr', 0.0002, 'Learning rate.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay.')
flags.DEFINE_float('min_peaks', 0, 'Remove low quality cells with few peaks.')
flags.DEFINE_float('min_cells', 0.01, 'Remove low quality peaks.')
flags.DEFINE_integer('n_centroids', help='cluster number', default=30)
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('gpu', 0, 'Select gpu device number when training')
flags.DEFINE_integer('seed', 18, 'Random seed for repeat results')
flags.DEFINE_integer('latent', 10, 'latent layer dim')
flags.DEFINE_integer('n_feature', 30000,
                     'Keep the number of highly variable peaks')
flags.DEFINE_integer('max_iter', 30000, 'Max iteration')
flags.DEFINE_boolean('verbose', True, 'Print loss of training process')
flags.DEFINE_multi_integer('encode_dim', [1024, 128], 'encoder structure')
flags.DEFINE_multi_integer('decode_dim', [], 'encoder structure')


def create_anndata(path):
  """Creates anndata object from raw data.

  Args:
    path: Path to the 10x formatted input files.

  Returns:
    anndata object for the experiment.
  """
  with tf.io.gfile.GFile(os.path.join(path, 'matrix.mtx'), mode='rb') as f:
    matrix = scipy.io.mmread(f)
  matrix = scipy.sparse.csr_matrix(matrix)
  adata = anndata.AnnData(matrix)
  adata = adata.transpose()
  with tf.io.gfile.GFile(os.path.join(path, 'barcodes.tsv'), mode='r') as f:
    barcodes = pd.read_csv(f, sep='\t', header=None)[0]
  adata.obs_names = barcodes
  with tf.io.gfile.GFile(os.path.join(path, 'bins.tsv'), mode='r') as f:
    bins = pd.read_csv(f, sep='\t', header=None)[0]
  adata.var_names = bins
  return adata


def load_dataset(
    path,
    batch_name='batch',
    min_genes=600,
    min_cells=3,
    n_top_genes=30000,
    batch_size=64,
    chunk_size=CHUNK_SIZE,
    log=None,
):
  """Loads data in appropriate formats."""
  adata = create_anndata(path)
  logging.info('Raw dataset shape: %s', format(adata.shape))
  if batch_name != 'batch':
    adata.obs['batch'] = adata.obs[batch_name]
  if 'batch' not in adata.obs:
    adata.obs['batch'] = 'batch'
  adata.obs['batch'] = adata.obs['batch'].astype('category')

  adata = scale.dataset.preprocessing_atac(
      adata,
      min_genes=min_genes,
      min_cells=min_cells,
      n_top_genes=n_top_genes,
      chunk_size=chunk_size,
      log=log,
  )

  scdata = scale.dataset.SingleCellDataset(adata)
  trainloader = DataLoader(
      scdata,
      batch_size=batch_size,
      drop_last=True,
      shuffle=True,
      num_workers=4)
  testloader = DataLoader(
      scdata, batch_size=batch_size, drop_last=False, shuffle=False)

  return adata, trainloader, testloader


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Set random seed
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  if torch.cuda.is_available():  # cuda device
    device = 'cuda'
    torch.cuda.set_device(FLAGS.gpu)
  else:
    device = 'cpu'

  adata, trainloader, testloader = load_dataset(
      FLAGS.input_path,
      batch_name='batch',
      min_genes=FLAGS.min_peaks,
      min_cells=FLAGS.min_cells,
      batch_size=FLAGS.batch_size,
      n_top_genes=FLAGS.n_feature,
      log=None,
  )

  input_dim = adata.shape[1]

  tf.io.gfile.makedirs(FLAGS.output_path)

  dims = [input_dim, FLAGS.latent, FLAGS.encode_dim, FLAGS.decode_dim]
  model = SCALE(dims, n_centroids=FLAGS.n_centroids)
  print(model)

  print('\n## Training Model ##')
  model.init_gmm_params(testloader)
  model.fit(
      trainloader,
      lr=FLAGS.lr,
      weight_decay=FLAGS.weight_decay,
      verbose=FLAGS.verbose,
      device=device,
      max_iter=FLAGS.max_iter,
      outdir=FLAGS.output_path)

  adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')

  dr = pd.DataFrame(adata.obsm['latent'], index=adata.obs_names)

  with tf.io.gfile.GFile(os.path.join(FLAGS.output_path, 'SCALE.csv'),
                         'w') as f:
    dr.to_csv(f)


if __name__ == '__main__':
  app.run(main)
