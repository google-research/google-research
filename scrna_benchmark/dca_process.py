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

"""DCA run and evaluation.

Runs DCA with the specified input parameters and evaluates it.

This is weird 'hack' that launches a hyperparameter grid locally
because our grid is so large that we cannot launch one configuration
per machine.
"""

import collections
import csv
import itertools
import os
import tempfile

from absl import app
from absl import flags
from absl import logging
import anndata
import numpy as np
import pandas as pd
import scanpy.api as sc
import sklearn.cluster
import sklearn.metrics
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None,
                    'Path to the input loom or anndata file.')
flags.DEFINE_string('output_csv', None,
                    'Path to the folder containing the csv results.')
flags.DEFINE_string('log_path', None,
                    'Path to the folder containing the runs log.')
flags.DEFINE_integer(
    'seed', None,
    'Random seed to use in the run. If no value is given we will run for 1..5')

# Flags about the DCA run hyperparameters.
flags.DEFINE_enum('ae_type', None,
                  ['zinb-conddisp', 'zinb', 'nb-conddisp', 'nb'],
                  'Type of autoencoder to use.')
flags.DEFINE_boolean(
    'normalize_per_cell', None,
    'If true, library size normalization is performed '
    'using the sc.pp.normalize_per_cell function in '
    'Scanpy. If no value is given it will run for both True and False.')
flags.DEFINE_boolean(
    'scale', None, 'If true, the input of the autoencoder is centered '
    'using sc.pp.scale function of Scanpy. If no value is given it will '
    'run for both True and False.')
flags.DEFINE_boolean(
    'log1p', None, 'If true, the input of the autoencoder is log '
    'transformed with a pseudocount of one using sc.pp.log1p function of '
    'Scanpy. If no value is given it will run for both True and False.')
flags.DEFINE_list('hidden_size', None, 'Width of hidden layers.')
flags.DEFINE_float('hidden_dropout', None,
                   'Probability of weight dropout in the autoencoder.')
flags.DEFINE_boolean(
    'batchnorm', None,
    'Whether to use batchnorm or not. If no value is given it will run for '
    'both True and False.')
flags.DEFINE_integer('batch_size', None, 'Batch size to use in training.')
flags.DEFINE_integer(
    'epochs', None,
    'Number of epochs to train on. If no value is given it will run for 20, 50,'
    '100, 200, 300, 500, and 1000.')

# Flags about the environment the code is executed in and its output.
flags.DEFINE_boolean('from_gcs', True, 'Whether the input is hosted on GCS.')
flags.DEFINE_boolean('run_info', False, 'Whether to store the whole run_info.')
flags.DEFINE_boolean('save_h5ad', False, 'Whether the anndata should be saved.')
flags.DEFINE_boolean('seurat_readable', False,
                     'Whether to make the file Seurat readable.')

Conf = collections.namedtuple(
    'Conf',
    ['log1p', 'normalize_per_cell', 'scale', 'batchnorm', 'epochs', 'seed'])
Metrics = collections.namedtuple(
    'Metrics', ['silhouette', 'kmeans_silhouette', 'ami', 'ari'])
RunResult = collections.namedtuple('RunResult', [
    'method', 'seed', 'ae_type', 'normalize_per_cell', 'scale', 'log1p',
    'hidden_size', 'hidden_dropout', 'batchnorm', 'batch_size', 'epochs',
    'silhouette', 'kmeans_silhouette', 'kmeans_ami', 'kmeans_ari', 'n_cells',
    'tissue', 'n_clusters', 'loss', 'val_loss', 'run_info_fname', 'h5ad_fname'
])


def evaluate_method(adata, n_clusters):
  """Runs the AMI, ARI, and silhouette computation."""
  # If the training diverged, the embedding will have nan for infinity.
  if np.any(np.isnan(adata.obsm['X_dca'])):
    return Metrics(
        silhouette=float('nan'),
        kmeans_silhouette=float('nan'),
        ari=float('nan'),
        ami=float('nan'),
    )

  silhouette = sklearn.metrics.silhouette_score(adata.obsm['X_dca'],
                                                adata.obs['label'])

  kmeans = sklearn.cluster.KMeans(
      n_clusters=n_clusters, random_state=0).fit(adata.obsm['X_dca'])
  adata.obs['predicted_clusters'] = kmeans.labels_

  # If all kmeans clusters end up together (failure to converge), the silhouette
  # computation will crash.
  if len(np.unique(adata.obs['predicted_clusters'])) < 2:
    kmeans_silhouette = float('nan')
  else:
    kmeans_silhouette = sklearn.metrics.silhouette_score(
        adata.obsm['X_dca'], adata.obs['predicted_clusters'])
  ari = sklearn.metrics.adjusted_rand_score(adata.obs['label'],
                                            adata.obs['predicted_clusters'])
  ami = sklearn.metrics.adjusted_mutual_info_score(
      adata.obs['label'], adata.obs['predicted_clusters'])

  return Metrics(
      silhouette=silhouette,
      kmeans_silhouette=kmeans_silhouette,
      ami=ami,
      ari=ari)


def dca_process(adata, ae_type, normalize_per_cell, scale, log1p, hidden_size,
                hidden_dropout, batchnorm, epochs, batch_size, seed,
                seurat_readable):
  """Runs dca from scanpy."""
  sc.pp.dca(
      adata,
      ae_type=ae_type,
      normalize_per_cell=normalize_per_cell,
      scale=scale,
      log1p=log1p,
      hidden_size=hidden_size,
      hidden_dropout=hidden_dropout,
      mode='latent',
      optimizer='Adam',
      batchnorm=batchnorm,
      epochs=epochs,
      batch_size=batch_size,
      random_state=seed,
      return_info=True)
  if seurat_readable:
    adata.var['Gene'] = adata.var.index
    adata.obs['CellID'] = adata.obs['cell']
    adata.obsm['dca_cell_embeddings'] = adata.obsm['X_dca']
  return adata


def log_run(path, conf):
  """Logs a successful run in a CSV, as well as the header for new files."""
  conf_dict = dict(conf._asdict())
  # We need to check if the file exists before creating it.
  write_header = not tf.io.gfile.exists(path)
  with tf.io.gfile.GFile(path, 'a') as f:
    csv_writer = csv.DictWriter(f, fieldnames=conf_dict.keys())
    if write_header:
      csv_writer.writeheader()
    csv_writer.writerow(conf_dict)


def log_run_info(save_run_info, infos, log_folder, conf, tissue, ae_type,
                 hidden_size, hidden_dropout, batch_size):
  """Saves the training stats and returns the path to them."""
  if not save_run_info:
    return ''

  # Save the loss for this run.
  run_info_fname = os.path.join(
      log_folder, f'{tissue}.method=dca.seed={conf.seed}.ae_type={ae_type}.'
      f'normalize_per_cell={conf.normalize_per_cell}.scale={conf.scale}.'
      f'log1p={conf.log1p}.hidden_size={hidden_size}.'
      f'hidden_dropout={hidden_dropout}.batchnorm={conf.batchnorm}.'
      f'batch_size={batch_size}.epochs={conf.epochs}.runinfo.csv')
  run_info_df = pd.DataFrame.from_dict(infos)
  with tf.io.gfile.GFile(run_info_fname, 'w') as f:
    run_info_df.to_csv(f)
  return run_info_fname


def fetch_anndata(path, from_gcs):
  """Reads the input data and turns it into an anndata.AnnData object."""
  _, ext = os.path.splitext(path)

  # AnnData is based of HDF5 and doesn't have GCS file handlers
  # so we have to locally copy the file before reading it.
  if from_gcs:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_path = tmp_file.name
    tf.io.gfile.copy(path, tmp_path, overwrite=True)
    path = tmp_path

  if ext == '.h5ad':
    adata = anndata.read_h5ad(path)
  elif ext == '.loom':
    adata = anndata.read_loom(path)
  else:
    raise app.UsageError('Only supports loom and h5ad files.')

  return adata


def write_anndata(save_h5ad, adata, log_folder, conf, tissue, ae_type,
                  hidden_size, hidden_dropout, batch_size):
  """Writes anndata object with the proper name on GCS and returns the name."""
  # We need to write the anndata locally and copy it to GCS for the same
  # reason as before.
  if not save_h5ad:
    return ''

  with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=True) as tmp_file:
    adata.write(tmp_file.name)
    h5ad_fname = os.path.join(
        log_folder, f'{tissue}.method=dca.seed={conf.seed}.ae_type={ae_type}.'
        f'normalize_per_cell={conf.normalize_per_cell}.scale={conf.scale}.'
        f'log1p={conf.log1p}.hidden_size={hidden_size}.'
        f'hidden_dropout={hidden_dropout}.batchnorm={conf.batchnorm}.'
        f'batch_size={batch_size}.epochs={conf.epochs}.h5ad')
    tf.io.gfile.copy(tmp_file.name, h5ad_fname, overwrite=True)
  return h5ad_fname


def generate_conf(log1p, normalize_per_cell, scale, batchnorm, epochs, seed):
  """Generates the local parameter grid."""
  local_param_grid = {
      'log1p': [True, False] if log1p is None else [log1p],
      'normalize_per_cell': [True, False] if normalize_per_cell is None else
                            [normalize_per_cell],
      'scale': [True, False] if scale is None else [scale],
      'batchnorm': [True, False] if batchnorm is None else [batchnorm],
      'epochs': [20, 50, 100, 200, 300, 500, 1000]
                if epochs is None else [epochs],
      'seed': [0, 1, 2, 3, 4] if seed is None else [seed]
  }

  return [Conf(*vals) for vals in itertools.product(*local_param_grid.values())]


def fetch_previous_runs(log_path):
  """Reads in the state in which the previous run stopped."""
  previous_runs = set()
  if tf.io.gfile.exists(log_path):
    with tf.io.gfile.GFile(log_path, mode='r') as f:
      reader = csv.DictReader(f)
      for row in reader:
        # Note: we need to do this conversion because DictReader creates an
        # OrderedDict, and reads all values as str instead of bool or int.
        previous_runs.add(
            str(
                Conf(
                    log1p=row['log1p'] == 'True',
                    normalize_per_cell=row['normalize_per_cell'] == 'True',
                    scale=row['scale'] == 'True',
                    batchnorm=row['batchnorm'] == 'True',
                    epochs=int(row['epochs']),
                    seed=int(row['seed']),
                )))
  logging.info('Previous runs:')
  for run in previous_runs:
    logging.info(run)

  return previous_runs


def main(unused_argv):
  hidden_size = [int(l) for l in FLAGS.hidden_size]

  tissue, _ = os.path.splitext(os.path.basename(FLAGS.input_path))
  adata = fetch_anndata(FLAGS.input_path, FLAGS.from_gcs)

  confs = generate_conf(
      log1p=FLAGS.log1p,
      normalize_per_cell=FLAGS.normalize_per_cell,
      scale=FLAGS.scale,
      batchnorm=FLAGS.batchnorm,
      epochs=FLAGS.epochs,
      seed=FLAGS.seed)
  previous_runs = fetch_previous_runs(FLAGS.log_path)

  sc.pp.filter_genes(adata, min_cells=1)
  n_clusters = adata.obs['label'].nunique()
  total_runs = len(confs)

  for i, conf in enumerate(confs):
    if str(conf) in previous_runs:
      logging.info('Skipped %s', conf)
      continue

    adata = dca_process(
        adata,
        ae_type=FLAGS.ae_type,
        normalize_per_cell=conf.normalize_per_cell,
        scale=conf.scale,
        log1p=conf.log1p,
        hidden_size=hidden_size,
        hidden_dropout=FLAGS.hidden_dropout,
        batchnorm=conf.batchnorm,
        batch_size=FLAGS.batch_size,
        epochs=conf.epochs,
        seed=conf.seed,
        seurat_readable=FLAGS.seurat_readable)
    metrics = evaluate_method(adata, n_clusters)
    infos = adata.uns['dca_loss_history']

    log_folder = os.path.dirname(FLAGS.output_csv)

    run_info_fname = log_run_info(
        save_run_info=FLAGS.run_info,
        infos=infos,
        log_folder=log_folder,
        conf=conf,
        tissue=tissue,
        ae_type=FLAGS.ae_type,
        hidden_size=hidden_size,
        hidden_dropout=FLAGS.hidden_dropout,
        batch_size=FLAGS.batch_size)

    h5ad_fname = write_anndata(
        adata=adata,
        save_h5ad=FLAGS.save_h5ad,
        log_folder=log_folder,
        conf=conf,
        tissue=tissue,
        ae_type=FLAGS.ae_type,
        hidden_size=hidden_size,
        hidden_dropout=FLAGS.hidden_dropout,
        batch_size=FLAGS.batch_size)

    run_result = RunResult(
        method='dca',
        seed=conf.seed,
        ae_type=FLAGS.ae_type,
        normalize_per_cell=conf.normalize_per_cell,
        scale=conf.scale,
        log1p=conf.log1p,
        hidden_size=hidden_size,
        hidden_dropout=FLAGS.hidden_dropout,
        batchnorm=conf.batchnorm,
        batch_size=FLAGS.batch_size,
        epochs=conf.epochs,
        silhouette=metrics.silhouette,
        kmeans_silhouette=metrics.kmeans_silhouette,
        kmeans_ami=metrics.ami,
        kmeans_ari=metrics.ari,
        n_cells=adata.n_obs,
        tissue=tissue,
        n_clusters=n_clusters,
        loss=infos['loss'][-1],
        val_loss=infos['val_loss'][-1],
        run_info_fname=run_info_fname,
        h5ad_fname=h5ad_fname)
    log_run(FLAGS.output_csv, run_result)

    logging.info(conf)
    logging.info('Done with %s out of %s', i, total_runs)
    log_run(FLAGS.log_path, conf)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_path')
  flags.mark_flag_as_required('output_csv')
  flags.mark_flag_as_required('log_path')
  flags.mark_flag_as_required('ae_type')
  flags.mark_flag_as_required('hidden_size')
  flags.mark_flag_as_required('hidden_dropout')
  flags.mark_flag_as_required('batch_size')
  app.run(main)
