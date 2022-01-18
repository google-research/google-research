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

# Lint as: python3
"""scVI run and evaluation.

Runs scVI with the specified input parameters and evaluates it.

This is a weird 'hack' that launches a hyperparameter grid locally
because our grid is so large that we cannot launch one configuration
per machine.
"""

import collections
import csv
import itertools
import os
import random
import tempfile

from absl import app
from absl import flags
from absl import logging

import anndata
import numpy as np
import scanpy.api as sc
import scvi
import scvi.dataset
import scvi.inference
import sklearn.cluster
import sklearn.metrics
import tensorflow as tf
import torch

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

# Flags about the scVI run hyperparameters.
flags.DEFINE_integer(
    'n_layers', None, 'Number of hidden layers used for encoder and '
    'decoder NNs')
flags.DEFINE_integer('n_hidden', None, 'Number of nodes per hidden layer')
flags.DEFINE_enum('dispersion', None, ['gene', 'gene-cell'],
                  'What kind of dispersion to use.')
flags.DEFINE_float('dropout_rate', None, 'Dropout rate for neural networks')
flags.DEFINE_enum('reconstruction_loss', None, ['zinb', 'nb'],
                  'Generative distribution.')
flags.DEFINE_integer(
    'n_latent', None,
    'Dimensionality of the latent space. If no value is given, it will run for '
    '2, 8, 10, 16, 32, 50, 64, and 128.')
flags.DEFINE_integer(
    'epochs', None,
    'Number of epochs to train on. If no value is given, it will run for '
    '20, 50, 100, 200, 300, 500, and 1000.')
flags.DEFINE_float(
    'lr', None,
    'Learning rate. If no value is given, it will run for 1e-2, 1e-3, and 1e-4.'
)

# Flags about the environment the code is executed in and its output.
flags.DEFINE_boolean('from_gcs', True, 'Whether the input is hosted on GCS.')
flags.DEFINE_boolean('save_h5ad', False, 'Whether the anndata should be saved.')
flags.DEFINE_boolean('seurat_readable', False,
                     'Whether to make the file Seurat readable.')

Conf = collections.namedtuple('Conf', ['n_latent', 'epochs', 'lr', 'seed'])
Metrics = collections.namedtuple(
    'Metrics', ['silhouette', 'kmeans_silhouette', 'ami', 'ari'])
RunResult = collections.namedtuple('RunResult', [
    'method', 'seed', 'n_layers', 'n_hidden', 'dispersion', 'dropout_rate',
    'reconstruction_loss', 'n_latent', 'epochs', 'lr', 'silhouette',
    'kmeans_silhouette', 'kmeans_ami', 'kmeans_ari', 'n_cells', 'tissue',
    'n_clusters', 'elbo_train_set', 'elbo_test_set', 'h5ad_fname'
])


def evaluate_method(adata, n_clusters):
  """Runs the AMI, ARI, and silhouette computation."""
  # If the training diverged, the embedding will have nan for infinity.
  if np.any(np.isnan(adata.obsm['X_scvi'])):
    return Metrics(
        silhouette=float('nan'),
        kmeans_silhouette=float('nan'),
        ari=float('nan'),
        ami=float('nan'),
    )

  silhouette = sklearn.metrics.silhouette_score(adata.obsm['X_scvi'],
                                                adata.obs['label'])

  kmeans = sklearn.cluster.KMeans(
      n_clusters=n_clusters, random_state=0).fit(adata.obsm['X_scvi'])
  adata.obs['predicted_clusters'] = kmeans.labels_

  # If all kmeans clusters end up together (failure to converge), the silhouette
  # computation will crash.
  if len(np.unique(adata.obs['predicted_clusters'])) < 2:
    kmeans_silhouette = float('nan')
  else:
    kmeans_silhouette = sklearn.metrics.silhouette_score(
        adata.obsm['X_scvi'], adata.obs['predicted_clusters'])
  ari = sklearn.metrics.adjusted_rand_score(adata.obs['label'],
                                            adata.obs['predicted_clusters'])
  ami = sklearn.metrics.adjusted_mutual_info_score(
      adata.obs['label'], adata.obs['predicted_clusters'])

  return Metrics(
      silhouette=silhouette,
      kmeans_silhouette=kmeans_silhouette,
      ami=ami,
      ari=ari)


def compute_scvi_latent(scvi_dataset,
                        n_latent,
                        n_layers,
                        n_hidden,
                        dropout_rate,
                        dispersion,
                        reconstruction_loss,
                        n_epochs,
                        lr,
                        use_batches=False,
                        use_cuda=True):
  """Train and return a scVI latent space.

  Args:
    scvi_dataset: dataset.GeneExpressionDataset to work on.
    n_latent: Dimensionality of the latent space.
    n_layers: Number of hidden layers used for encoder and decoder NNs.
    n_hidden: Number of nodes per hidden layer.
    dropout_rate: Dropout rate for neural networks.
    dispersion: One of the following * 'gene' - dispersion parameter of NB is
      constant per gene across cells * 'gene-batch' - dispersion can differ
      between different batches * 'gene-label' - dispersion can differ between
      different labels * 'gene-cell' - dispersion can differ for every gene in
      every cell.
    reconstruction_loss: One of * 'nb' - Negative Binomial distribution. *
      'zinb' - Zero-Inflated Negative Binomial distribution.
    n_epochs: int, number of epochs to run, default 100.
    lr: float, learning rate, default 1e-3.
    use_batches: bool, whether to apply batch correction.
    use_cuda: bool, whether to use CUDA if available.

  Returns:
    latent: a numpy array with cooordiantes in the latent space.
    elbo_train_set: list of the ELBO on the train set.
    elbo_test_set: list of the ELBO on the test set.
  """

  # Train a model.
  vae = scvi.models.VAE(
      n_input=scvi_dataset.nb_genes,
      n_batch=scvi_dataset.n_batches * use_batches,
      n_latent=n_latent,
      n_layers=n_layers,
      n_hidden=n_hidden,
      dropout_rate=dropout_rate,
      dispersion=dispersion,
      reconstruction_loss=reconstruction_loss,
  )
  trainer = scvi.inference.UnsupervisedTrainer(
      vae, scvi_dataset, train_size=0.8, use_cuda=use_cuda, frequency=5)
  trainer.train(n_epochs=n_epochs, lr=lr)

  elbo_train_set = trainer.history['elbo_train_set']
  elbo_test_set = trainer.history['elbo_test_set']

  # Extract latent space
  posterior = trainer.create_posterior(
      trainer.model, scvi_dataset,
      indices=np.arange(len(scvi_dataset))).sequential()

  latent, _, _ = posterior.get_latent()

  return latent, elbo_train_set, elbo_test_set


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


def write_anndata(save_h5ad, adata, log_folder, conf, tissue, n_layers,
                  n_hidden, dispersion, dropout_rate, reconstruction_loss):
  """Writes anndata object with the proper name on GCS and returns the name."""
  # We need to write the anndata locally and copy it to GCS for the same
  # reason as before.
  if not save_h5ad:
    return ''

  with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=True) as tmp_file:
    adata.write(tmp_file.name)
    h5ad_fname = os.path.join(
        log_folder, f'{tissue}.method=scvi.seed={conf.seed}.'
        f'n_latent={conf.n_latent}.n_layers={n_layers}.n_hidden={n_hidden}.'
        f'dispersion={dispersion}.dropout_rate={dropout_rate}.'
        f'reconstruction_loss={reconstruction_loss}.epochs={conf.epochs}.'
        f'lr={conf.lr}.h5ad')
    tf.io.gfile.copy(tmp_file.name, h5ad_fname, overwrite=True)
  return h5ad_fname


def generate_conf(n_latent, epochs, lr, seed):
  """Generates the local parameter grid."""
  local_param_grid = {
      'n_latent': [2, 8, 10, 16, 32, 50, 64, 128]
                  if n_latent is None else [n_latent],
      'epochs': [20, 50, 100, 200, 300, 500, 1000]
                if epochs is None else [epochs],
      'lr': [1e-2, 1e-3, 1e-4] if lr is None else [lr],
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
                    n_latent=int(row['n_latent']),
                    epochs=int(row['epochs']),
                    lr=float(row['lr']),
                    seed=int(row['seed']),
                )))

  logging.info('Previous runs:')
  for run in previous_runs:
    logging.info(run)

  return previous_runs


def main(unused_argv):
  tissue, _ = os.path.splitext(os.path.basename(FLAGS.input_path))
  adata = fetch_anndata(FLAGS.input_path, FLAGS.from_gcs)
  sc.pp.filter_genes(adata, min_cells=1)
  scvi_dataset = scvi.dataset.AnnDatasetFromAnnData(adata)

  n_clusters = adata.obs['label'].nunique()

  confs = generate_conf(
      n_latent=FLAGS.n_latent,
      epochs=FLAGS.epochs,
      lr=FLAGS.lr,
      seed=FLAGS.seed)
  total_runs = len(confs)
  previous_runs = fetch_previous_runs(FLAGS.log_path)

  for i, conf in enumerate(confs):
    if str(conf) in previous_runs:
      logging.info('Skipped %s', conf)
      continue

    np.random.seed(conf.seed)
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    latent, elbo_train_set, elbo_test_set = compute_scvi_latent(
        scvi_dataset=scvi_dataset,
        n_latent=conf.n_latent,
        n_layers=FLAGS.n_layers,
        n_hidden=FLAGS.n_hidden,
        dropout_rate=FLAGS.dropout_rate,
        dispersion=FLAGS.dispersion,
        n_epochs=conf.epochs,
        lr=conf.lr,
        reconstruction_loss=FLAGS.reconstruction_loss,
    )
    adata.obsm['X_scvi'] = latent
    metrics = evaluate_method(adata, n_clusters)

    log_folder = os.path.dirname(FLAGS.output_csv)

    h5ad_fname = write_anndata(
        adata=adata,
        save_h5ad=FLAGS.save_h5ad,
        log_folder=log_folder,
        conf=conf,
        tissue=tissue,
        n_layers=FLAGS.n_layers,
        n_hidden=FLAGS.n_hidden,
        dispersion=FLAGS.dispersion,
        dropout_rate=FLAGS.dropout_rate,
        reconstruction_loss=FLAGS.reconstruction_loss)

    run_result = RunResult(
        method='scvi',
        seed=conf.seed,
        n_latent=conf.n_latent,
        n_layers=FLAGS.n_layers,
        n_hidden=FLAGS.n_hidden,
        dispersion=FLAGS.dispersion,
        dropout_rate=FLAGS.dropout_rate,
        reconstruction_loss=FLAGS.reconstruction_loss,
        epochs=conf.epochs,
        lr=conf.lr,
        silhouette=metrics.silhouette,
        kmeans_silhouette=metrics.kmeans_silhouette,
        kmeans_ami=metrics.ami,
        kmeans_ari=metrics.ari,
        n_cells=adata.n_obs,
        tissue=tissue,
        n_clusters=n_clusters,
        elbo_train_set=elbo_train_set[-1],
        elbo_test_set=elbo_test_set[-1],
        h5ad_fname=h5ad_fname)
    log_run(FLAGS.output_csv, run_result)

    logging.info(conf)
    logging.info('Done with %s out of %s', i, total_runs)
    log_run(FLAGS.log_path, conf)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_path')
  flags.mark_flag_as_required('output_csv')
  flags.mark_flag_as_required('log_path')
  flags.mark_flag_as_required('n_layers')
  flags.mark_flag_as_required('n_hidden')
  flags.mark_flag_as_required('dispersion')
  flags.mark_flag_as_required('dropout_rate')
  flags.mark_flag_as_required('reconstruction_loss')
  app.run(main)
