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

"""Resize bins for the 10X formatted dataset."""

import os
from typing import Sequence, Any

from absl import app
from absl import flags
import anndata
import pandas as pd
import scipy.io
import scipy.sparse
import tensorflow as tf

from schptm_benchmark import resize_bins_lib

FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', None, 'Path to the 10x formatted folder.')
flags.DEFINE_string('output_dir', None, 'Path to the output directory.')
flags.DEFINE_integer('binsize', None, 'Number of bp per bin (in kbp).')
flags.DEFINE_enum('mode', 'bins', ['bins', 'annotation'],
                  'Number of bp per bin (in kbp)')
flags.DEFINE_string('annotation', None, 'Path to the annotation.')


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


def save_anndata(adata, output_dir,
                 input_path):
  """Saves AnnData object in 10X format."""
  tf.io.gfile.makedirs(output_dir)
  with tf.io.gfile.GFile(os.path.join(output_dir, 'matrix.mtx'), mode='w') as f:
    scipy.io.mmwrite(f, adata.X.transpose())  # pytype: disable=attribute-error  # scipy
  new_bins = pd.DataFrame(adata.var_names, columns=['var_names'])
  with tf.io.gfile.GFile(os.path.join(output_dir, 'bins.tsv'), mode='w') as f:
    new_bins.to_csv(
        f,
        sep='\t',
        index=False,
        header=False,
        columns=['var_names', 'var_names'])
  tf.io.gfile.copy(
      os.path.join(input_path, 'barcodes.tsv'),
      os.path.join(output_dir, 'barcodes.tsv'),
      overwrite=True)


def main(argv):
  del argv

  adata = create_anndata(FLAGS.input_path)
  if FLAGS.mode == 'bins':
    adata = resize_bins_lib.merge_bins(adata, FLAGS.binsize * (10**3))
  elif FLAGS.mode == 'annotation':
    adata = resize_bins_lib.bins_from_annotation(adata, FLAGS.annotation)

  save_anndata(adata, FLAGS.output_dir, FLAGS.input_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_path')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('binsize')
  app.run(main)
