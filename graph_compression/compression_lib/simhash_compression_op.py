# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Simhash Matrix Compression operator."""

from __future__ import absolute_import
from __future__ import division

import copy

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import decompose_matrix


class SimhashMatrixCompressor(compression_op.LowRankDecompMatrixCompressor):
  """Simhash decomposition compressor.

  Implements matrix compression interface for the simhash algorithm
  from decompose_matrix.
  """

  def __init__(self, spec=None):
    """Initializer.

    Args:
      spec: hparams object with default value given by
        self.get_default_hparams().
    """
    super(SimhashMatrixCompressor, self).__init__(spec)
    self._spec.set_hparam('name', 'simhash_compressor')
    self._seed = 42

  def static_matrix_compressor(self, a_matrix):
    """Simhash decomposition of a_matrix.

    Args:
      a_matrix: input matrix.

    Returns:
      List [b_matrix] which is the simhash approximation of a_matrix. Rank is
      taken from spec.rank and interpreted to be a compression factor.
    """
    # Tag used for all print statements within this method.
    logging_tag = 'Inside simhash static_matrix_compressor:'

    # self._spec.rank can be considered a compression factor out of a 100,
    # where 100 corresponds to the full size of the original matrix. For
    # example, self._spec.rank of 200 means that the original rank is
    # compressed by a factor of 2.
    rank = np.int(np.floor(a_matrix.shape[1] * 100 / self._spec.rank))
    logging.info('%s compression factor, old rank, and new rank are %s, %s, %s',
                 logging_tag, self._spec.rank, a_matrix.shape[1], rank)

    r, s, d = decompose_matrix.np_simhash_decompose(
        a_matrix, rank, seed=self._seed)

    logging.info('%s r,s,d shapes are: %s, %s, %s', logging_tag, r.shape,
                 s.shape, d.shape)
    a_matrix_approx = np.dot(r, np.dot(s, d))
    logging.info('%s a_matrix_approx norm: %s', logging_tag,
                 np.linalg.norm(a_matrix_approx))
    return [a_matrix_approx.astype(np.float32)]

  def tpu_matrix_compressor(self, a_matrix):
    """Simhash decomposition of a_matrix using tpu operations.

    For training on tpus, we only use basic tf operations (as py_func is not
    supported).

    Args:
      a_matrix: input matrix.

    Returns:
      A list of one matrix [b_matrix] which is simhash approximation of
      a_matrix. Rank is taken from spec.rank and taken to be the compression
      factor.
    """
    # self._spec.rank can be considered a compression factor out of a 100,
    # where 100 corresponds to the full size of the original matrix. For
    # example, self._spec.rank of 200 means that the original rank is
    # compressed by a factor of 2.
    rank = ((a_matrix.shape[1] * 100) // self._spec.rank) + 1

    logging.info(
        'In tpu_matrix_compressor factor old and new rank are %s, %s, %s',
        self._spec.rank, a_matrix.shape[1], rank)

    r, s, d = decompose_matrix.tf_simhash_decompose(
        a_matrix, rank, seed=self._seed)

    a_matrix_approx = tf.matmul(r, tf.matmul(s, d))
    logging.info('Inside tpu_matrix_compressor: u,s,v shapes are: %s, %s, %s',
                 r.shape, s.shape, d.shape)

    return [a_matrix_approx]

  def default_matrix(self):
    """Returns default matrix of zeros of size specified in spec."""
    a_matrix = np.zeros(shape=[self._spec.num_rows, self._spec.num_cols])
    return a_matrix.astype(np.float32)


class SimhashCompressionOp(compression_op.CompressionOp):
  """Implements a simhash compression OP.

  Does this based on simhash compression algorithm by
  replacing a variable a_matrix with alpha * a_matrix + (1-alpha) * b_matrix.
  See doc referenced in the README for details.
  """

  def add_compression_summaries(self):
    """Adds summaries of alpha value and last update step."""
    with tf.name_scope(self._spec.name + '_summaries'):
      tf.summary.scalar('last_alpha_update_step', self._last_alpha_update_step)
      tf.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.summary.scalar(self.a_matrix_tfvar.op.name + '/a_matrix_norm',
                        tf.norm(self.a_matrix_tfvar))
      tf.summary.scalar(self.b_matrix_tfvar.op.name + '/b_matrix_norm',
                        tf.norm(self.b_matrix_tfvar))

  def _compressor_op(self, matrix_compressor, a_matrix_tfvar):
    """Creates compressor op based on simhash matrix_compressor.

    Meant to create the factors once at begin_compression_step tailored
    for simhash (which has only one output b_matrix).

    Args:
      matrix_compressor: specifies the matrix compressor object.
      a_matrix_tfvar: the tf tensor to be compressed.
    """
    # py_func is not supported on TPU so need non py_func implementation
    # The following line seems to be needed otherwise it says machines with tpu
    # don't support pyfunc.
    use_tpu = self._spec.use_tpu

    if use_tpu:
      [b_matrix_out] = matrix_compressor.tpu_matrix_compressor(a_matrix_tfvar)
    else:
      [b_matrix_out] = tf.py_func(matrix_compressor.static_matrix_compressor,
                                  [a_matrix_tfvar], [tf.float32])

    b_matrix_assign_op = tf.assign(
        self.b_matrix_tfvar, b_matrix_out, name='_b_matrix_assign_op')
    with tf.control_dependencies([b_matrix_assign_op]):
      return tf.no_op('compresor_b_matrix_update')

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               simhash_compressor,
                               scope='default_scope'):
    """Returns simhash compressed tensorflow operator.

    Does this for variable a_matrix_tfvar for compression method specified in
    simhash_compressor by replacing a variable a_matrix with
    alpha * a_matrix + (1-alpha) * b_matrix.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      simhash_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = simhash_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)

    [a_matrix_approx] = simhash_compressor.static_matrix_compressor(a_matrix)

    with tf.variable_scope(scope):
      self.b_matrix_tfvar = tf.get_variable(
          'B',
          dtype=tf.float32,
          initializer=a_matrix_approx.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      self.alpha = tf.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

    self.a_matrix_tfvar = a_matrix_tfvar

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * self.b_matrix_tfvar

    self.update_op = self._create_update_op()

    self.add_compression_summaries()
    return [self.final_op, self.update_op]


class SimhashApplyCompression(compression_op.ApplyCompression):
  """Wrapper class for Simhash.

  This is to repeatedly invoke above compression operator to different
  layers in a model.

  Intialized by specifying the compressor and compression_spec.

  After that apply_compression can be called several times for different
  matrices in the model.

  Finally all_update_op returns the combined update OP from all these
  compressions.

  Adds random_shift's to the begin_compression_step to stagger the
  compression of the different matrices being compressed.
  """

  def __init__(self, scope, compression_spec, compressor, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      compression_spec: compression hyper parameters.
      compressor: matrix compressor object of class MatrixCompressorInferface.
      global_step: tf variable that has the global step.
    """
    super(SimhashApplyCompression, self).__init__(
        scope=scope,
        compression_spec=compression_spec,
        compressor=compressor,
        global_step=global_step)
    self._compression_op_spec_orig = copy.deepcopy(self._compression_op_spec)

  def apply_compression(self, a_matrix_tfvar, scope='default_scope'):
    """Applies matrix compression OP on a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that represents the compressed version of a_matrix_tfvar.
    """
    orig_spec = self._compression_op_spec_orig
    delta = orig_spec.end_compression_step - orig_spec.begin_compression_step
    random_shift = np.random.randint(0, np.round(delta / 2))

    self._compression_op_spec.set_hparam(
        'begin_compression_step',
        self._compression_op_spec_orig.begin_compression_step + random_shift)

    logging.info('random_shift is %s', random_shift)
    logging.info('New and old begin_compression_step are: %s, %s',
                 self._compression_op_spec.begin_compression_step,
                 self._compression_op_spec_orig.begin_compression_step)

    c = SimhashCompressionOp(
        spec=self._compression_op_spec, global_step=self._global_step)
    self._compression_ops.append(c)
    [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
        a_matrix_tfvar, self._matrix_compressor, scope=scope)
    self._update_ops.append(a_matrix_update_op)

    return a_matrix_compressed
