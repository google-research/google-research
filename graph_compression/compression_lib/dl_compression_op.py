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

"""Dictionary learning compressor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from dictionary_learning import dictionary_learning
from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import compression_op_utils

FLAGS = flags.FLAGS


class DLMatrixCompressor(compression_op.LowRankDecompMatrixCompressor):
  """Implements a dictionary learning compression OP."""

  def __init__(self, spec=None):
    compression_op.LowRankDecompMatrixCompressor.__init__(self, spec=spec)
    self._seed = 42

  def static_matrix_compressor(self, matrix, n_iterations=1):
    """Performance dictionary learning on an input matrix.

    Args:
      matrix: input matrix, numpy 2d array;
      n_iterations: number of iterations to performance in dictionary learning,
        int.

    Returns:
      code: code matrix, numpy 2d array, see dictionary_learning module for more
        details;
      dictionary: dictionary matrix, numpy 2d array, see dictionary_learning
        module for more details.
    """
    logging.info(
        'Inside dl static_matrix_compressor: matrix shape is %s norm is %d: ',
        matrix.shape, np.linalg.norm(matrix))
    logging.info(self._spec.to_json())
    print('matrix.shape: ', matrix.shape)
    [code, dictionary] = dictionary_learning.dictionary_learning(
        matrix,
        row_percentage=100 / self._spec.rank,
        col_percentage=100 / self._spec.rank,
        n_iterations=n_iterations,
        seed=15,
        use_lsh=self._spec.use_lsh)

    logging.info(
        'Inside dl static_matrix_compressor: code, dictionary shapes are: %s %s',
        code.shape, dictionary.shape)
    col_percentage = 100 / self._spec.rank
    self.uncompressed_size = matrix.size
    self.compressed_size = int(code.size * col_percentage) + dictionary.size

    print(
        'Inside dl_matrix_compressor: a_matrix,b_matrix,c_matrix shapes are: ',
        matrix.shape, code.shape, dictionary.shape,
        '; compressed and uncompressed size are: ', self.uncompressed_size,
        self.compressed_size)

    return [code.astype(np.float32), dictionary.astype(np.float32)]

  # for tpu since py_func is not supported we use tf operations only
  def tpu_matrix_compressor(self, matrix):
    # Not implemented.
    return 0

  def default_matrix(self):
    return np.zeros(shape=[self._spec.num_rows, self._spec.num_cols])


class DLCompressionOp(compression_op.CompressionOp):
  """Implements a dictionary learning compression OP.

  Does this based on dictionary learning compression algorithm by
  replacing a variable a_matrix with alpha * a_matrix + (1-alpha) *
  tf.matmul(b_matrix, c_matrix).

  See doc referenced in the README for details.
  """

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator.

    Does this for variable a_matrix_tfvar for
    compression method specified in matrix_compressor that must be based on some
    matrix factorization compression algorithm by replacing a variable
    a_matrix by alpha*a_matrix + (1-alpha)b_matrix*c_matrix

    Args:
      a_matrix_tfvar: TF variable representihg a tensor variable in a model
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      scope: TF scope used for creating new TF variables

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)
    # Get nonzero indices and values.
    b_matrix_indices = np.transpose(np.nonzero(b_matrix))
    b_matrix_values = b_matrix[np.nonzero(b_matrix)]

    self.uncompressed_size = matrix_compressor.uncompressed_size
    self.compressed_size = matrix_compressor.compressed_size

    with tf.variable_scope(scope):
      self.b_matrix_indices_tfvar = tf.get_variable(
          'b_matrix_indices_tfvar',
          dtype=tf.int64,
          initializer=b_matrix_indices.astype(np.int64),
          trainable=False)
      self.b_matrix_values_tfvar = tf.get_variable(
          'b_matrix_values_tfvar',
          dtype=tf.float32,
          initializer=b_matrix_values.astype(np.float32),
          trainable=True)

      self.b_matrix_tfvar = tf.SparseTensor(
          indices=self.b_matrix_indices_tfvar,
          values=self.b_matrix_values_tfvar,
          dense_shape=tf.constant(b_matrix.shape, dtype=tf.int64))

      self.c_matrix_tfvar = tf.get_variable(
          'c_matrix',
          dtype=tf.float32,
          initializer=c_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.alpha = tf.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

    self.a_matrix_tfvar = a_matrix_tfvar

    if self._spec.update_option == compression_op_utils.UpdateOptions.TF_UPDATE:
      self.update_op = self._create_update_op()
    else:
      self.setup_update_explicit()

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.b_matrix_tfvar,
                                                        self.c_matrix_tfvar)

    self.add_compression_summaries()
    logging.info('DL compressor: get_apply_compression.')
    return [self.final_op, self.update_op]

  def get_apply_embedding_lookup(self, ids):
    """Returns compressed tensorflow operator for embedding_lookup.

    This method returns a TensorFlow node that performs embedding lookup as
    alpha * tf.nn.embedding_lookup(a_matrix_tfvar, ids) +
    (1 - alpha) * tf.nn.embedding_lookup(b_matrix_tfvar, ids) if c_matrix is
    not presented, and alpha * tf.nn.embedding_lookup(a_matrix_tfvar, ids) +
    (1 - alpha) * tf.matmul(tf.nn.embedding_lookup(b_matrix_tfvar, ids),
    c_matrix) if c_matrix is presented, where b_matrix_tfvar and c_matrix_tfvar
    are the factor matrices for the compressed embedding table.

    Args:
      ids: A Tensor with type int32 or int64 containing the ids to be looked up
        in the embedding table (the a_matrix_tfvar variable).

    Returns:
      embedding_op: a TensorFlow node that performs compressed embedding lookup.
    """
    logging.info('DL compressor: get_apply_embedding_lookup.')
    if self.matrix_compressor.get_spec().is_c_matrix_present:
      embedding_op = self.alpha * tf.nn.embedding_lookup(
          self.a_matrix_tfvar, ids) + (1 - self.alpha) * tf.matmul(
              tf.nn.embedding_lookup(
                  tf.sparse.to_dense(self.b_matrix_tfvar), ids),
              self.c_matrix_tfvar)
    else:
      embedding_op = self.alpha * tf.nn.embedding_lookup(
          self.a_matrix_tfvar, ids) + (1 - self.alpha) * tf.nn.embedding_lookup(
              tf.sparse.to_dense(self.b_matrix_tfvar), ids)

    return embedding_op

  def get_apply_matmul(self, left_operand):
    """Returns compressed TensorFlow node for matmul.

    This method performs matmul (on the right) with the compressed matrix.

    Args:
      left_operand: a Tensor that is the left operand in matmul.

    Returns:
      matmul_op: a TensorFlow node that performs matmul of left_operand with the
      compressed a_matrix_tfvar.
    """
    # Applies matmul on the right
    logging.info('DL compressor: get_apply_matmul.')
    if self.matrix_compressor.get_spec().is_c_matrix_present:
      matmul_op = self.alpha * tf.matmul(input, tf.transpose(
          self.a_matrix_tfvar)) + (1 - self.alpha) * tf.matmul(
              tf.matmul(input, tf.transpose(self.c_matrix_tfvar)),
              tf.sparse.to_dense(self.b_matrix_tfvar),
              transpose_b=True,
              b_is_sparse=True)
    else:
      matmul_op = self.alpha * tf.matmul(input, tf.transpose(
          self.a_matrix_tfvar)) + (1 - self.alpha) * tf.matmul(
              input,
              tf.sparse.to_dense(self.b_matrix_tfvar),
              transpose_b=True,
              b_is_sparse=True)

    return matmul_op

  def add_compression_summaries(self):
    """Adds summaries of alpha value, new variables, and last update step."""
    with tf.name_scope(self._spec.name + '_summaries'):
      tf.compat.v2.summary.scalar(
          self._last_alpha_update_step.op.name + '/last_alpha_update_step',
          self._last_alpha_update_step)
      tf.compat.v2.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.compat.v2.summary.scalar(
          self.a_matrix_tfvar.op.name + '/a_matrix_norm',
          tf.norm(self.a_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          self.b_matrix_indices_tfvar.op.name + '/b_matrix_indices_size',
          tf.size(self.b_matrix_indices_tfvar))
      tf.compat.v2.summary.scalar(
          self.b_matrix_values_tfvar.op.name + '/b_matrix_values_norm',
          tf.norm(self.b_matrix_values_tfvar))
      tf.compat.v2.summary.scalar(
          self.c_matrix_tfvar.op.name + '/c_matrix_norm',
          tf.norm(self.c_matrix_tfvar))

  def run_update_step(self, session, step_number=None):
    """Returns the combine update tf OP."""
    logging.info('running run_update_step self._global_step is %s name is %s',
                 self._global_step, self.a_matrix_tfvar.op.name)
    if step_number is None:
      if self._spec.run_update_interval_check != 0:
        print('running run_update_step self._global_step is %s',
              self._global_step)
        logging.info(
            'running run_update_step step_num is null self.globalstep is %s',
            self._global_step)
        step_number = session.run(self._global_step)
        print('running run_update_step step_num is %s', step_number)
        logging.info('running run_update_step step_num is %s', step_number)
      else:
        step_number = 1

    logging.info(
        'In compression op.run_update_step: '
        'step_number is %s begin end update_count are: %s %s %s ',
        step_number, self._spec.begin_compression_step,
        self._spec.end_compression_step, self.run_update_count)
    if (step_number >= self._spec.begin_compression_step and
        step_number < self._spec.end_compression_step):
      logging.info(
          'In compression op.run_update_step: '
          'step_number is %s begin end update_count are: %s %s %s ',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count)
      self.run_update_count = self.run_update_count + 1
      logging.info('inside compression interval')

      # Need to persist these python state variables in TF as if a task gets
      # aborted things get out of sync.
      self._last_update_step = session.run(self._last_alpha_update_step)
      logging.info(
          'In compression op.run_update_step:'
          'step_number is %s begin end update_count,'
          'last_alpha last_alpha_update are: %s %s %s %s',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count,
          self._last_update_step)
      if self._last_update_step == -1:
        logging.info(
            'In compression op.run_update_step:'
            'step_number is %s begin end update_count are: %s %s %s ',
            step_number, self._spec.begin_compression_step,
            self._spec.end_compression_step, self.run_update_count)
        print('inside compression interval: initial decomposition step')
        a_matrix = session.run(self.a_matrix_tfvar)
        logging.info(
            'In compression op.run_update_step  a_matrix.shape is %s norm is %d',
            a_matrix.shape, np.linalg.norm(a_matrix))
        if self.matrix_compressor.get_spec().is_c_matrix_present:
          logging.info(
              'In compression op.run_update_step:'
              'step_number is %s begin end update_count are: %s %s %s ',
              step_number, self._spec.begin_compression_step,
              self._spec.end_compression_step, self.run_update_count)
          [b_matrix,
           c_matrix] = self.matrix_compressor.static_matrix_compressor(a_matrix)
          b_matrix_indices = np.transpose(np.nonzero(b_matrix))
          b_matrix_values = b_matrix[np.nonzero(b_matrix)]

          session.run(
              tf.assign(
                  self.b_matrix_indices_tfvar,
                  b_matrix_indices,
                  validate_shape=False))
          session.run(
              tf.assign(
                  self.b_matrix_values_tfvar,
                  b_matrix_values,
                  validate_shape=False))
          session.run(tf.assign(self.c_matrix_tfvar, c_matrix))
        else:
          [b_matrix] = self.matrix_compressor.static_matrix_compressor(a_matrix)
          session.run(tf.assign(self.b_matrix_tfvar, b_matrix))
      logging.info(
          'In compression op.run_update_step:'
          'step_number is %s begin end update_count are: %s %s %s ',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count)

      alpha = session.run(self.alpha)
      self.last_alpha_value = alpha
      new_alpha = max(alpha - self._spec.alpha_decrement_value, 0)
      if self.last_alpha_value > 0:
        make_a_zero = False
        if make_a_zero and new_alpha == 0:
          logging.info('Making a_matrix all zero for %s',
                       self.a_matrix_tfvar.op.name)
          a_matrix = np.zeros(shape=self.a_matrix_tfvar.shape)
          session.run(tf.assign(self.a_matrix_tfvar, a_matrix))
        logging.info('in run_update_step decrementing alpha, alpha value is %d',
                     self.last_alpha_value)
        logging.info(
            'running run_update_step self._global_step '
            'is %s new and old alpha are %d %d',
            self._global_step, alpha, new_alpha)
        session.run(tf.assign(self.alpha, new_alpha))

        self.last_alpha_value = new_alpha
        self._last_update_step = step_number
        session.run(tf.assign(self._last_alpha_update_step, step_number))
    logging.info(
        'In compression op.run_update_step:'
        'step_number is %s begin end update_count are: %s %s %s ',
        step_number, self._spec.begin_compression_step,
        self._spec.end_compression_step, self.run_update_count)

