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

"""Simhash Matrix Compression operator."""

from __future__ import absolute_import
from __future__ import division

import copy

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from graph_compression.compression_lib import compression_op
from graph_compression.compression_lib import compression_op_utils
from graph_compression.compression_lib import decompose_matrix
from graph_compression.compression_lib import kmeans_quantize
from model_pruning.python import hparam
from model_pruning.python import pruning


class SimhashMatrixCompressor(compression_op.LowRankDecompMatrixCompressor):
  """Simhash decomposition compressor.

  Implements matrix compression interface for the simhash algorithm
  from decompose_matrix.
  """

  def __init__(self, spec=None):
    """Initializer.

    Args:
      spec: hparams object with default value given by
        self.get_default_hparams()
    """
    super(SimhashMatrixCompressor, self).__init__(spec)
    self._spec.set_hparam('name', 'simhash_compressor')
    self._spec.is_c_matrix_present = False
    self._spec.is_b_matrix_trainable = False
    self._seed = 42

  def static_matrix_compressor(self, a_matrix):
    """Simhash decomposition of a_matrix.

    Args:
      a_matrix: input matrix

    Returns:
      List [b_matrix] which is the simhash approximation of a_matrix. Rank is
      taken from spec.rank and interpreted to be a compression factor.
    """
    # Tag used for all print statements within this method.
    logging_tag = 'Inside simhash static_matrix_compressor:'

    rank = ((np.min(a_matrix.shape) * 100) // self._spec.rank) + 1
    tf.logging.info('%s compression factor, old rank, and new rank '
                    'are %s %s %s',
                    logging_tag,
                    self._spec.rank,
                    a_matrix.shape[1], rank)

    r, s, d = decompose_matrix.np_simhash_decompose(
        a_matrix, rank, seed=self._seed)

    self.uncompressed_size = np.size(a_matrix)
    self.compressed_size = np.size(r)

    logging.info(
        '%s r,s,d shapes are: %s, %s, %s, compressed and uncompressed size are %s %s',
        logging_tag, r.shape, s.shape, d.shape, self.uncompressed_size,
        self.compressed_size)
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
    with tf.compat.v1.name_scope(self._spec.name + '_summaries'):
      tf.compat.v2.summary.scalar('last_alpha_update_step',
                                  self._last_alpha_update_step)
      tf.compat.v2.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.compat.v2.summary.scalar(
          self.a_matrix_tfvar.op.name + '/a_matrix_norm',
          tf.norm(tensor=self.a_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          self.b_matrix_tfvar.op.name + '/b_matrix_norm',
          tf.norm(tensor=self.b_matrix_tfvar))

  def _compressor_op(self, matrix_compressor, a_matrix_tfvar):
    """Creates compressor op based on simhash matrix_compressor.

    Meant to create the factors once at begin_compression_step tailored
    for simhash (which has only one output b_matrix).

    Args:
      matrix_compressor: specifies the matrix compressor object.
      a_matrix_tfvar: the tf tensor to be compressed.

    Returns:
      a tf.no_op object with assign ops as control dependencies.
    """
    # py_func is not supported on TPU so need non py_func implementation
    # The following line seems to be needed otherwise it says machines with tpu
    # don't support pyfunc.
    use_tpu = self._spec.use_tpu

    if use_tpu:
      [b_matrix_out] = matrix_compressor.tpu_matrix_compressor(a_matrix_tfvar)
    else:
      [b_matrix_out
      ] = tf.compat.v1.py_func(matrix_compressor.static_matrix_compressor,
                               [a_matrix_tfvar], [tf.float32])

    b_matrix_assign_op = tf.compat.v1.assign(
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

    with tf.compat.v1.variable_scope(scope):
      self.b_matrix_tfvar = tf.compat.v1.get_variable(
          'B',
          dtype=tf.float32,
          initializer=a_matrix_approx.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      self.alpha = tf.compat.v1.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

    self.a_matrix_tfvar = a_matrix_tfvar

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * self.b_matrix_tfvar

    if self._spec.update_option == 0:
      self.update_op = self._create_update_op()
    else:
      self.setup_update_explicit()

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          simhash_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope'):
    """Returns simhash compressed tensorflow operator for a customized layer.

    Does this for variable a_matrix_tfvar for compression method specified in
    simhash_compressor by replacing a variable a_matrix with
    alpha * a_matrix + (1-alpha) * b_matrix.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      simhash_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customized layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = simhash_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)

    [a_matrix_approx] = simhash_compressor.static_matrix_compressor(a_matrix)

    p = layer_obj.params
    with tf.variable_scope(scope) as scope:
      b_matrix_pc = weight_params_fn(a_matrix_approx.shape,
                                     weight_init_obj.Constant(1.0), p.dtype)
      alpha_pc = weight_params_fn([], weight_init_obj.Constant(1.0), tf.float32)

      layer_obj.CreateVariable(
          'alpha', alpha_pc, theta_fn=None, trainable=False)
      layer_obj.CreateVariable(
          'b_matrix_tfvar',
          b_matrix_pc,
          theta_fn=None,
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)

    self.b_matrix_tfvar = layer_obj.vars.b_matrix_tfvar
    self.alpha = layer_obj.vars.alpha
    self.a_matrix_tfvar = a_matrix_tfvar

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * self.b_matrix_tfvar

    if self._spec.update_option == 0:
      self.update_op = self._create_update_op()
    else:
      self.setup_update_explicit()

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_mix_operator(self, theta, concat):
    """Performs matrix multiplication for customized layer.

    This performs the compressed equivalent of tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    return tf.matmul(concat, (theta.alpha * theta.wm +
                              (1 - theta.alpha) * theta.b_matrix_tfvar))


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
    logging.info('New and old begin_compression_step are: %s, %s',
                 self._compression_op_spec.begin_compression_step,
                 self._compression_op_spec_orig.begin_compression_step)

    if self._compression_op_spec.compression_option == 4:
      c = KMeansCompressionOp(
          spec=self._compression_op_spec, global_step=self._global_step)
    elif self._compression_op_spec.compression_option == 8:
      c = KMeansPruningCompressionOp(
          spec=self._compression_op_spec, global_step=self._global_step)
    else:
      c = SimhashCompressionOp(
          spec=self._compression_op_spec, global_step=self._global_step)

    self._compression_ops.append(c)
    [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
        a_matrix_tfvar, self._matrix_compressor, scope=scope)
    self._update_ops.append(a_matrix_update_op)

    self.uncompressed_size = self.uncompressed_size + c.uncompressed_size
    self.compressed_size = self.compressed_size + c.compressed_size

    return a_matrix_compressed

  def customized_apply_compression(self,
                                   a_matrix_tfvar,
                                   layer_obj,
                                   weight_params_fn,
                                   weight_init_obj,
                                   scope='default_scope',
                                   spec=None):
    """Applies matrix compression OP on a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      layer_obj: a customeried layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      spec: spec to be used for the compression op. this is optional.
            if not provided, self._compression_op_spec is used.

    Returns:
      A TF node that represents the compressed version of a_matrix_tfvar.
    """
    compression_op_spec = spec if spec else self._compression_op_spec
    if compression_op_spec.compression_option == 4:
      c = KMeansCompressionOp(
          spec=compression_op_spec, global_step=self._global_step)
    elif compression_op_spec.compression_option == 8:
      c = KMeansPruningCompressionOp(
          spec=compression_op_spec, global_step=self._global_step)
    else:
      c = SimhashCompressionOp(
          spec=compression_op_spec, global_step=self._global_step)

    self._compression_ops.append(c)
    [a_matrix_compressed,
     a_matrix_update_op] = c.get_customized_apply_compression_op(
         a_matrix_tfvar,
         self._matrix_compressor,
         layer_obj,
         weight_params_fn,
         weight_init_obj,
         scope=scope)
    self._update_ops.append(a_matrix_update_op)

    self.uncompressed_size += c.uncompressed_size
    self.compressed_size += c.compressed_size

    return a_matrix_compressed

  def get_mix_operator(self, theta, concat):
    """Return mixed operator.

    See KmeansPruningCompressionOp.get_mix_operator for details.

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    return self._compression_ops[-1].get_mix_operator(theta, concat)

  def get_update_ops(self):
    for c in self._compression_ops:
      self._update_ops.append(c.get_update_op())
    return self._update_ops

  def all_update_op(self):
    _ = self.get_update_ops()
    self._all_update_op = compression_op.CompressionOp.all_update_op(
        self._update_ops, self._scope)
    return self._all_update_op


class KmeansMatrixCompressor(compression_op.LowRankDecompMatrixCompressor):
  """K-means decomposition compressor.

  Implements matrix compression interface for the kmeans quantize algorithm.
  """

  def __init__(self, spec=None):
    """Initializer.

    Args:
      spec: hparams object with default value given by
        self.get_default_hparams()
    """
    compression_op.LowRankDecompMatrixCompressor.__init__(self, spec=spec)
    self._spec.set_hparam('name', 'kmeans_compressor')
    # c_matrix is the encoding array, which is untrainable.
    self._spec.is_c_matrix_trainable = False
    self._seed = 42

  def static_matrix_compressor(self, a_matrix):
    """K-means decomposition of a_matrix.

    Args:
      a_matrix: input matrix

    Returns:
      [codebook, a_matrix_encoding]: rows of codebook are centroid vectors, and
      a_matrix_encoding is an array of centroid indices for blocks in a_matrix.
    """
    [codebook, a_matrix_encoding] = kmeans_quantize.kmeans_quantize_block(
        a_matrix,
        levels=self._spec.rank,
        pruning_factor=self._spec.pruning_fraction,
        block_size=self._spec.block_size,
        is_padded=True)
    return [codebook, a_matrix_encoding]


class KMeansCompressionOp(compression_op.CompressionOp):
  """Implements a kmeans compression OP."""

  def add_compression_summaries(self):
    """Adds summaries of alpha value and last update step."""
    with tf.name_scope(self._spec.name + '_summaries'):
      logging.info('add_compression_summaries scope name is %s',
                   self._spec.name)
      tf.compat.v2.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.compat.v2.summary.scalar(
          self.a_matrix_tfvar.op.name + '/a_matrix_norm',
          tf.norm(self.a_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          self.b_matrix_tfvar.op.name + '/b_matrix_norm',
          tf.norm(tf.reshape(self.b_matrix_tfvar, [-1]), ord=1))
      tf.compat.v2.summary.scalar(
          self.c_matrix_tfvar.op.name + '/c_matrix_norm',
          tf.reduce_sum(self.c_matrix_tfvar))

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator - kmeans.

    Replaces a_matrix by alpha * a_matrix + (1 - alpha) *
    tf.nn.embedding(b_matrix, c_matrix).

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

    self.uncompressed_size = matrix_compressor.uncompressed_size
    self.compressed_size = matrix_compressor.compressed_size

    with tf.variable_scope(scope):
      self.b_matrix_tfvar = tf.get_variable(
          'b_matrix',
          dtype=tf.float32,
          initializer=b_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)

      # Use uint8 if number of k-centers is small enough.
      if self._spec.rank <= 256:
        c_matrix_tfvar_dtype = tf.uint8
        c_matrix_type = np.uint8
      else:
        c_matrix_tfvar_dtype = tf.int32
        c_matrix_type = np.int32

      self.c_matrix_tfvar = tf.get_variable(
          'c_matrix',
          dtype=c_matrix_tfvar_dtype,
          initializer=c_matrix.astype(c_matrix_type),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.alpha = tf.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

      self.a_matrix_tfvar = a_matrix_tfvar

      if self._spec.update_option == 0:
        self.update_op = self._create_update_op()
      else:
        self.update_op = self.setup_update_explicit()

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.reshape(
            tf.nn.embedding_lookup(self.b_matrix_tfvar,
                                   tf.cast(self.c_matrix_tfvar, tf.int32)),
            a_matrix_tfvar.shape)

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope'):
    """Returns kmeans compressed tensorflow operator for a customized layer.

    Replaces a_matrix by alpha * a_matrix + (1 - alpha) *
    tf.nn.embedding(b_matrix, c_matrix).

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customeried layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)

    self.uncompressed_size = matrix_compressor.uncompressed_size
    self.compressed_size = matrix_compressor.compressed_size

    # Use uint8 if number of k-centers is small enough.
    c_matrix_tfvar_dtype = tf.int32
    if self._spec.rank <= 256:
      c_matrix_tfvar_dtype = tf.uint8

    p = layer_obj.params
    with tf.variable_scope(scope) as scope:
      b_matrix_pc = weight_params_fn(b_matrix.shape,
                                     weight_init_obj.Constant(1.0), p.dtype)
      c_matrix_pc = weight_params_fn(c_matrix.shape,
                                     weight_init_obj.Constant(1),
                                     c_matrix_tfvar_dtype)
      alpha_pc = weight_params_fn([], weight_init_obj.Constant(1.0), tf.float32)

      layer_obj.CreateVariable(
          'alpha', alpha_pc, theta_fn=None, trainable=False)
      layer_obj.CreateVariable(
          'b_matrix_tfvar',
          b_matrix_pc,
          theta_fn=None,
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      layer_obj.CreateVariable(
          'c_matrix_tfvar',
          c_matrix_pc,
          theta_fn=None,
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)

      self.b_matrix_tfvar = layer_obj.vars.b_matrix_tfvar
      self.c_matrix_tfvar = layer_obj.vars.c_matrix_tfvar
      self.alpha = layer_obj.vars.alpha
      self.a_matrix_tfvar = a_matrix_tfvar

      if self._spec.update_option == 0:
        self.update_op = self._create_update_op()
      else:
        self.update_op = tf.no_op()

    self.final_op = self.a_matrix_tfvar

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_mix_operator(self, theta, concat):
    """Performs matrix multiplication for customized layer.

    This performs the compressed equivalent of tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    return (
        theta.alpha * tf.matmul(concat, theta.wm) +
        (1 - theta.alpha) * tf.matmul(
            concat,
            tf.nn.embedding_lookup(theta.b_matrix_tfvar,
                                   tf.cast(theta.c_matrix_tfvar, tf.int32))))

  def flat_embedding_lookup(self, emb_table, flat_ids, vocab_size,
                            matmul_axis=1,
                            fprop_type='matmul'):
    if fprop_type == 'matmul':
      lhs = tf.equal(
          tf.expand_dims(flat_ids, matmul_axis),
          tf.cast(tf.range(vocab_size, dtype=tf.int32), flat_ids.dtype))
      if emb_table.dtype == tf.uint8:
        return tf.cast(
            tf.matmul(tf.cast(lhs, tf.int32), tf.cast(emb_table, tf.int32)),
            tf.uint8)
      else:
        return tf.matmul(tf.cast(lhs, emb_table.dtype), emb_table)
    else:
      return tf.nn.embedding_lookup(emb_table, tf.cast(flat_ids, tf.int32))

  def get_embedding_lookup_operator(self, theta, flat_ids, fprop_type='matmul'):
    """Perform gather based embedding lookup.

    Args:
      theta: layer parameter class, theta should have an attribute theta.wm
        which is the embedding table.
      flat_ids: one dimensional tensor of ids, tf.Tensor of tf.int32 type.
      fprop_type: embedding lookup type: should be 'matmul' or  'gather'.

    Returns:
      Compressed version of tf.nn.embedding_lookup(theta.wm, flat_ids).
    """
    single_emb_result = self.flat_embedding_lookup(theta.wm, flat_ids,
                                                   theta.wm.shape[0],
                                                   matmul_axis=1,
                                                   fprop_type=fprop_type)
    double_emb_result = self.flat_embedding_lookup(
        theta.b_matrix_tfvar,
        self.flat_embedding_lookup(theta.c_matrix_tfvar, flat_ids,
                                   theta.c_matrix_tfvar.shape[0],
                                   matmul_axis=1,
                                   fprop_type=fprop_type),
        256,
        matmul_axis=2,
        fprop_type=fprop_type)
    double_emb_result = compression_op_utils.flatten_last_dims(
        double_emb_result, ndims=2)

    return (theta.alpha * single_emb_result +
            (1 - theta.alpha) * double_emb_result)


class KMeansPruningCompressionOp(compression_op.CompressionOp):
  """Implements a kmeans compression OP.

  This op performs pruning followed by kmeans vector quantization.
  """

  def __init__(self, scope='default_scope', spec=None, global_step=None):
    super(KMeansPruningCompressionOp, self).__init__(scope, spec, global_step)

    pruning_spec = copy.deepcopy(self._spec)
    pruning_spec.prune_option = 'weight'
    self.pruning_obj = pruning.Pruning(
        pruning_spec, global_step=self._global_step)

  @staticmethod
  def get_default_hparams():
    """Get a tf.HParams object with the default values for the hyperparameters.

      name: string
        name of the compression specification. Used for adding summaries and ops
        under a common tensorflow name_scope.
      alpha_decrement_value: float
        a positive real number by which alpha is decremented at each update.
      begin_compression_step: integer
        the global step at which to begin compression.
      end_compression_step: integer
        the global step at which to terminate compression. Defaults to -1
        implying that compression continues till the training stops.
      use_tpu: False
        indicates whether to use TPU.
      compression_option: integer
        indicates what type of factorization (if any) is used.
      rank: integer
        indicates what type of factorization (if any) is used.
      update_option: integer
        indicates how the update logic is being run. More specifically:
        0 - run the update logic in TF; needed when using GPU/TPU.
        1 - run the update logic in regular python as opposed to TF.
        2 - run the update logic in TF and in regular python.
      TODO(wanxin): add doc strings for pruning hparams.

    Returns:
      tf.HParams object initialized to default values.

    """
    return hparam.HParams(
        name='model_compression',
        alpha_decrement_value=0.01,
        begin_compression_step=0,
        end_compression_step=-1,
        compression_frequency=10,
        use_tpu=False,
        compression_option=0,
        rank=7,
        update_option=2,
        run_update_interval_check=1,
        block_size=1,
        pruning_fraction=0.0,
        begin_pruning_step=0,
        end_pruning_step=-1,
        weight_sparsity_map=[''],
        block_dims_map=[''],
        threshold_decay=0.0,
        pruning_frequency=10,
        nbins=256,
        block_height=1,
        block_width=1,
        block_pooling_function='AVG',
        initial_sparsity=0.0,
        target_sparsity=0.5,
        sparsity_function_begin_step=0,
        sparsity_function_end_step=100,
        sparsity_function_exponent=3.0,
        gradient_decay_rate=0.99,
        prune_option='weight',
        do_transpose=False)

  def add_compression_summaries(self):
    """Adds summaries of alpha value and last update step."""
    with tf.name_scope(self._spec.name + '_summaries'):
      logging.info('add_compression_summaries scope name is %s',
                   self._spec.name)
      tf.compat.v2.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.compat.v2.summary.scalar(
          self.a_matrix_tfvar.op.name + '/a_matrix_norm',
          tf.norm(self.a_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          self.b_matrix_tfvar.op.name + '/d_matrix_norm',
          tf.norm(tf.reshape(self.b_matrix_tfvar, [-1]), ord=1))
      tf.compat.v2.summary.scalar(
          self.c_matrix_tfvar.op.name + '/c_matrix_norm',
          tf.reduce_sum(self.c_matrix_tfvar))

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator - kmeans.

    Replaces a_matrix by alpha * a_matrix + (1 - alpha) *
    tf.nn.embedding(b_matrix, c_matrix).

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
    if getattr(self._spec, 'do_transpose', False):
      a_matrix = np.transpose(a_matrix)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)

    self.uncompressed_size = matrix_compressor.uncompressed_size
    self.compressed_size = matrix_compressor.compressed_size

    with tf.variable_scope(scope):
      self.b_matrix_tfvar = tf.get_variable(
          'b_matrix',
          dtype=tf.float32,
          initializer=b_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      self.c_matrix_tfvar = tf.get_variable(
          'c_matrix',
          dtype=tf.int32,
          initializer=c_matrix.astype(np.int32),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.alpha = tf.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

      self.a_matrix_tfvar = a_matrix_tfvar
      [self.pruned_a_matrix_tfvar, self.mask] = pruning.apply_mask_and_return(
          self.a_matrix_tfvar, scope)

    def maybe_apply_compression():
      """Decide whether global step is within compression range.

      Returns:
      is_step_within_compression_range: bool.
      """
      with tf.compat.v1.name_scope(self._spec.name):
        global_step = self._global_step
        def real_global_step_fn():
          return global_step
        def mock_global_step_fn():
          return self._spec.begin_compression_step
        global_step = tf.cond(
            tf.constant(global_step is None, dtype=tf.bool),
            mock_global_step_fn,
            real_global_step_fn)
        is_step_within_compression_range = tf.logical_and(
            tf.greater_equal(
                tf.cast(global_step, tf.int32),
                self._spec.begin_compression_step),
            tf.logical_or(
                tf.less_equal(
                    tf.cast(global_step, tf.int32),
                    self._spec.end_compression_step),
                tf.less(self._spec.end_compression_step, 0)))
        return is_step_within_compression_range

    if getattr(self._spec, 'do_transpose', False):
      self.pruning_and_compression_op = self.alpha * self.pruned_a_matrix_tfvar + (
          1 - self.alpha) * tf.math.multiply(tf.transpose(
              tf.reshape(tf.nn.embedding_lookup(
                  self.b_matrix_tfvar, self.c_matrix_tfvar),
                         tf.transpose(a_matrix_tfvar).shape)), self.mask,
                                             name='pruned_compressed_weight')
    else:
      self.pruning_and_compression_op = self.alpha * self.pruned_a_matrix_tfvar + (
          1 - self.alpha) * tf.math.multiply(
              tf.reshape(tf.nn.embedding_lookup(
                  self.b_matrix_tfvar, self.c_matrix_tfvar),
                         a_matrix_tfvar.shape),
              self.mask, name='pruned_compressed_weight')

    def pruned_a_matrix_fn():
      return self.pruned_a_matrix_tfvar

    def quantized_pruned_a_matrix_fn():
      return self.pruning_and_compression_op

    self.final_op = tf.cond(maybe_apply_compression(),
                            quantized_pruned_a_matrix_fn, pruned_a_matrix_fn)

    if self._spec.add_summary:
      self.add_compression_summaries()
    self.pruning_obj.add_pruning_summaries()
    self.mask_update_op = self.pruning_obj.conditional_mask_update_op()
    self.update_op = self.mask_update_op
    return [self.final_op, self.update_op]

  def _create_layer_variable(self,
                             layer_obj,
                             var_name,
                             var_pc,
                             var_theta_fn=None,
                             trainable=False):
    if not hasattr(layer_obj.vars, var_name):
      layer_obj.CreateVariable(
          var_name, var_pc, theta_fn=var_theta_fn, trainable=trainable)

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope'):
    """Returns pruning + kmeans compressed operator for a customized layer.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customeried layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    if getattr(self._spec, 'do_transpose', False):
      a_matrix = np.transpose(a_matrix)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)

    self.uncompressed_size = matrix_compressor.uncompressed_size
    self.compressed_size = matrix_compressor.compressed_size

    p = layer_obj.params
    with tf.variable_scope(scope) as scope:
      # Create pruning relevant variables.
      mask_pc = weight_params_fn(a_matrix.shape, weight_init_obj.Constant(1.0),
                                 p.dtype)
      threshold_pc = weight_params_fn([], weight_init_obj.Constant(0.0),
                                      tf.float32)
      self._create_layer_variable(layer_obj, 'mask', mask_pc, None, False)
      self._create_layer_variable(layer_obj, 'threshold', threshold_pc, None,
                                  False)
      if layer_obj.vars.mask not in tf.get_collection(pruning.MASK_COLLECTION):
        tf.add_to_collection(pruning.WEIGHT_COLLECTION, layer_obj.vars.wm)
        tf.add_to_collection(pruning.MASK_COLLECTION, layer_obj.vars.mask)
        tf.add_to_collection(pruning.THRESHOLD_COLLECTION,
                             layer_obj.vars.threshold)
      if self.pruning_obj.get_spec().prune_option in [
          'first_order_gradient', 'second_order_gradient'
      ]:
        grad_pc = weight_params_fn(a_matrix.shape,
                                   weight_init_obj.Constant(0.0), p.dtype)
        self._create_layer_variable(layer_obj, 'gradient', grad_pc, None, False)
        self._create_layer_variable(layer_obj, 'old_weight', grad_pc, None,
                                    False)
        self._create_layer_variable(layer_obj, 'old_old_weight', grad_pc, None,
                                    False)
        tf.add_to_collection(pruning.WEIGHT_GRADIENT_COLLECTION,
                             layer_obj.vars.gradient)
        tf.add_to_collection(pruning.OLD_WEIGHT_COLLECTION,
                             layer_obj.vars.old_weight)
        tf.add_to_collection(pruning.OLD_OLD_WEIGHT_COLLECTION,
                             layer_obj.vars.old_old_weight)

      b_matrix_pc = weight_params_fn(b_matrix.shape,
                                     weight_init_obj.Constant(1.0), p.dtype)
      c_matrix_pc = weight_params_fn(c_matrix.shape,
                                     weight_init_obj.Constant(1), tf.int32)
      alpha_pc = weight_params_fn([], weight_init_obj.Constant(1.0), tf.float32)

      self._create_layer_variable(layer_obj, 'alpha', alpha_pc, None, False)
      self._create_layer_variable(
          layer_obj,
          'b_matrix_tfvar',
          b_matrix_pc,
          None,
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      self._create_layer_variable(
          layer_obj,
          'c_matrix_tfvar',
          c_matrix_pc,
          None,
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)

      self.b_matrix_tfvar = layer_obj.vars.b_matrix_tfvar
      self.c_matrix_tfvar = layer_obj.vars.c_matrix_tfvar
      self.alpha = layer_obj.vars.alpha
      self.a_matrix_tfvar = a_matrix_tfvar
      self.mask = layer_obj.vars.mask
      self.threshold = layer_obj.vars.threshold

      self.pruned_a_matrix_tfvar = tf.multiply(layer_obj.vars.wm,
                                               layer_obj.vars.mask,
                                               'masked_weight')

    def maybe_apply_compression():
      """Decide whether global step is within compression range.

      Returns:
        is_step_within_compression_range: bool.
      """
      with tf.compat.v1.name_scope(self._spec.name):
        # Compress if current step is more than begin_compression_step and
        # less than end_compression_step (unless it's negative)
        global_step = tf.train.get_global_step()
        def real_global_step_fn():
          return tf.cast(tf.train.get_global_step(), tf.int32)
        def mock_global_step_fn():
          return self._spec.begin_compression_step
        def is_global_step_none(global_step):
          return tf.constant(global_step is None, dtype=tf.bool)
        global_step = tf.cond(is_global_step_none(global_step),
                              mock_global_step_fn,
                              real_global_step_fn)
        is_step_within_compression_range = tf.logical_and(
            tf.greater_equal(
                tf.cast(global_step, tf.int32),
                self._spec.begin_compression_step),
            tf.logical_or(
                tf.less_equal(
                    tf.cast(global_step, tf.int32),
                    self._spec.end_compression_step),
                tf.less(self._spec.end_compression_step, 0)))
        return is_step_within_compression_range

    if getattr(self._spec, 'do_transpose', False):
      self.pruning_and_compression_op = (
          self.alpha * self.pruned_a_matrix_tfvar +
          (1 - self.alpha) * tf.math.multiply(
              tf.transpose(
                  tf.reshape(
                      tf.nn.embedding_lookup(self.b_matrix_tfvar,
                                             self.c_matrix_tfvar),
                      tf.transpose(a_matrix_tfvar).shape)),
              self.mask,
              name='pruned_compressed_weight'))
    else:
      self.pruning_and_compression_op = (
          self.alpha * self.pruned_a_matrix_tfvar +
          (1 - self.alpha) * tf.math.multiply(
              tf.reshape(
                  tf.nn.embedding_lookup(self.b_matrix_tfvar,
                                         self.c_matrix_tfvar),
                  a_matrix_tfvar.shape),
              self.mask,
              name='pruned_compressed_weight'))

    def pruned_a_matrix_fn():
      return self.pruned_a_matrix_tfvar

    def quantized_pruned_a_matrix_fn():
      return self.pruning_and_compression_op

    self.final_op = tf.cond(maybe_apply_compression(),
                            quantized_pruned_a_matrix_fn, pruned_a_matrix_fn)

    if self._spec.add_summary:
      self.add_compression_summaries()
    self.pruning_obj.add_pruning_summaries()
    self.update_op = tf.no_op()
    return [self.final_op, self.update_op]

  def get_update_op(self):
    return self.pruning_obj.conditional_mask_update_op()

  def run_update_step(self, session, step_number=None):
    """Returns the combine update tf OP."""
    logging.info('running run_update_step self._global_step is %s name is %s',
                 self._global_step, self.a_matrix_tfvar.op.name)
    # TODO(wanxin): Resolve tensor infetchable issue and update mask here.
    if step_number is None:
      if self._spec.run_update_interval_check != 0:
        logging.info(
            'running run_update_step step_num is null self.globalstep is %s',
            self._global_step)
        step_number = session.run(self._global_step)
        logging.info('running run_update_step step_num is %s', step_number)
      else:
        step_number = 1

    logging.info(
        'In compression op.run_update_step: '
        'step_number is %s, begin, end and update_count are: %s %s %s ',
        step_number, self._spec.begin_compression_step,
        self._spec.end_compression_step, self.run_update_count)
    if (step_number >= self._spec.begin_compression_step and
        (step_number < self._spec.end_compression_step or
         self._spec.end_compression_step == -1)):
      logging.info(
          'In compression op.run_update_step:'
          'step_number is %s, begin, end and update_count are: %s %s %s ',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count)
      self.run_update_count += 1
      logging.info('inside compression interval')

      # Need to persist these python state variables in TF as if a task gets
      # aborted things get out of sync.
      self._last_update_step = session.run(self._last_alpha_update_step)
      logging.info(
          'In compression op.run_update_step: '
          'step_number is %s, begin, end, update_count, last_alpha_update'
          ' are: %s %s %s %s', step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count,
          self._last_update_step)
      if self._last_update_step == -1:
        logging.info(
            'In compression op.run_update_step: step_number is %s, '
            'begin, end, update_count are: %s %s %s ', step_number,
            self._spec.begin_compression_step, self._spec.end_compression_step,
            self.run_update_count)
        print('inside compression interval: initial decomposition step')
        a_matrix = session.run(self.a_matrix_tfvar)
        pruned_a_matrix = session.run(
            tf.multiply(self.a_matrix_tfvar, self.mask))
        logging.info(
            'In compression op.run_update_step: '
            'a_matrix.shape is %s norm is %d', a_matrix.shape,
            np.linalg.norm(a_matrix))
        if self.matrix_compressor.get_spec().is_c_matrix_present:
          logging.info(
              'In compression op.run_update_step: '
              'step_number is %s, begin, end and update_count are: %s %s %s ',
              step_number, self._spec.begin_compression_step,
              self._spec.end_compression_step, self.run_update_count)
          if getattr(self._spec, 'do_transpose', False):
            [b_matrix, c_matrix
             ] = self.matrix_compressor.static_matrix_compressor(
                 pruned_a_matrix.T)
          else:
            [b_matrix, c_matrix
            ] = self.matrix_compressor.static_matrix_compressor(pruned_a_matrix)
          session.run(tf.assign(self.b_matrix_tfvar, b_matrix))
          session.run(tf.assign(self.c_matrix_tfvar, c_matrix))
        else:
          [b_matrix
          ] = self.matrix_compressor.static_matrix_compressor(pruned_a_matrix)
          session.run(tf.assign(self.b_matrix_tfvar, b_matrix))
      logging.info(
          'In compression op.run_update_step: '
          'step_number is %s, begin, end and update_count are: %s %s %s ',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count)

      alpha = session.run(self.alpha)
      self.last_alpha_value = alpha
      if self.last_alpha_value > 0:
        make_a_zero = False
        new_alpha = max(alpha - self._spec.alpha_decrement_value, 0)
        if make_a_zero and new_alpha == 0:
          logging.info('Making a_matrix all zero for %s',
                       self.a_matrix_tfvar.op.name)
          a_matrix = np.zeros(shape=self.a_matrix_tfvar.shape)
          session.run(tf.assign(self.a_matrix_tfvar, a_matrix))
        logging.info('in run_update_step decrementing alpha, alpha value is %d',
                     self.last_alpha_value)

        logging.info(
            'running run_update_step self._global_step is %s new and old alpha are %d %d',
            self._global_step, alpha, new_alpha)
        session.run(tf.assign(self.alpha, new_alpha))
        self.last_alpha_value = new_alpha
        self._last_update_step = step_number
        session.run(tf.assign(self._last_alpha_update_step, step_number))
    logging.info(
        'In compression op.run_update_step: '
        'step_number is %s, begin, end  and update_count are: %s %s %s ',
        step_number, self._spec.begin_compression_step,
        self._spec.end_compression_step, self.run_update_count)

  def get_mix_operator(self, theta, concat):
    """Performs matrix multiplication for customized layer.

    This performs the compressed equivalent of tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    def maybe_choose_compression_path():
      """Decide whether global step is after begin compression step.

      Returns:
        is_step_after_begin_compression: bool.
      """
      with tf.compat.v1.name_scope(self._spec.name):
        global_step = theta.global_step
        is_step_after_begin_compression = tf.greater_equal(
            tf.cast(global_step, tf.int32), self._spec.begin_compression_step)
        return is_step_after_begin_compression

    pruning_result = tf.matmul(concat, tf.multiply(theta.wm, theta.mask))
    if getattr(self._spec, 'do_transpose', False):
      pruning_compression_result = (
          theta.alpha * pruning_result + (1 - theta.alpha) * tf.matmul(
              concat,
              tf.multiply(tf.transpose(
                  tf.reshape(
                      tf.nn.embedding_lookup(theta.b_matrix_tfvar,
                                             theta.c_matrix_tfvar),
                      tf.transpose(theta.wm).shape)), theta.mask)))
    else:
      pruning_compression_result = (
          theta.alpha * pruning_result + (1 - theta.alpha) * tf.matmul(
              concat,
              tf.multiply(
                  tf.reshape(
                      tf.nn.embedding_lookup(theta.b_matrix_tfvar,
                                             theta.c_matrix_tfvar),
                      theta.wm.shape),
                  theta.mask)))

    def pruning_result_fn():
      return pruning_result

    def quantized_pruned_result_fn():
      return pruning_compression_result

    return tf.cond(maybe_choose_compression_path(), quantized_pruned_result_fn,
                   pruning_result_fn)
