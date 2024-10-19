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

"""Matrix compression operator.

Helper functions to have an automated process to take any matrix compression
algorithm and create a tensorflow operator that can be applied on a tensorflow
matrix variable to compress it on the fly during training.

The class MatrixCompressorInferface can be used to implement any matrix
compression algorithm in the method static_matrix_compressor. The other class
CompressionOpInterface is used to create a tensorflow operator that injects
any matrix compression method dynamically into a tensorflow layer. This is
done by specifying in the spec during initialization a
MatrixCompressorInferface object that implements the method.
The get_apply_compression_op return such a tensorflow operator.
Further a tensorflow operator to update variables needs to be invoked
periodically depending on the method. Such an operator is created using
the get_update_op method.

Derived classes of these interfaces can be used to create compression OPs that
implement different compression methods. Such OPs have been implemented using
derived classes such as LowRankDecompMatrixCompressor, CompressionOp for low
rank decomposition, SimhashMatrixCompressor, SimhashCompressionOp for simhash,
DLMatrixCompressor for dictionary learning.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from graph_compression.compression_lib import compression_op_utils as comp_op_utils
from model_pruning.python import hparam as contrib_hparam


def _get_var_name(var):
  if tf.executing_eagerly_outside_functions():
    return var.name.rsplit(':', 1)[0]
  else:
    return var.op.name


class MatrixCompressorInferface(object):
  """Interface for any matrix compressor algorithm.

  This MatrixCompressorInferface class can be implemented by any third party to
  implement any compression algorithm.
  """

  @abc.abstractmethod
  def __init__(self, spec):
    pass

  def static_matrix_compressor(self, a_matrix):
    """Implements the matrix compression algorithm of choice to compress.

    Args:
      a_matrix: input matrix.

    Returns:
      The factor(s) or any compressed representation of a_matrix.
    """
    raise NotImplementedError()

  def default_matrix(self):
    """Returns default matrix for initialization.

    Size is taken from spec.
    """
    raise NotImplementedError()


class LowRankDecompMatrixCompressor(MatrixCompressorInferface):
  """Low rank decomposition compressor.

  Implements matrix compression interface for the low rank decomposition
  algorithm.
  """

  def __init__(self, spec):
    """Initializer.

    Args:
      spec: hparams object with default value given by
        self.get_default_hparams().
    """
    super(LowRankDecompMatrixCompressor, self).__init__(spec)
    self._spec = spec

    self.uncompressed_size = 0
    self.compressed_size = 0

  def get_spec(self):
    return self._spec

  @staticmethod
  def get_default_hparams():
    """Get a tf.HParams object with the default values for the hyperparameters.

      name: string
        name of the low-rank matrix decompressor specification.
      rank: integer
        rank of the low-rank decomposition that is performed.
      num_rows: integer
        number of rows of given matrix.
      num_cols: integer
        number of columns of given matrix.
      use_tpu: False
        experimental flag; indicates whether to use tensorflow operations (True)
        or python operations (False). For TPU, TF operations are preferred.
      compressor_option: integer
        indicates what type of factorization (if any) is used.
      is_b_matrix_trainable: bool
        indicates whether the b_matrix matrix in the factorization is to be
        trained.
      is_c_matrix_trainable: bool
        indicates whether the c_matrix matrix in the factorization is to be
        trained.

    Returns:
      tf.HParams object initialized to default values.
    """
    return contrib_hparam.HParams(
        name='model_compression',
        rank=100,
        num_rows=10,
        num_cols=10,
        use_tpu=False,
        compressor_option=0,
        is_b_matrix_trainable=True,
        is_c_matrix_trainable=True,
        is_d_matrix_trainable=True,
        is_c_matrix_present=True,
        block_size=1,
        pruning_fraction=0.0,
        use_lsh=False)

  def static_matrix_compressor(self, a_matrix):
    """Low-rank decomposition of a_matrix.

    Args:
      a_matrix: input matrix.

    Returns:
      A list [b_matrix,c_matrix] which is the low-rank decomposition of
      a_matrix. Rank is taken from spec.rank.
    """
    u, s, vh = np.linalg.svd(a_matrix, full_matrices=False)
    logging.info(
        'Inside static_matrix_compressor: u,s,vh shapes are: %s, %s, %s',
        u.shape, s.shape, vh.shape)
    # If matrix dimension is smaller than rank specified then adjust rank
    rank = comp_op_utils.compute_compressed_rank_from_matrix_shape(
        a_matrix.shape, self._spec.rank)
    b_matrix = u[:, :rank]
    c_matrix = vh[:rank, :]
    s_mat = np.diag(np.sqrt(s[:rank]))
    b_matrix = np.matmul(b_matrix, s_mat)
    c_matrix = np.matmul(s_mat, c_matrix)
    logging.info(
        'Inside static_matrix_compressor: a_matrix,b_matrix,c_matrix shapes '
        'are: %s, %s, %s', a_matrix.shape, b_matrix.shape, c_matrix.shape)

    self.uncompressed_size = a_matrix.size
    self.compressed_size = b_matrix.size + c_matrix.size

    return [b_matrix, c_matrix]

  def tpu_matrix_compressor(self, a_matrix):
    """Low-rank decomposition of a_matrix using tpu operations.

    For training on tpus, we only use basic tf operations (as py_func is not
    supported).

    Args:
      a_matrix: input matrix.

    Returns:
      A list of two matrices [b_matrix,c_matrix] which is the low-rank
      decomposition of a_matrix. Rank is taken from spec.rank.
    """
    s, u, v = tf.linalg.svd(a_matrix, full_matrices=False)
    logging.info('Inside tpu_matrix_compressor: u,s,v shapes are: %s, %s, %s',
                 u.shape, s.shape, v.shape)
    rank = comp_op_utils.compute_compressed_rank_from_matrix_shape(
        tuple(a_matrix.shape.dims), self._spec.rank)
    b_matrix = u[:, :rank]
    c_matrix = tf.transpose(a=v)[:rank, :]
    s_mat = tf.linalg.tensor_diag(tf.sqrt(s[:rank]))
    b_matrix = tf.matmul(b_matrix, s_mat)
    c_matrix = tf.matmul(s_mat, c_matrix)
    logging.info(
        'Inside tpu_matrix_compressor: a_matrix,b_matrix,c_matrix'
        'shapes are: %s, %s, %s', a_matrix.shape, b_matrix.shape,
        c_matrix.shape)
    return [b_matrix, c_matrix]

  def default_matrix(self):
    """Returns default matrix of zeros of size specified in spec."""

    a_matrix = np.zeros(shape=[self._spec.num_rows, self._spec.num_cols])
    return a_matrix


class CompressionOpInterface(object):
  """Interface for a compression op.

  Class to take a matrix compression algorithm and create a tensorflow
  compression operator to inject that compression dynamically during training.
  The compression algorithm is specified using an object of
  MatrixCompressorInferface class.
  """

  @abc.abstractmethod
  def __init__(self, scope='default_scope', spec=None, global_step=None):
    pass

  def _setup_global_step(self, global_step):
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = tf.train.get_or_create_global_step()
    logging.info('graph_global_step: %s', graph_global_step)
    return tf.cast(graph_global_step, tf.int32)

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator.

    Does it for variable a_matrix_tfvar for compression method specified in
    matrix_compressor.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    raise NotImplementedError()


class CompressionOp(CompressionOpInterface):
  """Implements a compression OP.

  Does this based on any matrix factorization compression algorithm by
  replacing a variable a_matrix by alpha*a_matrix +
  (1-alpha)b_matrix*c_matrix. See the doc linked in the directory README for
  details.
  """

  def __init__(self, scope='default_scope', spec=None, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      spec: compression hyper parameters default value given by
        self.get_default_hparams().
      global_step: tf variable that has the global step.
    """
    super(CompressionOp, self).__init__(scope, spec, global_step)
    # Compression specification
    self._spec = spec if spec else self.get_default_hparams()
    logging.info('Compression spec in init CompressionOp is: ')
    self.print_hparams()

    # Sanity check for compression hparams
    self._validate_spec()
    self._global_step = self._setup_global_step(global_step)

    # public member variables to track the compressor, the variables and
    # other tf nodes corresponding to this OP.
    self.matrix_compressor = None
    self.a_matrix_tfvar = None
    self.b_matrix_tfvar = None
    self.c_matrix_tfvar = None
    self.alpha = None
    self.final_op = None

    self.update_op = None
    self._last_alpha_update_step = self._setup_last_alpha_update_step()
    self._last_update_step = -1
    self._alpha_update_tf_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0
    self.run_update_count = 0
    self.last_alpha_value = 1

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
        0: TF_UPDATE - run the update logic in TF; needed when using GPU/TPU
        1: PYTHON_UPDATE - run the update logic in regular python as opposed
                           to TF.
        2: TF_AND_PYTHON_UPDATE - run the update logic in TF and in regular
                                  python.
        3: NO_UPDATE - no alpha update; not required for some compression ops.

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_hparam.HParams(
        name='model_compression',
        alpha_decrement_value=0.01,
        begin_compression_step=0,
        end_compression_step=-1,
        compression_frequency=10,
        use_tpu=False,
        compression_option=comp_op_utils.CompressionOptions
        .LOWRANK_MATRIX_COMPRESSION,
        rank=7,
        update_option=comp_op_utils.UpdateOptions.TF_UPDATE,
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
        add_summary=True)

  def add_compression_summaries(self):
    """Adds summaries of alpha value, new variables, and last update step."""
    with tf.compat.v1.name_scope(self._spec.name + '_summaries'):
      tf.compat.v2.summary.scalar(
          _get_var_name(self._last_alpha_update_step) +
          '/last_alpha_update_step', self._last_alpha_update_step)
      tf.compat.v2.summary.scalar(
          _get_var_name(self.alpha) + '/alpha', self.alpha)
      tf.compat.v2.summary.scalar(
          _get_var_name(self.a_matrix_tfvar) + '/a_matrix_norm',
          tf.norm(tensor=self.a_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          _get_var_name(self.b_matrix_tfvar) + '/b_matrix_norm',
          tf.norm(tensor=self.b_matrix_tfvar))
      tf.compat.v2.summary.scalar(
          _get_var_name(self.c_matrix_tfvar) + '/c_matrix_norm',
          tf.norm(tensor=self.c_matrix_tfvar))

  def _setup_last_alpha_update_step(self):
    """Setup to track last alpha update step."""
    with tf.compat.v1.variable_scope(
        self._spec.name, use_resource=True) as scope:
      try:
        last_alpha_update_step = tf.compat.v1.get_variable(
            'last_alpha_update_step',
            initializer=-1,
            trainable=False,
            dtype=tf.int32)
      except ValueError:
        scope.reuse_variables()
        last_alpha_update_step = tf.compat.v1.get_variable(
            'last_alpha_update_step', dtype=tf.int32)
    return last_alpha_update_step

  def _alpha_update_op(self):
    """Update alpha along with last_alpha_update_step."""
    with tf.compat.v1.name_scope(self._spec.name):
      with tf.control_dependencies([
          tf.compat.v1.assign(
              self._last_alpha_update_step,
              tf.cast(self._global_step, tf.int32),
              name='last_alpha_update_step_assign')
      ]):
        with tf.control_dependencies([self._alpha_assign_op()]):
          logging.info('Updating alpha.')
          return tf.no_op('alpha_update')

  def _alpha_assign_op(self):
    new_alpha = tf.maximum(self.alpha - self._spec.alpha_decrement_value, 0)
    alpha_assign_op = tf.compat.v1.assign(
        self.alpha, new_alpha, name='alpha_assign_op')
    return alpha_assign_op

  def _compressor_op(self, matrix_compressor, a_matrix_tfvar):
    """Creates compressor op based on matrix_compressor.

    Meant to create the factors once at begin_compression_step.

    Args:
      matrix_compressor: specifies the matrix compressor object.
      a_matrix_tfvar: the tf tensor to be compressed.

    Returns:
      A TF op which does the compressor update on b and c.
    """
    # py_func is not supported on TPU so need non py_func implementation
    use_tpu = self._spec.use_tpu
    # Seeing some tf.py_func error because of which the
    # following may be needed, so enforcing TF operation updates.
    if use_tpu:
      [b_matrix_out,
       c_matrix_out] = matrix_compressor.tpu_matrix_compressor(a_matrix_tfvar)
    else:
      [b_matrix_out, c_matrix_out
      ] = tf.compat.v1.py_func(matrix_compressor.static_matrix_compressor,
                               [a_matrix_tfvar], [tf.float32, tf.float32])

    b_matrix_assign_op = tf.compat.v1.assign(
        self.b_matrix_tfvar, b_matrix_out, name='b_matrix_assign_op')
    c_matrix_assign_op = tf.compat.v1.assign(
        self.c_matrix_tfvar, c_matrix_out, name='c_matrix_assign_op')
    with tf.control_dependencies([b_matrix_assign_op, c_matrix_assign_op]):
      logging.info('Updating b_matrix,c_matrix.')
      return tf.no_op('compresor_b_matrix_and_c_matrix_update')

  def _compressor_and_alpha_update_op(self):
    """Applies compressor and also updates alpha."""

    def compressor_op():
      return self._compressor_op(self.matrix_compressor, self.a_matrix_tfvar)

    def tf_no_op():
      return tf.no_op()

    cond_compressor_op = tf.cond(
        pred=self._last_alpha_update_step < 0,
        true_fn=compressor_op,
        false_fn=tf_no_op)

    with tf.control_dependencies([cond_compressor_op]):
      with tf.control_dependencies([self._alpha_update_op()]):
        return tf.no_op('alpha_update')

  def _create_update_op(self):
    """Creates tensoflow update op for the compression."""

    def maybe_update_alpha():
      """Operator to update alpha.

      Checks if global_step is between begin_compression_step and
      end_compression_step.
      """
      with tf.compat.v1.name_scope(self._spec.name):
        # prune if current step is more than begin_compression_step and
        # less than end_compression_step (unless it's negative)
        is_step_within_compression_range = tf.logical_and(
            tf.greater_equal(
                tf.cast(self._global_step, tf.int32),
                self._spec.begin_compression_step),
            tf.logical_or(
                tf.less_equal(
                    tf.cast(self._global_step, tf.int32),
                    self._spec.end_compression_step),
                tf.less(self._spec.end_compression_step, 0)))
        is_compression_step = tf.less_equal(
            tf.add(self._last_alpha_update_step,
                   self._spec.compression_frequency),
            tf.cast(self._global_step, tf.int32))
        return tf.logical_and(is_step_within_compression_range,
                              is_compression_step)

    def no_update_op():
      return tf.no_op()

    def compressor_and_alpha_update_op_fn():
      return self._compressor_and_alpha_update_op()

    cond_alpha_update_op = tf.cond(
        pred=maybe_update_alpha(),
        true_fn=compressor_and_alpha_update_op_fn,
        false_fn=no_update_op)
    self.update_op = cond_alpha_update_op
    return self.update_op

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator.

    Does this for variable a_matrix_tfvar for
    compression method specified in matrix_compressor that must be based on some
    matrix factorization compression algorithm by replacing a variable
    a_matrix by alpha*a_matrix + (1-alpha)b_matrix*c_matrix.

    Args:
      a_matrix_tfvar: TF variable representihg a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      scope: TF scope used for creating new TF variables.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)
    with tf.compat.v1.variable_scope(scope, use_resource=True):
      self.b_matrix_tfvar = tf.compat.v1.get_variable(
          'b_matrix',
          dtype=tf.float32,
          initializer=b_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      self.c_matrix_tfvar = tf.compat.v1.get_variable(
          'c_matrix',
          dtype=tf.float32,
          initializer=c_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.alpha = tf.compat.v1.get_variable(
          'alpha', dtype=tf.float32, trainable=False, initializer=1.0)

      self.a_matrix_tfvar = a_matrix_tfvar

      if self._spec.update_option == comp_op_utils.UpdateOptions.TF_UPDATE:
        self.update_op = self._create_update_op()
      else:
        self.setup_update_explicit()

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.matmul(self.b_matrix_tfvar, self.c_matrix_tfvar)

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope',
                                          a_matrix_tfvar_shape=None):
    """Returns compressed tensorflow operator for a customized model/layer.

    Does this for variable a_matrix_tfvar for
    compression method specified in matrix_compressor that must be based on some
    matrix factorization compression algorithm by replacing a variable
    a_matrix by alpha*a_matrix + (1-alpha)b_matrix*c_matrix.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customeried layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      a_matrix_tfvar_shape: A list specifying the shape of the tensor to
        compress. In some cases when a_matrix_tfvar is set to None,
        this field is used to pass in the shape of the matrix to compress.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    shape = (a_matrix_tfvar.shape if a_matrix_tfvar is not None
             else a_matrix_tfvar_shape)
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=shape)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)

    p = layer_obj.params
    with tf.variable_scope(scope) as scope:
      b_matrix_pc = weight_params_fn(b_matrix.shape,
                                     weight_init_obj.Constant(1.0), p.dtype)
      c_matrix_pc = weight_params_fn(c_matrix.shape,
                                     weight_init_obj.Constant(1.0), p.dtype)
      alpha_pc = weight_params_fn([], weight_init_obj.Constant(1.0), tf.float32)

      layer_obj.CreateVariable('alpha', alpha_pc, trainable=False)
      layer_obj.CreateVariable(
          'b_matrix_tfvar',
          b_matrix_pc,
          trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      layer_obj.CreateVariable(
          'c_matrix_tfvar',
          c_matrix_pc,
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)

      self.b_matrix_tfvar = layer_obj.vars.b_matrix_tfvar
      self.c_matrix_tfvar = layer_obj.vars.c_matrix_tfvar
      self.alpha = layer_obj.vars.alpha
      self.a_matrix_tfvar = a_matrix_tfvar

      if self._spec.update_option == comp_op_utils.UpdateOptions.TF_UPDATE:
        self.update_op = self._create_update_op()
      else:
        self.setup_update_explicit()

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.matmul(self.b_matrix_tfvar, self.c_matrix_tfvar)

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_mix_operator(self, theta, concat, layer_obj=None):
    """Performs matrix multiplication for customized layer.

    This performs the compressed equivalent of tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation.
      layer_obj: reference to the customized layer object. default is None.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    del layer_obj  # Unused by get_mix_operator here
    return (theta.alpha * tf.matmul(concat, theta.wm) +
            (1 - theta.alpha) * tf.matmul(
                tf.matmul(concat, theta.b_matrix_tfvar), theta.c_matrix_tfvar))

  def get_apply_embedding_lookup(self, ids):
    """Returns compressed tensorflow operator for embedding_lookup.

    This method returns a TensorFlow node that performs embedding lookup as
    alpha * tf.nn.embedding_lookup(a_matrix_tfvar, ids) +
    (1 - alpha) * tf.nn.embedding_lookup(b_matrix_tfvar, ids) if c_matrix is
    not present, and alpha * tf.nn.embedding_lookup(a_matrix_tfvar, ids) +
    (1 - alpha) * tf.matmul(tf.nn.embedding_lookup(b_matrix_tfvar, ids),
    c_matrix) if c_matrix is present, where b_matrix_tfvar and c_matrix_tfvar
    are the factor matrices for the compressed embedding table.

    Args:
      ids: A Tensor with type int32 or int64 containing the ids to be looked up
        in the embedding table (the a_matrix_tfvar variable).

    Returns:
      embedding_op: a TensorFlow node that performs compressed embedding lookup.
    """
    if self.matrix_compressor.get_spec().is_c_matrix_present:
      embedding_op = self.alpha * tf.nn.embedding_lookup(
          self.a_matrix_tfvar, ids) + (1 - self.alpha) * tf.matmul(
              tf.nn.embedding_lookup(self.b_matrix_tfvar, ids),
              self.c_matrix_tfvar)
    else:
      embedding_op = self.alpha * tf.nn.embedding_lookup(
          self.a_matrix_tfvar, ids) + (1 - self.alpha) * tf.nn.embedding_lookup(
              self.b_matrix_tfvar, ids)

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
    if self.matrix_compressor.get_spec().is_c_matrix_present:
      matmul_op = self.alpha * tf.matmul(
          left_operand, tf.transpose(
              self.a_matrix_tfvar)) + (1 - self.alpha) * tf.matmul(
                  tf.matmul(left_operand, tf.transpose(self.c_matrix_tfvar)),
                  tf.transpose(self.b_matrix_tfvar))
    else:
      matmul_op = self.alpha * tf.matmul(
          left_operand, tf.transpose(
              self.a_matrix_tfvar)) + (1 - self.alpha) * tf.matmul(
                  left_operand, tf.transpose(self.b_matrix_tfvar))

    return matmul_op

  @staticmethod
  def all_update_op(update_ops_list, scope='default_scope'):
    """Method to create a complete update op.

    Args:
      update_ops_list: list of individual update ops.
      scope: tf scope for creating update op.

    Returns:
      A TensorFlow op that updates the compression related variables.
    """
    with tf.compat.v1.name_scope(scope):
      with tf.control_dependencies(update_ops_list):
        logging.info('Updating all compression_ops.')
        return tf.no_op('update_all_compression_ops')

  def _validate_spec(self):
    spec = self._spec
    if spec.begin_compression_step < 0:
      raise ValueError('Illegal value for begin_compression_step')

    if spec.begin_compression_step >= spec.end_compression_step:
      if spec.end_compression_step != -1:
        raise ValueError(
            'Compression must begin before it can end. begin_step=%d, '
            'end_step=%d. Set end_compression_step to -1 if compression is '
            'required till training stops' %
            (spec.begin_compression_step, spec.end_compression_step))

  def _setup_global_step(self, global_step):
    graph_global_step = global_step
    if graph_global_step is None:
      graph_global_step = tf.train.get_global_step()
    logging.info('graph_global_step: %s', graph_global_step)
    return tf.cast(graph_global_step, tf.int32)

  def print_hparams(self):
    logging.info(self._spec.to_json())

  def setup_update_explicit(self):
    self._alpha_update_tf_op = self._alpha_update_op()
    return self._alpha_update_tf_op

  def run_update_step(self, session, step_number=None):
    """Returns the combine update tf OP."""
    logging.info('running run_update_step self._global_step is %s name is %s',
                 self._global_step, self.a_matrix_tfvar.op.name)
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
        step_number < self._spec.end_compression_step):
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
          ' are: %s %s %s %s',
          step_number, self._spec.begin_compression_step,
          self._spec.end_compression_step, self.run_update_count,
          self._last_update_step)
      if self._last_update_step == -1:
        logging.info(
            'In compression op.run_update_step: step_number is %s, '
            'begin, end, update_count are: %s %s %s ',
            step_number, self._spec.begin_compression_step,
            self._spec.end_compression_step, self.run_update_count)
        print('inside compression interval: initial decomposition step')
        a_matrix = session.run(self.a_matrix_tfvar)
        logging.info(
            'In compression op.run_update_step: '
            'a_matrix.shape is %s norm is %d',
            a_matrix.shape, np.linalg.norm(a_matrix))
        if self.matrix_compressor.get_spec().is_c_matrix_present:
          logging.info(
              'In compression op.run_update_step: '
              'step_number is %s, begin, end and update_count are: %s %s %s ',
              step_number, self._spec.begin_compression_step,
              self._spec.end_compression_step, self.run_update_count)
          [b_matrix,
           c_matrix] = self.matrix_compressor.static_matrix_compressor(a_matrix)
          session.run(tf.assign(self.b_matrix_tfvar, b_matrix))
          session.run(tf.assign(self.c_matrix_tfvar, c_matrix))
        else:
          [b_matrix] = self.matrix_compressor.static_matrix_compressor(a_matrix)
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

  def get_update_op(self):
    return self.update_op

  def run_update_step_keras(self, step_number):
    """Keras version of run_update_step.

    Run matrix and alpha update step if criterion is met.

    Args:
      step_number: step number in the training process.
    Note: This method should only be called during training.
    """
    if (step_number >= self._spec.begin_compression_step and
        (step_number < self._spec.end_compression_step or
         self._spec.end_compression_step == -1)):
      if self.last_alpha_update_step.numpy() == -1:
        a_matrix = self.a_matrix_tfvar.numpy()
        b_matrix, c_matrix = self.matrix_compressor.static_matrix_compressor(
            a_matrix)
        self.b_matrix_tfvar.assign(b_matrix)
        self.c_matrix_tfvar.assign(c_matrix)
      if self.alpha.numpy() > 0:
        self.alpha.assign(
            max(self.alpha.numpy() - self._spec.alpha_decrement_value, 0))
        self.last_alpha_update_step.assign(step_number)


class InputOutputCompressionOp(CompressionOpInterface):
  """Implements an input (and/or) output compression OP.

  Replaces a vector-matrix multiplication with a compressed vector-smaller
  matrix multiplication. The compression can happen on the input vector (input
  compression) or on the output of the vector-matrix multiplication (output
  compression).

  Input compression projects the input vector into a lower-dimensional space
  using a linear transform and replaces the original weight matrix by a smaller
  matrix as a result.

  Output compression replaces the product of an input vector and a weight matrix
  by taking the product of the input with a smaller weight matrix and then
  projecting this smaller sized output back up to the original dimensionality of
  the product.

  compress_input flag indicates if we want the input to be compressed. set to
  True by default.
  compress_output flag indicates if we want the output to be compressed. set to
  False by default.
  """

  def __init__(self, scope='default_scope', spec=None, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      spec: compression hyper parameters default value given by
        self.get_default_hparams().
      global_step: tf variable that has the global step.
    """
    super(InputOutputCompressionOp, self).__init__(scope, spec, global_step)
    # Compression specification
    self._spec = spec if spec else self.get_default_hparams()
    logging.info('Compression spec in init InputOutputCompressionOp is: ')
    self.print_hparams()
    self._global_step = self._setup_global_step(global_step)

    # public member variables to track the compressor, the variables and
    # other tf nodes corresponding to this OP.
    self.matrix_compressor = None
    self.a_matrix_tfvar = None
    self.b_matrix_tfvar = None
    self.c_matrix_tfvar = None
    self.final_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0

  @staticmethod
  def get_default_hparams():
    """Get a tf.HParams object with the default values for the hyperparameters.

      name: string
        name of the compression specification. Used for adding summaries and ops
        under a common tensorflow name_scope.
      use_tpu: False
        indicates whether to use TPU.
      compression_option: integer
        indicates what type of factorization (if any) is used.
      rank: integer
        indicates what type of factorization (if any) is used.
      update_option: integer
        indicates how the update logic is being run. More specifically:
        0: TF_UPDATE - run the update logic in TF; needed when using GPU/TPU
        1: PYTHON_UPDATE - run the update logic in regular python as opposed
                           to TF.
        2: TF_AND_PYTHON_UPDATE - run the update logic in TF and in regular
                                  python.
        3: NO_UPDATE - no alpha update; not required for some compression ops.
      TODO(wanxin): add doc strings for pruning hparams.

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_hparam.HParams(
        name='input_compression',
        compression_frequency=10,
        use_tpu=False,
        compression_option=comp_op_utils.CompressionOptions
        .INPUTOUTPUT_COMPRESSION,
        update_option=comp_op_utils.UpdateOptions.NO_UPDATE,
        begin_compression_step=1000,
        end_compression_step=2000,
        is_b_matrix_trainable=True,
        is_c_matrix_trainable=True,
        is_d_matrix_trainable=True,
        rank=4,
        compress_input=True,
        compress_output=False,
        input_compression_factor=1,
        input_block_size=32,
        output_compression_factor=1,
        output_block_size=1,
        add_summary=True)

  def add_compression_summaries(self):
    """Adds summaries."""
    with tf.name_scope(self._spec.name + '_summaries'):
      logging.info('add_compression_summaries scope name is %s',
                   self._spec.name)
      tf.compat.v2.summary.scalar(
          _get_var_name(self.a_matrix_tfvar) + '/a_matrix_norm',
          tf.norm(self.a_matrix_tfvar))
      if self._spec.compress_input:
        tf.compat.v2.summary.scalar(
            _get_var_name(self.b_matrix_tfvar) + '/b_matrix_norm',
            tf.norm(tf.reshape(self.b_matrix_tfvar, [-1]), ord=1))
      if self._spec.compress_output:
        tf.compat.v2.summary.scalar(
            _get_var_name(self.d_matrix_tfvar) + '/d_matrix_norm',
            tf.norm(tf.reshape(self.d_matrix_tfvar, [-1]), ord=1))
      tf.compat.v2.summary.scalar(
          _get_var_name(self.c_matrix_tfvar) + '/c_matrix_norm',
          tf.norm(self.c_matrix_tfvar))

  def print_hparams(self):
    logging.info(self._spec.to_json())

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator for input/output compression.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      scope: TF scope used for creating new TF variables

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    if self._spec.compress_input:
      # input to be compressed. create b matrix with Xavier initialitation.
      b_matrix_shape = [
          self._spec.input_block_size,
          self._spec.input_block_size // self._spec.input_compression_factor
      ]
      b_limit = np.sqrt(
          3.0 * (1 / np.max([1.,
                             (b_matrix_shape[0] + b_matrix_shape[1]) / 2.])))
      b_matrix = np.random.uniform(-b_limit, b_limit, size=b_matrix_shape)
    if self._spec.compress_output:
      # output to be compressed. create d matrix with Xavier initialization.
      d_matrix_shape = [
          self._spec.output_block_size // self._spec.output_compression_factor,
          self._spec.output_block_size
      ]
      d_limit = np.sqrt(
          3.0 * (1 / np.max([1.,
                             (d_matrix_shape[0] + d_matrix_shape[1]) / 2.])))
      d_matrix = np.random.uniform(-d_limit, d_limit, size=d_matrix_shape)

    # create c_matrix according to whether input is being compressed
    # and whether the output is being compressed. Xavier init.
    c_matrix_shape = [
        a_matrix_tfvar.shape[0] // self._spec.input_compression_factor,
        a_matrix_tfvar.shape[1] // self._spec.output_compression_factor
    ]
    c_limit = np.sqrt(3.0 * (1 / np.max([
        1., (c_matrix_shape[0] + c_matrix_shape[1]) /
        (2. * self._spec.input_compression_factor)
    ])))
    c_matrix = np.random.uniform(-c_limit, c_limit, size=c_matrix_shape)

    # convert b,c,d from numpy arrays to tf tensors
    with tf.compat.v1.variable_scope(scope, use_resource=True):
      if self._spec.compress_input:
        self.b_matrix_tfvar = tf.compat.v1.get_variable(
            'b_matrix',
            dtype=tf.float32,
            initializer=b_matrix.astype(np.float32),
            trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      if self._spec.compress_output:
        self.d_matrix_tfvar = tf.compat.v1.get_variable(
            'd_matrix',
            dtype=tf.float32,
            initializer=d_matrix.astype(np.float32),
            trainable=self.matrix_compressor.get_spec().is_d_matrix_trainable)
      self.c_matrix_tfvar = tf.compat.v1.get_variable(
          'c_matrix',
          dtype=tf.float32,
          initializer=c_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.a_matrix_tfvar = a_matrix_tfvar

    # update_op and final_op not necessary for InputOutputCompressionOp.
    self.update_op = tf.no_op()
    self.final_op = tf.no_op()

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope',
                                          a_matrix_tfvar_shape=None):
    """Returns input (and) or output compressed operator for a babelfish layer.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customeried layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      a_matrix_tfvar_shape: A list specifying the shape of the tensor to
        compress. In some cases when a_matrix_tfvar is set to None,
        this field is used to pass in the shape of the matrix to compress.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    shape = (a_matrix_tfvar.shape if a_matrix_tfvar is not None
             else a_matrix_tfvar_shape)
    self.matrix_compressor = matrix_compressor
    with tf.variable_scope(scope) as scope:
      if self._spec.compress_input:
        # input-side compression being applied.
        # create b with appropriate shape and init params.
        b_matrix_pc = weight_params_fn([
            self._spec.input_block_size,
            self._spec.input_block_size // self._spec.input_compression_factor
        ], weight_init_obj.Xavier(1.0), layer_obj.params.dtype)
      if self._spec.compress_output:
        # output-side compression being applied.
        # create d with appropriate shape and init params.
        d_matrix_pc = weight_params_fn([
            self._spec.output_block_size //
            self._spec.output_compression_factor, self._spec.output_block_size
        ], weight_init_obj.Xavier(1.0), layer_obj.params.dtype)
      # shape of c determined by whether input-side and output-side compression
      # are turned on.
      c_matrix_pc = weight_params_fn([
          shape[0] // self._spec.input_compression_factor,
          shape[1] // self._spec.output_compression_factor
      ], weight_init_obj.Xavier(1.0), layer_obj.params.dtype)

      # create the TF variables using babelfish variable creation function
      if self._spec.compress_input:
        layer_obj.CreateVariable(
            'b_matrix_tfvar',
            b_matrix_pc,
            trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
      if self._spec.compress_output:
        layer_obj.CreateVariable(
            'd_matrix_tfvar',
            d_matrix_pc,
            trainable=self.matrix_compressor.get_spec().is_d_matrix_trainable)
      layer_obj.CreateVariable(
          'c_matrix_tfvar',
          c_matrix_pc,
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)

      if self._spec.compress_input:
        self.b_matrix_tfvar = layer_obj.vars.b_matrix_tfvar
      if self._spec.compress_output:
        self.d_matrix_tfvar = layer_obj.vars.d_matrix_tfvar
      self.c_matrix_tfvar = layer_obj.vars.c_matrix_tfvar
      self.a_matrix_tfvar = a_matrix_tfvar

    self.final_op = tf.no_op()
    if self._spec.add_summary:
      self.add_compression_summaries()
    self.update_op = tf.no_op()
    return [self.final_op, self.update_op]

  def get_apply_compression_op_keras(self,
                                     a_matrix_tfvar,
                                     matrix_compressor,
                                     layer):
    """Returns compressed tensorflow operator for input compression.

    Args:
      a_matrix_tfvar: TF variable representihg a tensor variable in a model
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer: keras layer object calling this function. Must support add_weight
         method.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    self.matrix_compressor = matrix_compressor
    if self._spec.compress_input:
      # input-side compression being applied.
      # create b with appropriate shape and init params.
      b_matrix_shape = [
          self._spec.input_block_size,
          self._spec.input_block_size // self._spec.input_compression_factor
      ]
      self.b_matrix_tfvar = layer.add_weight(
          'b_matrix',
          shape=b_matrix_shape,
          initializer=layer.kernel_initializer,
          regularizer=layer.kernel_regularizer,
          constraint=layer.kernel_constraint,
          dtype=layer.dtype,
          trainable=True)
    if self._spec.compress_output:
      # output-side compression being applied.
      # create d with appropriate shape and init params.
      d_matrix_shape = [
          self._spec.output_block_size // self._spec.output_compression_factor,
          self._spec.output_block_size
      ]
      self.d_matrix_tfvar = layer.add_weight(
          'd_matrix',
          shape=d_matrix_shape,
          initializer=layer.kernel_initializer,
          regularizer=layer.kernel_regularizer,
          constraint=layer.kernel_constraint,
          dtype=layer.dtype,
          trainable=True)
    c_matrix_shape = [
        a_matrix_tfvar.shape[0] // self._spec.input_compression_factor,
        a_matrix_tfvar.shape[1] // self._spec.output_compression_factor
    ]
    self.c_matrix_tfvar = layer.add_weight(
        'c_matrix',
        shape=c_matrix_shape,
        initializer=layer.kernel_initializer,
        regularizer=layer.kernel_regularizer,
        constraint=layer.kernel_constraint,
        dtype=layer.dtype,
        trainable=True)

    self.a_matrix_tfvar = a_matrix_tfvar

    self.update_op = tf.no_op()
    self.final_op = tf.no_op()

    print('****************returning these self.final_op, self.update_op',
          self.final_op, self.update_op)
    # self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_apply_matmul(self, left_operand):
    """Returns input (and/or) output compressed TensorFlow node for matmul.

    This method performs matmul according to the compression
    procedure.

    Args:
      left_operand: a Tensor that is the left operand in matmul.

    Returns:
      matmul_op: a TensorFlow node that performs matmul of left_operand with the
      compressed a_matrix_tfvar.
    """
    if self._spec.compress_input:
      # block the left operand into blocks of size input_block_size.
      blocked_left_operand = tf.reshape(
          left_operand,
          tf.concat([
              tf.shape(left_operand)[:-1],
              [tf.shape(left_operand)[-1] // self._spec.input_block_size],
              [self._spec.input_block_size]
          ], axis=0))
      # project blocked_left_operand down using b.
      projected_blocked_left_operand = tf.matmul(blocked_left_operand,
                                                 self.b_matrix_tfvar)
      # flatten the block dimension in projected_blocked_left_operand.
      compressed_left_operand = tf.reshape(
          projected_blocked_left_operand,
          tf.concat([
              tf.shape(left_operand)[:-1],
              [
                  tf.shape(left_operand)[-1] //
                  self._spec.input_compression_factor
              ]
          ], axis=0))
    else:
      # input not being compressed
      compressed_left_operand = left_operand

    # multiply compressed_left_operand with c.
    intermediate_result = tf.matmul(compressed_left_operand,
                                    self.c_matrix_tfvar)

    if self._spec.compress_output:
      # block intermediate_result
      block_size = self._spec.output_block_size // self._spec.output_compression_factor
      blocked_intermediate_result = tf.reshape(
          intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [tf.shape(intermediate_result)[-1] // block_size],
              [block_size]
          ], axis=0))
      # project blocked_intermediate_result up using d.
      projected_blocked_intermediate_result = tf.matmul(
          blocked_intermediate_result, self.d_matrix_tfvar)
      # flatten block dimension in projected_blocked_intermediate_result.
      compressed_result = tf.reshape(
          projected_blocked_intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [
                  tf.shape(intermediate_result)[-1] *
                  self._spec.output_compression_factor
              ]
          ], axis=0))
    else:
      # output not being compressed
      compressed_result = intermediate_result
    return compressed_result

  def get_mix_operator(self, theta, concat, layer_obj=None):
    """Performs matrix multiplication on compressed input for Babelfish LSTM layers.

    This performs the input (and/or) output compressed equivalent of
    tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation. a rank 2 tensor.
      layer_obj: reference to the customized layer object. default is None.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    if self._spec.compress_input:
      # block concat into blocks of size input_block_size.
      blocked_concat = tf.reshape(
          concat,
          tf.concat([
              tf.shape(concat)[:-1],
              [tf.shape(concat)[-1] // self._spec.input_block_size],
              [self._spec.input_block_size]
          ],
                    axis=0))
      # project blocked_left_operand down using b.
      if layer_obj is not None:
        b_matrix_tfvar = layer_obj.QWeight(theta.b_matrix_tfvar,
                                           'b_compression')
      else:
        b_matrix_tfvar = theta.b_matrix_tfvar
      projected_blocked_concat = tf.matmul(blocked_concat, b_matrix_tfvar)
      # flatten the block dimension in projected_blocked_concat.
      compressed_concat = tf.reshape(
          projected_blocked_concat,
          tf.concat([
              tf.shape(concat)[:-1],
              [tf.shape(concat)[-1] // self._spec.input_compression_factor]
          ], axis=0))
    else:
      compressed_concat = concat

    # multiply compressed concat with c.
    if layer_obj is not None:
      c_matrix_tfvar = layer_obj.QWeight(theta.c_matrix_tfvar, 'c_compression')
    else:
      c_matrix_tfvar = theta.c_matrix_tfvar
    intermediate_result = tf.matmul(compressed_concat, c_matrix_tfvar)

    if self._spec.compress_output:
      # block intermediate_result into blocks
      block_size = self._spec.output_block_size // self._spec.output_compression_factor
      blocked_intermediate_result = tf.reshape(
          intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [tf.shape(intermediate_result)[-1] // block_size], [block_size]
          ],
                    axis=0))
      # project blocked_intermediate_result up using d.
      if layer_obj is not None:
        d_matrix_tfvar = layer_obj.QWeight(theta.d_matrix_tfvar,
                                           'd_compression')
      else:
        d_matrix_tfvar = theta.d_matrix_tfvar
      projected_intermediate_result = tf.matmul(blocked_intermediate_result,
                                                d_matrix_tfvar)
      # flatten the block dimension
      compressed_result = tf.reshape(
          projected_intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [
                  tf.shape(intermediate_result)[-1] *
                  self._spec.output_compression_factor
              ]
          ], axis=0))
    else:
      compressed_result = intermediate_result
    return compressed_result

  def get_matmul_operator(self,
                          inputs,
                          wm,
                          layer_obj,
                          transpose_a=False,
                          transpose_b=False):
    """Performs matrix multiplication on compressed input for customized Softmax layers.

    This performs the input (and/or) output compressed equivalent of
    tf.matmul(inputs, wm).

    Args:
      inputs: the left operand of the matmul operation. a rank 2 tensor.
      wm: the right operand of the matmul operator. a rank 2 tensor.
      layer_obj: a lingvo layer object that handles variable creation.
      transpose_a: whether inputs tensor needs to be transposed before matmul.
      transpose_b: whether wm tensor needs to be transposed before matmul.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    theta = layer_obj.theta
    if transpose_a:
      inputs = tf.transpose(inputs)
    if transpose_b:
      wm = tf.transpose(wm)
    if self._spec.compress_input:
      # block inputs into blocks of size input_block_size.
      blocked_inputs = tf.reshape(inputs, [
          -1,
          tf.shape(inputs)[1] // self._spec.input_block_size,
          self._spec.input_block_size
      ])
      # project blocked_inputs down using b.
      if layer_obj is not None:
        b_matrix_tfvar = layer_obj.QWeight(theta.b_matrix_tfvar,
                                           'b_compression')
      else:
        b_matrix_tfvar = theta.b_matrix_tfvar
      projected_blocked_inputs = tf.matmul(blocked_inputs, b_matrix_tfvar)
      # flatten the block dimension in projected_blocked_inputs.
      compressed_inputs = tf.reshape(
          projected_blocked_inputs,
          [tf.shape(inputs)[0], -1])
    else:
      compressed_inputs = inputs

    # multiply compressed inputs with c.
    if layer_obj is not None:
      c_matrix_tfvar = layer_obj.QWeight(theta.c_matrix_tfvar, 'c_compression')
    else:
      c_matrix_tfvar = theta.c_matrix_tfvar
    intermediate_result = tf.matmul(compressed_inputs, c_matrix_tfvar)

    if self._spec.compress_output:
      # block intermediate_result into blocks
      block_size = self._spec.output_block_size // self._spec.output_compression_factor
      blocked_intermediate_result = tf.reshape(
          intermediate_result,
          [tf.shape(intermediate_result)[0], -1, block_size])
      # project blocked_intermediate_result up using d.
      if layer_obj is not None:
        d_matrix_tfvar = layer_obj.QWeight(theta.d_matrix_tfvar,
                                           'd_compression')
      else:
        d_matrix_tfvar = theta.d_matrix_tfvar
      projected_intermediate_result = tf.matmul(blocked_intermediate_result,
                                                d_matrix_tfvar)
      # flatten the block dimension
      compressed_result = tf.reshape(
          projected_intermediate_result,
          [tf.shape(projected_intermediate_result)[0], -1])
    else:
      compressed_result = intermediate_result
    return compressed_result

  def get_einsum_operator(self,
                          inputs,
                          layerobj):
    """Performs compressed matrix multiplication for customized ProjectionLayer.

    This performs the input (and/or) output compressed equivalent of
    tf.matmul(inputs, weight).

    Args:
      inputs: the left operand of the matmul operation.
      layerobj: the ProjectionLayer object from where get_einsum_operator
                is called.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    theta = layerobj.theta
    if self._spec.compress_input:
      # block inputs into blocks of size input_block_size.
      blocked_inputs = tf.reshape(
          inputs,
          tf.concat([
              tf.shape(inputs)[:-1],
              [tf.shape(inputs)[-1] // self._spec.input_block_size],
              [self._spec.input_block_size]
          ],
                    axis=0))
      # project blocked_inputs down using b.
      projected_blocked_inputs = tf.matmul(blocked_inputs, theta.b_matrix_tfvar)
      # flatten the block dimension in projected_blocked_concat.
      compressed_inputs = tf.reshape(
          projected_blocked_inputs,
          tf.concat([
              tf.shape(inputs)[:-1],
              [tf.shape(inputs)[-1] // self._spec.input_compression_factor]
          ], axis=0))
    else:
      compressed_inputs = inputs

    # multiply compressed inputs with c.
    intermediate_result = tf.matmul(compressed_inputs, theta.c_matrix_tfvar)

    if self._spec.compress_output:
      # block intermediate_result into blocks
      block_size = self._spec.output_block_size // self._spec.output_compression_factor
      blocked_intermediate_result = tf.reshape(
          intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [tf.shape(intermediate_result)[-1] // block_size], [block_size]
          ],
                    axis=0))
      # project blocked_intermediate_result up using d.
      projected_intermediate_result = tf.matmul(blocked_intermediate_result,
                                                theta.d_matrix_tfvar)
      # flatten the block dimension
      compressed_result = tf.reshape(
          projected_intermediate_result,
          tf.concat([
              tf.shape(intermediate_result)[:-1],
              [
                  tf.shape(intermediate_result)[-1] *
                  self._spec.output_compression_factor
              ]
          ], axis=0))
    else:
      compressed_result = intermediate_result
    return compressed_result


class BlockCompressionOp(CompressionOpInterface):
  """Implements a block diagonal compression OP.

  Replaces the weight matrix with a block-diagonal matrix. This produces an
  effect similar to input/output compression but without the need to reshape
  the input vector.


  block_method: string.
    "mask" creates block diagonal matrices by masking out entries outside of the
    diagonal blocks.
    "loop" stores the blocks in a rank 3 tensor and loops through them during
    the multiplication step.
  block_compression_factor: integer.
    Factor by which number of (non-zero) parameters in weight matrix is reduced
    by.
  """

  def __init__(self, scope='default_scope', spec=None, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      spec: compression hyper parameters default value given by
        self.get_default_hparams().
      global_step: tf variable that has the global step.
    """
    super(BlockCompressionOp, self).__init__(scope, spec, global_step)
    # Compression specification
    self._spec = spec if spec else self.get_default_hparams()
    logging.info('Compression spec in init BlockCompressionOp is: ')
    self.print_hparams()
    self._global_step = self._setup_global_step(global_step)

    # public member variables to track the compressor, the variables and
    # other tf nodes corresponding to this OP.
    self.matrix_compressor = None
    self.a_matrix_tfvar = None
    self.b_matrix_tfvar = None
    self.c_matrix_tfvar = None
    self.final_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0

  def print_hparams(self):
    logging.info(self._spec.to_json())

  @staticmethod
  def get_default_hparams():
    """Get a tf.HParams object with the default values for the hyperparameters.

      name: string
        name of the compression specification. Used for adding summaries and ops
        under a common tensorflow name_scope.
      use_tpu: False
        indicates whether to use TPU.
      compression_option: integer
        indicates what type of factorization (if any) is used.
      rank: integer
        indicates what type of factorization (if any) is used.
      update_option: integer
        indicates how the update logic is being run. More specifically:
        0: TF_UPDATE - run the update logic in TF; needed when using GPU/TPU
        1: PYTHON_UPDATE - run the update logic in regular python as opposed
                           to TF.
        2: TF_AND_PYTHON_UPDATE - run the update logic in TF and in regular
                                  python.
        3: NO_UPDATE - no alpha update; not required for some compression ops.
      TODO(wanxin): add doc strings for pruning hparams.

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_hparam.HParams(
        name='block_compression',
        compression_frequency=10,
        use_tpu=False,
        compression_option=comp_op_utils.CompressionOptions.BLOCK_COMPRESSION,
        update_option=comp_op_utils.UpdateOptions.NO_UPDATE,
        begin_compression_step=1000,
        end_compression_step=2000,
        is_c_matrix_trainable=True,
        compress_input=False,
        compress_output=False,
        input_compression_factor=1,
        output_compression_factor=1,
        block_compression_factor=1,
        block_method='loop',
        add_summary=True,
        use_collection=False,)

  def add_compression_summaries(self):
    """Adds summaries."""
    with tf.name_scope(self._spec.name + '_summaries'):
      logging.info('add_compression_summaries scope name is %s',
                   self._spec.name)
      if self.a_matrix_tfvar is not None:
        tf.compat.v2.summary.scalar(
            _get_var_name(self.a_matrix_tfvar) + '/a_matrix_norm',
            tf.norm(self.a_matrix_tfvar))
      if self.c_matrix_tfvar is not None:
        tf.compat.v2.summary.scalar(
            _get_var_name(self.c_matrix_tfvar) + '/c_matrix_norm',
            tf.norm(self.c_matrix_tfvar))

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns compressed tensorflow operator for block diagonal compression.

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
    if self._spec.block_method == 'mask':
      num_blocks = self._spec.block_compression_factor
      # create c_matrix
      c_limit = np.sqrt(3.0 * (1 / np.max([
          1., (a_matrix_tfvar.shape[0] + a_matrix_tfvar.shape[1]) /
          (2. * num_blocks)
      ])))
      c_matrix = np.random.uniform(-c_limit, c_limit, size=a_matrix_tfvar.shape)
      # create block diagonal mask for c_matrix
      num_rows, num_cols = a_matrix_tfvar.shape.as_list()
      r_block, c_block = num_rows // num_blocks, num_cols // num_blocks
      c_mask = np.array([[
          float(j // c_block == i // r_block) for j in range(num_cols)
      ] for i in range(num_rows)])
    elif self._spec.block_method == 'loop':
      # create c_matrix, which is a rank 3 tensor
      c_limit = np.sqrt(3.0 * (1 / np.max([
          1., (a_matrix_tfvar.shape[0] + a_matrix_tfvar.shape[1]) /
          (2. * num_blocks)
      ])))
      num_blocks = self._spec.block_compression_factor
      c_matrix_shape = [
          num_blocks, a_matrix_tfvar.shape[0] // num_blocks,
          a_matrix_tfvar.shape[1] // num_blocks
      ]
      c_matrix = np.random.uniform(-c_limit, c_limit, size=c_matrix_shape)
    # convert c_matrix and c_mask from numpy arrays to tf tensors
    with tf.compat.v1.variable_scope(scope, use_resource=True):
      self.c_matrix_tfvar = tf.compat.v1.get_variable(
          'c_matrix',
          dtype=tf.float32,
          initializer=c_matrix.astype(np.float32),
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      self.a_matrix_tfvar = a_matrix_tfvar
      if self._spec.block_method == 'mask':
        self.c_mask_tfvar = tf.compat.v1.get_variable(
            'c_mask',
            dtype=tf.float32,
            initializer=c_mask.astype(np.float32),
            trainable=False)

    # update_op and final_op not necessary for BlockCompressionOp.
    self.update_op = tf.no_op()
    self.final_op = tf.no_op()

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope',
                                          a_matrix_tfvar_shape=None):
    """Returns compressed operator for block diagonal compression.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customized layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      a_matrix_tfvar_shape: A list specifying the shape of the tensor to
        compress. In some cases when a_matrix_tfvar is set to None,
        this field is used to pass in the shape of the matrix to compress.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    shape = (a_matrix_tfvar.shape if a_matrix_tfvar is not None
             else a_matrix_tfvar_shape)
    self.matrix_compressor = matrix_compressor
    with tf.variable_scope(scope) as scope:
      if self._spec.block_method == 'mask':
        # c_matrix has same shape as a_matrix
        c_matrix_pc = weight_params_fn(shape,
                                       weight_init_obj.Xavier(1.0),
                                       layer_obj.params.dtype)
        # create block diagonal mask for c_matrix
        num_blocks = self._spec.block_compression_factor
        num_rows, num_cols = shape.as_list()
        r_block, c_block = num_rows // num_blocks, num_cols // num_blocks
        c_mask = np.array(
            [[float(j // c_block == i // r_block)
              for j in range(num_cols)]
             for i in range(num_rows)])
        c_mask_pc = weight_params_fn(shape,
                                     weight_init_obj.Constant(c_mask),
                                     layer_obj.params.dtype)
      elif self._spec.block_method == 'loop':
        # c_matrix is a rank 3 tensor consisting of num_blocks blocks
        num_blocks = self._spec.block_compression_factor
        c_matrix_pc = weight_params_fn([
            num_blocks, shape[0] // num_blocks,
            shape[1] // num_blocks
        ], weight_init_obj.Xavier(1.0), layer_obj.params.dtype)

      # create the c_matrix and c_mask variables
      layer_obj.CreateVariable(
          'c_matrix_tfvar',
          c_matrix_pc,
          trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
      if self._spec.block_method == 'mask':
        layer_obj.CreateVariable('c_mask_tfvar', c_mask_pc, trainable=False)

      self.c_matrix_tfvar = layer_obj.vars.c_matrix_tfvar
      self.a_matrix_tfvar = a_matrix_tfvar
      if self._spec.block_method == 'mask':
        self.c_mask_tfvar = layer_obj.vars.c_mask_tfvar

    self.final_op = tf.no_op()
    if self._spec.add_summary:
      self.add_compression_summaries()
    self.update_op = tf.no_op()
    return [self.final_op, self.update_op]

  def get_apply_matmul(self, left_operand):
    """Returns block diagonal compressed TensorFlow node for matmul.

    This method performs matmul according to the compression
    procedure.

    Args:
      left_operand: a Tensor that is the left operand in matmul.

    Returns:
      matmul_op: a TensorFlow node that performs matmul of left_operand with the
      compressed a_matrix_tfvar.
    """
    if self._spec.block_method == 'mask':
      return tf.matmul(left_operand,
                       tf.multiply(self.c_matrix_tfvar, self.c_mask_tfvar))
    elif self._spec.block_method == 'loop':
      num_blocks = self._spec.block_compression_factor
      input_splitted = tf.split(left_operand, num_blocks, axis=-1)
      output_splitted = []
      for i, input_i in enumerate(input_splitted):
        output_splitted.append(tf.matmul(input_i, self.c_matrix_tfvar[i, :, :]))
      return tf.concat(output_splitted, axis=-1)

  def get_mix_operator(self, theta, concat, layer_obj=None):
    """Performs matrix multiplication on customized LSTM layers.

    This performs the block diagonal compressed equivalent of
    tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation. a rank 2 tensor.
      layer_obj: reference to the customized layer object. default is None.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    del layer_obj  # Unused within get_mix_operator here.
    if self._spec.block_method == 'mask':
      return tf.matmul(concat,
                       tf.multiply(theta.c_matrix_tfvar, theta.c_mask_tfvar))
    elif self._spec.block_method == 'loop':
      num_blocks = self._spec.block_compression_factor
      input_splitted = tf.split(concat, num_blocks, axis=-1)
      output_splitted = []
      for i, input_i in enumerate(input_splitted):
        output_splitted.append(
            tf.matmul(input_i, theta.c_matrix_tfvar[i, :, :]))
      return tf.concat(output_splitted, axis=-1)

  def get_matmul_operator(self,
                          inputs,
                          wm,
                          layer_obj,
                          transpose_a=False,
                          transpose_b=False):
    """Performs matrix multiplication for customized Softmax layers.

    This performs the block diagonal compressed equivalent of
    tf.matmul(inputs, wm).

    Args:
      inputs: the left operand of the matmul operation. a rank 2 tensor.
      wm: the right operand of the matmul operator. a rank 2 tensor.
      layer_obj: a lingvo layer object that handles variable creation.
      transpose_a: whether inputs tensor needs to be transposed before matmul.
      transpose_b: whether wm tensor needs to be transposed before matmul.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    theta = layer_obj.theta
    if transpose_a:
      inputs = tf.transpose(inputs)
    if transpose_b:
      wm = tf.transpose(wm)

    if self._spec.block_method == 'mask':
      return tf.matmul(inputs,
                       tf.multiply(theta.c_matrix_tfvar, theta.c_mask_tfvar))
    elif self._spec.block_method == 'loop':
      num_blocks = self._spec.block_compression_factor
      input_splitted = tf.split(inputs, num_blocks, axis=-1)
      output_splitted = []
      for i, input_i in enumerate(input_splitted):
        output_splitted.append(
            tf.matmul(input_i, theta.c_matrix_tfvar[i, :, :]))
      return tf.concat(output_splitted, axis=-1)

  def get_einsum_operator(self, inputs, layerobj):
    """Performs compressed matrix multiplication for customized ProjectionLayer.

    This performs the block diagonal compressed equivalent of
    tf.matmul(inputs, weight).

    Args:
      inputs: the left operand of the matmul operation.
      layerobj: the ProjectionLayer object from where get_einsum_operator is
        called.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    theta = layerobj.theta
    if self._spec.block_method == 'mask':
      return tf.matmul(inputs,
                       tf.multiply(theta.c_matrix_tfvar, theta.c_mask_tfvar))
    elif self._spec.block_method == 'loop':
      num_blocks = self._spec.block_compression_factor
      input_splitted = tf.split(inputs, num_blocks, axis=-1)
      output_splitted = []
      for i, input_i in enumerate(input_splitted):
        output_splitted.append(
            tf.matmul(input_i, theta.c_matrix_tfvar[i, :, :]))
      return tf.concat(output_splitted, axis=-1)


class MixedBlockCompressionOp(CompressionOp):
  """Implements a mixed block compression OP.

  Replaces the weight matrix with a block-diagonal matrix and outputs a linear
  combination of the outputs of the different blocks.


  compression_factor: integer.
    Factor by which number of (non-zero) parameters in weight matrix is reduced
    by.
  num_bases: integer.
    Numer of bases to use for computing the linear combination of the block
    outputs.
  """

  def __init__(self, scope='default_scope', spec=None, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      spec: compression hyper parameters default value given by
        self.get_default_hparams().
      global_step: tf variable that has the global step.
    """
    super(MixedBlockCompressionOp, self).__init__(scope, spec, global_step)
    # Compression specification
    self._spec = spec if spec else self.get_default_hparams()
    logging.info('Compression spec in init MixedBlockCompressionOp is: ')
    self.print_hparams()
    self._global_step = self._setup_global_step(global_step)

    # public member variables to track the compressor, the variables and
    # other tf nodes corresponding to this OP.
    self.matrix_compressor = None
    self.a_matrix_tfvar = None
    self.block_matrices = None
    self.linear_mixer = None
    self.final_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0

  @staticmethod
  def get_default_hparams():
    """Get a tf.HParams object with the default values for the hyperparameters.

      name: string
        name of the compression specification. Used for adding summaries and ops
        under a common tensorflow name_scope.
      use_tpu: False
        indicates whether to use TPU.
      compression_option: integer
        indicates what type of factorization (if any) is used.
      rank: integer
        indicates what type of factorization (if any) is used.
      update_option: integer
        indicates how the update logic is being run.
      TODO(wanxin): add doc strings for pruning hparams.

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_hparam.HParams(
        name='mixed_block_compression',
        compression_frequency=10,
        use_tpu=False,
        compression_option=comp_op_utils.CompressionOptions
        .MIXED_BLOCK_COMPRESSION,
        update_option=comp_op_utils.UpdateOptions.NO_UPDATE,
        begin_compression_step=1000,
        end_compression_step=2000,
        is_c_matrix_trainable=True,
        compression_factor=1,
        num_bases=1,
        add_summary=True,
        use_collection=False)

  def add_compression_summaries(self):
    """Adds summaries."""
    with tf.name_scope(self._spec.name + '_summaries'):
      logging.info('add_compression_summaries scope name is %s',
                   self._spec.name)
      tf.compat.v2.summary.scalar(
          _get_var_name(self.block_matrices) + '/block_matrices_norm',
          tf.norm(self.block_matrices))
      tf.compat.v2.summary.scalar(
          _get_var_name(self.linear_mixer) + '/linear_mixer_norm',
          tf.norm(self.linear_mixer))

  def get_apply_compression_op(self,
                               a_matrix_tfvar,
                               matrix_compressor,
                               scope='default_scope'):
    """Returns tensorflow operator for mixed block diagonal compression.

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

    # create block_matrices np array of shape [m, n, num_blocks]
    bm_limit = np.sqrt(3.0 * (1 / np.max(
        [1., (a_matrix_tfvar.shape[0] + a_matrix_tfvar.shape[1]) / 2.])))
    num_blocks = self._spec.compression_factor
    num_bases = self._spec.num_bases
    block_matrices_shape = [
        a_matrix_tfvar.shape[0] // num_blocks,
        a_matrix_tfvar.shape[1] // num_blocks,
        num_blocks
    ]
    block_matrices = np.random.uniform(
        -bm_limit, bm_limit, size=block_matrices_shape)

    # create linear_mixer np array of shape [num_blocks, num_blocks, num_bases]
    linear_mixer_shape = [num_blocks, num_blocks, num_bases]
    lm_limit = np.sqrt(3.0 * (1 / num_blocks))
    linear_mixer = np.random.uniform(
        -lm_limit, lm_limit, size=linear_mixer_shape)

    # convert block_matrices and linear_mixer from numpy arrays to tf tensors
    with tf.compat.v1.variable_scope(scope, use_resource=True):
      self.block_matrices = tf.compat.v1.get_variable(
          'block_matrices',
          dtype=tf.float32,
          initializer=block_matrices.astype(np.float32))
      self.a_matrix_tfvar = a_matrix_tfvar
      self.linear_mixer = tf.compat.v1.get_variable(
          'linear_mixer',
          dtype=tf.float32,
          initializer=linear_mixer.astype(np.float32))

    # update_op and final_op not necessary for MixedBlockCompressionOp.
    self.update_op = tf.no_op()
    self.final_op = tf.no_op()

    if self._spec.add_summary:
      self.add_compression_summaries()
    return [self.final_op, self.update_op]

  def get_customized_apply_compression_op(self,
                                          a_matrix_tfvar,
                                          matrix_compressor,
                                          layer_obj,
                                          weight_params_fn,
                                          weight_init_obj,
                                          scope='default_scope',
                                          a_matrix_tfvar_shape=None):
    """Returns compressed operator for block diagonal compression.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      matrix_compressor: MatrixCompressorInferface object to specify the
        compression algorithm. Must return two matrices b_matrix,c_matrix in its
        compression.
      layer_obj: a customized layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      a_matrix_tfvar_shape: A list specifying the shape of the tensor to
        compress. In some cases when a_matrix_tfvar is set to None,
        this field is used to pass in the shape of the matrix to compress.

    Returns:
      A TF node that has the compressed version of a_matrix_tfvar.
    """
    shape = a_matrix_tfvar_shape if a_matrix_tfvar_shape else a_matrix_tfvar.shape
    self.matrix_compressor = matrix_compressor
    with tf.variable_scope(scope) as scope:
      # block_matrices is a rank 3 tensor of shape [m, n, num_blocks]
      num_blocks = self._spec.compression_factor
      num_bases = self._spec.num_bases
      block_matrices_pc = weight_params_fn([
          shape[0] // num_blocks,
          shape[1] // num_blocks,
          num_blocks
      ], weight_init_obj.Xavier(1.0), layer_obj.params.dtype)

      # linear_mixer is a rank 3 tensor of
      # shape [num_blocks, num_blocks, num_bases]
      linear_mixer_pc = weight_params_fn([num_blocks, num_blocks, num_bases],
                                         weight_init_obj.Xavier(1.0),
                                         layer_obj.params.dtype)

      # create block_matrices and linear_mixer variables
      layer_obj.CreateVariable(
          'block_matrices',
          block_matrices_pc,
          trainable=True)
      layer_obj.CreateVariable(
          'linear_mixer',
          linear_mixer_pc,
          trainable=True)

      self.block_matrices = layer_obj.vars.block_matrices
      self.linear_mixer = layer_obj.vars.linear_mixer
      if a_matrix_tfvar is not None:  # will not exist if using no_original_w
        self.a_matrix_tfvar = a_matrix_tfvar

    if self._spec.add_summary:
      self.add_compression_summaries()

    # Do not need final_op and update_op for MixedBlockCompressionOp.
    self.final_op = tf.no_op()
    self.update_op = None
    return [self.final_op, self.update_op]

  def get_apply_matmul(self, left_operand):
    """Returns mixed block diagonal compressed TensorFlow node for matmul.

    This method performs matmul according to the compression
    procedure.

    Args:
      left_operand: a Tensor that is the left operand in matmul.

    Returns:
      matmul_op: a TensorFlow node that performs matmul of left_operand with the
      compressed a_matrix_tfvar.
    """
    # The linear mixer tensor is small one, typically of shape [2,2,1].
    # Performing einsum or a matmul with such a small tensor on TPUs
    # turned out to be worse for latency than writing out the matmul/einsum
    # using a loop. Hence we implement the latter strategy here.
    num_blocks = self._spec.compression_factor
    num_bases = self._spec.num_bases

    # block the left_operand tensor into num_blocks
    blocked_input = tf.reshape(left_operand, [
        tf.shape(left_operand)[0],
        tf.shape(left_operand)[1] // num_blocks,
        num_blocks,
    ])

    # compute the matmul of each block with corresponding block_matrix
    intermediate_splitted = tf.einsum('abc,bdc->adc', blocked_input,
                                      self.block_matrices)

    # compute the linear combinations of the above intermediate outputs
    output_splitted = []
    for k in range(num_bases):
      output_splitted.append([])
      for i in range(num_blocks):
        output_splitted[-1].append(intermediate_splitted[:, :, 0] *
                                   self.linear_mixer[i, 0, k])
        for j in range(1, num_blocks):
          output_splitted[-1][-1] = output_splitted[-1][
              -1] + intermediate_splitted[:, :, j] * self.linear_mixer[i, j, k]

    reduced_output_splitted = []
    for i in range(num_blocks):
      reduced = output_splitted[0][i]
      for k in range(1, num_bases):
        reduced += output_splitted[k][i]
      reduced_output_splitted.append(reduced)

    output = tf.concat(reduced_output_splitted, axis=-1)
    return output

  def get_mix_operator(self, theta, concat, layer_obj=None):
    """Performs matrix multiplication on customized LSTM layers.

    This performs the block diagonal compressed equivalent of
    tf.matmul(concat, theta.wm).

    Args:
      theta: object in customized layer that contains weight tensors, etc.
      concat: the left operand of the matmul operation. a rank 2 tensor.
      layer_obj: reference to the customized layer object. default is None.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(concat, theta.wm).
    """
    # The linear mixer tensor is small one, typically of shape [2,2,1].
    # Performing einsum or a matmul with such a small tensor on TPUs
    # turned out to be worse for latency than writing out the matmul/einsum
    # using a loop. Hence we implement the latter strategy here.

    num_blocks = self._spec.compression_factor
    num_bases = self._spec.num_bases

    # block the concat tensor into num_blocks
    blocked_input = tf.reshape(concat, [
        tf.shape(concat)[0],
        tf.shape(concat)[1] // num_blocks,
        num_blocks,
    ])

    # compute the matmul of each block with corresponding block_matrix
    intermediate_splitted = tf.einsum('abc,bdc->adc', blocked_input,
                                      theta.block_matrices)

    # compute the linear combinations of the above intermediate outputs
    output_splitted = []
    for k in range(num_bases):
      output_splitted.append([])
      for i in range(num_blocks):
        output_splitted[-1].append(intermediate_splitted[:, :, 0] *
                                   theta.linear_mixer[i, 0, k])
        for j in range(1, num_blocks):
          output_splitted[-1][-1] = output_splitted[-1][
              -1] + intermediate_splitted[:, :, j] * theta.linear_mixer[i, j, k]

    reduced_output_splitted = []
    for i in range(num_blocks):
      reduced = output_splitted[0][i]
      for k in range(1, num_bases):
        reduced += output_splitted[k][i]
      reduced_output_splitted.append(reduced)

    output = tf.concat(reduced_output_splitted, axis=-1)
    return output

  def get_matmul_operator(self,
                          inputs,
                          wm,
                          layer_obj,
                          transpose_a=False,
                          transpose_b=False):
    """Performs matrix multiplication for customized Softmax layers.

    This performs the block diagonal compressed equivalent of
    tf.matmul(inputs, wm).

    Args:
      inputs: the left operand of the matmul operation. a rank 2 tensor.
      wm: the right operand of the matmul operator. a rank 2 tensor.
      layer_obj: a lingvo layer object that handles variable creation.
      transpose_a: whether inputs tensor needs to be transposed before matmul.
      transpose_b: whether wm tensor needs to be transposed before matmul.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    # MixedBlockCompressionOp does not need to use the wm, transpose_a
    # and transpose_b arguments to the function.
    del wm, transpose_a, transpose_b

    theta = layer_obj.theta
    # The linear mixer tensor is small one, typically of shape [2,2,1].
    # Performing einsum or a matmul with such a small tensor on TPUs
    # turned out to be worse for latency than writing out the matmul/einsum
    # using a loop. Hence we implement the latter strategy here.
    num_blocks = self._spec.compression_factor
    num_bases = self._spec.num_bases

    # block the inputs tensor into num_blocks
    blocked_input = tf.reshape(inputs, [
        tf.shape(inputs)[0],
        tf.shape(inputs)[1] // num_blocks,
        num_blocks,
    ])

    # compute the matmul of each block with corresponding block_matrix
    intermediate_splitted = tf.einsum('abc,bdc->adc', blocked_input,
                                      theta.block_matrices)

    # compute the linear combinations of the above intermediate outputs
    output_splitted = []
    for k in range(num_bases):
      output_splitted.append([])
      for i in range(num_blocks):
        output_splitted[-1].append(intermediate_splitted[:, :, 0] *
                                   theta.linear_mixer[i, 0, k])
        for j in range(1, num_blocks):
          output_splitted[-1][-1] = output_splitted[-1][
              -1] + intermediate_splitted[:, :, j] * theta.linear_mixer[i, j, k]

    reduced_output_splitted = []
    for i in range(num_blocks):
      reduced = output_splitted[0][i]
      for k in range(1, num_bases):
        reduced += output_splitted[k][i]
      reduced_output_splitted.append(reduced)

    output = tf.concat(reduced_output_splitted, axis=-1)
    return output

  def get_einsum_operator(self, inputs, layerobj):
    """Performs compressed matrix multiplication for customized ProjectionLayer.

    This performs the block diagonal compressed equivalent of
    tf.matmul(inputs, weight).

    Args:
      inputs: the left operand of the matmul operation.
      layerobj: the ProjectionLayer object from where get_einsum_operator is
        called.

    Returns:
      A TensorFlow node that has compressed version of
      tf.matmul(inputs, wm).
    """
    # The linear mixer tensor is small one, typically of shape [2,2,1].
    # Performing einsum or a matmul with such a small tensor on TPUs
    # turned out to be worse for latency than writing out the matmul/einsum
    # using a loop. Hence we implement the latter strategy here.
    theta = layerobj.theta
    num_blocks = self._spec.compression_factor
    num_bases = self._spec.num_bases

    # split the inputs tensor into num_blocks along its last axis
    input_splitted = tf.split(inputs, num_blocks, axis=-1)

    # compute the matmul of each block with corresponding block_matrix
    intermediate_splitted = []
    for i, input_i in enumerate(input_splitted):
      intermediate_splitted.append(
          tf.matmul(input_i, theta.block_matrices[:, :, i]))

    # compute the linear combinations of the above intermediate outputs
    output_splitted = []
    for k in range(num_bases):
      output_splitted.append([])
      for i in range(num_blocks):
        output_splitted[-1].append(intermediate_splitted[0] *
                                   theta.linear_mixer[i, 0, k])
        for j in range(1, num_blocks):
          output_splitted[-1][-1] = output_splitted[
              -1][-1] + intermediate_splitted[j] * theta.linear_mixer[i, j, k]

    reduced_output_splitted = []
    for i in range(num_blocks):
      reduced = output_splitted[0][i]
      for k in range(1, num_bases):
        reduced += output_splitted[k][i]
      reduced_output_splitted.append(reduced)

    output = tf.concat(reduced_output_splitted, axis=-1)
    return output
