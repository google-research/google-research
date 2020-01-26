# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import abc
import copy

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from graph_compression.compression_lib import compression_op_utils as comp_op_utils
from tensorflow.contrib import training as contrib_training


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
    return contrib_training.HParams(
        name='model_compression',
        rank=100,
        num_rows=10,
        num_cols=10,
        use_tpu=False,
        compressor_option=0,
        is_b_matrix_trainable=True,
        is_c_matrix_trainable=True)

  def static_matrix_compressor(self, a_matrix):
    """Low-rank decomposition of a_matrix.

    Args:
      a_matrix: input matrix.

    Returns:
      A list [b_matrix,c_matrix] which is the low-rank decomposition of
      a_matrix. Rank is taken from spec.rank.
    """
    u, s, vh = np.linalg.svd(a_matrix)
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
    s, u, v = tf.linalg.svd(a_matrix)
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

  def get_update_op(self):
    """Update operator.

    Returns:
      TF operator that implements the update steps that may need to
      be applied periodically.
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
    self._spec = spec if spec else self.get_compression_hparams()
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

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_training.HParams(
        name='model_compression',
        alpha_decrement_value=0.01,
        begin_compression_step=0,
        end_compression_step=-1,
        compression_frequency=10,
        use_tpu=False,
        compression_option=0,
        rank=7,
        update_option=0)

  def add_compression_summaries(self):
    """Adds summaries of alpha value, new variables, and last update step."""
    with tf.compat.v1.name_scope(self._spec.name + '_summaries'):
      tf.compat.v1.summary.scalar(
          self._last_alpha_update_step.op.name + '/last_alpha_update_step',
          self._last_alpha_update_step)
      tf.compat.v1.summary.scalar(self.alpha.op.name + '/alpha', self.alpha)
      tf.compat.v1.summary.scalar(
          self.a_matrix_tfvar.op.name + '/a_matrix_norm',
          tf.norm(tensor=self.a_matrix_tfvar))
      tf.compat.v1.summary.scalar(
          self.b_matrix_tfvar.op.name + '/b_matrix_norm',
          tf.norm(tensor=self.b_matrix_tfvar))
      tf.compat.v1.summary.scalar(
          self.c_matrix_tfvar.op.name + '/c_matrix_norm',
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
              self._global_step,
              name='last_alpha_update_step_assign')
      ]):
        with tf.control_dependencies([self._alpha_assign_op()]):
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
    """
    # py_func is not supported on TPU so need non py_func implementation
    use_tpu = self._spec.use_tpu
    # Seeing some tf.py_func error because of which the
    # following may be needed, so enforcing TF operation updates.
    use_tpu = True
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
            tf.greater_equal(self._global_step,
                             self._spec.begin_compression_step),
            tf.logical_or(
                tf.less_equal(self._global_step,
                              self._spec.end_compression_step),
                tf.less(self._spec.end_compression_step, 0)))
        is_compression_step = tf.less_equal(
            tf.add(self._last_alpha_update_step,
                   self._spec.compression_frequency), self._global_step)
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

    self.update_op = self._create_update_op()

    self.final_op = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.matmul(self.b_matrix_tfvar, self.c_matrix_tfvar)

    self.add_compression_summaries()
    return [self.final_op, self.update_op]

  @staticmethod
  def all_update_op(update_ops_list, scope='default_scope'):
    """Method to create a complete update op.

    Args:
        update_ops_list: list of individual update ops.
        scope: tf scope for creating update op.
    """
    with tf.compat.v1.name_scope(scope):
      with tf.control_dependencies(update_ops_list):
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
      graph_global_step = tf.get_global_step()
    logging.info('graph_global_step: %s', graph_global_step)
    return tf.cast(graph_global_step, tf.int32)

  def print_hparams(self):
    logging.info(self._spec.to_json())


class ApplyCompression(object):
  """Wrapper class.

  This is to repeatedly invoke above compression operator to different
  layers in a model.

  Intialized by specifying the compressor and compression_spec.

  After that apply_compression can be called several times for different
  matrices in the model.

  Finally all_update_op returns the combined update OP from all these
  compressions.
  """

  def __init__(self, scope, compression_spec, compressor, global_step=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      compression_spec: compression hyper parameters.
      compressor: matrix compressor object of class MatrixCompressorInferface.
      global_step: tf variable that has the global step.
    """
    self._compression_op_spec = compression_spec
    self._scope = scope
    self._global_step = global_step
    self._matrix_compressor = compressor
    self._compression_ops = []
    self._update_ops = []
    self._all_update_op = None

  def apply_compression(self, a_matrix_tfvar, scope='default_scope'):
    """Applies matrix compression OP on a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      scope: TF scope used for creating new TF variables.

    Returns:
      TF node that represents the compressed version of a_matrix_tfvar.
    """
    c = CompressionOp(
        spec=self._compression_op_spec, global_step=self._global_step)
    self._compression_ops.append(c)
    [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
        a_matrix_tfvar, self._matrix_compressor, scope=scope)
    self._update_ops.append(a_matrix_update_op)
    return a_matrix_compressed

  def all_update_op(self):
    """Returns the combine update tf OP."""
    self._all_update_op = CompressionOp.all_update_op(self._update_ops,
                                                      self._scope)
    return self._all_update_op

  def get_operator_hparam(self, hparam):
    """Returns the value of queried hparam of the compression operator."""
    return self._compression_op_spec.get(hparam)

  def get_compression_ops(self):
    """Returns the compression operators used during the update steps.

    Returns:
      A list of CompressionOp objects.
    """
    return copy.copy(self._compression_ops)


class CompressionOpEager(tf.keras.layers.Layer):
  """CompressionOp class that supports eager execution.

  It replaces the alpha_update_op in CompressionOp by an explicit
  run_update_step method, and relies on clients to pass in a step_number.
  """

  def __init__(self, last_alpha_update_step=-1, spec=None):
    super(CompressionOpEager, self).__init__()

    self._spec = spec if spec else self.get_default_hparams()
    logging.info('Compression spec in init CompressionOpEager is: %s',
                 self._spec)

    self.last_alpha_update_step = tf.Variable(
        last_alpha_update_step, dtype=tf.int32, trainable=False)
    self.alpha = tf.Variable(1.0, dtype=tf.float32, trainable=False)

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

    Returns:
      tf.HParams object initialized to default values.

    """
    return contrib_training.HParams(
        name='model_compression',
        alpha_decrement_value=0.01,
        begin_compression_step=0,
        end_compression_step=-1,
        compression_frequency=10,
        use_tpu=False,
        compression_option=0,
        rank=7,
        update_option=0)

  def set_up_variables(self, a_matrix_tfvar, matrix_compressor):
    """Creates compression specific variables.

    Args:
      a_matrix_tfvar: the matrix to be compressed, Tensor
      matrix_compressor: a matrix compressor object (instance of a sub-class of
        MatrixCompressorInterface)
    """
    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    b_matrix, c_matrix = matrix_compressor.static_matrix_compressor(a_matrix)
    self.b_matrix_tfvar = tf.Variable(
        b_matrix,
        name='b_matrix',
        dtype=tf.float32,
        trainable=self.matrix_compressor.get_spec().is_b_matrix_trainable)
    self.c_matrix_tfvar = tf.Variable(
        c_matrix,
        name='c_matrix',
        dtype=tf.float32,
        trainable=self.matrix_compressor.get_spec().is_c_matrix_trainable)
    self.a_matrix_tfvar = a_matrix_tfvar

  @tf.function
  def _get_apply_compression(self, alpha, a_matrix_tfvar, b_matrix_tfvar,
                             c_matrix_tfvar):
    final_op = alpha * a_matrix_tfvar + (1 - alpha) * tf.matmul(
        b_matrix_tfvar, c_matrix_tfvar)
    return final_op

  def get_apply_compression(self):
    self.final_op = self._get_apply_compression(self.alpha, self.a_matrix_tfvar,
                                                self.b_matrix_tfvar,
                                                self.c_matrix_tfvar)
    return self.final_op

  def run_update_step(self, step_number):
    """Run matrix and alpha update step if criterion is met.

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
