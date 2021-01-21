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
import copy
from absl import logging

import numpy as np
from tensor2tensor.utils.hparam import HParams
import tensorflow.compat.v2 as tf


class MatrixCompressorInferface(object):
  """Interface for any matrix compressor algorithm.

  This MatrixCompressorInferface class can be implemented by any third party to
  implement any compression algorithm.
  """

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
    return HParams(
        name='model_compression',
        rank=100,
        num_rows=10,
        num_cols=10,
        use_tpu=False,
        compressor_option=0,
        is_b_matrix_trainable=True,
        is_c_matrix_trainable=True,
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
    u, s, vh = np.linalg.svd(a_matrix)

    # If matrix dimension is smaller than rank specified then adjust rank
    rank = max(min(np.min(a_matrix.shape), self._spec.rank), 1)
    # rank = comp_op_utils.compute_compressed_rank_from_matrix_shape(
    #     a_matrix.shape, self._spec.rank)
    b_matrix = u[:, :rank]
    c_matrix = vh[:rank, :]
    s_mat = np.diag(np.sqrt(s[:rank]))
    b_matrix = np.matmul(b_matrix, s_mat)
    c_matrix = np.matmul(s_mat, c_matrix)
    logging.info(
        'Inside static_matrix_compressor: a_matrix,b_matrix,c_matrix shapes '
        'are: %s, %s, %s', a_matrix.shape, b_matrix.shape, c_matrix.shape)

    self.uncompressed_size = tf.size(a_matrix)
    self.compressed_size = b_matrix.size + c_matrix.size

    return [b_matrix, c_matrix]


class CompressionOpInterface(object):
  """Interface for a compression op.

  Class to take a matrix compression algorithm and create a tensorflow
  compression operator to inject that compression dynamically during training.
  The compression algorithm is specified using an object of
  MatrixCompressorInferface class.
  """

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

  def __init__(self,
               scope='default_scope',
               spec=None,
               global_step=None,
               layer=None):
    """Initializer.

    Args:
      scope: TF scope used for creating new TF variables.
      spec: compression hyper parameters default value given by
        self.get_default_hparams().
      global_step: tf variable that has the global step.
      layer: Layer to compress.
    """
    super(CompressionOp, self).__init__(scope, spec, global_step)
    # Compression specification
    self._spec = spec

    # Sanity check for compression hparams
    self._validate_spec()
    self._global_step = global_step

    # public member variables to track the compressor, the variables and
    # other tf nodes corresponding to this OP.
    self.matrix_compressor = None
    self.a_matrix_tfvar = None
    self.b_matrix_tfvar = None
    self.c_matrix_tfvar = None
    self.alpha = None
    self.layer = layer
    self.last_alpha_update_step = None
    self.uncompressed_size = 0
    self.compressed_size = 0

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

    Returns:
      tf.HParams object initialized to default values.

    """
    return HParams(
        name='model_compression',
        alpha_decrement_value=0.01,
        begin_compression_step=0,
        end_compression_step=-1,
        compression_frequency=10,
        use_tpu=False,
        compression_option=0,
        rank=100,
        update_option=0,
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
        prune_option='weight')

  def setup_variables(self, a_matrix_tfvar, matrix_compressor, layer):
    """Create compressed layer weight matrices."""

    self.matrix_compressor = matrix_compressor
    a_matrix = np.zeros(shape=a_matrix_tfvar.shape)
    [b_matrix, c_matrix] = matrix_compressor.static_matrix_compressor(a_matrix)

    self.b_matrix_tfvar = layer.add_weight(
        'b_matrix',
        shape=b_matrix.shape,
        initializer=layer.kernel_initializer,
        regularizer=layer.kernel_regularizer,
        constraint=layer.kernel_constraint,
        dtype=layer.dtype,
        trainable=True)
    self.c_matrix_tfvar = layer.add_weight(
        'c_matrix',
        shape=c_matrix.shape,
        initializer=layer.kernel_initializer,
        regularizer=layer.kernel_regularizer,
        constraint=layer.kernel_constraint,
        dtype=layer.dtype,
        trainable=True)

    self.alpha = layer.add_weight(
        'alpha',
        shape=(),
        initializer=tf.keras.initializers.Ones(),
        dtype=layer.dtype,
        trainable=False)

    self.last_alpha_update_step = layer.add_weight(
        'last_alpha_update_step',
        shape=(),
        initializer=tf.keras.initializers.Constant(value=-1),
        dtype=tf.int32,
        trainable=False)

    self.a_matrix_tfvar = a_matrix_tfvar
    self.layer.alpha = self.alpha

  def compressed_matmul_keras(self, inputs):
    """Matmul with a convex combination of original and compressed weights."""
    compressed_mat = self.alpha * self.a_matrix_tfvar + (
        1 - self.alpha) * tf.matmul(self.b_matrix_tfvar, self.c_matrix_tfvar)
    return tf.matmul(inputs, compressed_mat)

  def maybe_run_update_step(self):
    """Creates TensorFlow update op for compression."""

    def maybe_update_alpha():
      """Maybe update the alpha param.

      Checks if global_step is between begin_compression_step and
      end_compression_step, and if the current training step is a
      compression step.

      Returns:
        Boolean tensor whether the training step is a compression step.
      """
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
          tf.add(self.last_alpha_update_step, self._spec.compression_frequency),
          tf.cast(self._global_step, tf.int32))
      return tf.logical_and(is_step_within_compression_range,
                            is_compression_step)

    def no_update_op():
      pass

    def compressor_and_alpha_update_op_fn():
      return self._compressor_and_alpha_update_op()

    tf.cond(
        pred=maybe_update_alpha(),
        true_fn=compressor_and_alpha_update_op_fn,
        false_fn=no_update_op)
    return

  def _compressor_op(self, matrix_compressor, a_matrix_tfvar):
    """Creates compressor op based on matrix_compressor.

    Meant to create the factors once at begin_compression_step.
    Args:
      matrix_compressor: specifies the matrix compressor object.
      a_matrix_tfvar: the tf tensor to be compressed.
    """

    [b_matrix_out, c_matrix_out
    ] = tf.compat.v1.py_function(matrix_compressor.static_matrix_compressor,
                                 [a_matrix_tfvar], [tf.float32, tf.float32])

    self.b_matrix_tfvar.assign(b_matrix_out)
    self.c_matrix_tfvar.assign(c_matrix_out)
    return

  def _update_alpha_op(self):
    self.alpha.assign_sub(self._spec.alpha_decrement_value, 0)
    self.alpha.assign(tf.math.maximum(self.alpha, 0))
    return

  def _compressor_and_alpha_update_op(self):
    """Applies compressor and also updates alpha."""

    self._compressor_op(self.matrix_compressor, self.a_matrix_tfvar)
    self._update_alpha_op()
    self.last_alpha_update_step.assign(tf.cast(self._global_step, tf.int32))

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
    logging.info('Entering ApplyCompression constructor')
    self._compression_op_spec = compression_spec
    self._scope = scope
    self._global_step = global_step
    self._matrix_compressor = compressor
    self._compression_ops = []
    self._update_ops = []
    self._all_update_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0

  def apply_compression_keras(self,
                              a_matrix_tfvar,
                              scope='default_scope',
                              layer=None):
    """keras version of apply_compression.

    Applies matrix compression OP on
    a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      scope: TF scope used for creating new TF variables.
      layer: keras layer object calling this function. Must support an
        add_weight method.

    Returns:
      TF node that represents the compressed version of a_matrix_tfvar.
    """
    if self._compression_op_spec.compression_option == 9:
      raise NotImplementedError('InputCompression not Supported.')
    else:
      c = CompressionOp(
          scope=scope,
          spec=self._compression_op_spec,
          global_step=self._global_step,
          layer=layer)
    c.setup_variables(a_matrix_tfvar, self._matrix_compressor, layer=layer)
    return c

  def get_operator_hparam(self, hparam):
    """Returns the value of queried hparam of the compression operator."""
    return self._compression_op_spec.get(hparam)

  def get_compression_ops(self):
    """Returns the compression operators used during the update steps.

    Returns:
      A list of CompressionOp objects.
    """
    return copy.copy(self._compression_ops)

  def get_spec(self):
    """Get the spec / hparams used to create the Pruning object."""
    return self._compression_op_spec
