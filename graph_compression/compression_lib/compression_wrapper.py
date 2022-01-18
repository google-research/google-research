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

"""Helper class that wraps around multiple different compression operators.

This is the Python 2 only version of compression_wrapper.py.

This allows for easier testing of different operators. Rather than importing
each operator separately, this class can be used and different
compression_option values can be passed in to specifiy the operator type.

compression_option:
  0 - No Compression
  1 - LowRankDecompMatrixCompression
  2 - SimhashMatrixCompression
  3 - DLMatrixCompression
  4 - KmeansMatrixCompression
  8 - KmeansAndPruningMatrixCompression
  9 - InputOutputCompression
  10 - BlockCompression
"""

from __future__ import absolute_import

import copy

from absl import logging
import tensorflow.compat.v1 as tf
from graph_compression.compression_lib import compression_op as comp_op
from graph_compression.compression_lib import compression_op_utils as comp_op_utils
from graph_compression.compression_lib import dl_compression_op as dl_comp_op
from graph_compression.compression_lib import simhash_compression_op as simhash_comp_op


CompressionOptions = comp_op_utils.CompressionOptions
UpdateOptions = comp_op_utils.UpdateOptions

# Map from CompressionOptions to corresponding compression op classes.
COMP_OP_MAP = {
    CompressionOptions.LOWRANK_MATRIX_COMPRESSION:
        comp_op.CompressionOp,
    CompressionOptions.SIMHASH_MATRIX_COMPRESSION:
        simhash_comp_op.SimhashCompressionOp,
    CompressionOptions.DL_MATRIX_COMPRESSION:
        dl_comp_op.DLCompressionOp,
    CompressionOptions.KMEANS_MATRIX_COMPRESSION:
        simhash_comp_op.KMeansCompressionOp,
    CompressionOptions.KMEANS_AND_PRUNING_MATRIX_COMPRESSION:
        simhash_comp_op.KMeansPruningCompressionOp,
    CompressionOptions.INPUTOUTPUT_COMPRESSION:
        comp_op.InputOutputCompressionOp,
    CompressionOptions.BLOCK_COMPRESSION:
        comp_op.BlockCompressionOp,
    CompressionOptions.MIXED_BLOCK_COMPRESSION:
        comp_op.MixedBlockCompressionOp,
}


def get_apply_compression(compression_op_spec, global_step):
  """Returns apply_compression operation matching compression_option input."""
  compressor_spec = comp_op.LowRankDecompMatrixCompressor.get_default_hparams()
  if compression_op_spec.__contains__('rank'):
    compressor_spec.set_hparam('rank', compression_op_spec.rank)
  if compression_op_spec.__contains__('block_size'):
    compressor_spec.set_hparam('block_size', compression_op_spec.block_size)
  logging.info('Compressor spec %s', compressor_spec.to_json())
  logging.info('Compression operator spec %s', compression_op_spec.to_json())

  if compression_op_spec.compression_option not in list(CompressionOptions):
    # if unknown compression_option is given, default to low rank compression.
    logging.info(
        'Compression_option %s not in expected options: %s. '
        'Will use low_rank decomp by default.',
        str(compression_op_spec.compression_option),
        ','.join([str(opt) for opt in CompressionOptions]))
    compression_op_spec.compression_option = CompressionOptions.LOWRANK_MATRIX_COMPRESSION

  apply_compression = None
  if compression_op_spec.compression_option == CompressionOptions.LOWRANK_MATRIX_COMPRESSION:
    compressor = comp_op.LowRankDecompMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.SIMHASH_MATRIX_COMPRESSION:
    compressor_spec.set_hparam('is_b_matrix_trainable', False)
    compressor = simhash_comp_op.SimhashMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.KMEANS_MATRIX_COMPRESSION:
    compressor_spec.set_hparam('is_b_matrix_trainable', True)
    compressor = simhash_comp_op.KmeansMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.KMEANS_AND_PRUNING_MATRIX_COMPRESSION:
    compressor_spec.set_hparam('is_b_matrix_trainable', True)
    compressor = simhash_comp_op.KmeansMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.INPUTOUTPUT_COMPRESSION:
    compressor_spec.set_hparam('is_b_matrix_trainable', True)
    compressor_spec.set_hparam('is_c_matrix_trainable', True)
    compressor_spec.set_hparam('is_d_matrix_trainable', True)
    compressor = comp_op.LowRankDecompMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.BLOCK_COMPRESSION:
    compressor_spec.set_hparam('is_c_matrix_trainable', True)
    compressor = comp_op.LowRankDecompMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.MIXED_BLOCK_COMPRESSION:
    compressor_spec.set_hparam('is_c_matrix_trainable', True)
    compressor = comp_op.LowRankDecompMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == CompressionOptions.DL_MATRIX_COMPRESSION:
    compressor = dl_comp_op.DLMatrixCompressor(spec=compressor_spec)
    apply_compression = ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)

  return apply_compression


class ApplyCompression(object):
  """Wrapper class which handles creation of compression ops for different layers in a model.

  This is to repeatedly invoke various compression operators for different
  layers in a model.

  Intialized by specifying the default compressor and compression_spec to use.
  The spec and compressor for initializing the operator for each layer can
  be unique to that layer.

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
    logging.debug('Entering ApplyCompression constructor.')
    self._compression_op_spec = compression_spec
    self._scope = scope
    self._global_step = global_step
    self._matrix_compressor = compressor
    self._compression_ops = []
    self._update_ops = []
    self._all_update_op = None

    self.uncompressed_size = 0
    self.compressed_size = 0

  def apply_compression(self,
                        a_matrix_tfvar,
                        scope='default_scope',
                        spec=None,
                        compressor=None):
    """Applies matrix compression OP on a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      scope: TF scope used for creating new TF variables.
      spec: spec to be used for the compression op. this is optional. if
            not provided, self._compression_op_spec is used.
      compressor: matrix_compressor to for the compression op. this is optional.
                  if not provided, self._matrix_compressor is used.

    Returns:
      TF node that represents the compressed version of a_matrix_tfvar.
    """
    compression_op_spec = spec if spec else self._compression_op_spec
    matrix_compressor = compressor if compressor else self._matrix_compressor
    if compression_op_spec.compression_option in COMP_OP_MAP:
      c = COMP_OP_MAP[compression_op_spec.compression_option](
          scope=scope, spec=compression_op_spec, global_step=self._global_step)
    else:
      c = None

    self._compression_ops.append(c)
    [a_matrix_compressed, a_matrix_update_op] = c.get_apply_compression_op(
        a_matrix_tfvar, matrix_compressor, scope=scope)
    if compression_op_spec.update_option in [
        UpdateOptions.TF_UPDATE, UpdateOptions.TF_AND_PYTHON_UPDATE
    ]:
      self._update_ops.append(a_matrix_update_op)

    self.uncompressed_size += c.uncompressed_size
    self.compressed_size += c.compressed_size

    return a_matrix_compressed

  def customized_apply_compression(self,
                                   a_matrix_tfvar,
                                   layer_obj,
                                   weight_params_fn,
                                   weight_init_obj,
                                   scope='default_scope',
                                   spec=None,
                                   compressor=None):
    """Applies matrix compression OP on a_matrix_tfvar as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      layer_obj: a customized layer object that handles variable creation.
      weight_params_fn: functional handle to create model parameters.
      weight_init_obj: a weight initialization object.
      scope: TF scope used for creating new TF variables.
      spec: spec to be used for the compression op. this is optional.
            if not provided, self._compression_op_spec is used.
      compressor: matrix_compressor to for the compression op. this is optional.
                  if not provided, self._matrix_compressor is used.

    Returns:
      TF node that represents the compressed version of a_matrix_tfvar.
    """
    compression_op_spec = spec if spec else self._compression_op_spec
    matrix_compressor = compressor if compressor else self._matrix_compressor
    if compression_op_spec.compression_option in COMP_OP_MAP:
      c = COMP_OP_MAP[compression_op_spec.compression_option](
          scope=scope, spec=compression_op_spec, global_step=self._global_step)
    else:
      c = None

    self._compression_ops.append(c)
    [a_matrix_compressed,
     a_matrix_update_op] = c.get_customized_apply_compression_op(
         a_matrix_tfvar,
         matrix_compressor,
         layer_obj,
         weight_params_fn,
         weight_init_obj,
         scope=scope)
    if compression_op_spec.update_option in [
        UpdateOptions.TF_UPDATE, UpdateOptions.TF_AND_PYTHON_UPDATE
    ]:
      self._update_ops.append(a_matrix_update_op)

    self.uncompressed_size = self.uncompressed_size + c.uncompressed_size
    self.compressed_size = self.compressed_size + c.compressed_size

    return a_matrix_compressed

  def apply_compression_keras(self,
                              a_matrix_tfvar,
                              scope='default_scope',
                              layer=None,
                              spec=None,
                              compressor=None):
    """Keras version of the `apply_compression` method.

    Applies the matrix compression OP on `a_matrix_tfvar` as specified in spec.

    Args:
      a_matrix_tfvar: TF variable representing a tensor variable in a model.
      scope: TF scope used for creating new TF variables.
      layer: keras layer object calling this function. Must support an
         add_weight method.
      spec: spec to be used for the compression op. this is optional.
            if not provided, self._compression_op_spec is used.
      compressor: matrix_compressor to for the compression op. this is optional.
                  if not provided, self._matrix_compressor is used.

    Returns:
      TF node that represents the compressed version of `a_matrix_tfvar`.
    """
    compression_op_spec = spec if spec else self._compression_op_spec
    matrix_compressor = compressor if compressor else self._matrix_compressor
    if compression_op_spec.compression_option in COMP_OP_MAP:
      c = COMP_OP_MAP[compression_op_spec.compression_option](
          scope=scope, spec=compression_op_spec, global_step=self._global_step)
    else:
      c = None

    self._compression_ops.append(c)
    [a_matrix_compressed,
     a_matrix_update_op] = c.get_apply_compression_op_keras(
         a_matrix_tfvar, matrix_compressor, layer=layer)
    self._update_ops.append(a_matrix_update_op)

    self.uncompressed_size += c.uncompressed_size
    self.compressed_size += c.compressed_size

    return a_matrix_compressed

  def get_last_compression_op(self):
    return self._compression_ops[-1]

  def get_mix_operator(self, theta, concat):
    # TODO(nishanthd): remove this function. should ideally never be used.
    return self._compression_ops[-1].get_mix_operator(theta, concat)

  def get_matmul_operator(self,
                          a,
                          b,
                          lstmobj,
                          transpose_a=False,
                          transpose_b=False):
    # TODO(nishanthd): remove this function. should ideally never be used.
    return self._compression_ops[-1].get_matmul_operator(
        a, b, lstmobj, transpose_a, transpose_b)

  def get_einsum_operator(self, inputs, weight, equation, layerobj):
    return self._compression_ops[-1].get_einsum_operator(
        inputs, weight, equation, layerobj)

  def all_update_op(self):
    """Returns the combine update tf OP."""
    # TODO(nishanthd): implement all_update_op logic inside the wrapper
    with tf.compat.v1.name_scope(self._scope):
      with tf.control_dependencies(self._update_ops):
        logging.info('Updating all compression_ops.')
        self._all_update_op = tf.no_op('update_all_compression_ops')
    return self._all_update_op

  def run_update_step(self, session=None, step_number=None):
    """Returns the combine update tf OP."""
    logging.debug('Running ApplyCompression\'s run_update_step step_num is %s',
                  step_number)

    for c_op in self._compression_ops:
      if c_op._spec.update_option != UpdateOptions.NO_UPDATE:  # pylint: disable=protected-access
        logging.debug('Running run_update_step step_num is %s', step_number)
        c_op.run_update_step(session=session, step_number=step_number)
        logging.info('Finished running run_update_step step_num is %s',
                     step_number)

  def get_operator_hparam(self, hparam):
    """Returns the value of queried hparam of the compression operator."""
    # TODO(nishanthd): check if this function is necessary.
    # perhaps change it to a version which reads the spec from a compression_op
    # or from an index into the list of self._compression_ops.
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
