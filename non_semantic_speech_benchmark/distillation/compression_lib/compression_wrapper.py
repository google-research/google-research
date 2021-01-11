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

"""Helper class that wraps around multiple different compression operators.

This allows for easier testing of different operators. Rather than importing
each operator separately, this class can be used and different
compression_option values can be passed in to specifiy the operator type.

compression_option:
  1 - LowRankDecompMatrixCompressor
  2 - SimhashMatrixCompressor
  3 - DLMatrixCompressor
  4 - KmeansMatrixCompressor
  8 - KmeansAndPruningMatrixCompressor
  9 - InputCompressor
"""

from __future__ import absolute_import

from absl import logging
from non_semantic_speech_benchmark.distillation.compression_lib import compression_op as comp_op

_COMPRESSION_OPTIONS = [1, 2, 3, 4, 8, 9]


def get_apply_compression(compression_op_spec, global_step):
  """Returns apply_compression operation matching compression_option input."""
  compressor_spec = comp_op.LowRankDecompMatrixCompressor.get_default_hparams()
  if compression_op_spec.__contains__('rank'):
    compressor_spec.set_hparam('rank', compression_op_spec.rank)
  if compression_op_spec.__contains__('block_size'):
    compressor_spec.set_hparam('block_size', compression_op_spec.block_size)
  logging.info('Compressor spec %s', compressor_spec.to_json())
  logging.info('Compression operator spec %s', compression_op_spec.to_json())

  if compression_op_spec.compression_option not in _COMPRESSION_OPTIONS:
    logging.info(
        'Compression_option %s not in expected options: %s. '
        'Will use low_rank decomp by default.',
        str(compression_op_spec.compression_option),
        ','.join([str(opt) for opt in _COMPRESSION_OPTIONS]))
    compression_op_spec.compression_option = 1

  apply_compression = None
  if compression_op_spec.compression_option == 1:
    compressor = comp_op.LowRankDecompMatrixCompressor(spec=compressor_spec)
    apply_compression = comp_op.ApplyCompression(
        scope='default_scope',
        compression_spec=compression_op_spec,
        compressor=compressor,
        global_step=global_step)
  elif compression_op_spec.compression_option == 2:
    raise NotImplementedError()
  elif compression_op_spec.compression_option == 3:
    raise NotImplementedError()
  elif compression_op_spec.compression_option == 4:
    raise NotImplementedError()
  elif compression_op_spec.compression_option == 8:
    raise NotImplementedError()
  elif compression_op_spec.compression_option == 9:
    raise NotImplementedError()

  return apply_compression
