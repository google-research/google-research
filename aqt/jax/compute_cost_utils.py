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

"""Estimation functions for compute costs of ML models."""

import contextlib
import functools
import re
from typing import Dict, Iterable, List, Optional, Tuple
from absl import flags
from jax._src.lax import convolution as lax_convolution
from jax._src.lax import lax
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import masking
from jax.interpreters import xla
import numpy as onp

from aqt.jax import hlo_utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.xla.service import hlo_pb2
# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'metadata_enabled',
    default=False,
    help=('Whether to annotate quantization info in HLO metadata.'))

# We intend to use quantization information to estimate the compute cost of
# our ML model. However, currently JAX does not support transferring high-level
# information from the model to its HLO representation, e.g. via metadata.
# Therefore, using the context managers below, this info is appended to the name
# of the op in the original primitives in lax via monkey-patching. After the
# op is instantiated, the original value of the primitive is restored.
# This is not a permanent solution.


# Based on: https://jax.readthedocs.io/en/latest/jax.lax.html
class DotMetadataMonkeyPatch(contextlib.ContextDecorator):
  """Context for passing quantization data to the dot operation."""

  def __init__(self, *, lhs_prec, rhs_prec,
               rhs_is_weight):
    annotation = _quantization_annotation(lhs_prec, rhs_prec, rhs_is_weight)
    self._op_name = 'dot_general_quant' + annotation

  def __enter__(self):
    # pylint: disable=protected-access
    # The following primitive accepts a name argument which is passed into
    # the HLO metadata field. Here, it is the only argument changed from
    # the original lax implementation.
    self._dot_general_p_original = lax.dot_general_p
    lax.dot_general_p = lax.standard_primitive(
        shape_rule=lax._dot_general_shape_rule,
        dtype_rule=lax._dot_general_dtype_rule,
        name=self._op_name,
        translation_rule=lax._dot_general_translation_rule)
    ad.defbilinear(lax.dot_general_p, lax._dot_general_transpose_lhs,
                   lax._dot_general_transpose_rhs)
    batching.primitive_batchers[lax.dot_general_p] = lax._dot_general_batch_rule
    masking.masking_rules[lax.dot_general_p] = lax._dot_general_masking_rule
    # pylint: enable=protected-access

  def __exit__(self, *exc):
    # Restore original primitive
    lax.dot_general_p = self._dot_general_p_original


# Based on: https://jax.readthedocs.io/en/latest/jax.lax.html
class ConvMetadataMonkeyPatch(contextlib.ContextDecorator):
  """Context for passing quantization data to the conv operation."""

  def __init__(self, *, weight_prec,
               act_prec):
    annotation = _quantization_annotation(
        act_prec, weight_prec, rhs_is_weight=True)
    self._op_name = 'conv_general_dilated_quant' + annotation

  def __enter__(self):
    # pylint: disable=protected-access
    self._conv_general_dilated_p_original = (
        lax_convolution.conv_general_dilated_p)
    # The following primitive accepts a name argument which is passed into
    # the HLO metadata field. Here, it is the only argument changed from
    # the original lax implementation.
    lax_convolution.conv_general_dilated_p = lax.standard_primitive(
        shape_rule=lax_convolution._conv_general_dilated_shape_rule,
        dtype_rule=lax_convolution._conv_general_dilated_dtype_rule,
        name=self._op_name,
        translation_rule=functools.partial(
            lax_convolution._conv_general_dilated_translation_rule,
            expand_complex_convolutions=False))
    xla.register_translation(
        lax_convolution.conv_general_dilated_p,
        functools.partial(
            lax_convolution._conv_general_dilated_translation_rule,
            expand_complex_convolutions=True),
        platform='cpu')
    xla.register_translation(
        lax_convolution.conv_general_dilated_p,
        functools.partial(
            lax_convolution._conv_general_dilated_translation_rule,
            expand_complex_convolutions=True),
        platform='gpu')
    ad.defbilinear(lax_convolution.conv_general_dilated_p,
                   lax_convolution._conv_general_dilated_transpose_lhs,
                   lax_convolution._conv_general_dilated_transpose_rhs)
    batching.primitive_batchers[lax_convolution.conv_general_dilated_p] = (
        lax_convolution._conv_general_dilated_batch_rule)
    masking.masking_rules[lax_convolution.conv_general_dilated_p] = (
        lax_convolution._conv_general_dilated_masking_rule)
    # pylint: enable=protected-access

  def __exit__(self, *exc):
    # Restore original primitive
    lax_convolution.conv_general_dilated_p = (
        self._conv_general_dilated_p_original)


# TODO(abdolrashidi): Add support for QuantOps.FloatQuant for cost estimation.
def _quantization_annotation(lhs_prec, rhs_prec,
                             rhs_is_weight):
  """Returns an annotation to be appended to the name of the quantizable op."""
  bfloat16_prec = 'bf16'

  def _replace_with_bf16_if_prec_is_none(prec):
    return prec if prec is not None else bfloat16_prec

  lhs_prec = _replace_with_bf16_if_prec_is_none(lhs_prec)
  rhs_prec = _replace_with_bf16_if_prec_is_none(rhs_prec)

  quant_annotation = '_lhs{}_rhs{}_lw{}'.format(lhs_prec, rhs_prec,
                                                int(rhs_is_weight))
  return quant_annotation


def _find_lhs_shape(instr,
                    computations):
  """Find the lhs shape of an instruction in HLO computations."""
  for computation in computations:
    for i in computation.instructions:
      # instr.operand_ids[0] contains the input shape.
      if i.id == instr.operand_ids[0]:
        # Input and output batch sizes must match
        assert i.shape.dimensions[0] == instr.shape.dimensions[0]
        return i.shape.dimensions
  return None


def _find_rhs_shape(instr,
                    computations):
  """Find the weight shape of an instruction in HLO computations."""
  for computation in computations:
    for i in computation.instructions:
      # instr.operand_ids[1] contains the weight shape.
      if i.id == instr.operand_ids[1]:
        return i.shape.dimensions
  return None


def _estimate_weights(
    instr,
    computations):
  """Estimate the number of weights in a conv or dot instruction."""
  weight_shape = _find_rhs_shape(instr, computations)
  assert weight_shape is not None
  return onp.prod(weight_shape)


def _estimate_conv_mults(
    instr,
    computations):
  """Estimate the number of multiplications in a convolution instruction."""
  lhs_shape = _find_lhs_shape(instr, computations)
  assert lhs_shape is not None
  input_channels = lhs_shape[-1]
  kernel_size = onp.prod([dim.size for dim in instr.window.dimensions])
  output_conv_dims = [
      instr.shape.dimensions[i]
      for i in instr.convolution_dimension_numbers.output_spatial_dimensions
  ]
  output_image_size = onp.prod(output_conv_dims)
  output_channels = instr.shape.dimensions[
      instr.convolution_dimension_numbers.output_feature_dimension]
  return output_image_size * kernel_size * input_channels * output_channels


def _estimate_dot_mults(
    instr,
    computations):
  """Estimate the number of multiplications in a dot instruction."""
  lhs_shape = _find_lhs_shape(instr, computations)
  assert lhs_shape is not None
  input_channels = lhs_shape[-1]
  output_channels = instr.shape.dimensions[-1]
  return input_channels * output_channels


def _extract_quant_info(
    instr):
  """Extracts lhs and rhs quantization precision from op metadata."""
  if instr.opcode not in _get_supported_ops():
    raise NotImplementedError('Unexpected op detected')
  # If annotated, metadata.op_type would have the following format:
  # '[original op name]_quant_w[weight prec]_a[act prec]'
  if 'quant' not in instr.metadata.op_type:
    raise NotImplementedError('Unable to parse {}'.format(
        instr.metadata.op_type))
  [(lhs_prec_str, rhs_prec_str, rhs_is_weight_str)
  ] = re.findall('_lhs(.*)_rhs(.*)_lw(.*)', instr.metadata.op_type)

  def _extract_prec(prec_str):
    if prec_str.startswith('bf'):
      prec_str = prec_str[2:]
    if prec_str.isnumeric():
      prec = int(prec_str)
    else:
      raise NotImplementedError('Unable to parse {}'.format(
          instr.metadata.op_type))
    return prec

  lhs_prec = _extract_prec(lhs_prec_str)
  rhs_prec = _extract_prec(rhs_prec_str)
  rhs_is_weight = bool(int(rhs_is_weight_str))
  if lhs_prec <= 0 or rhs_prec <= 0:
    raise ValueError('HLO metadata precision annotatation must be a positive '
                     'integer.')
  return lhs_prec, rhs_prec, rhs_is_weight


def _list_supported_ops_from_hlo(
    hlo_proto):
  """Gather and return a list of supported quantizable ops in the HLO."""
  supported_ops = _get_supported_ops()
  target_instructions = []
  computations = hlo_proto.computations
  for computation in computations:
    for instr in computation.instructions:
      if instr.opcode in supported_ops:
        target_instructions.append(instr)
  return target_instructions


def _get_supported_ops():
  """Get the supported ops for compute and memory cost estimation."""
  # Output dictionary key is HLO instruction opcode.
  # 'estimate_instr_mult': function used to estimate the number of matrix
  # multiplications.
  # 'estimate_instr_weights': function used to estimate the number of weights.
  return {
      'convolution': {
          'estimate_instr_mult': _estimate_conv_mults,
          'estimate_instr_weights': _estimate_weights,
      },
      'dot': {
          'estimate_instr_mult': _estimate_dot_mults,
          'estimate_instr_weights': _estimate_weights,
      },
  }


def estimate_compute_cost(
    hlo_proto):
  """Estimates the compute cost for the input HLO proto.

  Args:
    hlo_proto: the model's HLO representation (e.g. derived from
      hlo_utils.load_hlo_proto()) It contains all the instructions used in the
      model.

  Returns:
    a dictionary with two key-value pairs:
      'compute_cost': The sum of all compute costs for each layer
      'compute_cost_ratio_to_bfloat16': The ratio of the estimated compute cost
      to the cost in the case of no quantization (bfloat16)

  """
  # To estimate the overall compute cost of ops, we should multiply the
  # number of multiplications by bits_weights * bits_acts in each layer.

  # Gather the supported quantizable ops and their quantization parameters
  target_instructions = _list_supported_ops_from_hlo(hlo_proto)

  # Begin compute cost calculation
  compute_cost_quadratic = 0
  compute_cost_linear = 0
  bfloat16_cost_quadratic = 0  # computed as reference for comparison purposes
  bfloat16_cost_linear = 0

  supported_ops = _get_supported_ops()
  for instr in target_instructions:
    # Estimate multiplications in the op
    opcode = instr.opcode
    multiplication_count = supported_ops[opcode]['estimate_instr_mult'](
        instr, hlo_proto.computations)

    # For the model's compute cost, we use the number of multiplications used
    # in the quantizable layers used in the model, which dominate the cost
    # compared to that of other operations used in them, such as additions.
    # Also, the cost of each multiplication is proportional to the number of
    # bits in each operand, which are subject to change due to quantization.
    # Therefore, we use the weight and activation precisions in our estimation
    # as well.

    bfloat16_lhs_prec = 16
    bfloat16_rhs_prec = 16

    lhs_prec, rhs_prec, _ = _extract_quant_info(instr)
    if lhs_prec > 16:
      raise ValueError(f'Unexpected lhs precision {lhs_prec}.')
    if rhs_prec > 16:
      raise ValueError(f'Unexpected rhs precision {rhs_prec}.')

    bfloat16_cost_quadratic += multiplication_count * (
        bfloat16_lhs_prec * bfloat16_rhs_prec)
    compute_cost_quadratic += multiplication_count * lhs_prec * rhs_prec

    # Nvidia A100 only supports 16x16, 8x8 and 4x4
    # Cost of 8x8 is half of 16x16
    # Cost of 4x4 is 1/4 of 16x16
    bfloat16_cost_linear += multiplication_count * max(
        bfloat16_lhs_prec, bfloat16_rhs_prec)

    prec = max(lhs_prec, rhs_prec)
    if prec <= 4:
      prec = 4
    elif prec <= 8:
      prec = 8
    elif prec <= 16:
      prec = 16

    compute_cost_linear += multiplication_count * prec
  # Return the results
  cost_ratio_to_bfloat16_quadratic = compute_cost_quadratic / bfloat16_cost_quadratic
  cost_ratio_to_bfloat16_linear = compute_cost_linear / bfloat16_cost_linear

  result_dict = {
      # quadratic compute cost. We didn't rename the keys to
      # compute_cost_quadratic to presev backwards compatibility.
      'compute_cost':
          float(compute_cost_quadratic),
      'compute_cost_ratio_to_bfloat16':
          float(cost_ratio_to_bfloat16_quadratic),
      # Linear compute cost
      'compute_cost_linear':
          float(compute_cost_linear),
      'compute_cost_ratio_to_bfloat16_linear':
          float(cost_ratio_to_bfloat16_linear),
  }

  return result_dict


def estimate_memory_cost(hlo_proto):
  """Estimates the memory cost for the input HLO proto.

  Args:
    hlo_proto: the model's HLO representation (e.g. derived from
      hlo_utils.load_hlo_proto()) It contains all the instructions used in the
      model.

  Returns:
    a dictionary with two key-value pairs:
      'memory_cost': The sum of all memory costs for each layer
      'memory_cost_ratio_to_bfloat16': The ratio of the estimated memory cost to
         the cost in the case of no quantization (bfloat16)

  """
  # For the memory cost, we estimate the number of bits used for the weights,
  # which would also require the quantization precision for those weights.

  # Gather the supported quantizable ops and their quantization parameters
  target_instructions = _list_supported_ops_from_hlo(hlo_proto)

  # Begin memory cost calculation
  memory_cost = 0
  bfloat16_cost = 0  # computed as reference for comparison purposes

  supported_ops = _get_supported_ops()
  for instr in target_instructions:
    # Estimate number of weights in the op
    opcode = instr.opcode
    weight_count = supported_ops[opcode]['estimate_instr_weights'](
        instr, hlo_proto.computations)

    # Multiply number of weights by the number of bits used for each.
    bfloat16_weight_prec = 16

    _, rhs_prec, rhs_is_weight = _extract_quant_info(instr)

    if rhs_prec > 16:
      raise ValueError(f'Unexpected rhs precision {rhs_prec}.')

    if rhs_is_weight:
      memory_cost += weight_count * rhs_prec
      bfloat16_cost += weight_count * bfloat16_weight_prec

  # For dynamic matmuls, memory_cost is zero. The following check is to avoid
  # division by zero. We set the ratio for 0/0 to 1.
  if bfloat16_cost == 0 and memory_cost == 0:
    cost_ratio_to_bfloat16 = 1
  else:
    cost_ratio_to_bfloat16 = memory_cost / bfloat16_cost

  result_dict = {
      'memory_cost': float(memory_cost),
      'memory_cost_ratio_to_bfloat16': float(cost_ratio_to_bfloat16)
  }
  return result_dict


def estimate_costs_of_dot_and_conv_ops_from_jax_fn(
    fn, *fn_args, **fn_kwargs):
  """Wrapper function around estimate_compute_and_memory_cost().

  Will generate hlo proto from jax function, and call
  estimate_compute_and_memory_cost() on it.

  Args:
    fn: the function for which the HLO is to be produced.
    *fn_args: the function's args.
    **fn_kwargs: the function's kwargs.

  Returns:
    A dictionary with compute cost and memory cost data.
    See estimate_compute_and_memory_cost() docstring for details.
  """
  FLAGS.metadata_enabled = True
  hlo_module_proto = hlo_utils.load_hlo_proto_from_jax_fn(
      fn, *fn_args, **fn_kwargs)
  cost_dict = estimate_compute_cost(hlo_module_proto)
  memory_cost_dict = estimate_memory_cost(hlo_module_proto)
  cost_dict.update(memory_cost_dict)
  FLAGS.metadata_enabled = False
  return cost_dict
