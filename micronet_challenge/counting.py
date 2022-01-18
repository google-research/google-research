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

"""This module defines an API for counting parameters and operations.


## Defining the Operation Count API
- `input_size` is an int, since square image assumed.
- `strides` is a tuple, but assumed to have same stride in both dimensions.
- Supported `paddings` are `same' and `valid`.
- `use_bias` is boolean.
- `activation` is one of the following `relu`, `swish`, `sigmoid`, None
- kernel_shapes for `Conv2D` dimensions must be in the following order:
  `k_size, k_size, c_in, c_out`
- kernel_shapes for `FullyConnected` dimensions must be in the following order:
  `c_in, c_out`
- kernel_shapes for `DepthWiseConv2D` dimensions must be like the following:
  `k_size, k_size, c_in==c_out, 1`

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

"""Operation definition for 2D convolution.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  kernel_shape: list, of length 4. Shape of the convolutional kernel.
  strides: list, of length 2. Stride with which the kernel is applied.
  padding: str, padding added to the input image.
  use_bias: bool, if true a bias term is added to the output.
  activation: str, type of activation applied to the output.
"""  # pylint: disable=pointless-string-statement
Conv2D = collections.namedtuple(
    'Conv2D', ['input_size', 'kernel_shape', 'strides', 'padding', 'use_bias',
               'activation'])

"""Operation definition for 2D depthwise convolution.

Only difference compared to Conv2D is the kernel_shape[3] = 1.
"""
DepthWiseConv2D = collections.namedtuple(
    'DepthWiseConv2D', ['input_size', 'kernel_shape', 'strides', 'padding',
                        'use_bias', 'activation'])

"""Operation definition for Global Average Pooling.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  n_channels: int, Number of output dimensions.
"""
GlobalAvg = collections.namedtuple('GlobalAvg', ['input_size', 'n_channels'])

"""Operation definitions for elementwise multiplication and addition.

Attributes:
  input_size: int, Dimensions of the input image (square assumed).
  n_channels: int, Number of output dimensions.
"""
Scale = collections.namedtuple('Scale', ['input_size', 'n_channels'])
Add = collections.namedtuple('Add', ['input_size', 'n_channels'])


"""Operation definitions for elementwise multiplication and addition.

Attributes:
  kernel_shape: list, of length 2. Shape of the weight matrix.
  use_bias: bool, if true a bias term is added to the output.
  activation: str, type of activation applied to the output.
"""
FullyConnected = collections.namedtuple(
    'FullyConnected', ['kernel_shape', 'use_bias', 'activation'])


def get_flops_per_activation(activation):
  """Returns the number of multiplication ands additions of an activation.

  Args:
    activation: str, type of activation applied to the output.
  Returns:
    n_muls, n_adds
  """
  if activation == 'relu':
    # For the purposes of the "freebie" quantization scoring, ReLUs can be
    # assumed to be performed on 16-bit inputs. Thus, we track them as
    # multiplications in our accounting, which can also be assumed to be
    # performed on reduced precision inputs.
    return 1, 0
  elif activation == 'swish':  # Swish: x / (1 + exp(-bx))
    return 3, 1
  elif activation == 'sigmoid':  # Sigmoid: exp(x) / (1 + exp(x))
    return 2, 1
  else:
    raise ValueError('activation: %s is not valid' % activation)


def get_sparse_size(tensor_shape, param_bits, sparsity):
  """Given a tensor shape returns #bits required to store the tensor sparse.

  If sparsity is greater than 0, we do have to store a bit mask to represent
  sparsity.
  Args:
    tensor_shape: list<int>, shape of the tensor
    param_bits: int, number of bits the elements of the tensor represented in.
    sparsity: float, sparsity level. 0 means dense.
  Returns:
    int, number of bits required to represented the tensor in sparse format.
  """
  n_elements = np.prod(tensor_shape)
  c_size = n_elements * param_bits * (1 - sparsity)
  if sparsity > 0:
    c_size += n_elements  # 1 bit binary mask.
  return c_size


def get_conv_output_size(image_size, filter_size, padding, stride):
  """Calculates the output size of convolution.

  The input, filter and the strides are assumed to be square.
  Arguments:
    image_size: int, Dimensions of the input image (square assumed).
    filter_size: int, Dimensions of the kernel (square assumed).
    padding: str, padding added to the input image. 'same' or 'valid'
    stride: int, stride with which the kernel is applied (square assumed).
  Returns:
    int, output size.
  """
  if padding == 'same':
    pad = filter_size // 2
  elif padding == 'valid':
    pad = 0
  else:
    raise NotImplementedError('Padding: %s should be `same` or `valid`.'
                              % padding)
  out_size = np.ceil((image_size - filter_size + 1. + 2 * pad) / stride)
  return int(out_size)


def count_ops(op, sparsity, param_bits):
  """Given a operation class returns the flop and parameter statistics.

  Args:
    op: namedtuple, operation definition.
    sparsity: float, sparsity of parameterized operations. Sparsity only effects
      Conv and FC layers; since activations are dense.
    param_bits: int, number of bits required to represent a parameter.
  Returns:
    param_count: number of bits required to store parameters
    n_mults: number of multiplications made per input sample.
    n_adds: number of multiplications made per input sample.
  """
  flop_mults = flop_adds = param_count = 0
  if isinstance(op, Conv2D):
    # Square kernel expected.
    assert op.kernel_shape[0] == op.kernel_shape[1]
    k_size, _, c_in, c_out = op.kernel_shape

    # Size of the possibly sparse convolutional tensor.
    param_count += get_sparse_size(
        [k_size, k_size, c_in, c_out], param_bits, sparsity)

    # Square stride expected.
    assert op.strides[0] == op.strides[1]
    stride = op.strides[0]

    # Each application of the kernel can be thought as a dot product between
    # the flattened kernel and patches of the image.
    vector_length = (k_size * k_size * c_in) * (1 - sparsity)
    # Number of elements in the output is OUT_SIZE * OUT_SIZE * OUT_CHANNEL
    n_output_elements = get_conv_output_size(op.input_size, k_size, op.padding,
                                             stride) ** 2 * c_out
    # Each output is the product of a one dot product. Dot product of two
    # vectors of size n needs n multiplications and n - 1 additions.
    flop_mults += vector_length * n_output_elements
    flop_adds += (vector_length - 1) * n_output_elements

    if op.use_bias:
      # For each output channel we need a bias term.
      param_count += c_out * param_bits
      # If we have bias we need one more addition per dot product.
      flop_adds += n_output_elements

    if op.activation:
      # We would apply the activaiton to every single output element.
      n_muls, n_adds = get_flops_per_activation(op.activation)
      flop_mults += n_muls * n_output_elements
      flop_adds += n_adds * n_output_elements

  elif isinstance(op, DepthWiseConv2D):
    # Square kernel expected.
    assert op.kernel_shape[0] == op.kernel_shape[1]
    # Last dimension of the kernel should be 1.
    assert op.kernel_shape[3] == 1
    k_size, _, channels, _ = op.kernel_shape

    # Size of the possibly sparse convolutional tensor.
    param_count += get_sparse_size(
        [k_size, k_size, channels], param_bits, sparsity)

    # Square stride expected.
    assert op.strides[0] == op.strides[1]
    stride = op.strides[0]

    # Each application of the kernel can be thought as a dot product between
    # the flattened kernel and patches of the image.
    vector_length = (k_size * k_size) * (1 - sparsity)
    # Number of elements in the output tensor.

    n_output_elements = get_conv_output_size(op.input_size, k_size, op.padding,
                                             stride) ** 2 * channels

    # Each output is the product of a one dot product. Dot product of two
    # vectors of size n needs n multiplications and n - 1 additions.
    flop_mults += vector_length * n_output_elements
    flop_adds += (vector_length - 1) * n_output_elements

    if op.use_bias:
      # For each output channel we need a bias term.
      param_count += channels * param_bits
      # If we have bias we need one more addition per dot product.
      flop_adds += n_output_elements

    if op.activation:
      # We would apply the activaiton to every single output element.
      n_muls, n_adds = get_flops_per_activation(op.activation)
      flop_mults += n_muls * n_output_elements
      flop_adds += n_adds * n_output_elements
  elif isinstance(op, GlobalAvg):
    # For each output channel we will make a division.
    flop_mults += op.n_channels
    # We have to add values over spatial dimensions.
    flop_adds += (op.input_size * op.input_size - 1) * op.n_channels
  elif isinstance(op, Scale):
    # Number of elements many multiplications.
    flop_mults += op.input_size * op.input_size *  op.n_channels
  elif isinstance(op, Add):
    # Number of elements many additions.
    flop_adds += op.input_size * op.input_size *  op.n_channels
  elif isinstance(op, FullyConnected):
    c_in, c_out = op.kernel_shape
    # Size of the possibly sparse weight matrix.
    param_count += get_sparse_size(
        [c_in, c_out], param_bits, sparsity)

    # number of non-zero elements for the sparse dot product.
    n_elements = c_in * (1 - sparsity)
    flop_mults += n_elements * c_out
    # We have one less addition than the number of multiplications per output
    # channel.
    flop_adds += (n_elements - 1) * c_out

    if op.use_bias:
      param_count += c_out * param_bits
      flop_adds += c_out
    if op.activation:
      n_muls, n_adds = get_flops_per_activation(op.activation)
      flop_mults += n_muls * c_out
      flop_adds += n_adds * c_out
  else:
    raise ValueError('Encountered unknown operation %s.' % str(op))
  return param_count, flop_mults, flop_adds


# Info
def get_info(op):
  """Given an op extracts some common information."""
  input_size, kernel_size, in_channels, out_channels = [-1] * 4
  if isinstance(op, (DepthWiseConv2D, Conv2D)):
    # square kernel assumed.
    kernel_size, _, in_channels, out_channels = op.kernel_shape
    input_size = op.input_size
  elif isinstance(op, GlobalAvg):
    in_channels = op.n_channels
    out_channels = 1
    input_size = op.input_size
  elif isinstance(op, (Add, Scale)):
    in_channels = op.n_channels
    out_channels = op.n_channels
    input_size = op.input_size
  elif isinstance(op, FullyConnected):
    in_channels, out_channels = op.kernel_shape
    input_size = 1
  else:
    raise ValueError('Encountered unknown operation %s.' % str(op))
  return input_size, kernel_size, in_channels, out_channels


class MicroNetCounter(object):
  """Counts operations using given information.

  """
  _header_str = '{:25} {:>10} {:>13} {:>13} {:>13} {:>15} {:>10} {:>10} {:>10}'
  _line_str = ('{:25s} {:10d} {:13d} {:13d} {:13d} {:15.3f} {:10.3f}'
               ' {:10.3f} {:10.3f}')

  def __init__(self, all_ops, add_bits_base=32, mul_bits_base=32):
    self.all_ops = all_ops
    # Full precision add is counted one.
    self.add_bits_base = add_bits_base
    # Full precision multiply is counted one.
    self.mul_bits_base = mul_bits_base

  def _aggregate_list(self, counts):
    return np.array(counts).sum(axis=0)

  def process_counts(self, total_params, total_mults, total_adds,
                     mul_bits, add_bits):
    # converting to Mbytes.
    total_params = int(total_params) / 8. / 1e6
    total_mults = total_mults * mul_bits / self.mul_bits_base / 1e6
    total_adds = total_adds * add_bits / self.add_bits_base  / 1e6
    return total_params, total_mults, total_adds

  def _print_header(self):
    output_string = self._header_str.format(
        'op_name', 'inp_size', 'kernel_size', 'in channels', 'out channels',
        'params(MBytes)', 'mults(M)', 'adds(M)', 'MFLOPS')
    print(output_string)
    print(''.join(['=']*125))

  def _print_line(self, name, input_size, kernel_size, in_channels,
                  out_channels, param_count, flop_mults, flop_adds, mul_bits,
                  add_bits, base_str=None):
    """Prints a single line of operation counts."""
    op_pc, op_mu, op_ad = self.process_counts(param_count, flop_mults,
                                              flop_adds, mul_bits, add_bits)
    if base_str is None:
      base_str = self._line_str
    output_string = base_str.format(
        name, input_size, kernel_size, in_channels, out_channels, op_pc,
        op_mu, op_ad, op_mu + op_ad)
    print(output_string)

  def print_summary(self, sparsity, param_bits, add_bits, mul_bits,
                    summarize_blocks=True):
    """Prints all operations with given options.

    Args:
      sparsity: float, between 0,1 defines how sparse each parametric layer is.
      param_bits: int, bits in which parameters are stored.
      add_bits: float, number of bits used for accumulator.
      mul_bits: float, number of bits inputs represented for multiplication.
      summarize_blocks: bool, if True counts within a block are aggregated and
        reported in a single line.

    """
    self._print_header()
    # Let's count starting from zero.
    total_params, total_mults, total_adds = [0] * 3
    for op_name, op_template in self.all_ops:
      if op_name.startswith('block'):
        if not summarize_blocks:
          # If debug print the ops inside a block.
          for block_op_name, block_op_template in op_template:
            param_count, flop_mults, flop_adds = count_ops(block_op_template,
                                                           sparsity, param_bits)
            temp_res = get_info(block_op_template)
            input_size, kernel_size, in_channels, out_channels = temp_res
            self._print_line('%s_%s' % (op_name, block_op_name), input_size,
                             kernel_size, in_channels, out_channels,
                             param_count, flop_mults, flop_adds, mul_bits,
                             add_bits)
        # Count and sum all ops within a block.
        param_count, flop_mults, flop_adds = self._aggregate_list(
            [count_ops(template, sparsity, param_bits)
             for _, template in op_template])
        # Let's extract the input_size and in_channels from the first operation.
        input_size, _, in_channels, _ = get_info(op_template[0][1])
        # Since we don't know what is inside a block we don't know the following
        # fields.
        kernel_size = out_channels = -1
      else:
        # If it is a single operation just count.
        param_count, flop_mults, flop_adds = count_ops(op_template, sparsity,
                                                       param_bits)
        temp_res = get_info(op_template)
        input_size, kernel_size, in_channels, out_channels = temp_res
      # At this point param_count, flop_mults, flop_adds should be read.
      total_params += param_count
      total_mults += flop_mults
      total_adds += flop_adds
      # Print the operation.
      self._print_line(op_name, input_size, kernel_size, in_channels,
                       out_channels, param_count, flop_mults, flop_adds,
                       mul_bits, add_bits)

    # Print Total values.
    # New string since we are passing empty strings instead of integers.
    out_str = ('{:25s} {:10s} {:13s} {:13s} {:13s} {:15.3f} {:10.3f} {:10.3f} '
               '{:10.3f}')
    self._print_line(
        'total', '', '', '', '', total_params, total_mults, total_adds,
        mul_bits, add_bits, base_str=out_str)
