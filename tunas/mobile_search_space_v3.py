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

# Lint as: python2, python3
"""Mobile classification search space built around MobileNet V3.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, List, NamedTuple, Optional, Sequence, Text, Tuple, Union

from tunas import basic_specs
from tunas import mobile_model_archive
from tunas import schema
from tunas import schema_io
from tunas import search_space_utils

# Reference models we compare against from the published literature.
MOBILENET_V2 = 'mobilenet_v2'
MNASNET_B1 = 'mnasnet_b1'
PROXYLESSNAS_MOBILE = 'proxylessnas_mobile'
MOBILENET_V3_LARGE = 'mobilenet_v3_large'
MOBILENET_MULTI_AVG = 'mobilenet_multi_avg'
MOBILENET_MULTI_MAX = 'mobilenet_multi_max'

# Key search spaces reported in our paper.
PROXYLESSNAS_SEARCH = 'proxylessnas_search'
PROXYLESSNAS_ENLARGED_SEARCH = 'proxylessnas_enlarged_search'
MOBILENET_V3_LIKE_SEARCH = 'mobilenet_v3_like_search'

ALL_SSDS = (
    MOBILENET_V2,
    MNASNET_B1,
    PROXYLESSNAS_MOBILE,
    MOBILENET_V3_LARGE,
    PROXYLESSNAS_SEARCH,
    PROXYLESSNAS_ENLARGED_SEARCH,
    MOBILENET_V3_LIKE_SEARCH,
    # Multi-hardware models from paper
    # "Discovering Multi-Hardware Mobile Models via Architecture Search".
    MOBILENET_MULTI_AVG,
    MOBILENET_MULTI_MAX,
)

MOBILENET_V3_LIKE_SSDS = (
    MOBILENET_V3_LARGE,
    MOBILENET_V3_LIKE_SEARCH,
    MOBILENET_MULTI_AVG,
    MOBILENET_MULTI_MAX,
)

_IntOrIntPair = Union[int, Tuple[int, int]]

TunableKernelSize = Union[
    int,
    Tuple[int, int],
    schema.OneOf[int],
    schema.OneOf[Tuple[int, int]]]


@schema_io.register_namedtuple('mobile_search_space_v3.ActivationSpec')
class ActivationSpec(NamedTuple('ActivationSpec', [('name', Text)])):
  """Neural network activation function.

  Attributes:
    name: Name of the activation function to apply, like 'relu' or 'swish6'.
  """
  pass


# List of supported activation functions.
RELU = ActivationSpec('relu')
RELU6 = ActivationSpec('relu6')
SWISH6 = ActivationSpec('swish6')
SIGMOID = ActivationSpec('sigmoid')


@schema_io.register_namedtuple('mobile_search_space_v3.ConvSpec')
class ConvSpec(
    NamedTuple('ConvSpec',
               [('kernel_size', TunableKernelSize),
                ('strides', _IntOrIntPair),
                ('use_batch_norm', bool)])):
  """2D convolution followed by an optional batch norm.

  Attributes:
    kernel_size: Kernel size.
    strides: Output strides.
    use_batch_norm: If true, we'll insert a batch norm op after the convolution.
  """

  def __new__(cls,
              kernel_size,
              strides,
              use_batch_norm = True):
    return super(ConvSpec, cls).__new__(
        cls, kernel_size, strides, use_batch_norm)


@schema_io.register_namedtuple('mobile_search_space_v3.SeparableConvSpec')
class SeparableConvSpec(
    NamedTuple('SeparableConvSpec',
               [('kernel_size', TunableKernelSize),
                ('strides', _IntOrIntPair),
                ('activation', ActivationSpec)])):
  """2D depthwise separable convolution followed by a batch norm.

  Attributes:
    kernel_size: Kernel size for the depthwise convolution.
    strides: Strides to use for the depthwise convolution.
    activation: Activation function to apply between the depthwise and pointwise
        convolutions.
  """

  def __new__(cls,
              kernel_size,
              strides,
              activation = RELU):
    return super(SeparableConvSpec, cls).__new__(
        cls, kernel_size, strides, activation)


@schema_io.register_namedtuple('mobile_search_space_v3.DepthwiseBottleneckSpec')
class DepthwiseBottleneckSpec(
    NamedTuple('DepthwiseBottleneckSpec',
               [('kernel_size', TunableKernelSize),
                ('expansion_filters',
                 Union[schema.OneOf[int],
                       schema.OneOf[basic_specs.FilterMultiplier],
                       basic_specs.FilterMultiplier]),
                ('use_squeeze_and_excite', Union[bool, schema.OneOf[bool]]),
                ('strides', _IntOrIntPair),
                ('activation', ActivationSpec),
                ('se_inner_activation', ActivationSpec),
                ('se_gating_activation', ActivationSpec)])):
  """Inverted bottleneck: a depthwise convolution between two pointwise convs.

  Attributes:
    kernel_size: Kernel size to use for the depthwise convolution.
    expansion_filters: Number of filters to use in the intermediate layers of
        the network.
    use_squeeze_and_excite: Set to true to add a squeeze-and-excite layer
        immediately after the depthwise convolution.
    strides: Strides to use for the depthwise convolution.
    activation: Activation function to use between the depthwise and pointwise
        convolutions.
    se_inner_activation: Activation function to apply between the inner layers
        of the squeeze-and-excite function. Used only when
        use_squeeze_and_excite is true.
    se_gating_activation: Activation function to apply to the output of the
        squeeze-and-excite feed-forward network. Used only when
        use_squeeze_and_excite is true.
  """

  def __new__(
      cls,
      kernel_size,
      expansion_filters,
      use_squeeze_and_excite,
      strides,
      activation = RELU,
      se_inner_activation = RELU,
      se_gating_activation = SIGMOID):
    return super(DepthwiseBottleneckSpec, cls).__new__(
        cls, kernel_size, expansion_filters, use_squeeze_and_excite, strides,
        activation, se_inner_activation, se_gating_activation)


# NOTE: There's a bug in gpylint that triggers an error if we try to
# use typing.NamedTuple with an empty argument list. We work around the problem
# by using collections.namedtuple instead.
@schema_io.register_namedtuple('mobile_search_space_v3.GlobalAveragePoolSpec')
class GlobalAveragePoolSpec(
    collections.namedtuple('GlobalAveragePoolSpec', [])):
  """Global average pooling layer."""
  pass


@schema_io.register_namedtuple('mobile_search_space_v3.ResidualSpec')
class ResidualSpec(NamedTuple('ResidualSpec', [('layer', Any)])):
  """Residual connection.

  Attributes:
    layer: The layer to apply the residual connection to. The input and output
        shapes must match.
  """
  pass


@schema_io.register_namedtuple('mobile_search_space_v3.DetectionEndpointSpec')
class DetectionEndpointSpec(
    collections.namedtuple('DetectionEndpointSpec', [])):
  """Mark the position of an endpoint for object detection."""
  pass


def _merge_strides(
    lhs,
    rhs
):
  """Merge two sets of strides.

  Args:
    lhs: A tuple (x, y) where each element is either an integer or None.
    rhs: A tuple (x, y) where each element is either an integer or None.

  Returns:
    A merged tuple (x, y). For example:
    * merge((1, 1), (None, None)) = (1, 1)
    * merge((None, None), (None, None)) = None
    * merge((1, 1), (2, 2)) triggers an error, since the two strides
      are incompatible.
  """

  if lhs[0] is not None and rhs[0] is not None and lhs[0] != rhs[0]:
    raise ValueError('Stride mismatch: {} vs {}'.format(lhs, rhs))
  if lhs[1] is not None and rhs[1] is not None and lhs[1] != rhs[1]:
    raise ValueError('Stride mismatch: {} vs {}'.format(lhs, rhs))
  x = lhs[0] if lhs[0] is not None else rhs[0]
  y = lhs[1] if lhs[1] is not None else rhs[1]
  return (x, y)


def get_strides(layer_spec):
  """Compute the output strides for a given layer."""
  strides = (None, None)
  if isinstance(layer_spec, ConvSpec):
    strides = search_space_utils.normalize_strides(layer_spec.strides)
  elif isinstance(layer_spec, SeparableConvSpec):
    strides = search_space_utils.normalize_strides(layer_spec.strides)
  elif isinstance(layer_spec, DepthwiseBottleneckSpec):
    strides = search_space_utils.normalize_strides(layer_spec.strides)
  elif isinstance(layer_spec, basic_specs.ZeroSpec):
    strides = (1, 1)
  elif isinstance(layer_spec, GlobalAveragePoolSpec):
    # Cannot be determined statically.
    strides = (None, None)
  elif isinstance(layer_spec, ActivationSpec):
    strides = (1, 1)
  elif isinstance(layer_spec, ResidualSpec):
    strides = get_strides(layer_spec.layer)
    if strides != (1, 1):
      raise ValueError('Residual layer must have stride 1: {}'.format(
          layer_spec))
  elif isinstance(layer_spec, schema.OneOf):
    for choice in layer_spec.choices:
      strides = _merge_strides(strides, get_strides(choice))
  else:
    raise ValueError('Unsupported layer_spec type: {}'.format(
        type(layer_spec)))

  return strides


def choose_filters(choices):
  """Choose one of the filters from the given choices."""
  return schema.OneOf(choices, basic_specs.FILTERS_TAG)


def _proxylessnas_search_base(base_filters,
                              collapse_shared_ops = False):
  """Reproduction of ProxylessNAS search space with custom output filters."""
  block = basic_specs.block
  residual = ResidualSpec
  global_avg_pool = GlobalAveragePoolSpec

  def conv(kernel, s, bn=True):
    return ConvSpec(
        kernel_size=kernel,
        strides=s,
        use_batch_norm=bn)

  def sepconv(s):
    choices = []
    for kernel_size in (3, 5, 7):
      choices.append(
          SeparableConvSpec(
              kernel_size=kernel_size,
              strides=s,
              activation=RELU))
    return schema.OneOf(choices, basic_specs.OP_TAG)

  def bneck(s, skippable):
    """Construct a spec for an inverted bottleneck layer."""
    possible_filter_multipliers = [3.0, 6.0]
    possible_kernel_sizes = [3, 5, 7]
    choices = []

    if collapse_shared_ops:
      kernel_size = schema.OneOf(possible_kernel_sizes, basic_specs.OP_TAG)
      expansion_filters = schema.OneOf(
          [basic_specs.FilterMultiplier(multiplier)
           for multiplier in possible_filter_multipliers],
          basic_specs.FILTERS_TAG)
      choices.append(
          DepthwiseBottleneckSpec(
              kernel_size=kernel_size,
              expansion_filters=expansion_filters,
              use_squeeze_and_excite=False,
              strides=s,
              activation=RELU))
    else:
      for multiplier in possible_filter_multipliers:
        for kernel_size in possible_kernel_sizes:
          choices.append(
              DepthwiseBottleneckSpec(
                  kernel_size=kernel_size,
                  expansion_filters=basic_specs.FilterMultiplier(multiplier),
                  use_squeeze_and_excite=False,
                  strides=s,
                  activation=RELU))

    if skippable:
      choices.append(basic_specs.ZeroSpec())
    return schema.OneOf(choices, basic_specs.OP_TAG)

  blocks = [
      # Stem
      block([
          conv(kernel=3, s=2),
          RELU,
      ], filters=base_filters[0]),

      block([
          # NOTE: The original MobileNet V2 paper used an inverted bottleneck
          # layer with an expansion factor of 1 here. Under the definition used
          # by the paper, an inverted bottleneck layer with an expansion factor
          # of 1 was equivalent to a depthwise separable convolution, which is
          # what we use. (Our definition of an inverted bottleneck layer with
          # an expansion factor of 1 is slightly different from the one used in
          # the MobileNet paper.)
          sepconv(s=1),
          DetectionEndpointSpec(),
      ], filters=base_filters[1]),

      # Body
      block([
          bneck(s=2, skippable=False),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          DetectionEndpointSpec(),
      ], filters=base_filters[2]),

      block([
          bneck(s=2, skippable=False),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          DetectionEndpointSpec(),
      ], filters=base_filters[3]),

      block([
          bneck(s=2, skippable=False),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
      ], filters=base_filters[4]),

      block([
          bneck(s=1, skippable=False),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          DetectionEndpointSpec(),
      ], filters=base_filters[5]),

      block([
          bneck(s=2, skippable=False),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          residual(bneck(s=1, skippable=True)),
          DetectionEndpointSpec(),
      ], filters=base_filters[6]),

      block([
          bneck(s=1, skippable=False),
          DetectionEndpointSpec(),
      ], filters=base_filters[7]),

      # Head
      block([
          conv(kernel=1, s=1),
          RELU,
          global_avg_pool(),
      ], filters=base_filters[8]),
  ]
  return basic_specs.ConvTowerSpec(blocks=blocks, filters_base=8)


def proxylessnas_search():
  return _proxylessnas_search_base(
      mobile_model_archive.PROXYLESSNAS_MOBILE_FILTERS,
      collapse_shared_ops=True)


def proxylessnas_enlarged_search():
  base_filters = (16, 16, 16, 32, 64, 128, 256, 512, 1024)
  multipliers = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0)

  model_spec = _proxylessnas_search_base(base_filters, collapse_shared_ops=True)
  return search_space_utils.scale_conv_tower_spec(model_spec, multipliers)


def mobilenet_v2():
  """Specification for MobileNet V2 w/ relative expansion filters."""
  model_spec = _proxylessnas_search_base(
      mobile_model_archive.MOBILENET_V2_FILTERS)
  model_spec = search_space_utils.prune_model_spec(
      model_spec,
      {basic_specs.OP_TAG: mobile_model_archive.MOBILENET_V2_OPERATIONS})
  return model_spec


def mnasnet_b1():
  model_spec = _proxylessnas_search_base(
      mobile_model_archive.MNASNET_FILTERS)
  model_spec = search_space_utils.prune_model_spec(
      model_spec,
      {basic_specs.OP_TAG: mobile_model_archive.MNASNET_OPERATIONS})
  return model_spec


def proxylessnas_mobile():
  model_spec = _proxylessnas_search_base(
      mobile_model_archive.PROXYLESSNAS_MOBILE_FILTERS)
  model_spec = search_space_utils.prune_model_spec(
      model_spec,
      {basic_specs.OP_TAG: mobile_model_archive.PROXYLESSNAS_MOBILE_OPERATIONS})
  return model_spec


def _mobilenet_v3_large_base(
    use_relative_filter_sizes):
  """Specification for MobileNet V3 - Large model."""
  block = basic_specs.block
  residual = ResidualSpec
  global_avg_pool = GlobalAveragePoolSpec

  def conv(kernel, s, bn=True):
    return ConvSpec(
        kernel_size=kernel,
        strides=s,
        use_batch_norm=bn)

  def sepconv(kernel, s, act):
    return SeparableConvSpec(
        kernel_size=kernel,
        strides=s,
        activation=act)

  def bneck(kernel, input_size, exp_size, se, s, act):
    if use_relative_filter_sizes:
      # The expanded filter size will be computed relative to the input filter
      # size. Separate logic in the model builder code ensures that the expanded
      # filter size will be an integer multiple of `model_spec.filters_base`.
      filters = basic_specs.FilterMultiplier(exp_size / input_size)
    else:
      filters = exp_size

    return DepthwiseBottleneckSpec(
        kernel_size=kernel,
        expansion_filters=choose_filters([filters]),
        use_squeeze_and_excite=se,
        strides=s,
        activation=act)

  blocks = [
      # Stem
      block([
          conv(kernel=3, s=2),
          SWISH6,
          residual(sepconv(kernel=3, s=1, act=RELU)),
          DetectionEndpointSpec(),
      ], filters=16),

      # Body
      block([
          bneck(kernel=3, input_size=16, exp_size=64, se=False, s=2, act=RELU),
          residual(bneck(kernel=3, input_size=24, exp_size=72, se=False, s=1,
                         act=RELU)),
          DetectionEndpointSpec(),
      ], filters=24),

      block([
          bneck(kernel=5, input_size=24, exp_size=72, se=True, s=2, act=RELU),
          residual(bneck(kernel=5, input_size=40, exp_size=120, se=True, s=1,
                         act=RELU)),
          residual(bneck(kernel=5, input_size=40, exp_size=120, se=True, s=1,
                         act=RELU)),
          DetectionEndpointSpec(),
      ], 40),

      block([
          bneck(kernel=3, input_size=40, exp_size=240, se=False, s=2,
                act=SWISH6),
          residual(bneck(kernel=3, input_size=80, exp_size=200, se=False, s=1,
                         act=SWISH6)),
          residual(bneck(kernel=3, input_size=80, exp_size=184, se=False, s=1,
                         act=SWISH6)),
          residual(bneck(kernel=3, input_size=80, exp_size=184, se=False, s=1,
                         act=SWISH6)),
      ], 80),

      block([
          bneck(kernel=3, input_size=80, exp_size=480, se=True, s=1,
                act=SWISH6),
          residual(bneck(kernel=3, input_size=112, exp_size=672, se=True, s=1,
                         act=SWISH6)),
          DetectionEndpointSpec(),
      ], 112),

      block([
          bneck(kernel=5, input_size=112, exp_size=672, se=True, s=2,
                act=SWISH6),
          residual(bneck(kernel=5, input_size=160, exp_size=960, se=True, s=1,
                         act=SWISH6)),
          residual(bneck(kernel=5, input_size=160, exp_size=960, se=True, s=1,
                         act=SWISH6)),
          DetectionEndpointSpec(),
      ], 160),

      # Head
      block([
          conv(kernel=1, s=1),
          SWISH6,
          global_avg_pool(),
      ], 960),

      block([
          conv(kernel=1, s=1, bn=False),
          SWISH6,
      ], 1280),
  ]
  return basic_specs.ConvTowerSpec(blocks=blocks, filters_base=8)


def mobilenet_v3_large():
  """Returns a reproduction of the MobileNetV3-Large model."""
  return _mobilenet_v3_large_base(use_relative_filter_sizes=False)


def mobilenet_multi_avg():
  """Specification for MobileNet Multi-AVG model.

  From the paper:
  "Discovering Multi-Hardware Mobile Models via Architecture Search"

  Returns:
    A ConvTowerSpec namedtuple for the Mobilenet Multi-AVG model.
  """
  block = basic_specs.block
  residual = ResidualSpec
  global_avg_pool = GlobalAveragePoolSpec

  def conv(kernel, s, bn=True):
    return ConvSpec(
        kernel_size=kernel,
        strides=s,
        use_batch_norm=bn)

  def bneck(kernel, exp_size, s):
    return DepthwiseBottleneckSpec(
        kernel_size=kernel,
        expansion_filters=choose_filters([exp_size]),
        use_squeeze_and_excite=False,
        strides=s,
        activation=RELU)

  blocks = [
      # Stem
      block([
          conv(kernel=3, s=2),
          RELU,
          DetectionEndpointSpec(),
      ], filters=32),

      # Body
      block([
          bneck(kernel=3, exp_size=96, s=2),
          residual(bneck(kernel=3, exp_size=64, s=1)),
          DetectionEndpointSpec(),
      ], filters=32),

      block([
          bneck(kernel=5, exp_size=160, s=2),
          residual(bneck(kernel=3, exp_size=192, s=1)),
          residual(bneck(kernel=3, exp_size=128, s=1)),
          residual(bneck(kernel=3, exp_size=192, s=1)),
          DetectionEndpointSpec(),
      ], 64),

      block([
          bneck(kernel=5, exp_size=384, s=2),
          residual(bneck(kernel=3, exp_size=384, s=1)),
          residual(bneck(kernel=3, exp_size=384, s=1)),
          residual(bneck(kernel=3, exp_size=384, s=1)),
      ], 128),

      block([
          bneck(kernel=3, exp_size=768, s=1),
          residual(bneck(kernel=3, exp_size=640, s=1)),
          DetectionEndpointSpec(),
      ], 160),

      block([
          bneck(kernel=3, exp_size=960, s=2),
          residual(bneck(kernel=5, exp_size=768, s=1)),
          residual(bneck(kernel=5, exp_size=768, s=1)),
          residual(bneck(kernel=5, exp_size=768, s=1)),
          DetectionEndpointSpec(),
      ], 192),

      # Head
      block([
          conv(kernel=1, s=1),
          RELU,
          global_avg_pool(),
      ], 960),

      block([
          conv(kernel=1, s=1, bn=False),
          RELU,
      ], 1280),
  ]
  return basic_specs.ConvTowerSpec(blocks=blocks, filters_base=32)


def mobilenet_multi_max():
  """Specification for MobileNet Multi-MAX model.

  From the paper:
  "Discovering Multi-Hardware Mobile Models via Architecture Search"

  Returns:
    A ConvTowerSpec namedtuple for the Mobilenet Multi-MAX model.
  """
  block = basic_specs.block
  residual = ResidualSpec
  global_avg_pool = GlobalAveragePoolSpec

  def conv(kernel, s, bn=True):
    return ConvSpec(
        kernel_size=kernel,
        strides=s,
        use_batch_norm=bn)

  def bneck(kernel, exp_size, s):
    return DepthwiseBottleneckSpec(
        kernel_size=kernel,
        expansion_filters=choose_filters([exp_size]),
        use_squeeze_and_excite=False,
        strides=s,
        activation=RELU)

  blocks = [
      # Stem
      block([
          conv(kernel=3, s=2),
          RELU,
          DetectionEndpointSpec(),
      ], filters=32),

      # Body
      block([
          bneck(kernel=3, exp_size=96, s=2),
          DetectionEndpointSpec(),
      ], filters=32),

      block([
          bneck(kernel=5, exp_size=192, s=2),
          residual(bneck(kernel=3, exp_size=128, s=1)),
          residual(bneck(kernel=3, exp_size=128, s=1)),
          DetectionEndpointSpec(),
      ], 64),

      block([
          bneck(kernel=5, exp_size=384, s=2),
          residual(bneck(kernel=3, exp_size=512, s=1)),
          residual(bneck(kernel=3, exp_size=384, s=1)),
          residual(bneck(kernel=3, exp_size=384, s=1)),
      ], 128),

      block([
          bneck(kernel=3, exp_size=768, s=1),
          residual(bneck(kernel=3, exp_size=384, s=1)),
          DetectionEndpointSpec(),
      ], 128),

      block([
          bneck(kernel=3, exp_size=768, s=2),
          residual(bneck(kernel=5, exp_size=640, s=1)),
          residual(bneck(kernel=3, exp_size=800, s=1)),
          residual(bneck(kernel=5, exp_size=640, s=1)),
          DetectionEndpointSpec(),
      ], 160),

      # Head
      block([
          conv(kernel=1, s=1),
          RELU,
          global_avg_pool(),
      ], 960),

      block([
          conv(kernel=1, s=1, bn=False),
          RELU,
      ], 1280),
  ]
  return basic_specs.ConvTowerSpec(blocks=blocks, filters_base=32)


def _mobilenet_v3_large_search_base(
    block_filters_multipliers,
    expansion_multipliers,
    search_squeeze_and_excite = False,
    always_use_relu = False,
    use_relative_expansion_filters = False,
    base_filters=(16, 24, 40, 80, 112, 160, 960, 1280)
):
  """Experimental search space built around MobileNet V3 - Large model."""
  swish6_or_relu = RELU if always_use_relu else SWISH6

  def block(layers, filters):
    all_filters = sorted({
        search_space_utils.scale_filters(filters, multiplier, base=8)
        for multiplier in block_filters_multipliers
    })
    return basic_specs.Block(layers=layers, filters=choose_filters(all_filters))

  residual = ResidualSpec
  global_avg_pool = GlobalAveragePoolSpec

  def initial_conv(s, bn=True):
    return ConvSpec(
        kernel_size=schema.OneOf([3, 5], basic_specs.OP_TAG),
        strides=s,
        use_batch_norm=bn)

  def sepconv(s, act):
    return SeparableConvSpec(
        kernel_size=schema.OneOf([3, 5, 7], basic_specs.OP_TAG),
        strides=s,
        activation=act)

  def bneck(input_size, se, s, act):
    """Construct a DepthwiseBottleneckSpec namedtuple."""
    if use_relative_expansion_filters:
      expansion_filters = sorted({
          basic_specs.FilterMultiplier(expansion)
          for expansion in expansion_multipliers
      })
    else:
      expansion_filters = sorted({
          search_space_utils.scale_filters(input_size, expansion, base=8)
          for expansion in expansion_multipliers
      })
    if search_squeeze_and_excite:
      # Replace the default value of the argument 'se' with a OneOf node.
      se = schema.OneOf([False, True], basic_specs.OP_TAG)
    return DepthwiseBottleneckSpec(
        kernel_size=schema.OneOf([3, 5, 7], basic_specs.OP_TAG),
        expansion_filters=choose_filters(expansion_filters),
        use_squeeze_and_excite=se,
        strides=s,
        activation=act)

  def optional(layer):
    return schema.OneOf([layer, basic_specs.ZeroSpec()], basic_specs.OP_TAG)

  blocks = [
      # Stem
      block([
          initial_conv(s=2),
          swish6_or_relu,
          residual(optional(sepconv(s=1, act=RELU))),
          DetectionEndpointSpec(),
      ], filters=base_filters[0]),

      # Body
      block([
          bneck(input_size=base_filters[0], se=False, s=2, act=RELU),
          residual(optional(bneck(input_size=base_filters[1], se=False, s=1,
                                  act=RELU))),
          residual(optional(bneck(input_size=base_filters[1], se=False, s=1,
                                  act=RELU))),
          residual(optional(bneck(input_size=base_filters[1], se=False, s=1,
                                  act=RELU))),
          DetectionEndpointSpec(),
      ], filters=base_filters[1]),

      block([
          bneck(input_size=base_filters[1], se=True, s=2, act=RELU),
          residual(optional(bneck(input_size=base_filters[2], se=True, s=1,
                                  act=RELU))),
          residual(optional(bneck(input_size=base_filters[2], se=True, s=1,
                                  act=RELU))),
          residual(optional(bneck(input_size=base_filters[2], se=True, s=1,
                                  act=RELU))),
          DetectionEndpointSpec(),
      ], base_filters[2]),

      block([
          bneck(input_size=base_filters[2], se=False, s=2, act=swish6_or_relu),
          residual(optional(bneck(input_size=base_filters[3], se=False, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[3], se=False, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[3], se=False, s=1,
                                  act=swish6_or_relu))),
      ], base_filters[3]),

      block([
          bneck(input_size=base_filters[3], se=True, s=1, act=swish6_or_relu),
          residual(optional(bneck(input_size=base_filters[4], se=True, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[4], se=True, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[4], se=True, s=1,
                                  act=swish6_or_relu))),
          DetectionEndpointSpec(),
      ], base_filters[4]),

      block([
          bneck(input_size=base_filters[4], se=True, s=2, act=swish6_or_relu),
          residual(optional(bneck(input_size=base_filters[5], se=True, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[5], se=True, s=1,
                                  act=swish6_or_relu))),
          residual(optional(bneck(input_size=base_filters[5], se=True, s=1,
                                  act=swish6_or_relu))),
          DetectionEndpointSpec(),
      ], base_filters[5]),

      # Head
      block([
          ConvSpec(kernel_size=1, strides=1, use_batch_norm=True),
          swish6_or_relu,
          global_avg_pool(),
      ], base_filters[6]),

      block([
          ConvSpec(kernel_size=1, strides=1, use_batch_norm=False),
          swish6_or_relu,
      ], base_filters[7]),
  ]
  return basic_specs.ConvTowerSpec(blocks=blocks, filters_base=8)


def mobilenet_v3_like_search():
  """Like exp11, but use base filter sizes which are increasing powers of 2."""
  return _mobilenet_v3_large_search_base(
      block_filters_multipliers=(0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
      expansion_multipliers=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
      use_relative_expansion_filters=True,
      search_squeeze_and_excite=True,
      base_filters=(16, 16, 32, 64, 128, 256, 512, 1024))


def get_search_space_spec(ssd):
  """Returns the search space with the specified name."""
  if ssd == MOBILENET_V2:
    return mobilenet_v2()
  elif ssd == MNASNET_B1:
    return mnasnet_b1()
  elif ssd == PROXYLESSNAS_MOBILE:
    return proxylessnas_mobile()
  elif ssd == MOBILENET_V3_LARGE:
    return mobilenet_v3_large()
  elif ssd == MOBILENET_MULTI_AVG:
    return mobilenet_multi_avg()
  elif ssd == MOBILENET_MULTI_MAX:
    return mobilenet_multi_max()
  elif ssd == PROXYLESSNAS_SEARCH:
    return proxylessnas_search()
  elif ssd == PROXYLESSNAS_ENLARGED_SEARCH:
    return proxylessnas_enlarged_search()
  elif ssd == MOBILENET_V3_LIKE_SEARCH:
    return mobilenet_v3_like_search()
  else:
    raise ValueError('Unsupported SSD')
