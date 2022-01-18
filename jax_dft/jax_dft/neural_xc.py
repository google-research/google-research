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

# python3
"""xc functional parameterized by neural network."""

import functools

import jax
from jax import lax
from jax import nn
from jax.experimental import stax
import jax.numpy as jnp
from jax.scipy import ndimage

from jax_dft import scf
from jax_dft import utils


_STAX_ACTIVATION = {
    'relu': stax.Relu,
    'elu': stax.Elu,
    'softplus': stax.Softplus,
    'swish': stax.elementwise(nn.swish),
}


def negativity_transform():
  """Layer construction function for negativity transform.

  This layer is used as the last layer of xc energy density network since
  exhange and correlation must be negative according to exact conditions.

  Note we use a 'soft' negativity transformation here. The range after this
  transformation is (-inf, 0.278].

  Returns:
    (init_fn, apply_fn) pair.
  """
  def negative_fn(x):
    return -nn.swish(x)

  return stax.elementwise(negative_fn)


def _exponential_function(displacements, width):
  """Exponential function.

  Args:
    displacements: Float numpy array.
    width: Float, parameter of exponential function.

  Returns:
    Float numpy array with same shape of displacements.
  """
  return jnp.exp(-jnp.abs(displacements) / width) / (2 * width)


def _exponential_function_channels(displacements, widths):
  """Exponential function over channels.

  Args:
    displacements: Float numpy array with shape (spatial_size, spatial_size).
    widths: Float numpy array with shape (num_channels,).

  Returns:
    Float numpy array with shape (spatial_size, spatial_size, num_channels).
  """
  return jax.vmap(_exponential_function, in_axes=(None, 0), out_axes=2)(
      displacements, widths)


def exponential_global_convolution(
    num_channels,
    grids,
    minval,
    maxval,
    downsample_factor=0,
    eta_init=nn.initializers.normal()):
  """Layer construction function for exponential global convolution.

  Args:
    num_channels: Integer, the number of channels.
    grids: Float numpy array with shape (num_grids,).
    minval: Float, the min value in the uniform sampling for exponential width.
    maxval: Float, the max value in the uniform sampling for exponential width.
    downsample_factor: Integer, the factor of downsampling. The grids are
        downsampled with step size 2 ** downsample_factor.
    eta_init: Initializer function in nn.initializers.

  Returns:
    (init_fn, apply_fn) pair.
  """
  grids = grids[::2 ** downsample_factor]
  displacements = jnp.expand_dims(
      grids, axis=0) - jnp.expand_dims(grids, axis=1)
  dx = utils.get_dx(grids)

  def init_fn(rng, input_shape):
    if num_channels <= 0:
      raise ValueError(f'num_channels must be positive but got {num_channels}')
    if len(input_shape) != 3:
      raise ValueError(
          f'The ndim of input should be 3, but got {len(input_shape)}')
    if input_shape[1] != len(grids):
      raise ValueError(
          f'input_shape[1] should be len(grids), but got {input_shape[1]}')
    if input_shape[2] != 1:
      raise ValueError(
          f'input_shape[2] should be 1, but got {input_shape[2]}')
    output_shape = input_shape[:-1] + (num_channels,)
    eta = eta_init(rng, shape=(num_channels,))
    return output_shape, (eta,)

  def apply_fn(params, inputs, **kwargs):
    """Applies layer.

    Args:
      params: Layer parameters, (eta,).
      inputs: Float numpy array with shape
          (batch_size, num_grids, num_in_channels).
      **kwargs: Other key word arguments. Unused.

    Returns:
      Float numpy array with shape (batch_size, num_grids, num_channels).
    """
    del kwargs
    eta, = params
    # shape (num_grids, num_grids, num_channels)
    kernels = _exponential_function_channels(
        displacements, widths=minval + (maxval - minval) * nn.sigmoid(eta))
    # shape (batch_size, num_grids, num_channels)
    return jnp.squeeze(
        # shape (batch_size, 1, num_grids, num_channels)
        jnp.tensordot(inputs, kernels, axes=(1, 0)) * dx,
        axis=1)

  return init_fn, apply_fn


def global_conv_block(num_channels, grids, minval, maxval, downsample_factor):
  """Global convolution block.

  First downsample the input, then apply global conv, finally upsample and
  concatenate with the input. The input itself is one channel in the output.

  Args:
    num_channels: Integer, the number of channels.
    grids: Float numpy array with shape (num_grids,).
    minval: Float, the min value in the uniform sampling for exponential width.
    maxval: Float, the max value in the uniform sampling for exponential width.
    downsample_factor: Integer, the factor of downsampling. The grids are
        downsampled with step size 2 ** downsample_factor.

  Returns:
    (init_fn, apply_fn) pair.
  """
  layers = []
  layers.extend([linear_interpolation_transpose()] * downsample_factor)
  layers.append(exponential_global_convolution(
      num_channels=num_channels - 1,  # one channel is reserved for input.
      grids=grids,
      minval=minval,
      maxval=maxval,
      downsample_factor=downsample_factor))
  layers.extend([linear_interpolation()] * downsample_factor)
  global_conv_path = stax.serial(*layers)
  return stax.serial(
      stax.FanOut(2),
      stax.parallel(stax.Identity, global_conv_path),
      stax.FanInConcat(axis=-1),
  )


def self_interaction_weight(reshaped_density, dx, width):
  """Gets self-interaction weight.

  When the density integral is one, the self-interaction weight is 1. The weight
  goes to zero when the density integral deviates from one.

  Args:
    reshaped_density: Float numpy array with any shape. The total size should be
       num_grids.
    dx: Float, grid spacing.
    width: Float, the width of the Gaussian function.

  Returns:
    Float, the self-interaction weight.
  """
  density_integral = jnp.sum(reshaped_density) * dx
  return jnp.exp(-jnp.square((density_integral - 1) / width))


def self_interaction_layer(grids, interaction_fn):
  """Layer construction function for self-interaction.

  The first input is density and the second input is the feature to mix.

  When the density integral is one, this layer outputs -0.5 * Hartree potential
  in the same shape of the density which will cancel the Hartree term.
  When the density integral is not one, the output is a linear combination of
  two inputs to this layer. The weight are determined by
  self_interaction_weight().

  Args:
    grids: Float numpy array with shape (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    (init_fn, apply_fn) pair.
  """
  dx = utils.get_dx(grids)

  def init_fn(rng, input_shape):
    del rng
    if len(input_shape) != 2:
      raise ValueError(
          f'self_interaction_layer must have two inputs, '
          f'but got {len(input_shape)}')
    if input_shape[0] != input_shape[1]:
      raise ValueError(
          f'The input shape to self_interaction_layer must be equal, '
          f'but got {input_shape[0]} and {input_shape[1]}')
    return input_shape[0], (jnp.array(1.),)

  def apply_fn(params, inputs, **kwargs):  # pylint: disable=missing-docstring
    del kwargs
    width, = params
    reshaped_density, features = inputs
    beta = self_interaction_weight(
        reshaped_density=reshaped_density, dx=dx, width=width)
    hartree = -0.5 * scf.get_hartree_potential(
        density=reshaped_density.reshape(-1),
        grids=grids,
        interaction_fn=interaction_fn).reshape(reshaped_density.shape)
    return hartree * beta + features * (1 - beta)

  return init_fn, apply_fn


def wrap_network_with_self_interaction_layer(network, grids, interaction_fn):
  """Wraps a network with self-interaction layer.

  Args:
    network: an (init_fn, apply_fn) pair.
     * init_fn: The init_fn of the neural network. It takes an rng key and
         an input shape and returns an (output_shape, params) pair.
     * apply_fn: The apply_fn of the neural network. It takes params,
         inputs, and an rng key and applies the layer.
    grids: Float numpy array with shape (num_grids,).
    interaction_fn: function takes displacements and returns
        float numpy array with the same shape of displacements.

  Returns:
    (init_fn, apply_fn) pair.
  """
  return stax.serial(
      stax.FanOut(2),
      stax.parallel(stax.Identity, network),
      self_interaction_layer(grids, interaction_fn),
  )


# pylint: disable=invalid-name
def GeneralConvWithoutBias(
    dimension_numbers, out_chan, filter_shape,
    strides=None, padding='VALID', W_init=None):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, _ = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or jax.nn.initializers.he_normal(
      rhs_spec.index('I'), rhs_spec.index('O'))
  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(
        input_shape, kernel_shape, strides, padding, dimension_numbers)
    W = W_init(rng, kernel_shape)
    return output_shape, (W,)
  def apply_fun(params, inputs, **kwargs):
    del kwargs
    W, = params
    return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                    dimension_numbers=dimension_numbers)
  return init_fun, apply_fun

Conv1D = functools.partial(GeneralConvWithoutBias, ('NHC', 'HIO', 'NHC'))

# pylint: enable=invalid-name


def _resample_1d(inputs, new_size):
  if inputs.ndim != 1:
    raise ValueError(f'inputs must be 1d but has shape {inputs.shape}')
  x = jnp.linspace(0, inputs.size - 1, num=new_size)
  return ndimage.map_coordinates(inputs, [x], order=1, mode='nearest')


def linear_interpolation():
  """Layer construction function for 1d linear interpolation.

  The input is 'NHC' and H is the spatial dimension. After upsampling, the
  size of the spatial dimension increases from `m` to `2 * m - 1`.

  Example of what this linear transformation looks like (for m=5):

    [[1. , 0. , 0. , 0. , 0. ],
     [0.5, 0.5, 0. , 0. , 0. ],
     [0. , 1. , 0. , 0. , 0. ],
     [0. , 0.5, 0.5, 0. , 0. ],
     [0. , 0. , 1. , 0. , 0. ],
     [0. , 0. , 0.5, 0.5, 0. ],
     [0. , 0. , 0. , 1. , 0. ],
     [0. , 0. , 0. , 0.5, 0.5],
     [0. , 0. , 0. , 0. , 1. ]]

  Returns:
    (init_fn, apply_fn) pair.
  """
  def init_fn(rng, input_shape):
    del rng
    output_shape = input_shape[0], 2 * input_shape[1] - 1, input_shape[2]
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    del params, kwargs
    upsample = functools.partial(_resample_1d, new_size=2 * inputs.shape[1] - 1)
    return jax.vmap(jax.vmap(upsample, 0, 0), 2, 2)(inputs)

  return init_fn, apply_fn


def _with_edge(x, scale):
  """Rescale values on the edge of a 1D array."""
  return jnp.concatenate([scale * x[:1], x[1:-1], scale * x[-1:]])


def linear_interpolation_transpose():
  """Layer construction function for the tranpsose of 1d linear interpolation.

  Assumes constant boundary conditions. Rows are rescaled so that they sum to 1.

  The input is 'NHC' and H is the spatial dimension. After downsampling, the
  size of the spatial dimension decreases from `2 * m - 1` to `m`.

  Example of what this linear transformation looks like (for m=5):

    [[0.75, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
     [0.  , 0.25, 0.5 , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  ],
     [0.  , 0.  , 0.  , 0.25, 0.5 , 0.25, 0.  , 0.  , 0.  ],
     [0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.5 , 0.25, 0.  ],
     [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.75]]

  Returns:
    (init_fn, apply_fn) pair.

  Raises:
    ValueError: if the size of the input spatial domain is not odd.
  """
  def init_fn(rng, input_shape):
    del rng
    if input_shape[1] % 2 == 0:
      raise ValueError(f'input shape must be odd {input_shape[1]}')
    m = (input_shape[1] + 1) // 2
    output_shape = input_shape[0], m, input_shape[2]
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):  # pylint: disable=missing-docstring
    del params, kwargs
    m = (inputs.shape[1] + 1) // 2
    upsample = functools.partial(_resample_1d, new_size=inputs.shape[1])
    dummy_inputs = jnp.zeros((m,), dtype=inputs.dtype)  # needed for tracing
    _, vjp_fun = jax.vjp(upsample, dummy_inputs)

    def downsample(x):
      # scale by 0.5 so downsampling preserves the average activation
      output = 0.5 * vjp_fun(x)[0]
      # need to add back in the edges to account for boundary conditions.
      output = output.at[:1].add(0.25 * x[:1])
      output = output.at[-1:].add(0.25 * x[-1:])
      return output

    return jax.vmap(jax.vmap(downsample, 0, 0), 2, 2)(inputs)

  return init_fn, apply_fn


def upsampling_block(num_filters, activation):
  """An upsampling block.

  Given the input feature with spatial dimension size `m`, upsamples the size to
  `2 * m - 1` and then applies a convolution.

  Args:
    num_filters: Integer, the number of filters of the convolution layers.
    activation: String, the activation function to use in the network.

  Returns:
    (init_fn, apply_fn) pair.
  """
  return stax.serial(
      linear_interpolation(),
      Conv1D(num_filters, filter_shape=(3,), padding='SAME'),
      _STAX_ACTIVATION[activation],
  )


def downsampling_block(num_filters, activation):
  """A downsampling block.

  Given the input feature with spatial dimension size `m`, applies a convolution
  and then reduce the size to `(m + 1) / 2`.

  Args:
    num_filters: Integer, the number of filters of the three convolution layers.
    activation: String, the activation function to use in the network.

  Returns:
    (init_fn, apply_fn) pair.
  """
  return stax.serial(
      Conv1D(num_filters, filter_shape=(3,), padding='SAME'),
      linear_interpolation_transpose(),
      _STAX_ACTIVATION[activation],
  )


def _build_unet_shell(layer, num_filters, activation):
  """Builds a shell in the U-net structure.

  *--------------*
  |              |                     *--------*     *----------------------*
  | downsampling |---------------------|        |     |                      |
  |   block      |   *------------*    | concat |-----|   upsampling block   |
  |              |---|    layer   |----|        |     |                      |
  *--------------*   *------------*    *--------*     *----------------------*

  Args:
    layer: (init_fn, apply_fn) pair in the bottom of the U-shape structure.
    num_filters: Integer, the number of filters used for downsampling and
        upsampling.
    activation: String, the activation function to use in the network.

  Returns:
    (init_fn, apply_fn) pair.
  """
  return stax.serial(
      downsampling_block(num_filters, activation=activation),
      stax.FanOut(2),
      stax.parallel(stax.Identity, layer),
      stax.FanInConcat(axis=-1),
      upsampling_block(num_filters, activation=activation)
  )


def build_unet(
    num_filters_list, core_num_filters, activation,
    num_channels=0, grids=None, minval=None, maxval=None,
    apply_negativity_transform=True):
  """Builds U-net.

  This neural network is used to parameterize a many-to-many mapping.

  Args:
    num_filters_list: List of integers, the number of filters for each
      downsampling_block.
      The number of filters for each upsampling_block is in reverse order.
      For example, if num_filters_list=[16, 32, 64], there are 3
      downsampling_block with number of filters from left to right: 16, 32, 64.
      There are 3 upsampling_block with number of filters from left to right:
      64, 32 ,16.
    core_num_filters: Integer, the number of filters for the convolution layer
      at the bottom of the U-shape structure.
    activation: String, the activation function to use in the network.
    num_channels: Integer, the number of channels.
    grids: Float numpy array with shape (num_grids,).
    minval: Float, the min value in the uniform sampling for exponential width.
    maxval: Float, the max value in the uniform sampling for exponential width.
    apply_negativity_transform: Boolean, whether to add negativity_transform at
        the end.

  Returns:
    (init_fn, apply_fn) pair.
  """
  layer = stax.serial(
      Conv1D(core_num_filters, filter_shape=(3,), padding='SAME'),
      _STAX_ACTIVATION[activation],
      Conv1D(core_num_filters, filter_shape=(3,), padding='SAME'),
      _STAX_ACTIVATION[activation])
  for num_filters in num_filters_list[::-1]:
    layer = _build_unet_shell(layer, num_filters, activation=activation)
  network = stax.serial(
      layer,
      # Use 1x1 convolution filter to aggregate channels.
      Conv1D(1, filter_shape=(1,), padding='SAME'))
  layers_before_network = []
  if num_channels > 0:
    layers_before_network.append(
        exponential_global_convolution(num_channels, grids, minval, maxval))
  if apply_negativity_transform:
    return stax.serial(*layers_before_network, network, negativity_transform())
  else:
    return stax.serial(*layers_before_network, network)


def build_global_local_conv_net(
    num_global_filters, num_local_filters, num_local_conv_layers, activation,
    grids, minval, maxval, downsample_factor, apply_negativity_transform=True):
  """Builds global-local convolutional network.

  Args:
    num_global_filters: Integer, the number of global filters in one cell.
    num_local_filters: Integer, the number of local filters in one cell.
    num_local_conv_layers: Integer, the number of local convolution layer in
        one cell.
    activation: String, the activation function to use in the network.
    grids: Float numpy array with shape (num_grids,).
    minval: Float, the min value in the uniform sampling for exponential width.
    maxval: Float, the max value in the uniform sampling for exponential width.
    downsample_factor: Integer, the factor of downsampling. The grids are
        downsampled with step size 2 ** downsample_factor.
    apply_negativity_transform: Boolean, whether to add negativity_transform at
        the end.

  Returns:
    (init_fn, apply_fn) pair.
  """
  layers = []
  layers.append(
      global_conv_block(
          num_channels=num_global_filters,
          grids=grids,
          minval=minval,
          maxval=maxval,
          downsample_factor=downsample_factor))
  layers.extend([
      Conv1D(num_local_filters, filter_shape=(3,), padding='SAME'),
      _STAX_ACTIVATION[activation]] * num_local_conv_layers)
  layers.append(
      # Use unit convolution filter to aggregate channels.
      Conv1D(1, filter_shape=(1,), padding='SAME'))
  if apply_negativity_transform:
    layers.append(negativity_transform())
  return stax.serial(*layers)


def build_sliding_net(
    window_size, num_filters_list, activation, apply_negativity_transform=True):
  """Builds neural network sliding over the input.

  The receptive field of this network is window_size.

  Args:
    window_size: Integer, the window size of the input, window_size >= 1.
    num_filters_list: List of integers, the number of filters for each layer.
    activation: String, the activation function to use in the network.
    apply_negativity_transform: Boolean, whether to add negativity_transform at
        the end.

  Returns:
    (init_fn, apply_fn) pair.

  Raises:
    ValueError: If window_size is less than 1.
  """
  if window_size < 1:
    raise ValueError(
        f'window_size cannot be less than 1, but got {window_size}')

  layers = []
  for i, num_filters in enumerate(num_filters_list):
    if i == 0:
      filter_shape = (window_size,)
    else:
      filter_shape = (1,)
    layers.extend([
        Conv1D(num_filters, filter_shape=filter_shape, padding='SAME'),
        _STAX_ACTIVATION[activation],
    ])
  layers.append(Conv1D(1, filter_shape=(1,), padding='SAME'))
  if apply_negativity_transform:
    layers.append(negativity_transform())
  return stax.serial(*layers)


def _check_network_output(output, num_features):
  """Checks whether the shape of the network output is (-1, num_features).

  Args:
    output: Numpy array, output of the network.
    num_features: Integer, number of output features.

  Raises:
    ValueError: If the shape of the output is not (-1, num_features).
  """
  shape = output.shape
  if output.ndim != 2 or shape[1] != num_features:
    raise ValueError(
        'The output shape of the network should be (-1, {}) but got {}'
        .format(num_features, shape))


def _is_power_of_two(number):
  """Checks whether a number is power of 2.

  If a number is power of 2, all the digits of its binary are zero except for
  the leading digit.
  For example:
    1 -> 1
    2 -> 10
    4 -> 100
    8 -> 1000
    16 -> 10000

  All the digits after the leading digit are zero after subtracting 1 from
  number.
  For example:
    0 -> 0
    1 -> 1
    3 -> 11
    7 -> 111
    15 -> 1111

  Therefore, given a non-zero number, (number & (number - 1)) is zero
  if number is power of 2.

  Args:
    number: Integer.

  Returns:
    Boolean.
  """
  return number and not number & (number - 1)


def _spatial_shift_input(features, num_spatial_shift):
  """Applies spatial shift to the input features.

  For each sample in the batch, num_spatial_shift copies are created, each of
  which is shifted to the left by 0 to num_spatial_shift - 1 in the
  num_grids dimension.

  Args:
    features: Float numpy array with shape
        (batch_size, num_grids, num_features).
    num_spatial_shift: Integer, the number of spatial shift (include the
        original input).

  Returns:
    Float numpy array with shape
        (batch_size * num_spatial_shift, num_grids, num_features)
  """
  output = []
  for sample_features in features:
    for offset in range(num_spatial_shift):
      output.append(
          jax.vmap(functools.partial(utils.shift, offset=offset), 1, 1)
          (sample_features))
  return jnp.stack(output)


def _reverse_spatial_shift_output(array):
  """Applies reverse spatial shift to the network output array.

  For each sample in the batch, the elements are shifted to the right by
  0 to num_spatial_shift - 1 in the num_grids dimension.

  Args:
    array: Float numpy array with shape (num_spatial_shift, num_grids).

  Returns:
    Float numpy array with shape (num_spatial_shift, num_grids).
  """
  output = []
  for offset, sample_array in enumerate(array):
    output.append(utils.shift(sample_array, offset=-offset))  # reverse offset.
  return jnp.stack(output)


def local_density_approximation(network):
  """Local Density Approximation (LDA) parameterized by neural network.

  This functional takes density as input.

  The output shape of the network must be (-1, 1).

  Args:
    network: an (init_fn, apply_fn) pair.
     * init_fn: The init_fn of the neural network. It takes an rng key and
         an input shape and returns an (output_shape, params) pair.
     * apply_fn: The apply_fn of the neural network. It takes params,
         inputs, and an rng key and applies the layer.

  Returns:
    init_fn: A function takes an rng key and returns initial params.
    xc_energy_density_fn: A function takes density (1d array) and params,
        returns xc energy density with the same shape of density.
  """
  network_init_fn, network_apply_fn = network

  def init_fn(rng):
    _, params = network_init_fn(rng=rng, input_shape=(-1, 1))
    return params

  @jax.jit
  def xc_energy_density_fn(density, params):
    """Gets xc energy density.

    Args:
      density: Float numpy array with shape (num_grids,).
      params: Parameters of the network.

    Returns:
      Float numpy array with shape (num_grids,).
    """
    # The shape of the density is (num_grids,).
    # The density on each grid point is an input, so the network is applied to
    # each point of the density on grid.
    # It is equivalent to consider a density with shape (num_grids,) as data
    # of single feature with batch size num_grids.
    output = network_apply_fn(params, jnp.expand_dims(density, axis=1))
    _check_network_output(output, num_features=1)
    output = jnp.squeeze(output, axis=-1)
    return output

  return init_fn, xc_energy_density_fn


def global_functional(network, grids, num_spatial_shift=1):
  """Functional with global density information parameterized by neural network.

  This function takes the entire density as input and outputs the entire xc
  energy density.

  The network used in this network is a convolution neural network. This
  function will expand the batch dimension and channel dimension of the input
  density to fit the input shape of the network.

  There are two types of mapping can be applied depending on the architecture
  of the network.

  * many-to-one:
    This function is inspired by

    Schmidt, Jonathan, Carlos L. Benavides-Riveros, and Miguel AL Marques.
    "Machine Learning the Physical Nonlocal Exchange–Correlation Functional of
    Density-Functional Theory."
    The journal of physical chemistry letters 10 (2019): 6425-6431.

    The XC energy density at index x is determined by the density in
    [x - radius, x + radius].

    Note for radius=0, only the density at one point is used to predict the
    xc energy density at the same point. Thus it is equivalent to LDA.

    For radius=1, the density at one point and its nearest neighbors are used to
    predict the xc energy density at this point. It uses same information used
    by GGA, where the gradient of density is computed by the finite difference.

    For large radius, it can be considered as a non-local functional.

    Applying MLP on 1d array as a sliding window is not accelerator efficient.
    Instead, same operations can be performed by using 1d convolution with
    filter size 2 * radius + 1 as the first layer, and 1d convolution with
    filter size 1 for the rest of the layers. The channel dimension in the
    rest of the layers acts as the hidden nodes in MLP.

  * many-to-many:
    The XC energy density at index x is determined by the entire density. This
    mapping can be parameterized by a U-net structure to capture both the low
    level and high level features from the input.

  Args:
    network: an (init_fn, apply_fn) pair.
     * init_fn: The init_fn of the neural network. It takes an rng key and
         an input shape and returns an (output_shape, params) pair.
     * apply_fn: The apply_fn of the neural network. It takes params,
         inputs, and an rng key and applies the layer.
    grids: Float numpy array with shape (num_grids,).
        num_grids must be 2 ** k + 1, where k is an non-zero integer.
    num_spatial_shift: Integer, the number of spatial shift (include the
        original input).

  Returns:
    init_fn: A function takes an rng key and returns initial params.
    xc_energy_density_fn: A function takes density (1d array) and params,
        returns xc energy density with the same shape of density.

  Raises:
    ValueError: If num_spatial_shift is less than 1
        or the num_grids is not 2 ** k + 1.
  """
  if num_spatial_shift < 1:
    raise ValueError(
        f'num_spatial_shift can not be less than 1 but got {num_spatial_shift}')

  network_init_fn, network_apply_fn = network
  num_grids = grids.shape[0]

  if not _is_power_of_two(num_grids - 1):
    raise ValueError(
        'The num_grids must be power of two plus one for global functional '
        'but got %d' % num_grids)

  def init_fn(rng):
    _, params = network_init_fn(rng=rng, input_shape=(-1, num_grids, 1))
    return params

  @jax.jit
  def xc_energy_density_fn(density, params):
    """Gets xc energy density.

    Args:
      density: Float numpy array with shape (num_grids,).
      params: Parameters of the network.

    Returns:
      Float numpy array with shape (num_grids,).
    """
    # Expand batch dimension and channel dimension. We use batch_size=1 here.
    # (1, num_grids, 1)
    input_features = density[jnp.newaxis, :, jnp.newaxis]
    if num_spatial_shift > 1:
      # The batch dimension size will be num_spatial_shift.
      # (num_spatial_shift, num_grids, num_input_features)
      input_features = _spatial_shift_input(
          input_features, num_spatial_shift=num_spatial_shift)

    output = network_apply_fn(params, input_features)

    # Remove the channel dimension.
    # (num_spatial_shift, num_grids)
    output = jnp.squeeze(output, axis=2)
    _check_network_output(output, num_grids)
    # Remove the batch dimension.
    if num_spatial_shift > 1:
      output = _reverse_spatial_shift_output(output)
    output = jnp.mean(output, axis=0)

    return output

  return init_fn, xc_energy_density_fn
