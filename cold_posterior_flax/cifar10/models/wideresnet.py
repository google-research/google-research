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

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=g-long-lambda
"""Wide ResNet model for CIFAR10 image classification.

Reference:

Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/wideresnet.py

Which was later moved (and has since diverged from this fork):
github.com/google-research/google-research/blob/master/flax_models/cifar/models/wide_resnet.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
that is widely used as a benchmark.
"""

from typing import List
from typing import Tuple
from typing import Type
from flax import nn
from flax.nn import initializers
import jax.numpy as jnp
import jax.scipy.stats
from cold_posterior_flax.cifar10.models import activations
from cold_posterior_flax.cifar10.models import conv_layers
from cold_posterior_flax.cifar10.models import evonorm
from cold_posterior_flax.cifar10.models import new_initializers
from cold_posterior_flax.cifar10.models import normalizers


def std_penalty(x, multiplier=1.0):
  """Returns a l2 loss on the std/var of x."""
  x = x.reshape(-1, x.shape[-1])
  mean = jnp.mean(x, axis=0)
  var = ((x - mean)**2).mean(0)
  # L2 loss on std < 1.0
  penalty = jnp.sqrt(jnp.mean((var - 1.0)**2))
  return multiplier * penalty


def get_conv(activation_f, bias_scale, weight_norm, compensate_padding,
             normalization):
  """Create Conv constructor based on architecture settings."""

  if weight_norm == 'fixed':
    conv = conv_layers.ConvFixedScale
  elif weight_norm == 'learned_b':
    conv = conv_layers.ConvLearnedScale
  elif weight_norm == 'ws_sqrt':
    conv = conv_layers.ConvWS.partial(kaiming_scaling=True)
  elif weight_norm == 'ws':
    conv = conv_layers.ConvWS.partial(kaiming_scaling=False)
  elif weight_norm == 'learned':
    conv = conv_layers.Conv
  elif weight_norm in ['none']:
    conv = conv_layers.Conv
  else:
    raise ValueError('weight_norm invalid option %s' % weight_norm)

  conv = conv.partial(compensate_padding=compensate_padding)

  if normalization in ['bn', 'bn_sync', 'gn_16', 'gn_32', 'gn_4', 'frn']:
    bias = False
  elif normalization in ['none']:
    bias = True
  else:
    raise ValueError('Does not exist')

  bias_init = None
  # TODO(basv): refactor to use config.activation_f.
  if activation_f.__name__ in [
      'relu',
      'tlu',
      'none',
      'tldu',
      'tlduz',
      'tlum',
      'relu_unitvar',
      'swish',
  ]:
    kernel_init = initializers.kaiming_normal()
    bias_init = jax.nn.initializers.normal(bias_scale)
  elif activation_f.__name__ in [
      'bias_relu_norm', 'bias_SELU_norm', 'SELU_norm_rebias',
      'bias_scale_relu_norm', 'bias_scale_SELU_norm', 'bias_scale_SELU_norm_gb'
  ]:
    # TODO(basv): parametrize normalized initializaton using lecun_normed().
    kernel_init = initializers.lecun_normal()
    bias = False
  elif activation_f.__name__ in [
      'selu', 'relu_norm', 'capped', 'evonorm_s0', 'evonorm_b0'
  ]:
    kernel_init = initializers.lecun_normal()
    bias_init = jax.nn.initializers.normal(bias_scale)
  elif activation_f.__name__ == 'tanh':
    # Scale according to:
    # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#calculate_gain
    kernel_init = initializers.variance_scaling((5.0 / 3)**2, 'fan_in',
                                                'truncated_normal')
    bias_init = jax.nn.initializers.normal(bias_scale)
  else:
    raise ValueError('Not prepared for activation_f', activation_f.__name__)
  conv = conv.partial(kernel_init=kernel_init, bias=bias, bias_init=bias_init)
  return conv


def get_activation_f(activation_f, train, softplus_scale, bias_scale):
  """Create activation function constructor."""
  if softplus_scale:
    scale_init = new_initializers.init_softplus_ones
  else:
    scale_init = jax.nn.initializers.ones

  name = activation_f
  if activation_f == 'relu':
    activation_f = lambda x, **kwargs: nn.activation.relu(x)
  elif activation_f == 'tanh':
    activation_f = lambda x, **kwargs: nn.activation.tanh(x)
  elif activation_f == 'tlu':
    activation_f = lambda x, **kwargs: activations.TLU(x)
  elif activation_f == 'evonorm_s0':
    activation_f = lambda x, **kwargs: evonorm.EvoNorm(
        x,
        layer=evonorm.LAYER_EVONORM_S0,
        use_running_average=not train,
        axis_name=None)
  elif activation_f == 'evonorm_b0':
    activation_f = lambda x, **kwargs: evonorm.EvoNorm(
        x,
        layer=evonorm.LAYER_EVONORM_B0,
        use_running_average=not train,
        axis_name=None)
  elif activation_f == 'capped':
    activation_f = lambda x, **kwargs: activations.Capped(x)
  elif activation_f == 'swish':
    activation_f = lambda x, **kwargs: activations.Swish(x)
  elif activation_f == 'relu_norm':
    activation_f = lambda x, **kwargs: activations.relu_norm(x)
  elif activation_f == 'selu':
    activation_f = lambda x, **kwargs: jax.nn.selu(x)
  elif activation_f == 'bias_relu_norm':
    activation_f = lambda x, **kwargs: activations.BiasReluNorm(
        x, bias_init=jax.nn.initializers.normal(bias_scale), **kwargs)
  elif activation_f == 'bias_scale_relu_norm':
    activation_f = lambda x, **kwargs: activations.BiasReluNorm(
        x,
        bias_init=jax.nn.initializers.normal(bias_scale),
        scale_init=scale_init,
        scale=True,
        softplus=softplus_scale,
        **kwargs)
  elif activation_f == 'bias_SELU_norm':
    activation_f = lambda x, **kwargs: activations.BiasSELUNorm(
        x, bias_init=jax.nn.initializers.normal(bias_scale), **kwargs)
  elif activation_f == 'bias_scale_SELU_norm':
    activation_f = lambda x, **kwargs: activations.BiasSELUNorm(
        x,
        bias_init=jax.nn.initializers.normal(bias_scale),
        scale_init=scale_init,
        scale=True,
        softplus=softplus_scale,
        **kwargs)
  elif activation_f == 'bias_scale_SELU_norm_gb':
    activation_f = lambda x, **kwargs: activations.BiasSELUNorm(
        x,
        bias_init=jax.nn.initializers.normal(bias_scale),
        scale_init=scale_init,
        scale=True,
        norm_grad_block=True,
        softplus=softplus_scale,
        **kwargs)
  elif activation_f == 'SELU_norm_rebias':
    activation_f = lambda x, **kwargs: activations.SELUNormReBias(
        x, bias_init=jax.nn.initializers.normal(bias_scale), **kwargs)
  elif activation_f == 'relu_unitvar':
    activation_f = lambda x, **kwargs: activations.relu_unitvar(x)
  elif activation_f == 'tlum':
    activation_f = lambda x, **kwargs: activations.TLUM(x)
  elif activation_f == 'tldu':
    activation_f = lambda x, **kwargs: activations.TLDU(x)
  elif activation_f == 'tlduz':
    activation_f = lambda x, **kwargs: activations.TLDUZ(x)
  elif activation_f == 'none':
    activation_f = lambda x, **kwargs: x
  else:
    raise ValueError('activation_f')
  activation_f.__name__ = name
  return activation_f


def get_norm(activation_f, normalization, train):
  """Create normalization layer based on activation function."""
  if activation_f.__name__ in [
      'relu',
      'tlu',
      'none',
      'tldu',
      'tlduz',
      'tlum',
      'relu_unitvar',
      'tanh',
      'selu',
      'swish',
      'capped',
      'evonorm_s0',
      'evonorm_b0',
  ]:
    use_bias = True
    use_scale = True
  elif activation_f.__name__ in [
      'bias_relu_norm', 'bias_SELU_norm', 'SELU_norm_rebias'
  ]:
    use_bias = False
    use_scale = True
  elif activation_f.__name__ in [
      'bias_scale_relu_norm', 'bias_scale_SELU_norm', 'bias_scale_SELU_norm_gb'
  ]:
    use_bias = False
    use_scale = False
  elif activation_f.__name__ in ['relu_norm']:
    use_bias = True
    use_scale = True
  else:
    raise ValueError('Not prepared for activation_f', activation_f.__name__)

  if normalization == 'bn':
    norm = nn.BatchNorm.partial(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        scale=use_scale,
        bias=use_bias)

  elif normalization == 'rescale':
    norm = normalizers.ReScale.partial()
  elif normalization == 'bn_sync':
    norm = nn.BatchNorm.partial(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name='batch',
        scale=use_scale,
        bias=use_bias)
  elif normalization == 'frn':
    norm = normalizers.FRN.partial(scale=use_scale, bias=use_bias)
  elif normalization == 'gn_4':
    norm = nn.GroupNorm.partial(num_groups=4, scale=use_scale, bias=use_bias)
  elif normalization == 'gn_16':
    norm = nn.GroupNorm.partial(num_groups=16, scale=use_scale, bias=use_bias)
  elif normalization == 'gn_32':
    norm = nn.GroupNorm.partial(num_groups=32, scale=use_scale, bias=use_bias)
  elif normalization == 'none':
    norm = lambda x, *args, **kwargs: x
  else:
    raise ValueError('Invalid normalization')
  return norm


class WideResnetBlock(nn.Module):
  """Defines a single wide ResNet block."""

  def apply(self,
            x,
            channels,
            strides=(1, 1),
            dropout_rate=0.0,
            normalization='bn',
            activation_f=None,
            std_penalty_mult=0,
            use_residual=1,
            train=True,
            bias_scale=0.0,
            weight_norm='none',
            compensate_padding=True):
    norm = get_norm(activation_f, normalization, train)

    conv = get_conv(activation_f, bias_scale, weight_norm, compensate_padding,
                    normalization)
    penalty = 0
    y = x
    y = norm(y, name='norm1')
    if std_penalty_mult > 0:
      penalty += std_penalty(y)
    y = activation_f(y, features=y.shape[-1])
    y = conv(
        y,
        channels,
        (3, 3),
        strides,
        padding='SAME',
        name='conv1',
    )
    y = norm(y, name='norm2')
    if std_penalty_mult > 0:
      penalty += std_penalty(y)
    y = activation_f(y, features=y.shape[-1])
    if dropout_rate > 0.0:
      y = nn.dropout(y, dropout_rate, deterministic=not train)
    y = conv(y, channels, (3, 3), padding='SAME', name='conv2')

    if use_residual == 1:
      # Apply an up projection in case of channel mismatch
      if (x.shape[-1] != channels) or strides != (1, 1):
        x = conv(x, y.shape[-1], (3, 3), strides, padding='SAME')
      result = x + y
    elif use_residual == 2:
      # Unit variance preserving residual.
      if (x.shape[-1] != channels) or strides != (1, 1):
        x = conv(x, y.shape[-1], (3, 3), strides, padding='SAME')

      result = (x + y) / jnp.sqrt(1**2 + 1**2)  # Sum of independent normals.
    else:
      result = y

    return result, penalty


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""

  def apply(
      self,
      x,
      blocks_per_group,
      channels,
      strides=(1, 1),
      dropout_rate=0.0,
      normalization='bn',
      activation_f=None,
      std_penalty_mult=0,
      use_residual=True,
      train=True,
      bias_scale=0.0,
      weight_norm='fixed',
  ):
    penalty = None
    for i in range(blocks_per_group):
      x, b_penalty = WideResnetBlock(
          x,
          channels,
          strides if i == 0 else (1, 1),
          dropout_rate,
          normalization,
          activation_f,
          std_penalty_mult,
          use_residual,
          train=train,
          bias_scale=bias_scale,
          weight_norm=weight_norm)
      penalty = b_penalty if penalty is None else penalty + b_penalty

    return x, penalty


class ResnetV1(nn.Module):
  """Resnet as used by Wenzel et al CIFAR10 experiments.

  Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L., Mandt, S.,
  et al. (2020, February 6). How Good is the Bayes Posterior in Deep Neural
  Networks Really? arXiv [stat.ML]. http://arxiv.org/abs/2002.02405.
  """

  def apply(self,
            x,
            depth,
            num_outputs,
            dropout_rate=0.0,
            normalization='bn',
            activation_f=None,
            std_penalty_mult=0,
            use_residual=1,
            train=True,
            bias_scale=0.0,
            weight_norm='none',
            filters=16,
            no_head=False,
            report_metrics=False,
            benchmark='cifar10',
            compensate_padding=True,
            softplus_scale=None):

    bn_index = iter(range(1000))
    conv_index = iter(range(1000))
    summaries = {}
    summary_ind = [0]

    def add_summary(name, val):
      """Summarize statistics of tensor."""
      if report_metrics:
        assert val.ndim == 4, ('Assuming 4D inputs with channels last, got %s' %
                               str(val.shape))
        assert val.shape[1] == val.shape[
            2], 'Assuming 4D inputs with channels last'
        summaries['%s_%d_mean_abs' % (name, summary_ind[0] // 2)] = jnp.mean(
            jnp.abs(jnp.mean(val, axis=(0, 1, 2))))
        summaries['%s_%d_mean_std' % (name, summary_ind[0] // 2)] = jnp.mean(
            jnp.std(val, axis=(0, 1, 2)))
        summary_ind[0] += 1

    penalty = 0

    activation_f = get_activation_f(activation_f, train, softplus_scale,
                                    bias_scale)
    norm = get_norm(activation_f, normalization, train)

    conv = get_conv(activation_f, bias_scale, weight_norm, compensate_padding,
                    normalization)

    def resnet_layer(
        inputs,
        penalty,
        filters,
        kernel_size=3,
        strides=1,
        activation=None,
    ):
      """2D Convolution-Batch Normalization-Activation stack builder."""
      x = inputs
      x = conv(
          x,
          filters, (kernel_size, kernel_size),
          strides=(strides, strides),
          padding='SAME',
          name='conv%d' % next(conv_index))
      x = norm(x, name='norm%d' % next(bn_index))
      add_summary('postnorm', x)
      if std_penalty_mult > 0:
        penalty += std_penalty(x)

      if activation:
        x = activation_f(x, features=x.shape[-1])
      add_summary('postact', x)
      return x, penalty

    # Main network code.
    num_res_blocks = (depth - 2) // 6

    if (depth - 2) % 6 != 0:
      raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

    inputs = x
    add_summary('input', x)
    add_summary('inputb', x)
    if benchmark in ['cifar10', 'cifar100']:
      x, penalty = resnet_layer(
          inputs, penalty, filters=filters, activation=True)
      head_kernel_init = nn.initializers.lecun_normal()
    elif benchmark in ['imagenet']:
      head_kernel_init = nn.initializers.zeros
      x, penalty = resnet_layer(
          inputs,
          penalty,
          filters=filters,
          activation=False,
          kernel_size=7,
          strides=2)
      # TODO(basv): evaluate max pool v/s avg_pool in an experiment?
      # if compensate_padding:
      #   x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding="VALID")
      # else:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    else:
      raise ValueError('Model def not prepared for benchmark %s' % benchmark)

    for stack in range(3):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # First layer but not first stack.
          strides = 2  # Downsample.
        y, penalty = resnet_layer(
            x,
            penalty,
            filters=filters,
            strides=strides,
            activation=True,
        )
        y, penalty = resnet_layer(
            y,
            penalty,
            filters=filters,
            activation=False,
        )
        if stack > 0 and res_block == 0:  # First layer but not first stack.
          # Linear projection residual shortcut to match changed dims.
          x, penalty = resnet_layer(
              x,
              penalty,
              filters=filters,
              kernel_size=1,
              strides=strides,
              activation=False,
          )

        if use_residual == 1:
          # Apply an up projection in case of channel mismatch
          x = x + y
        elif use_residual == 2:
          x = (x + y) / jnp.sqrt(1**2 + 1**2)  # Sum of independent normals.
        else:
          x = y

        add_summary('postres', x)
        x = activation_f(x, features=x.shape[-1])
        add_summary('postresact', x)
      filters *= 2

    # V1 does not use BN after last shortcut connection-ReLU.
    if not no_head:
      x = jnp.mean(x, axis=(1, 2))
      add_summary('postpool', x)
      x = x.reshape((x.shape[0], -1))

      x = nn.Dense(x, num_outputs, kernel_init=head_kernel_init)
    return x, penalty, summaries


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""

  def apply(self,
            x,
            blocks_per_group,
            channel_multiplier,
            num_outputs,
            dropout_rate=0.0,
            normalization='bn',
            activation_f=None,
            std_penalty_mult=0,
            use_residual=1,
            train=True,
            bias_scale=0.0,
            weight_norm='none',
            no_head=False,
            compensate_padding=True,
            softplus_scale=None):

    penalty = 0

    activation_f = get_activation_f(activation_f, train, softplus_scale,
                                    bias_scale)
    norm = get_norm(activation_f, normalization, train)
    conv = get_conv(activation_f, bias_scale, weight_norm, compensate_padding,
                    normalization)
    x = conv(
        x, 16 * channel_multiplier, (3, 3), padding='SAME', name='init_conv')
    x, g_penalty = WideResnetGroup(
        x,
        blocks_per_group,
        16 * channel_multiplier,
        dropout_rate=dropout_rate,
        normalization=normalization,
        activation_f=activation_f,
        std_penalty_mult=std_penalty_mult,
        use_residual=use_residual,
        train=train,
        bias_scale=bias_scale,
        weight_norm=weight_norm)
    penalty += g_penalty
    x, g_penalty = WideResnetGroup(
        x,
        blocks_per_group,
        32 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        normalization=normalization,
        activation_f=activation_f,
        std_penalty_mult=std_penalty_mult,
        use_residual=use_residual,
        train=train,
        bias_scale=bias_scale,
        weight_norm=weight_norm)
    penalty += g_penalty
    x, g_penalty = WideResnetGroup(
        x,
        blocks_per_group,
        64 * channel_multiplier, (2, 2),
        dropout_rate=dropout_rate,
        normalization=normalization,
        activation_f=activation_f,
        std_penalty_mult=std_penalty_mult,
        use_residual=use_residual,
        train=train,
        bias_scale=bias_scale,
        weight_norm=weight_norm)
    penalty += g_penalty

    x = norm(x, name='final_norm')
    if std_penalty_mult > 0:
      penalty += std_penalty(x)
    if not no_head:
      x = activation_f(x, features=x.shape[-1])
      x = nn.avg_pool(x, (8, 8))
      x = x.reshape((x.shape[0], -1))
      x = nn.Dense(x, num_outputs)
    return x, penalty, {}


class ResNetImageNetBlock(nn.Module):
  """ResNet block without bottleneck used in ResNet-18 and ResNet-34."""

  def apply(
      self,
      x,
      filters,
      *,
      train,
      strides = (1, 1),
      conv,
      norm,
      activation_f,
      use_residual,
      zero_inits,
  ):
    residual = x
    needs_projection = x.shape[-1] != filters or strides != (1, 1)
    if needs_projection:
      residual = conv(
          residual,
          filters,
          kernel_size=(1, 1),
          strides=strides,
          name='proj_conv')
      residual = norm(residual, name='proj_bn')

    x = conv(x, filters, kernel_size=(3, 3), strides=strides, name='conv1')
    x = norm(x, name='bn1')
    x = activation_f(x, features=x.shape[-1])
    x = conv(x, filters, kernel_size=(3, 3), name='conv2')
    # Initializing the scale to 0 has been common practice since "Fixup
    # Initialization: Residual Learning Without Normalization" Tengyu et al,
    # 2019, [https://openreview.net/forum?id=H1gsz30cKX].
    x = norm(
        x,
        scale_init=nn.initializers.zeros
        if zero_inits else nn.initializers.ones,
        name='bn2')

    if use_residual == 1:
      # Apply an up projection in case of channel mismatch
      x = residual + x
    elif use_residual == 2:
      x = (residual + x) / jnp.sqrt(1**2 + 1**2)  # (sum of independent normals)
    else:
      pass

    x = activation_f(x, features=x.shape[-1])
    return x


class BottleneckResNetImageNetBlock(ResNetImageNetBlock):
  """Bottleneck ResNet block used in ResNet-50 and larger."""

  def apply(
      self,
      x,
      filters,
      *,
      train,
      strides = (1, 1),
      conv,
      norm,
      activation_f,
      use_residual,
      zero_inits,
  ):
    residual = x
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    if needs_projection:
      residual = conv(
          residual,
          filters * 4,
          kernel_size=(1, 1),
          strides=strides,
          name='proj_conv')
      residual = norm(residual, name='proj_bn')

    x = conv(x, filters, kernel_size=(1, 1), name='conv1')
    x = norm(x, name='bn1')
    x = activation_f(x, features=x.shape[-1])
    x = conv(x, filters, kernel_size=(3, 3), strides=strides, name='conv2')
    x = norm(x, name='bn2')
    x = activation_f(x, features=x.shape[-1])
    x = conv(x, filters * 4, kernel_size=(1, 1), name='conv3')

    x = norm(
        x,
        scale_init=nn.initializers.zeros
        if zero_inits else nn.initializers.ones,
        name='bn3')

    if use_residual == 1:
      # Apply an up projection in case of channel mismatch.
      x = residual + x
    elif use_residual == 2:
      x = (residual + x) / jnp.sqrt(1**2 + 1**2)  # Sum of independent normals.

    elif use_residual == 3:
      features = x.shape[-1]
      scale = self.param('rescale', (features,), nn.initializers.zeros)
      x = (residual + scale * x) / jnp.sqrt(1**2 + scale**2)
    else:
      pass

    x = activation_f(x, features=x.shape[-1])
    return x


class ResNetStage(nn.Module):
  """ResNet stage consistent of multiple ResNet blocks."""

  def apply(self, x, stage_size, filters, *,
            block_class,
            first_block_strides, train, conv, norm,
            activation_f, use_residual, zero_inits):
    for i in range(stage_size):
      x = block_class(
          x,
          filters,
          strides=first_block_strides if i == 0 else (1, 1),
          train=train,
          name=f'block{i + 1}',
          conv=conv,
          norm=norm,
          activation_f=activation_f,
          use_residual=use_residual,
          zero_inits=zero_inits,
      )
    return x


class ResNetImageNet(nn.Module):
  """ResNetV1 for ImageNet."""

  def apply(
      self,
      x,
      *,
      train,
      num_classes,
      block_class = BottleneckResNetImageNetBlock,
      stage_sizes,
      width_factor = 1,
      normalization='bn',
      activation_f=None,
      std_penalty_mult=0,
      use_residual=1,
      bias_scale=0.0,
      weight_norm='none',
      compensate_padding=True,
      softplus_scale=None,
      no_head=False,
      zero_inits=True):
    """Construct ResNet V1 with `num_classes` outputs."""
    self._stage_sizes = stage_sizes
    if std_penalty_mult > 0:
      raise NotImplementedError(
          'std_penalty_mult not supported for ResNetImageNet')

    width = 64 * width_factor

    # Root block.
    activation_f = get_activation_f(activation_f, train, softplus_scale,
                                    bias_scale)
    norm = get_norm(activation_f, normalization, train)
    conv = get_conv(activation_f, bias_scale, weight_norm, compensate_padding,
                    normalization)
    x = conv(x, width, kernel_size=(7, 7), strides=(2, 2), name='init_conv')
    x = norm(x, name='init_bn')

    if compensate_padding:
      # NOTE: this leads to lower performance.
      x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding='SAME')
    else:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    # Stages.
    for i, stage_size in enumerate(stage_sizes):
      x = ResNetStage(
          x,
          stage_size,
          filters=width * 2**i,
          block_class=block_class,
          first_block_strides=(1, 1) if i == 0 else (2, 2),
          train=train,
          name=f'stage{i + 1}',
          conv=conv,
          norm=norm,
          activation_f=activation_f,
          use_residual=use_residual,
          zero_inits=zero_inits,
      )

    if not no_head:
      # Head.
      x = jnp.mean(x, axis=(1, 2))
      x = nn.Dense(
          x,
          num_classes,
          kernel_init=nn.initializers.zeros
          if zero_inits else nn.initializers.lecun_normal(),
          name='head')
    return x, 0, {}


ResNetImageNet18 = ResNetImageNet.partial(
    stage_sizes=[2, 2, 2, 2], block_class=ResNetImageNetBlock)
ResNetImageNet34 = ResNetImageNet.partial(
    stage_sizes=[3, 4, 6, 3], block_class=ResNetImageNetBlock)
ResNetImageNet50 = ResNetImageNet.partial(stage_sizes=[3, 4, 6, 3])
ResNetImageNet101 = ResNetImageNet.partial(stage_sizes=[3, 4, 23, 3])
ResNetImageNet152 = ResNetImageNet.partial(stage_sizes=[3, 8, 36, 3])
ResNetImageNet200 = ResNetImageNet.partial(stage_sizes=[3, 24, 36, 3])
