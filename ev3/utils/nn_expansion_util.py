# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Expansion functions for MLP and ResNet."""

import functools as ft
from typing import Any, Optional, Union

import chex
import flax
import jax
import jax.numpy as jnp

from ev3 import base
from ev3.utils import nn_util


def mlp_get_weights(
    params,
):
  n_layers = len(params['params'])
  w = [None] * n_layers
  b = [None] * n_layers
  for i in range(n_layers - 1):
    w[i] = params['params'][f'layer_{i}']['kernel']
    b[i] = params['params'][f'layer_{i}']['bias']
  w[-1] = params['params']['output']['kernel']
  b[-1] = params['params']['output']['bias']
  return w, b


def mlp_set_weights(
    params, w, b
):
  unfrozen_params = flax.core.frozen_dict.unfreeze(params)
  n_layers = len(params['params'])
  for i in range(n_layers - 1):
    unfrozen_params['params'][f'layer_{i}']['kernel'] = w[i]
    unfrozen_params['params'][f'layer_{i}']['bias'] = b[i]
  unfrozen_params['params']['output']['kernel'] = w[-1]
  unfrozen_params['params']['output']['bias'] = b[-1]
  return flax.core.frozen_dict.freeze(unfrozen_params)


def widen_mlp(
    nn_model,
    params,
    init_key,
    batch,
    widen_factor = 2.0,
    **expand_kwargs,
):
  """Widen an MLP nn_model.

  Args:
    nn_model: An MLP model.
    params: The params of the nn_model.
    init_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    widen_factor: A factor to widen the mlp layers.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """

  del expand_kwargs
  layer_widths = [int(widen_factor * w) + 1 for w in nn_model.layer_widths]
  nn_model2 = nn_util.MLP(
      layer_widths=layer_widths, num_labels=nn_model.num_labels
  )
  params2 = nn_model2.init(init_key, batch['feature'])
  wt, bias = mlp_get_weights(params)
  wt2, bias2 = mlp_get_weights(params2)

  for i, (w, w2) in enumerate(zip(wt, wt2)):
    w2 = w2.at[: w.shape[0], : w.shape[1]].set(w)
    if 0 < i < len(wt) - 1:
      w2 = w2.at[w.shape[0] :, : w.shape[1]].set(0)
    elif i == len(wt) - 1:
      w2 = w2.at[w.shape[0] :, :].set(0)
    wt2[i] = w2

    b = bias[i]
    b2 = bias2[i]
    b2 = b2.at[: b.shape[0]].set(b)
    bias2[i] = b2

  params2 = mlp_set_weights(params2, wt2, bias2)
  return nn_model2, params2


def deepen_mlp(
    nn_model,
    params,
    rand_key,
    batch,
    deepened_layer_idx = None,
    **expand_kwargs,
):
  """Deepen an MLP nn_model.

  Args:
    nn_model: An MLP model.
    params: The params of the nn_model.
    rand_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    deepened_layer_idx: An index to specify the layer after which to deepen the
      model. Default is None, and a random layer will be selected.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """
  del expand_kwargs

  init_key, layer_key = jax.random.split(rand_key, 2)

  # Duplicate one random layer if not specified.
  if deepened_layer_idx is None:
    deepened_layer_idx = jax.random.randint(
        layer_key, [1], 0, len(nn_model.layer_widths)
    )[0]
  else:
    chex.assert_scalar_in(deepened_layer_idx, 0, len(nn_model.layer_widths) - 1)
  deepened_layer_widths = list(
      nn_model.layer_widths[: deepened_layer_idx + 1]
  ) + list(nn_model.layer_widths[deepened_layer_idx:])

  nn_model2 = nn_util.MLP(
      layer_widths=deepened_layer_widths, num_labels=nn_model.num_labels
  )
  params2 = nn_model2.init(init_key, batch['feature'])
  wt, bias = mlp_get_weights(params)
  wt2, bias2 = mlp_get_weights(params2)

  # The deepened layer has identity weights, others keep the original weights.
  for i in range(deepened_layer_idx + 1):
    wt2[i] = wt[i]
    bias2[i] = bias[i]

  wt2[deepened_layer_idx + 1] = jnp.identity(
      wt2[deepened_layer_idx + 1].shape[0]
  )
  bias2[deepened_layer_idx + 1] = jnp.zeros(
      bias2[deepened_layer_idx + 1].shape[0]
  )

  for i in range(deepened_layer_idx + 2, len(wt2)):
    wt2[i] = wt[i - 1]
    bias2[i] = bias[i - 1]

  params2 = mlp_set_weights(params2, wt2, bias2)
  # Note: The current morphism only works for idempotent activations like ReLU.
  return nn_model2, params2


def expand_mlp(
    nn_model,
    params,
    init_key,
    batch,
    widen_factor = 2.0,
    deepened_layer_idx = None,
    **expand_kwargs,
):
  """Expand an MLP nn_model by deepening and widening.

  The currently implementation randomly choose from the two opeartions. For
  future work, we could make decisions based on the history, e.g., alternating
  between the opeartions, using some simple strategies to change the
  probabilities of each opeartions, or using some learning-based methods such as
  RL.

  Args:
    nn_model: An MLP model.
    params: The params of the nn_model.
    init_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    widen_factor: A factor to widen the mlp layers.
    deepened_layer_idx: An index to specify the layer after which to deepen the
      model. Default is None, and a random layer will be selected.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """
  decide_key, expand_key = jax.random.split(init_key, 2)
  operations = [widen_mlp, deepen_mlp]
  operation_idx = jax.random.randint(decide_key, [1], 0, 2)[0]
  return operations[operation_idx](
      nn_model,
      params,
      expand_key,
      batch,
      widen_factor=widen_factor,
      deepened_layer_idx=deepened_layer_idx,
      **expand_kwargs,
  )


def widen_resnet(
    resnet,
    params,
    init_key,
    batch,
    widen_factor = 2.0,
    widen_init='zeros',
    **expand_kwargs,
):
  """Widen a ResNet nn_model.

  Args:
    resnet: A ResNet model.
    params: The params of the nn_model.
    init_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    widen_factor: A factor to widen the mlp layers.
    widen_init: Init function for the expanded parts.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """

  del expand_kwargs

  stage_sizes = resnet.stage_sizes
  num_filters = resnet.num_filters
  new_num_filters = int(jnp.ceil(num_filters * widen_factor))
  new_resnet = nn_util.ResNetCIFAR(
      stage_sizes=stage_sizes,
      num_filters=new_num_filters,
      num_classes=resnet.num_classes,
  )
  new_params = new_resnet.init(init_key, batch['feature'])

  new_params['params']['conv_init']['kernel'] = (
      new_params['params']['conv_init']['kernel']
      .at[Ellipsis, :num_filters]
      .set(params['params']['conv_init']['kernel'])
  )

  for wt_key in ['scale', 'bias']:
    new_params['params']['bn_init'][wt_key] = (
        new_params['params']['bn_init'][wt_key]
        .at[:num_filters]
        .set(params['params']['bn_init'][wt_key])
    )
  for wt_key in ['mean', 'var']:
    new_params['batch_stats']['bn_init'][wt_key] = (
        new_params['batch_stats']['bn_init'][wt_key]
        .at[:num_filters]
        .set(params['batch_stats']['bn_init'][wt_key])
    )
  if widen_init == 'zeros':
    new_params['params']['bn_init']['scale'] = (
        new_params['params']['bn_init']['scale'].at[num_filters:].set(0)
    )

  for i in range(sum(stage_sizes)):
    block_key = 'ResNetBlock_' + str(i)
    for layer_key in ['Conv_0', 'Conv_1', 'conv_proj']:
      if layer_key in new_params['params'][block_key]:
        n1 = jnp.shape(params['params'][block_key][layer_key]['kernel'])[-2]
        n2 = jnp.shape(params['params'][block_key][layer_key]['kernel'])[-1]
        new_params['params'][block_key][layer_key]['kernel'] = (
            new_params['params'][block_key][layer_key]['kernel']
            .at[:, :, :n1, :n2]
            .set(params['params'][block_key][layer_key]['kernel'])
        )
    for layer_key in ['BatchNorm_0', 'BatchNorm_1', 'norm_proj']:
      if layer_key in new_params['params'][block_key]:
        n1 = jnp.shape(params['params'][block_key][layer_key]['scale'])[0]
        for wt_key in ['scale', 'bias']:
          new_params['params'][block_key][layer_key][wt_key] = (
              new_params['params'][block_key][layer_key][wt_key]
              .at[:n1]
              .set(params['params'][block_key][layer_key][wt_key])
          )
        for wt_key in ['mean', 'var']:
          new_params['batch_stats'][block_key][layer_key][wt_key] = (
              new_params['batch_stats'][block_key][layer_key][wt_key]
              .at[:n1]
              .set(params['batch_stats'][block_key][layer_key][wt_key])
          )
        if widen_init == 'zeros':
          new_params['params'][block_key][layer_key]['scale'] = (
              new_params['params'][block_key][layer_key]['scale'].at[n1:].set(0)
          )

  num_filters = jnp.shape(params['params']['Dense_0']['kernel'])[0]
  new_params['params']['Dense_0']['kernel'] = (
      new_params['params']['Dense_0']['kernel']
      .at[:num_filters]
      .set(params['params']['Dense_0']['kernel'])
  )
  new_params['params']['Dense_0']['bias'] = params['params']['Dense_0']['bias']

  return new_resnet, new_params


def deepen_resnet(
    resnet,
    params,
    rand_key,
    batch,
    deepened_layer_idx = None,
    **expand_kwargs,
):
  """Deepen a ResNet nn_model.

  Args:
    resnet: A ResNet model.
    params: The params of the nn_model.
    rand_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    deepened_layer_idx: An index to specify the layer after which to deepen the
      model. Default is None, and a random layer will be selected.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """
  del expand_kwargs

  init_key, layer_key = jax.random.split(rand_key, 2)

  if deepened_layer_idx is None:
    deepened_layer_idx = jax.random.randint(
        layer_key, [1], 0, len(resnet.stage_sizes)
    )[0]
  else:
    chex.assert_scalar_in(deepened_layer_idx, 0, len(resnet.stage_sizes) - 1)

  new_stage_sizes = resnet.stage_sizes[:]
  new_stage_sizes[deepened_layer_idx] += 1
  deepened_block_idx = sum(resnet.stage_sizes[: deepened_layer_idx + 1])

  new_resnet = nn_util.ResNetCIFAR(
      stage_sizes=new_stage_sizes,
      num_filters=resnet.num_filters,
      num_classes=resnet.num_classes,
  )
  new_params = new_resnet.init(init_key, batch['feature'])

  new_params['params']['conv_init']['kernel'] = params['params']['conv_init'][
      'kernel'
  ]
  for wt_key in ['scale', 'bias']:
    new_params['params']['bn_init'][wt_key] = params['params']['bn_init'][
        wt_key
    ]
  for wt_key in ['mean', 'var']:
    new_params['batch_stats']['bn_init'][wt_key] = params['batch_stats'][
        'bn_init'
    ][wt_key]

  old_block = 0
  for i in range(sum(new_stage_sizes)):
    if i == deepened_block_idx:
      continue
    old_block_key = 'ResNetBlock_' + str(old_block)
    block_key = 'ResNetBlock_' + str(i)

    for layer_key in new_params['params'][block_key]:
      for wt_key in ['kernel', 'scale', 'bias']:
        if wt_key in new_params['params'][block_key][layer_key]:
          new_params['params'][block_key][layer_key][wt_key] = params['params'][
              old_block_key
          ][layer_key][wt_key]

    for layer_key in new_params['batch_stats'][block_key]:
      for wt_key in ['mean', 'var']:
        new_params['batch_stats'][block_key][layer_key][wt_key] = params[
            'batch_stats'
        ][old_block_key][layer_key][wt_key]

    old_block += 1

  for wt_key in ['kernel', 'bias']:
    new_params['params']['Dense_0'][wt_key] = params['params']['Dense_0'][
        wt_key
    ]

  return new_resnet, new_params


def deepen_resnet_all_blocks(
    resnet,
    params,
    rand_key,
    batch,
    morphism = True,
    **expand_kwargs,
):
  """Deepen a ResNet nn_model.

  Args:
    resnet: A ResNet model.
    params: The params of the nn_model.
    rand_key: Random key to initialize the new model.
    batch: A batch of data to initialize the new model.
    morphism: Whether to apply a morphism.
    **expand_kwargs: Additional keyword arguments.

  Returns:
    Tuple of (new nn_model, new params).
  """
  del expand_kwargs

  new_stage_sizes = [i * 2 for i in resnet.stage_sizes]
  deepened_blocks = []
  for i, s in enumerate(new_stage_sizes):
    deepened_blocks += list(
        range(
            sum(new_stage_sizes[: i + 1]) - int(s / 2),
            sum(new_stage_sizes[: i + 1]),
        )
    )

  new_resnet = nn_util.ResNetCIFAR(
      stage_sizes=new_stage_sizes,
      num_filters=resnet.num_filters,
      num_classes=resnet.num_classes,
  )
  new_params = new_resnet.init(rand_key, batch['feature'])

  if morphism:
    new_params['params']['conv_init']['kernel'] = params['params']['conv_init'][
        'kernel'
    ]
    for wt_key in ['scale', 'bias']:
      new_params['params']['bn_init'][wt_key] = params['params']['bn_init'][
          wt_key
      ]
    for wt_key in ['mean', 'var']:
      new_params['batch_stats']['bn_init'][wt_key] = params['batch_stats'][
          'bn_init'
      ][wt_key]

    old_block = 0
    for i in range(sum(new_stage_sizes)):
      if i not in deepened_blocks:
        # Copy weights.
        old_block_key = 'ResNetBlock_' + str(old_block)
        block_key = 'ResNetBlock_' + str(i)

        for layer_key in new_params['params'][block_key]:
          for wt_key in ['kernel', 'scale', 'bias']:
            if wt_key in new_params['params'][block_key][layer_key]:
              new_params['params'][block_key][layer_key][wt_key] = params[
                  'params'
              ][old_block_key][layer_key][wt_key]

        for layer_key in new_params['batch_stats'][block_key]:
          for wt_key in ['mean', 'var']:
            new_params['batch_stats'][block_key][layer_key][wt_key] = params[
                'batch_stats'
            ][old_block_key][layer_key][wt_key]

        old_block += 1

    for wt_key in ['kernel', 'bias']:
      new_params['params']['Dense_0'][wt_key] = params['params']['Dense_0'][
          wt_key
      ]

  return new_resnet, new_params


def get_expand_fn(name, **expand_kwargs):
  if name == 'widen_resnet':
    return ft.partial(widen_resnet, **expand_kwargs)
  elif name == 'deepen_resnet':
    return ft.partial(deepen_resnet, **expand_kwargs)
  elif name == 'deepen_resnet_all_blocks':
    return ft.partial(deepen_resnet_all_blocks, **expand_kwargs)
  else:
    raise NotImplementedError(name)
