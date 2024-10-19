# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Defines model architectures."""
import functools

import numpy as np
import tensorflow.compat.v1 as tf

from poem.core import common
from poem.core import data_utils


def linear(input_features, output_size, weight_max_norm, weight_initializer,
           bias_initializer, name):
  """Builds a linear layer.

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    output_size: An integer for the number of output nodes.
    weight_max_norm: A float for the maximum weight norm to clip at. Use
      non-positive to ignore.
    weight_initializer: A function handle for kernel weight initializer.
    bias_initializer: A function handle for bias initializer.
    name: A string for the name scope.

  Returns:
    A tensor for the output logits. Shape = [..., output_size].
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    weights = tf.get_variable(
        name='weight',
        shape=[input_features.shape.as_list()[-1], output_size],
        initializer=weight_initializer)
    if weight_max_norm > 0.0:
      weights = tf.clip_by_norm(weights, clip_norm=weight_max_norm)

    bias = tf.get_variable(
        name='bias', shape=[output_size], initializer=bias_initializer)

  return tf.linalg.matmul(input_features, weights) + bias


def fully_connected(input_features, is_training, name, **kwargs):
  """Builds a fully connected layer.

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    is_training: A boolean indicator for whether in training mode.
    name: A string for the name scope.
    **kwargs: A dictionary for additional arguments. Supported arguments include
      `num_hidden_nodes`, `weight_initializer`, `bias_initializer`,
      `weight_max_norm`, `use_batch_norm`, `dropout_rate`, `num_fcs_per_block`,
      and `num_fc_blocks`.

  Returns:
    net: A tensor for output features. Shape = [..., output_dims].

  Raises:
    ValueError: If activation function is not supported.
  """
  net = linear(
      input_features,
      output_size=kwargs.get('num_hidden_nodes', 1024),
      weight_max_norm=kwargs.get('weight_max_norm', 0.0),
      weight_initializer=kwargs.get('weight_initializer',
                                    tf.initializers.he_normal()),
      bias_initializer=kwargs.get('bias_initializer',
                                  tf.initializers.he_normal()),
      name=name + '/Linear')

  if kwargs.get('use_batch_norm', True):
    net = tf.layers.batch_normalization(
        net,
        training=is_training,
        name=name + '/BatchNorm',
        reuse=tf.AUTO_REUSE)

  activation_fn_name = kwargs.get('activation_fn', common.ACTIVATION_FN_RELU)
  if activation_fn_name == common.ACTIVATION_FN_RELU:
    net = tf.nn.relu(net, name=name + '/Relu')
  elif (activation_fn_name is not None and
        activation_fn_name != common.ACTIVATION_FN_NONE):
    raise ValueError('Unsupported activation function: `%s`.' %
                     str(activation_fn_name))

  dropout_rate = kwargs.get('dropout_rate', 0.0)
  if is_training and dropout_rate > 0.0:
    net = tf.nn.dropout(net, rate=dropout_rate, name=name + '/Dropout')
  return net


def fully_connected_block(input_features, is_training, name, **kwargs):
  """Builds a fully connected layer block.

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    is_training: A boolean indicator for whether in training mode.
    name: A string for the name scope.
    **kwargs: A dictionary for additional arguments. Supported arguments include
      `num_hidden_nodes`, `weight_initializer`, `bias_initializer`,
      `weight_max_norm`, `use_batch_norm`, `dropout_rate`, `num_fcs_per_block`,
      and `num_fc_blocks`.

  Returns:
    net: A tensor for output features. Shape = [..., output_dims].
  """
  net = input_features
  for i in range(kwargs.get('num_fcs_per_block', 2)):
    net = fully_connected(
        net, is_training=is_training, name=name + '/FC_%d' % i, **kwargs)
  net += input_features
  return net


def multi_head_logits(input_features, output_sizes, name, **kwargs):
  """Builds a multi-head logit layer with potential bottleneck layer.

  Args:
    input_features: A tensor for input features. Shape =
      [..., sequence_length, feature_dim].
    output_sizes: A dictionary for output sizes in the format {output_name:
      output_size}, where `output_size` can be an integer or a list.
    name: A string for the name scope.
    **kwargs: A dictionary for additional arguments. Supported arguments include
      `num_hidden_nodes`, `weight_initializer`, `bias_initializer`,
      `weight_max_norm`, `use_batch_norm`, `dropout_rate`, `num_fcs_per_block`,
      and `num_fc_blocks`.

  Returns:
    outputs: A dictionary for the output logits.
  """
  outputs = {}
  for output_name, output_size in output_sizes.items():
    if isinstance(output_size, int):
      output_size = [output_size]
    outputs[output_name] = linear(
        input_features,
        output_size=np.prod(output_size),
        weight_max_norm=kwargs.get('weight_max_norm', 0.0),
        weight_initializer=kwargs.get('weight_initializer',
                                      tf.initializers.he_normal()),
        bias_initializer=kwargs.get('bias_initializer',
                                    tf.initializers.he_normal()),
        name=name + '/OutputLogits/' + output_name)
    if len(output_size) > 1:
      outputs[output_name] = data_utils.recursively_expand_dims(
          outputs[output_name], axes=[-1] * (len(output_size) - 1))
      outputs[output_name] = data_utils.reshape_by_last_dims(
          outputs[output_name], last_dim_shape=output_size)
  return outputs


def simple_base(input_features,
                sequential_inputs,
                is_training,
                name='SimpleModel',
                **kwargs):
  """Implements `simple baseline` model base architecture.

  Note that the code differs from the original architecture by disabling dropout
  and maximum weight norms by default.

  Reference:
    Martinez et al. A Simple Yet Effective Baseline for 3D Human Pose
    Estimation. https://arxiv.org/pdf/1705.03098.pdf.

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    sequential_inputs: A boolean flag indicating whether the input_features are
      sequential. If True, the input are supposed to be in shape
      [...,  sequence_length, feature_dim].
    is_training: A boolean for whether it is in training mode.
    name: A string for the name scope.
    **kwargs: A dictionary for additional arguments. Supported arguments include
      `num_hidden_nodes`, `weight_initializer`, `bias_initializer`,
      `weight_max_norm`, `use_batch_norm`, `dropout_rate`, `num_fcs_per_block`,
      and `num_fc_blocks`.

  Returns:
    A tensor for output activations. Shape = [..., output_dim].
  """
  if sequential_inputs:
    input_features = data_utils.flatten_last_dims(
        input_features, num_last_dims=2)

  net = fully_connected(
      input_features, is_training=is_training, name=name + '/InputFC', **kwargs)
  for i in range(kwargs.get('num_fc_blocks', 2)):
    net = fully_connected_block(
        net, is_training, name=name + '/FullyConnectedBlock_%d' % i,
        **kwargs)
  return net


def simple_model(input_features,
                 output_sizes,
                 sequential_inputs,
                 is_training,
                 name='SimpleModel',
                 num_bottleneck_nodes=0,
                 **kwargs):
  """Implements `simple base` model with outputs.

  Note that the code differs from the original architecture by disabling dropout
  and maximum weight norms by default.

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    output_sizes: A dictionary for output sizes in the format {output_name:
      output_size}, where `output_size` can be an integer or a list.
    sequential_inputs: A boolean flag indicating whether the input_features are
      sequential. If True, the input are supposed to be in shape
      [...,  sequence_length, feature_dim].
    is_training: A boolean for whether it is in training mode.
    name: A string for the name scope.
    num_bottleneck_nodes: An integer for size of the bottleneck layer to be
      added before the output layer(s). No bottleneck layer will be added if
      non-positive.
    **kwargs: A dictionary of additional arguments passed to `simple_base`.

  Returns:
    outputs: A dictionary for output tensors in the format {output_name:
      output_tensors}. Output tensor shape = [..., output_size].
    activations: A dictionary of addition activation tensors for pre-output
      model activations. Keys include 'base_activations' and optionally
      'bottleneck_activations'.
  """
  net = simple_base(
      input_features,
      sequential_inputs=sequential_inputs,
      is_training=is_training,
      name=name,
      **kwargs)
  activations = {'base_activations': net}

  if num_bottleneck_nodes > 0:
    net = linear(
        net,
        output_size=num_bottleneck_nodes,
        weight_max_norm=kwargs.get('weight_max_norm', 0.0),
        weight_initializer=kwargs.get('weight_initializer',
                                      tf.initializers.he_normal()),
        bias_initializer=kwargs.get('bias_initializer',
                                    tf.initializers.he_normal()),
        name=name + '/BottleneckLogits')
    activations['bottleneck_activations'] = net

  outputs = multi_head_logits(
      net,
      output_sizes=output_sizes,
      name=name,
      **kwargs)
  return outputs, activations


def simple_model_late_fuse(input_features,
                           output_sizes,
                           is_training,
                           name='SimpleModelLateFuse',
                           num_late_fusion_preprojection_nodes=0,
                           late_fusion_preprojection_activation_fn=None,
                           num_bottleneck_nodes=0,
                           **kwargs):
  """Implements `simple baseline` model base architecture on sequential inputs.

  The model first runs simpel_model on each individual set of keypoints and then
  performs late fusions.

  Args:
    input_features: A tensor for input features. Shape = [..., sequence_length,
      feature_dim].
    output_sizes: A dictionary for output sizes in the format {output_name:
      output_size}, where `output_size` can be an integer or a list.
    is_training: A boolean for whether it is in training mode.
    name: A string for the name scope.
    num_late_fusion_preprojection_nodes: An integer for the dimension to project
      each frame features to before late fusion. No preprojection will be added
      if non-positive.
    late_fusion_preprojection_activation_fn: A string for the activation
      function of the preprojection layer. If None or 'NONE', no activation
      function is used.
    num_bottleneck_nodes: An integer for size of the bottleneck layer to be
      added before the output layer(s). No bottleneck layer will be added if
      non-positive.
    **kwargs: A dictionary of additional arguments passed to
      `simple_base_late_fuse`.

  Returns:
    A tensor for output activations. Shape = [..., output_dim].
  """
  # First flatten temporal axis into batch.
  flatten_input_features = data_utils.flatten_first_dims(
      input_features, num_last_dims_to_keep=1)
  # Batch process each pose.
  net = simple_base(
      flatten_input_features,
      sequential_inputs=False,
      is_training=is_training,
      name=name,
      **kwargs)

  if num_late_fusion_preprojection_nodes > 0:
    params = dict(kwargs)
    params.update({
        'num_hidden_nodes': num_late_fusion_preprojection_nodes,
        'activation_fn': late_fusion_preprojection_activation_fn,
    })
    net = fully_connected(
        net,
        is_training=is_training,
        name=name + '/LateFusePreProject',
        **params)

  # Recover shape and concatenate temporal axis along feature dims.
  sequence_length = input_features.shape.as_list()[-2]
  feature_length = net.shape.as_list()[-1]
  net = tf.reshape(net, [-1, sequence_length * feature_length])
  # Late fusion.
  net = fully_connected(
      net, is_training=is_training, name=name + '/LateFuseProject', **kwargs)
  net = fully_connected_block(
      net, is_training=is_training, name=name + '/LateFuseBlock', **kwargs)
  activations = {'base_activations': net}

  if num_bottleneck_nodes > 0:
    net = linear(
        net,
        output_size=num_bottleneck_nodes,
        weight_max_norm=kwargs.get('weight_max_norm', 0.0),
        weight_initializer=kwargs.get('weight_initializer',
                                      tf.initializers.he_normal()),
        bias_initializer=kwargs.get('bias_initializer',
                                    tf.initializers.he_normal()),
        name=name + '/BottleneckLogits')
    activations['bottleneck_activations'] = net

  outputs = multi_head_logits(
      net, output_sizes=output_sizes, name=name, **kwargs)
  return outputs, activations


def create_model_helper(base_model_fn, sequential_inputs, is_training):
  """Helper function for creating model function given base model function.

  This function creates a model function that adaptively slices the input
  features for improved running speed.

  Note that the base model function is required to have interface:

    outputs, activations = model_fn(input_features, output_sizes),

  where `input_features` has shape [..., feature_dim] if `sequential_inputs` is
  False, or [..., sequence_length, feature_dim] otherwise.

  Args:
    base_model_fn: A function handle for base model.
    sequential_inputs: A boolean for whether the model input features are
      sequential.
    is_training: A boolean for whether it is in training mode.

  Returns:
    model_fn: A function handle for model.
  """

  def model_fn(input_features, output_sizes):
    """Applies model to input features and produces output of given sizes."""
    if is_training:
      # Flatten all the model-irrelevant dimensions, i.e., dimensions that
      # precede the sequence / feature channel dimensions). Note that we only do
      # this for training, for which the batch size is known.
      num_last_dims_to_keep = 2 if sequential_inputs else 1
      flattened_input_features = data_utils.flatten_first_dims(
          input_features, num_last_dims_to_keep=num_last_dims_to_keep)
      flattened_shape = data_utils.get_shape_by_first_dims(
          input_features, num_last_dims=num_last_dims_to_keep)

      outputs, activations = base_model_fn(flattened_input_features,
                                           output_sizes)

      # Unflatten back all the model-irrelevant dimensions.
      for key, output in outputs.items():
        outputs[key] = data_utils.unflatten_first_dim(
            output, shape_to_unflatten=flattened_shape)
      for key, activation in activations.items():
        activations[key] = data_utils.unflatten_first_dim(
            activation, shape_to_unflatten=flattened_shape)

    else:
      outputs, activations = base_model_fn(input_features, output_sizes)

    return outputs, activations

  return model_fn


def get_model(base_model_type, is_training, **kwargs):
  """Gets a base model builder function handle.

  Note that the returned model function has interface:

    outputs, activations = model_fn(input_features, output_sizes),

  where `input_features` has shape [batch_size, feature_dim] or [batch_size,
  num_instances, ..., feature_dim].

  Args:
    base_model_type: An enum string for base model type. See supported base
      model types in the `common` module.
    is_training: A boolean for whether it is in training mode.
    **kwargs: A dictionary of additional arguments.

  Returns:
    A function handle for base model.

  Raises:
    ValueError: If base model type is not supported.
  """
  # Single-input model(s).
  if base_model_type == common.BASE_MODEL_TYPE_SIMPLE:
    base_model_fn = functools.partial(
        simple_model,
        sequential_inputs=False,
        is_training=is_training,
        **kwargs)
    return create_model_helper(
        base_model_fn, sequential_inputs=False, is_training=is_training)

  # Sequence-input model(s).
  if base_model_type == common.BASE_MODEL_TYPE_TEMPORAL_SIMPLE:
    base_model_fn = functools.partial(
        simple_model, sequential_inputs=True, is_training=is_training, **kwargs)
  elif base_model_type == common.BASE_MODEL_TYPE_TEMPORAL_SIMPLE_LATE_FUSE:
    base_model_fn = functools.partial(
        simple_model_late_fuse, is_training=is_training, **kwargs)
  else:
    raise ValueError('Unsupported base model type: `%s`.' %
                     str(base_model_type))
  return create_model_helper(
      base_model_fn, sequential_inputs=True, is_training=is_training)


_add_prefix = lambda key, c: 'C%d/' % c + key


def _point_embedder(input_features, base_model_fn, num_embedding_components,
                    embedding_size):
  """Implements a point embedder.

  Output tensor shapes:
    KEY_EMBEDDING_MEANS: Shape = [..., num_embedding_components, embedding_dim].

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    base_model_fn: A function handle for base model.
    num_embedding_components: An integer for the number of embedding components.
    embedding_size: An integer for embedding dimensionality.

  Returns:
    outputs: A dictionary for output tensors See comment above for details.
    activations: A dictionary of addition activation tensors for pre-output
      model activations. Keys include 'base_activations' and optionally
      'bottleneck_activations'.
  """
  output_sizes = {
      _add_prefix(common.KEY_EMBEDDING_MEANS, c): embedding_size
      for c in range(num_embedding_components)
  }

  component_outputs, activations = base_model_fn(input_features, output_sizes)

  outputs = {
      common.KEY_EMBEDDING_MEANS:
          tf.stack([
              component_outputs[_add_prefix(common.KEY_EMBEDDING_MEANS, c)]
              for c in range(num_embedding_components)
          ],
                   axis=-2)
  }
  return outputs, activations


def _stddev_activation(x):
  """Activation function for standard deviation logits.

  Args:
    x: A tensor for standard deviation logits.

  Returns:
    A tensor for non-negative standard deviations.
  """
  return tf.nn.elu(x) + 1.0


def _gaussian_embedder(input_features,
                       base_model_fn,
                       num_embedding_components,
                       embedding_size,
                       scalar_stddev,
                       num_embedding_samples,
                       seed=None):
  """Implements a Gaussian (mixture) embedder.

  Output tensor shapes:
    KEY_EMBEDDING_MEANS: Shape = [..., num_embedding_components,
      embedding_dim].
    KEY_EMBEDDING_STDDEVS: Shape = [..., num_embedding_components,
      embedding_dim].
    KEY_EMBEDDING_SAMPLES: Shape = [..., num_embedding_components, num_samples,
      embedding_dim].

  Args:
    input_features: A tensor for input features. Shape = [..., feature_dim].
    base_model_fn: A function handle for base model.
    num_embedding_components: An integer for the number of Gaussian mixture
      components.
    embedding_size: An integer for embedding dimensionality.
    scalar_stddev: A boolean for whether to predict scalar standard deviations.
    num_embedding_samples: An integer for number of samples drawn Gaussian
      distributions. If non-positive, skips the sampling step.
    seed: An integer for random seed.

  Returns:
    outputs: A dictionary for output tensors See comment above for details.
    activations: A dictionary of addition activation tensors for pre-output
      model activations. Keys include 'base_activations' and optionally
      'bottleneck_activations'.
  """
  output_sizes = {}
  for c in range(num_embedding_components):
    output_sizes.update({
        _add_prefix(common.KEY_EMBEDDING_MEANS, c):
            embedding_size,
        _add_prefix(common.KEY_EMBEDDING_STDDEVS, c):
            1 if scalar_stddev else embedding_size,
    })
  component_outputs, activations = base_model_fn(input_features, output_sizes)

  for c in range(num_embedding_components):
    embedding_stddev_key = _add_prefix(common.KEY_EMBEDDING_STDDEVS, c)
    component_outputs[embedding_stddev_key] = _stddev_activation(
        component_outputs[embedding_stddev_key])

    if num_embedding_samples > 0:
      component_outputs[_add_prefix(common.KEY_EMBEDDING_SAMPLES, c)] = (
          data_utils.sample_gaussians(
              means=component_outputs[_add_prefix(common.KEY_EMBEDDING_MEANS,
                                                  c)],
              stddevs=component_outputs[embedding_stddev_key],
              num_samples=num_embedding_samples,
              seed=seed))

  outputs = {
      common.KEY_EMBEDDING_MEANS:
          tf.stack([
              component_outputs[_add_prefix(common.KEY_EMBEDDING_MEANS, c)]
              for c in range(num_embedding_components)
          ],
                   axis=-2),
      common.KEY_EMBEDDING_STDDEVS:
          tf.stack([
              component_outputs[_add_prefix(common.KEY_EMBEDDING_STDDEVS, c)]
              for c in range(num_embedding_components)
          ],
                   axis=-2),
  }
  if num_embedding_samples > 0:
    outputs[common.KEY_EMBEDDING_SAMPLES] = tf.stack([
        component_outputs[_add_prefix(common.KEY_EMBEDDING_SAMPLES, c)]
        for c in range(num_embedding_components)
    ],
                                                     axis=-3)

  return outputs, activations


def create_embedder_helper(base_model_fn, embedding_type,
                           num_embedding_components, embedding_size, **kwargs):
  """Helper function for creating an embedding model builder function handle.

  Args:
    base_model_fn: An enum string for base model type. See supported base model
      types in the `common` module.
    embedding_type: An enum string for embedding type. See supported embedding
      types in the `common` module.
    num_embedding_components: An integer for the number of embedding components.
    embedding_size: An integer for embedding dimensionality.
    **kwargs: A dictionary of additional arguments to embedder.

  Returns:
    A function handle for embedding model builder.

  Raises:
    ValueError: If base model type or embedding type is not supported.
  """
  if embedding_type == common.EMBEDDING_TYPE_POINT:
    return functools.partial(
        _point_embedder,
        base_model_fn=base_model_fn,
        num_embedding_components=num_embedding_components,
        embedding_size=embedding_size)

  if embedding_type == common.EMBEDDING_TYPE_GAUSSIAN:
    return functools.partial(
        _gaussian_embedder,
        base_model_fn=base_model_fn,
        num_embedding_components=num_embedding_components,
        embedding_size=embedding_size,
        scalar_stddev=False,
        num_embedding_samples=kwargs.get('num_embedding_samples'),
        seed=kwargs.get('seed', None))

  if embedding_type == common.EMBEDDING_TYPE_GAUSSIAN_SCALAR_VAR:
    return functools.partial(
        _gaussian_embedder,
        base_model_fn=base_model_fn,
        num_embedding_components=num_embedding_components,
        embedding_size=embedding_size,
        scalar_stddev=True,
        num_embedding_samples=kwargs.get('num_embedding_samples'),
        seed=kwargs.get('seed', None))

  raise ValueError('Unsupported embedding type: `%s`.' % str(embedding_type))


def get_embedder(base_model_type, embedding_type, num_embedding_components,
                 embedding_size, is_training, **kwargs):
  """Gets an embedding model builder function handle.

  Args:
    base_model_type: An enum string for base model type. See supported base
      model types in the `common` module.
    embedding_type: An enum string for embedding type. See supported embedding
      types in the `common` module.
    num_embedding_components: An integer for the number of embedding components.
    embedding_size: An integer for embedding dimensionality.
    is_training: A boolean for whether it is in training mode.
    **kwargs: A dictionary of additional arguments to pass to base model and
      embedder.

  Returns:
    A function handle for embedding model builder.
  """
  base_model_fn = get_model(base_model_type, is_training=is_training, **kwargs)
  return create_embedder_helper(
      base_model_fn,
      embedding_type,
      num_embedding_components=num_embedding_components,
      embedding_size=embedding_size,
      **kwargs)
