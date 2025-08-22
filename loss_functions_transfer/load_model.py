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

"""Entry point for building and loading models in graph mode."""
import collections
from typing import Tuple, Union

from . import losses
from . import resnet_model
import tensorflow.compat.v1 as tf

BASE_PATH = 'gs://gresearch/loss_functions_transfer'

Hyperparameters = collections.namedtuple(
    'Hyperparameters', ('model_kwargs', 'loss_kwargs'))
LOSS_HYPERPARAMETERS = {
    'softmax': Hyperparameters({}, {}),
    'label_smoothing': Hyperparameters({}, {'alpha': 0.1}),
    'dropout': Hyperparameters({'dropout_keep_prob': 0.7}, {}),
    'extra_final_layer_l2': Hyperparameters({}, {'lambda_': 8e-4}),
    'logit_penalty': Hyperparameters({}, {'beta': 6e-4}),
    'logit_normalization': Hyperparameters({}, {'tau': 0.04}),
    'cosine_softmax': Hyperparameters({'cosine_softmax': True}, {'tau': 0.05}),
    'sigmoid': Hyperparameters({}, {}),
    'squared_error': Hyperparameters({}, {'kappa': 9, 'm': 60,
                                          'loss_scale': 10}),
}


def build_model_and_compute_loss(
    loss_name, inputs, labels, is_training,
    num_classes = 1001, resnet_depth = 50, weight_decay = 8e-5,
    weights = 1
):
  """Constructs model graph and computes its loss.

  Args:
    loss_name: Name of the loss. See keys of LOSS_HYPERPARAMETERS for valid
      options.
    inputs: A batch of images in NHWC format (AKA channels_last).
    labels: Tensor of one-hot labels.
    is_training: Whether to construct the network in training mode.
    num_classes: Number of classes. Defaults to 1001, the appropriate choice for
      provided checkpoints. The first class is unused and should be dropped to
      get the standard 1000 ImageNet class outputs.
    resnet_depth: Depth of ResNet to construct. Defaults to 50, the appropriate
      choice for the provided checkpoints.
    weight_decay: Amount of weight decay to add to the loss. Added only if
      `is_training` is True. Defaults to 8e-5, the amount of weight decay used
      to train models in the paper.
    weights: Optional set of weights. Typically used to mask padding in a batch.

  Returns:
    loss: The value of the loss, including the regularizer.
    outputs: The network's output.
  """
  model_kwargs, loss_kwargs = LOSS_HYPERPARAMETERS[loss_name]
  model_fn = resnet_model.resnet_v1(
      resnet_depth=resnet_depth, num_classes=num_classes,
      data_format='channels_last', **model_kwargs)
  final_layer = model_fn(inputs, is_training)
  loss_fn = getattr(losses, loss_name)
  # `outputs` includes scaling/normalization performed before loss.
  loss, outputs = loss_fn(labels, final_layer, weights, **loss_kwargs)

  if is_training and weight_decay:
    l2_norm = tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if 'batch_normalization' not in v.name and 'bias' not in v.name])
    regularization_loss = tf.losses.add_loss(
        weight_decay * l2_norm, tf.GraphKeys.REGULARIZATION_LOSSES)
    loss += regularization_loss

  endpoints = {}
  default_graph = tf.get_default_graph()
  i = 1
  for op in default_graph.get_operations():
    if [True for y in op.inputs if y.name.startswith('add')]:
      endpoints[f'block{i}'] = op.outputs[0]
      i += 1
  endpoints['final_avg_pool'] = default_graph.get_tensor_by_name(
      'final_avg_pool:0')
  endpoints['final_layer'] = final_layer
  endpoints['outputs'] = outputs

  return loss, endpoints


def restore_checkpoint(loss_name, seed, sess,
                       base_path = BASE_PATH):
  """Restores model checkpoints from the ExponentialMovingAverage weights.

  Args:
    loss_name: Name of the loss. See keys of LOSS_HYPERPARAMETERS for valid
      options.
    seed: Model seed (between 0 and 7 inclusive).
    sess: TensorFlow session object.
    base_path: Base path to search for model in.
  """
  variables = tf.global_variables()
  saver = tf.train.Saver(
      {v.op.name + '/ExponentialMovingAverage': v for v in variables})
  saver.restore(
      sess, f'{base_path}/checkpoints/{loss_name}/seed{seed}/model.ckpt')
