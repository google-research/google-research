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

"""Implementation of GradCAM for CLIPWrapper."""

from typing import Any, Callable, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from clip_as_rnn import clip_wrapper
from clip_as_rnn import utils


class _ActivationsAndGradients:
  """Class for extracting activations and gradients."""

  def __init__(
      self,
      model,
      target_layers,
      reshape_transform = None,
      stride = 16,
  ):
    """Initializes ActivationsAndGradients.

    Args:
      model: The model to save its gradients.
      target_layers: A list of nn.Module to compute the gradients.
      reshape_transform: A callable function that reshapes and normalizes the
        image.
      stride: An integer representing the stride of the model.
    """
    self.model = model
    # This is only compatible with CLIPWrapper.
    if not isinstance(self.model, clip_wrapper.CLIPWrapper):
      raise NotImplementedError
    self.gradients = []
    self.activations = []
    self.reshape_transform = reshape_transform
    self.handles = []
    self.stride = stride
    for target_layer in target_layers:
      # Store tensors of the forward loop of the target layer.
      self.handles.append(
          target_layer.register_forward_hook(self.save_activation)
      )
      # Store gradients of the target layer.
      self.handles.append(
          target_layer.register_forward_hook(self.save_gradient)
      )

  def __call__(
      self, image_input, text_input, h, w
  ):
    """Get all the activations and gradients.

    Args:
      image_input: A torch.Tensor representing the image input.
      text_input: A torch.Tensor representing the text input.
      h: An integer representing the height of the input.
      w: An integer representing the width of the input

    Returns:
      An nn.Tensor representing the CLIP logits.
      An nn.Tensor representing the attetion weights of CLIP visual encoder.

    """
    self.height = h // self.stride
    self.width = w // self.stride
    self.gradients = []
    self.activations = []
    return self.model.forward_last_layer(image_input, text_input)

  def save_activation(
      self, module, input_tensor, output
  ):
    """This function inherits the PyTorch's save_activation hook.

    Args:
      module: A nn.Module used in the forward pass.
      input_tensor: A torch.Tensor as the input of the forward pass.
      output: A torch.Tensor as the output of the forward pass.
    """
    activation = output
    if module is None:
      raise ValueError("Module should not be None.")
    if input_tensor is None or output is None:
      raise ValueError("Input and output should not be None.")

    if self.reshape_transform is not None:
      activation = self.reshape_transform(activation, self.height, self.width)
    self.activations.append(activation.cpu().detach())

  def save_gradient(
      self, module, input_tensor, output
  ):
    """This function inherits the PyTorch's save_gradient hook.

    Args:
      module: A nn.Module used in the backward pass.
      input_tensor: A torch.Tensor as the input of the backward pass.
      output: A torch.Tensor as the output of the backward pass.
    """
    if module is None:
      raise ValueError("Module should not be None.")
    if input_tensor is None or output is None:
      raise ValueError("Input and output should not be None.")
    if not hasattr(output, "requires_grad") or not output.requires_grad:
      # You can only register hooks on tensor requires grad.
      return

    # Gradients are computed in reverse order.
    def _store_grad(grad):
      if self.reshape_transform is not None:
        grad = self.reshape_transform(grad, self.height, self.width)
      self.gradients = [grad.cpu().detach()] + self.gradients

    output.register_hook(_store_grad)


class CLIPOutputTarget:
  """The output target for CLIP model to record gradients.

  Attributes:
    category: An integer for the class id of the category.
  """

  def __init__(self, category):
    """Initializes CLIPOutputTarget.

    Args:
      category: An integer for the class id of the category.
    """
    self.category = category

  def __call__(self, model_output):
    """The output target for CLIP model to record gradients.

    Args:
      model_output: A torch.Tensor representing the CLIP logits output.

    Returns:
      A torch.Tensor representing the selected category of logits.

    """
    if len(model_output.shape) == 1:
      return model_output[self.category]
    return model_output[:, self.category]


class CAM:
  """Implementation of GradCAM for CLIPWrapper."""

  def __init__(
      self,
      model,
      target_layers,
      use_cuda = False,
      reshape_transform = None,
      stride = 16,
  ):
    """Initializes CAM.

    Args:
      model: the model to compute gradient.
      target_layers: target layers for gradient accumulation.
      use_cuda: a bool to indicate whether to use GPU for inference.
      reshape_transform: A callable function to reshape the CAM output.
      stride: a integer value indicating the stride of the network.
    """
    self.model = model.eval()
    self.target_layers = target_layers
    self.cuda = use_cuda
    self.model = model.cuda() if self.cuda else self.model
    self.reshape_transform = reshape_transform
    self.activations_and_grads = _ActivationsAndGradients(
        self.model, target_layers, reshape_transform, stride=stride
    )

  def __enter__(self):
    return self

  def __call__(
      self,
      image_input,
      text_input,
      targets,
      input_height = 0,
      input_width = 0,
  ):
    """This function defines how to get Class Activation Map (CAM).

    Args:
      image_input: A nn.Tensor representing the image input for CLIP model.
      text_input: A nn.Tensor representing the text input for CLIP model.
      targets: The target layer to be back-propagated.
      input_height: A integer of for the height of the target.
      input_width: A integer of for the width of the target.

    Returns:
      A np.ndarray representing the CAM map,
      A np.ndarray representing the CLIP logits,
      A np.ndarray representing the attention weight of CLIP.

    """
    logits_per_image, attention_weight = self.activations_and_grads(
        image_input, text_input, input_height, input_width)

    self.model.zero_grad()
    loss = sum(
        [target(logit) for target, logit in zip(targets, logits_per_image)]
    )
    loss.backward(retain_graph=True)
    cam_per_layer = self._compute_cam_per_layer()
    return (
        self._aggregate_multi_layers(cam_per_layer),
        logits_per_image,
        attention_weight,
    )

  def _get_cam(
      self, activations, grads
  ):
    """Get Class Activation Map (CAM) with gradCAM.

    Args:
      activations: The features from the target layer.
      grads: The gradient calculated from the target layer.

    Returns:
      cam: The CAM map calculated with gradCAM.

    """
    weights = np.mean(grads, axis=(2, 3))
    weighted_activations = weights[:, :, None, None] * activations
    cam = weighted_activations.sum(axis=1)
    return cam

  def _compute_cam_per_layer(self):
    """Compute CAM for each target layer.

    Returns:
        cam_per_target_layer: A list of np.ndarray for CAM maps.
    """

    activations_list = [
        a.cpu().data.numpy() for a in self.activations_and_grads.activations
    ]
    grads_list = [
        g.cpu().data.numpy() for g in self.activations_and_grads.gradients
    ]

    cam_per_target_layer = []
    # Loop over the saliency image from every layer.
    for i in range(len(self.target_layers)):
      layer_activations = None
      layer_grads = None
      if i < len(activations_list):
        layer_activations = activations_list[i]
      if i < len(grads_list):
        layer_grads = grads_list[i]

      cam = self._get_cam(layer_activations, layer_grads)
      cam = np.maximum(cam, 0).astype(np.float32)
      scaled = utils.normalize_and_scale(cam)
      cam_per_target_layer.append(scaled[:, None, :])

    return cam_per_target_layer

  def _aggregate_multi_layers(
      self, cam_per_target_layer
  ):
    cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
    result = np.mean(cam_per_target_layer, axis=1)
    return utils.normalize_and_scale(result)
