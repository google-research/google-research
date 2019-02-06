# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Saliency helper library to compute and pre-process saliency heatmaps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import saliency


def get_saliency_image(graph, sess, y, image, saliency_method):
  """generates saliency image.

  Args:
    graph: tensor flow graph.
    sess: the current session.
    y: the pre-softmax activation we want to assess attribution with respect to.
    image: float32 image tensor with size [1, None, None].
    saliency_method: string indicating saliency map type to generate.

  Returns:
    a saliency map and a smoothed saliency map.

  Raises:
    ValueError: if the saliency_method string does not match any included method
  """
  if saliency_method == 'integrated_gradients':
    integrated_placeholder = saliency.IntegratedGradients(graph, sess, y, image)
    return integrated_placeholder
  elif saliency_method == 'gradient':
    gradient_placeholder = saliency.GradientSaliency(graph, sess, y, image)
    return gradient_placeholder
  elif saliency_method == 'guided_backprop':
    gb_placeholder = saliency.GuidedBackprop(graph, sess, y, image)
    return gb_placeholder
  else:
    raise ValueError('No saliency method method matched. Verification of'
                     'input needed')


def generate_saliency_image(method,
                            image,
                            prediction,
                            gradient_placeholder=None,
                            backprop_placeholder=None,
                            integrated_gradient_placeholder=None,
                            baseline=None):
  """Generates a saliency image based upon the list of input methods.

  Args:
    method: string specifying saliency method.
    image: raw image float32 tensor.
    prediction: int32 tensor, network classification prediction for the image.
    gradient_placeholder: init saliency mask generator class.
    backprop_placeholder: init GB mask generator class.
    integrated_gradient_placeholder: init IG mask generator class.
    baseline: the integrated gradients baseline

  Returns:
    maps: a dictionary of saliency heatmaps.

  Raises:
    ValueError: if the saliency_method string does not match any included method
  """
  if method == 'IG':
    mask = integrated_gradient_placeholder.GetMask(
        image, prediction, x_baseline=baseline)

  elif method == 'IG_SG':
    mask = integrated_gradient_placeholder.GetSmoothedMask(
        image, prediction, x_baseline=baseline, magnitude=False)

  elif method == 'IG_SG_2':
    mask = integrated_gradient_placeholder.GetSmoothedMask(
        image, prediction, magnitude=True)

  elif method == 'SH':
    mask = gradient_placeholder.GetMask(image, prediction)

  elif method == 'SH_SG':
    mask = gradient_placeholder.GetSmoothedMask(
        image, prediction, magnitude=False)

  elif method == 'SH_SG_2':

    mask = gradient_placeholder.GetSmoothedMask(
        image, prediction, magnitude=True)

  elif method == 'GB':
    mask = backprop_placeholder.GetMask(image, prediction)

  elif method == 'GB_SG':
    mask = backprop_placeholder.GetSmoothedMask(
        image, prediction, magnitude=False)

  elif method == 'GB_SG_2':
    mask = backprop_placeholder.GetSmoothedMask(
        image, prediction, magnitude=True)

  else:
    raise ValueError('No saliency method method matched. Verification of'
                     'input needed')
  return mask
