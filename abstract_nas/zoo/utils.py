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

"""Utils for loading a model."""

from typing import Any, Callable, Dict, Sequence, Tuple

from abstract_nas.model.block import Block
from abstract_nas.model.concrete import Graph
from abstract_nas.zoo import cnn
from abstract_nas.zoo import efficientnet
from abstract_nas.zoo import resnetv1
from abstract_nas.zoo import vit

Tensor = Any
ModelFn = Callable[Ellipsis, Tuple[Graph, Dict[str, Tensor], Sequence[Block]]]

RESNET_FNS = {
    "resnet18": resnetv1.ResNet18,
    "resnet34": resnetv1.ResNet34,
    "resnet50": resnetv1.ResNet50,
    "resnet101": resnetv1.ResNet101,
    "resnet152": resnetv1.ResNet152,
    "resnet200": resnetv1.ResNet200,
}

RESNET_SPATIAL_RESOLUTIONS = {
    32: "small",
    224: "large",
}

VIT_FNS = {
    "ti/16": vit.ViT_Ti16,
    "s/16": vit.ViT_S16,
    "b/16": vit.ViT_B16,
    "l/16": vit.ViT_L16,
}

EFFICIENTNET_FNS = {
    "b0": efficientnet.EfficientNetB0,
    "b1": efficientnet.EfficientNetB1,
}


def get_cnn_fn(model, num_classes):
  """Returns a wrapped CNN model constructor."""
  model = model.lower()
  assert model == "cnn"

  if num_classes != 10:
    raise ValueError(
        f"Model {model} supports only 10 classes, {num_classes} given.")
  def cnn_model_fn(*args, **kwargs):
    return cnn.CifarNet(*args, **kwargs)
  return cnn_model_fn


def get_resnet_fn(model, num_classes, spatial_res):
  """Returns a wrapped ResNet model constructor."""
  model = model.lower()
  assert model in RESNET_FNS
  resnet_fn = RESNET_FNS[model]

  if spatial_res not in RESNET_SPATIAL_RESOLUTIONS:
    raise ValueError(f"Spatial resolution {spatial_res} not supported, select "
                     f"from {list(RESNET_SPATIAL_RESOLUTIONS.keys())}.")

  input_resolution = RESNET_SPATIAL_RESOLUTIONS[spatial_res]

  def wrapped_resnet_model_fn(*args, **kwargs):
    return resnet_fn(*args, num_classes=num_classes,
                     input_resolution=input_resolution, **kwargs)
  return wrapped_resnet_model_fn


def get_vit_fn(model, num_classes, spatial_res):
  """Returns a wrapped ViT model constructor."""
  model = model.lower()
  assert model in VIT_FNS
  vit_fn = VIT_FNS[model]

  def wrapped_vit_model_fn(*args, **kwargs):
    return vit_fn(*args, num_classes=num_classes,
                  image_size=spatial_res, **kwargs)
  return wrapped_vit_model_fn


def get_efficientnet_fn(model, num_classes,
                        spatial_res):
  """Returns a wrapped EfficientNet model constructor."""
  model = model.lower()
  assert model in EFFICIENTNET_FNS
  efficientnet_fn = EFFICIENTNET_FNS[model]

  model_config = efficientnet_fn.keywords["config"]
  expected_res = model_config.resolution
  assert spatial_res == expected_res

  def wrapped_efficientnet_model_fn(*args, **kwargs):
    return efficientnet_fn(*args, num_classes=num_classes, **kwargs)
  return wrapped_efficientnet_model_fn


def get_model_fn(model, num_classes, spatial_res):
  """Returns a model constructor for the given dataset."""

  model = model.lower()
  if model == "cnn": return get_cnn_fn(model, num_classes)
  if model in RESNET_FNS: return get_resnet_fn(model, num_classes, spatial_res)
  if model in VIT_FNS: return get_vit_fn(model, num_classes, spatial_res)
  if model in EFFICIENTNET_FNS: return get_efficientnet_fn(model, num_classes,
                                                           spatial_res)
  raise ValueError(f"Model {model} not recognized.")
