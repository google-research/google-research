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

"""Models for flare removal."""

from flare_removal.python import u_net
from flare_removal.python import vgg


def build_model(model_type, batch_size):
  """Returns a Keras model specified by name."""
  if model_type == 'unet':
    return u_net.get_model(
        input_shape=(512, 512, 3),
        scales=4,
        bottleneck_depth=1024,
        bottleneck_layers=2)
  elif model_type == 'can':
    return vgg.build_can(
        input_shape=(512, 512, 3), conv_channels=64, out_channels=3)
  else:
    raise ValueError(model_type)
