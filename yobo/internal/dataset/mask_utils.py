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

# pylint: skip-file
"""Mask utils."""

from typing import Optional, Union

from etils import epath
import jax.numpy as jnp
import numpy as np

_Array = Union[np.ndarray, jnp.ndarray]


def load_mask(
    proto_path,
    *,
    scale_factor = 1,
    xnp=np,
):
  """Load the mask from the given proto path."""
  if not proto_path.exists():
    return None
  bytes_ = proto_path.read_bytes()
  proto = detections_pb2.SemanticIndexDataEntry.FromString(bytes_)

  h, w = proto.image_height, proto.image_width
  h //= scale_factor
  w //= scale_factor

  mask = xnp.ones((h, w, 3))

  for detection in proto.detections:
    bbox = detection.bbox
    lo_x = round(bbox.lo.x / scale_factor)
    lo_y = round(bbox.lo.y / scale_factor)
    hi_x = round(bbox.hi.x / scale_factor)
    hi_y = round(bbox.hi.y / scale_factor)
    if xnp is jnp:
      mask = mask.at[lo_y:hi_y, lo_x:hi_x].set(0.0)
    else:
      mask[lo_y:hi_y, lo_x:hi_x] = 0.0

  return mask
