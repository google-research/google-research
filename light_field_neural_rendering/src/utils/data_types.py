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

"""Ray utility functions."""

from typing import Tuple, Optional

import chex


@chex.dataclass
class Rays:
  """Rays through a scene."""
  origins: chex.ArrayDevice  # The origins of the rays
  directions: chex.ArrayDevice  # The normalized directions
  base_radius: Optional[chex.ArrayDevice] = None  # If we want to cast a cone

  @property
  def batch_shape(self):
    """Returns the leading dimensions of the rays."""
    return self.origins.shape[:-1]


@chex.dataclass
class Views:
  """A view of a scene.

  This class will contain the rays and optionally the rgb values of the rays.
  The colors are store as uint8 and converted to float32 when required.
  """
  rays: Rays
  rgb: Optional[chex.ArrayDevice] = None

  @property
  def batch_shape(self):
    """Return the leading dimension of rays and rgb."""
    return self.rays.batch_shape


@chex.dataclass
class ReferenceViews:
  """Refernce views of a scene.

  This class will contain the reference view images along with camera
  information.
  """
  rgb: chex.ArrayDevice  # RGB
  ref_cameratoworld: chex.ArrayDevice  # shape [... 4 4]
  ref_worldtocamera: chex.ArrayDevice  # shape [... 4 4]
  intrinsic_matrix: chex.ArrayDevice  # shape [... 3 4]
  idx: chex.ArrayDevice  # shape [... n]


@chex.dataclass
class Batch:
  """Single Batch.

  A batch consists of information about a target view and optionally
  information about reference views.
  """
  target_view: Views
  reference_views: Optional[ReferenceViews] = None

  @property
  def batch_shape(self):
    """Return the leading dimension of target_view."""
    return self.target_view.batch_shape

  @property
  def num_reference_views(self):
    if self.reference_views is not None:
      return self.reference_views.rgb.shape[0]
    else:
      return 0
