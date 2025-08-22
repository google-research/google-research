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

"""Utilities for outputting images to files."""

from typing import Union

import chex
import numpy as np
import tqdm


def export_as_ppm(image, filename):
  """Exports an RGB image of dimension H x W to a PPM file."""
  image = np.asarray(image)
  image = np.ceil(255 * image).astype(np.int32)

  image_height, image_width, *_ = image.shape

  with open(filename, "w") as f:
    f.write("P3\n")
    f.write(f"{image_width} {image_height}\n")
    f.write("255\n")

    for i in tqdm.tqdm(reversed(range(image_height))):
      for j in range(image_width):
        r, g, b = image[i][j]
        f.write(f"{r} {g} {b}\n")
