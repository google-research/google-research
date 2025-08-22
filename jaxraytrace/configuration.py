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

"""Configuration file for rendering."""

import ml_collections


def get_config():
  """Returns a rendering configuration."""

  config = ml_collections.ConfigDict()
  config.height = 400
  config.aspect_ratio = 16 / 9
  config.origin = [0, 0, 0.5]
  config.view_direction = [0, 0, -1]
  config.view_up = [0, 1, 0]
  config.vertical_field_of_view = 90
  config.output_file = "outputs/rendered.ppm"
  config.num_antialiasing_samples = 100
  config.max_recursion_depth = 10
  config.rng_seed = 0
  config.gamma_correction = 2
  config.first_background_color = "skyblue"
  config.second_background_color = "white"
  config.objects = {
      "left_sphere": {
          "shape": "sphere",
          "center": (-1, 0, -1),
          "radius": 0.5,
          "material_type": "reflective",
          "material_color": "red",
      },
      "center_sphere": {
          "shape": "sphere",
          "center": (0, 0, -1),
          "radius": 0.5,
          "material_type": "diffuse",
          "material_color": "brown",
      },
      "right_sphere": {
          "shape": "sphere",
          "center": (1, 0, -1),
          "radius": 0.5,
          "material_type": "fuzzy",
          "material_color": "gray",
          "material_fuzz": 1.,
      },
      "earth": {
          "shape": "sphere",
          "center": (0, -100.5, -1),
          "radius": 100,
          "material_type": "diffuse",
          "material_color": "lightgreen",
      }
  }
  return config
