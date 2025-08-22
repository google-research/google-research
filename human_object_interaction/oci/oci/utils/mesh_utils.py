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

"""Mesh utils."""

# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=unused-argument
# pylint: disable=missing-class-docstring

from matplotlib import cm
from psbody.mesh import Mesh
import trimesh

# colormap = cm.get_cmap("plasma")


def convert_points_to_mesh(points,
                           radius=0.02,
                           colormap=cm.get_cmap("jet"),
                           return_ps_mesh=False):
  sphere = trimesh.primitives.Sphere(radius=radius)
  new_mesh = trimesh.Trimesh()
  # new_mesh = []
  for _, point in enumerate(points):
    new_sphere = trimesh.Trimesh(sphere.vertices + point, sphere.faces * 1)
    new_mesh += new_sphere
    # new_mesh.append(new_sphere)
    # color = colormap(px*1.0/ 20)
    # vc = trimesh.visual.color.VertexColor(color)

  if return_ps_mesh:
    new_mesh = Mesh(new_mesh.vertices, new_mesh.faces)

  return new_mesh


def convert_verts_to_mesh(vertices, faces):
  new_mesh = Mesh(vertices, faces)
  return new_mesh
