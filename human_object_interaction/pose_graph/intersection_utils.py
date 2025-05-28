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

"""Object Mesh intersection."""
import trimesh


def compute_intersection_with_object(object_mesh, smpl_mesh):
  query_pts = object_mesh.vertices * 1
  proxmity_struct = trimesh.proximity.ProximityQuery(smpl_mesh)
  sdf_values = proxmity_struct.signed_distance(query_pts)
  return sdf_values
