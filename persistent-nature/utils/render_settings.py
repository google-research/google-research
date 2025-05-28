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

"""some rendering settings for layout model."""
nerf_render_dense = dict(
    nerf_out_res=32, samples_per_ray=1024, nerf_far=32, patch_size=32
)
nerf_render_full = dict(
    nerf_out_res=256, samples_per_ray=1024, nerf_far=32, patch_size=32
)
nerf_render_interactive = dict(
    nerf_out_res=32, samples_per_ray=256, nerf_far=32, patch_size=32
)
nerf_render_supersample = dict(
    nerf_out_res=256, samples_per_ray=1024, nerf_far=32, patch_size=32
)
