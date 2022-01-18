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

"""Performs ray batch-level operations on NeRFs."""
import tensorflow as tf
from osf import light_utils
from osf import run_nerf_helpers
from osf import shadow_utils


def prepare_retvals(outputs_0, outputs, **kwargs):
  """Prepares values to return from the NeRF outputs."""
  ret = {}
  for k, v in outputs_0.items():
    if k.endswith('map'):
      k_name = k.split('_')[0]
      ret[f'{k_name}0'] = v
    else:
      ret[f'{k}0'] = v
  if kwargs['N_importance'] > 0:
    for k, v in outputs.items():
      # if k.endswith('map'):
      #   ret[k] = v
      # else:
      ret[k] = v
  for k in ret:
    tf.debugging.check_numerics(ret[k], 'output {}'.format(k))
  grid = None
  return ret, grid


def evaluate_secondary_rays(name, secondary_ray_batch, combine_keys, **kwargs):
  """Evaluate secondary rays."""
  n_secondary_rays = tf.shape(secondary_ray_batch)[0]  # [RS]

  name2secondary_results = {}
  for secondary_object_name in kwargs['object_names']:
    # Exclude the current object.
    if secondary_object_name == name:
      continue

    # Evaluate secondary rays for the current secondary object.
    _, secondary_intersect = run_nerf_helpers.run_single_object(
        name=secondary_object_name,
        ray_batch=secondary_ray_batch,
        use_random_lightdirs=False,
        **kwargs)  # [RS, M]

    # Scatter the results from intersecting secondary rays back to the full set
    # of secondary rays.
    secondary_results = run_nerf_helpers.scatter_results(
        intersect=secondary_intersect,
        N_rays=n_secondary_rays,
        keys=combine_keys)
    name2secondary_results[secondary_object_name] = secondary_results

  # Combine secondary ray results across objects.
  secondary_results = run_nerf_helpers.combine_results(
      name2results=name2secondary_results, keys=combine_keys)
  return secondary_results


def compute_shadows(name, intersect, **kwargs):
  """Renders shadows for a batch of rays.

  Args:
    name: str. The name of the object we are rendering shadows for.
    intersect: Dict. A set of results for primary interesecting rays for current
      object.
    **kwargs: Additional arguments.

  Returns:
    shadow_transmittance: [R, S, 1] tf.float32. Per-sample shadow
        transmittances.
  """
  # Create a batch of shadow rays. The number of shadow rays is the number of
  # primary rays multipled by the number of samples per primary ray.
  shadow_ray_batch = shadow_utils.create_shadow_ray_batch(
      ray_batch=intersect['ray_batch'],
      scene_info=kwargs['scene_info'],
      pts=intersect['pts'],
      light_pos=kwargs['light_pos'])

  # Evaluate shadow rays across all other objects.
  shadow_results = evaluate_secondary_rays(  # [RS, SO, K]
      name=name,
      secondary_ray_batch=shadow_ray_batch,
      combine_keys=['z_vals', 'normalized_alpha'],
      **kwargs)

  # Compute the shadow transmittance for each primary sample.
  shadow_transmittance = shadow_utils.compute_shadow_transmittance(
      intersect=intersect, shadow_results=shadow_results)
  return shadow_transmittance


def compute_indirect(name, intersect, **kwargs):
  """Renders indirect illumination for a batch of rays.

  Args:
    name: str. The name of the object we are rendering shadows for.
    intersect: Dict. A set of results for primary interesecting rays for current
      object.
    **kwargs: Additional arguments.

  Returns:
    indirect_radiance: [R, S, 3] tf.float32. Per-sample radiance from
      indirect illumination.
  """
  indirect_results = evaluate_secondary_rays(  # [RS, SO, K]
      name=name,
      secondary_ray_batch=intersect['light_ray_batch'],
      combine_keys=['z_vals', 'normalized_rgb', 'normalized_alpha'],
      **kwargs)

  # Compute the indirect radiance for each primary sample from intersecting
  # primary rays.
  indirect_radiance = light_utils.compute_indirect_radiance(
      intersect=intersect,
      results=indirect_results,
      light_rgb=kwargs['light_rgb'],
      white_bkgd=False)
  return indirect_radiance


def render_rays_sparse_object_loop(ray_batch, **kwargs):
  """Renders a sparse batch of rays.

  Args:
    ray_batch: [R, M] tf.float32.
    **kwargs: Additional arguments.

  Returns:
    ret: Dict.
    grid: Unused.
  """
  num_rays = ray_batch.shape[0]
  combine_keys = ['z_vals', 'normalized_rgb', 'normalized_alpha']

  name2results_0, name2results = {}, {}
  # Loop over objects for primary rays.
  for name in kwargs['object_names']:
    if kwargs['render_indirect']:  # Evaluate indirect illumination
      # Evaluate the current object, with random light ray directions for
      # indirect illumination computation.
      intersect_0, intersect = run_nerf_helpers.run_single_object(
          name=name, ray_batch=ray_batch, use_random_lightdirs=True, **kwargs)
      print(f'intersect light_ray_batch: {intersect["light_ray_batch"]}')
      # Compute radiance along each secondary/indirect ray.
      indirect_radiance = compute_indirect(
          name=name, intersect=intersect, **kwargs)  # [R, S, 3]
      # Multiply rho by the radiance along the secondary indirect rays.
      intersect['normalized_rgb'] *= indirect_radiance  # [R, S, 3]
    else:  # Evaluate direct illumination
      intersect_0, intersect = run_nerf_helpers.run_single_object(
          name=name, ray_batch=ray_batch, use_random_lightdirs=False, **kwargs)
      # Compute shadow rays if requested.
      if kwargs['render_shadows']:
        shadow_transmittance = compute_shadows(
            name=name, intersect=intersect, **kwargs)  # [R, S, 1]
        intersect['normalized_rgb'] *= shadow_transmittance  # [R, S, 3]

    # # Scatter intersect results into original set of rays.
    name2results_0[name] = run_nerf_helpers.scatter_results(
        intersect=intersect_0, N_rays=num_rays, keys=combine_keys)
    name2results[name] = run_nerf_helpers.scatter_results(
        intersect=intersect, N_rays=num_rays, keys=combine_keys)

  # Combine samples across objects by sorting.
  results_0 = run_nerf_helpers.combine_results(
      name2results=name2results_0, keys=combine_keys)
  results = run_nerf_helpers.combine_results(
      name2results=name2results, keys=combine_keys)

  # Compose the combined object outputs into the final rendered result.
  outputs_0 = run_nerf_helpers.compose_outputs(
      results=results_0,
      light_rgb=kwargs['light_rgb'],
      white_bkgd=kwargs['white_bkgd'])
  outputs = run_nerf_helpers.compose_outputs(
      results=results,
      light_rgb=kwargs['light_rgb'],
      white_bkgd=kwargs['white_bkgd'])

  # Prepare values to return.
  ret, grid = prepare_retvals(outputs_0, outputs, **kwargs)
  return ret, grid
