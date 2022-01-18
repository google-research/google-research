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

"""NeRF helpers."""
import math
import tensorflow as tf
from osf import box_utils
from osf import ray_utils
from osf import scene_utils


def default_ray_sampling(ray_batch, n_samples, perturb, lindisp):
  """Default NeRF ray sampling.

  This function takes a batch of rays and returns points along each ray that
  should be evaluated by the coarse NeRF model.

  Args:
    ray_batch: Array of shape [batch_size, ...]. All information necessary
      for sampling along a ray, including: ray origin, ray direction, min dist,
        max dist, and unit-magnitude viewing direction.
    n_samples: Number of samples to take on each ray.
    perturb: Whether to perturb the points with white noise.
    lindisp: bool. If True, sample linearly in inverse depth rather than in
      depth.

  Returns:
    z_vals: Positions of the sampled points on each ray as a scalar:
      [n_rays, n_samples, 1].
    pts: Actual sampled points in 3D: [n_rays, n_samples, 3].
  """
  # batch size
  n_rays = tf.shape(ray_batch)[0]

  # Extract ray origin, direction.
  rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [n_rays, 3] each

  # Extract lower, upper bound for ray distance.
  bounds = tf.reshape(ray_batch[Ellipsis, 6:8], [-1, 1, 2])
  near, far = bounds[Ellipsis, 0], bounds[Ellipsis, 1]  # [R, 1]

  # Decide where to sample along each ray. Under the logic, all rays will be
  # sampled at the same times.
  t_vals = tf.linspace(0., 1., n_samples)
  if not lindisp:
    # Space integration times linearly between 'near' and 'far'. Same
    # integration points will be used for all rays.
    z_vals = near * (1. - t_vals) + far * (t_vals)
  else:
    tf.debugging.assert_greater(near, 0)
    tf.debugging.assert_greater(far, 0)
    # Sample linearly in inverse depth (disparity).
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
  z_vals = tf.broadcast_to(z_vals, [n_rays, n_samples])

  # Perturb sampling time along each ray.
  if perturb > 0.:
    # get intervals between samples
    mids = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
    upper = tf.concat([mids, z_vals[Ellipsis, -1:]], -1)
    lower = tf.concat([z_vals[Ellipsis, :1], mids], -1)
    # stratified samples in those intervals
    t_rand = tf.random.uniform(tf.shape(z_vals))
    z_vals = lower + (upper - lower) * t_rand
  # Points in space to evaluate model at.
  pts = rays_o[Ellipsis, None, :] + rays_d[Ellipsis, None, :] * z_vals[Ellipsis, :, None]
  tf.debugging.assert_equal(tf.shape(z_vals)[0], tf.shape(pts)[0])
  return z_vals, pts


def raw2alpha(raw, dists, act_fn=tf.nn.relu):
  return 1.0 - tf.exp(-act_fn(raw) * dists)


def compute_alpha(z_vals,
                  raw_alpha,
                  raw_noise_std,
                  last_dist_method='infinity'):
  """Normalizes raw sigma predictions from the network into normalized alpha."""
  # Compute 'distance' (in time) between each integration time along a ray.
  dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]

  # The 'distance' from the last integration time is infinity.
  if last_dist_method == 'infinity':
    dists = tf.concat(
        [dists, tf.broadcast_to([1e10], tf.shape(dists[Ellipsis, :1]))],
        axis=-1)  # [n_rays, n_samples]
  elif last_dist_method == 'last':
    dists = tf.concat([dists, dists[Ellipsis, -1:]], axis=-1)

  # Multiply each distance by the norm of its corresponding direction ray
  # to convert to real world distance (accounts for non-unit directions).
  # dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

  raw_alpha = tf.squeeze(raw_alpha, axis=-1)  # [n_rays, n_samples]

  # Add noise to model's predictions for density. Can be used to
  # regularize network during training (prevents floater artifacts).
  noise = 0.
  if raw_noise_std > 0.:
    noise = tf.random.normal(tf.shape(raw_alpha)) * raw_noise_std

  # Convert from raw alpha to alpha values between [0, 1].
  # Predict density of each sample along each ray. Higher values imply
  # higher likelihood of being absorbed at this point.
  alpha = raw2alpha(raw_alpha + noise, dists)  # [n_rays, n_samples]
  return alpha[Ellipsis, None]


def broadcast_samples_dim(x, target):
  """Broadcast shape of 'x' to match 'target'.

  Given 'target' of shape [N, S, M] and 'x' of shape [N, K],
  broadcast 'x' to have shape [N, S, K].

  Args:
    x: array to broadcast.
    target: array to match.

  Returns:
    x, broadcasts to shape [..., num_samples, K].
  """
  s = target.shape[1]
  result = tf.expand_dims(x, axis=1)  # [N, 1, K]
  result_tile = tf.tile(result, [1, s, 1])  # [N, S, K]
  return result_tile


def sample_pdf(bins, weights, n_samples, det=False):
  """Function for sampling a probability distribution."""
  # Get pdf
  weights += 1e-5  # prevent nans
  pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
  cdf = tf.cumsum(pdf, -1)
  cdf = tf.concat([tf.zeros_like(cdf[Ellipsis, :1]), cdf], -1)

  # Take uniform samples
  u_shape = tf.concat([tf.shape(cdf)[:-1], tf.constant([n_samples])], axis=0)
  if det:
    u = tf.linspace(0., 1., n_samples)
    u = tf.broadcast_to(u, u_shape)
  else:
    u = tf.random.uniform(u_shape)

  # Invert CDF
  inds = tf.searchsorted(cdf, u, side='right')
  below = tf.maximum(0, inds - 1)
  above = tf.minimum(tf.shape(cdf)[-1] - 1, inds)
  inds_g = tf.stack([below, above], -1)
  cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(tf.shape(inds_g)) - 2)
  bins_g = tf.gather(
      bins, inds_g, axis=-1, batch_dims=len(tf.shape(inds_g)) - 2)

  denom = (cdf_g[Ellipsis, 1] - cdf_g[Ellipsis, 0])
  denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
  t = (u - cdf_g[Ellipsis, 0]) / denom
  samples = bins_g[Ellipsis, 0] + t * (bins_g[Ellipsis, 1] - bins_g[Ellipsis, 0])

  return samples


def default_ray_sampling_fine(ray_batch,
                              z_vals,
                              weights,
                              n_samples,
                              perturb,
                              keep_original_points=True,
                              compute_fine_indices=False):
  """Ray sampling of fine points."""
  n_orig_samples = z_vals.shape[1]

  # Extract ray origin, direction.
  rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [R, 3] each

  # Obtain additional integration times to evaluate based on the weights
  # assigned to colors in the coarse model.
  z_vals_mid = .5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
  z_samples = sample_pdf(
      z_vals_mid, weights[Ellipsis, 1:-1], n_samples, det=(perturb == 0.))
  z_samples = tf.stop_gradient(z_samples)

  # Obtain all points to evaluate color, density at.
  if keep_original_points:
    z_list = [z_vals, z_samples]  # [R, S], [R, I]
  else:
    z_list = [z_samples]
  z_vals = tf.sort(tf.concat(z_list, -1), -1)  # [R, S + I]

  fine_indices = None
  if compute_fine_indices:
    # The last `n_samples` values represent the indices of the fine samples
    # in the final sorted set of `z_samples`.
    z_argsort_indices = tf.argsort(tf.concat(z_list, -1), -1)  # [R, S + I]
    fine_indices = z_argsort_indices[:, -n_samples:]  # [R, I]
    fine_indices = tf.reshape(fine_indices, [-1, n_samples])  # [R, I]
  pts = rays_o[Ellipsis, None, :] + \
      rays_d[Ellipsis, None, :] * z_vals[Ellipsis, :, None]  # [R, S + I, 3]

  # The inputs may contain unknown batch size, and leads to results were the
  # first two dimensions are unknown. Set dimension one with the known dimension
  # size.
  n_total_samples = n_orig_samples + n_samples

  z_vals = tf.reshape(z_vals, [-1, n_total_samples])  # [R, S + I]
  z_samples = tf.reshape(z_samples, [-1, n_samples])  # [R, I]
  pts = tf.reshape(pts, [-1, n_total_samples, 3])  # [R, S + I, 3]

  tf.debugging.assert_equal(tf.shape(z_vals)[0], tf.shape(pts)[0])
  return z_vals, z_samples, pts, fine_indices


def apply_intersect_mask_to_tensors(intersect_mask, tensors):
  intersect_tensors = []
  for t in tensors:
    intersect_t = tf.boolean_mask(tensor=t, mask=intersect_mask)  # [Ro, ...]
    intersect_tensors.append(intersect_t)
  return intersect_tensors


def compute_object_intersect_tensors(name, ray_batch, scene_info, far,
                                     object2padding, swap_object_yz, **kwargs):
  """Compute intersecting rays."""
  rays_o = ray_utils.extract_slice_from_ray_batch(  # [R, 3]
      ray_batch=ray_batch,  # [R, M]
      key='origin')
  rays_d = ray_utils.extract_slice_from_ray_batch(  # [R, 3]
      ray_batch=ray_batch,  # [R, M]
      key='direction')
  rays_far = ray_utils.extract_slice_from_ray_batch(  # [R, 3]
      ray_batch=ray_batch,  # [R, M]
      key='far')
  rays_sid = ray_utils.extract_slice_from_ray_batch(  # [R, 1]
      ray_batch=ray_batch,  # [R, M]
      key='metadata')

  (
      box_dims,  # [R, 3], [R, 3], [R, 3, 3]
      box_center,
      box_rotation) = scene_utils.extract_object_boxes_for_scenes(
          name=name,
          scene_info=scene_info,
          sids=rays_sid,  # [R, 1]
          padding=object2padding[name],
          swap_yz=swap_object_yz,
          box_delta_t=kwargs['box_delta_t'])

  # Compute ray-bbox intersections.
  intersect_bounds, intersect_indices, intersect_mask = (  # [R', 2],[R',],[R,]
      box_utils.compute_ray_bbox_bounds_pairwise(
          rays_o=rays_o,  # [R, 3]
          rays_d=rays_d,  # [R, 3]
          rays_far=rays_far,  # [R, 1]
          box_length=box_dims[:, 0],  # [R,]
          box_width=box_dims[:, 1],  # [R,]
          box_height=box_dims[:, 2],  # [R,]
          box_center=box_center,  # [R, 3]
          box_rotation=box_rotation,  # [R, 3, 3]
          far_limit=far))

  # Apply the intersection mask to the ray batch.
  intersect_ray_batch = apply_intersect_mask_to_tensors(  # [R', M]
      intersect_mask=intersect_mask,  # [R,]
      tensors=[ray_batch])[0]  # [R, M]

  # Update the near and far bounds of the ray batch with the intersect bounds.
  intersect_ray_batch = ray_utils.update_ray_batch_bounds(  # [R', M]
      ray_batch=intersect_ray_batch,  # [R', M]
      bounds=intersect_bounds)  # [R', 2]
  return intersect_ray_batch, intersect_indices  # [R', M], [R', 1]


def compute_lightdirs(pts,
                      metadata,
                      scene_info,
                      use_lightdir_norm,
                      use_random_lightdirs,
                      light_pos,
                      light_origins=None):
  """Compute light directions."""
  n_rays = tf.shape(pts)[0]
  n_samples = tf.shape(pts)[1]
  if use_random_lightdirs:
    # https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    phi = tf.random.uniform(
        shape=[n_rays, n_samples], minval=0, maxval=2 * math.pi)  # [R, S]
    cos_theta = tf.random.uniform(
        shape=[n_rays, n_samples], minval=-1, maxval=1)  # [R, S]
    theta = tf.math.acos(cos_theta)  # [R, S]

    x = tf.math.sin(theta) * tf.math.cos(phi)
    y = tf.math.sin(theta) * tf.math.sin(phi)
    z = tf.math.cos(theta)

    light_origins = tf.zeros([n_rays, n_samples, 3],
                             dtype=tf.float32)  # [R, S, 3]
    light_dst = tf.concat([x[Ellipsis, None], y[Ellipsis, None], z[Ellipsis, None]],
                          axis=-1)  # [R, S, 3]
    # Transform ray origin/dst to points.
    light_origins = light_origins + pts
    light_dst = light_dst + pts
    light_origins_flat = tf.reshape(light_origins, [-1, 3])  # [RS, 3]
    light_dst_flat = tf.reshape(light_dst, [-1, 3])  # [RS, 3]
    light_dst_flat = tf.reshape(light_dst, [-1, 3])  # [RS, 3]
    metadata_tiled = tf.tile(metadata[:, None, :],
                             [1, n_samples, 1])  # [R, S, 1]
    metadata_flat = tf.reshape(metadata_tiled, [-1, 1])  # [RS, 1]
    # We want points to be the origin of the light rays in the batch because
    # we will be computing radiance along that ray.
    light_ray_batch = ray_utils.create_ray_batch(  # [RS, M]
        rays_o=light_origins_flat,  # [RS, 3]
        rays_dst=light_dst_flat,  # [RS, 3]
        rays_sid=metadata_flat)  # [RS, 1]
    lightdirs = light_dst
    return light_ray_batch, lightdirs

  # Compute light origins using scene info and control metadata.
  if light_origins is None:
    light_origins = scene_utils.extract_light_positions_for_sids(
        sids=metadata, scene_info=scene_info, light_pos=light_pos)

    # Make sure to use tf.shape instead of light_origins.shape:
    # https://github.com/tensorflow/models/issues/6245
    light_origins = tf.reshape(  # [n_rays, 1, 3]
        light_origins,
        [tf.shape(light_origins)[0], 1,
         tf.shape(light_origins)[1]])

  # Compute the incoming light direction for each point.
  # Note that we use points in the world coordinate space because we currently
  # assume that the light position is in world coordinate space.
  lightdirs = pts - light_origins  # [n_rays, n_samples, 3]

  # Make all directions unit magnitude.
  if use_lightdir_norm:
    lightdirs_norm = tf.linalg.norm(
        lightdirs, axis=-1, keepdims=True)  # [n_rays, n_samples, 1]
    lightdirs = tf.math.divide_no_nan(lightdirs,
                                      lightdirs_norm)  # [n_rays, n_samples, 3]
    lightdirs = tf.reshape(lightdirs, [n_rays, n_samples, 3])
    lightdirs = tf.cast(lightdirs, dtype=tf.float32)

  light_origins = tf.tile(light_origins, [1, n_samples, 1])
  metadata = tf.tile(metadata[:, None, :], [1, n_samples, 1])
  light_origins_flat = tf.reshape(light_origins, [-1, 3])  # [?, 3]
  pts_flat = tf.reshape(pts, [-1, 3])  # [?, 3]
  metadata_flat = tf.reshape(metadata, [-1, 1])  # [?, 1]
  light_ray_batch = ray_utils.create_ray_batch(  # [?S, M]
      rays_o=light_origins_flat,  # [?S, 3]
      rays_dst=pts_flat,  # [?S, 3]
      rays_sid=metadata_flat)  # [?S, 1]
  return light_ray_batch, lightdirs


def compute_view_light_dirs(ray_batch, pts, scene_info, use_viewdirs,
                            use_lightdir_norm, use_random_lightdirs, light_pos):
  """Compute viewing and lighting directions."""
  viewdirs = ray_batch[:, 8:11]  # [R, 3]
  metadata = ray_utils.extract_slice_from_ray_batch(
      ray_batch, key='metadata', use_viewdirs=use_viewdirs)  # [R, 1]
  viewdirs = broadcast_samples_dim(x=viewdirs, target=pts)  # [R, S, 3]
  light_ray_batch, lightdirs = compute_lightdirs(
      pts=pts,
      metadata=metadata,
      scene_info=scene_info,
      use_lightdir_norm=use_lightdir_norm,
      use_random_lightdirs=use_random_lightdirs,
      light_pos=light_pos)  # [R, S, 3]
  return viewdirs, light_ray_batch, lightdirs


def network_query_fn_helper(pts, ray_batch, network, network_query_fn, viewdirs,
                            lightdirs, use_viewdirs, use_lightdirs):
  """Query the NeRF network."""
  if not use_viewdirs:
    viewdirs = None
  if not use_lightdirs:
    lightdirs = None
  # Extract unit-normalized viewing direction.
  # [n_rays, 3]
  # viewdirs = ray_batch[:, 8:11] if use_viewdirs else None

  # Extract additional per-ray metadata.
  # [n_rays, metadata_channels]
  rays_data = ray_utils.extract_slice_from_ray_batch(
      ray_batch, key='example_id', use_viewdirs=use_viewdirs)

  # Query NeRF for the corresponding densities for the light points.
  raw = network_query_fn(pts, viewdirs, lightdirs, rays_data, network)
  return raw


def network_query_fn_helper_nodirs(pts,
                                   ray_batch,
                                   network,
                                   network_query_fn,
                                   use_viewdirs,
                                   use_lightdirs,
                                   use_lightdir_norm,
                                   scene_info,
                                   use_random_lightdirs,
                                   light_origins=None,
                                   **kwargs):
  """Same as network_query_fn_helper, but without input directions."""
  _ = kwargs

  if not use_viewdirs:
    viewdirs = None
  if not use_lightdirs:
    lightdirs = None

  # Extract unit-normalized viewing direction.
  if use_viewdirs:
    viewdirs = ray_batch[:, 8:11]  # [R, 3]
    viewdirs = broadcast_samples_dim(x=viewdirs, target=pts)  # [R, S, 3]
  else:
    viewdirs = None

  # Compute the light directions.
  # if use_lightdirs:
  light_ray_batch, lightdirs = compute_lightdirs(  # [R, S, 3]
      pts=pts,
      metadata=ray_utils.extract_slice_from_ray_batch(
          ray_batch, key='metadata', use_viewdirs=use_viewdirs),
      scene_info=scene_info,
      use_lightdir_norm=use_lightdir_norm,
      use_random_lightdirs=use_random_lightdirs,
      light_pos=kwargs['light_pos'],
      light_origins=light_origins)
  # else:
  #   light_ray_batch = None
  #   lightdirs = None

  # Extract additional per-ray metadata.
  rays_data = ray_utils.extract_slice_from_ray_batch(
      ray_batch, key='example_id', use_viewdirs=use_viewdirs)

  # Query NeRF for the corresponding densities for the light points.
  raw = network_query_fn(pts, viewdirs, lightdirs, rays_data, network)
  return light_ray_batch, raw


def create_w2o_transformations_tensors(name, scene_info, ray_batch,
                                       use_viewdirs, box_delta_t):
  """Create transformation tensor from world to object space."""
  metadata = ray_utils.extract_slice_from_ray_batch(
      ray_batch, key='metadata', use_viewdirs=use_viewdirs)  # [R, 1]
  w2o_rt_per_scene, w2o_r_per_scene = (
      scene_utils.extract_w2o_transformations_per_scene(
          name=name, scene_info=scene_info, box_delta_t=box_delta_t))
  w2o_rt = tf.gather_nd(  # [R, 4, 4]
      params=w2o_rt_per_scene,  # [N_scenes, 4, 4]
      indices=metadata)  # [R, 1]
  w2o_r = tf.gather_nd(  # [R, 4, 4]
      params=w2o_r_per_scene,  # [N_scenes, 4, 4]
      indices=metadata)  # [R, 1]
  return w2o_rt, w2o_r


def apply_batched_transformations(inputs, transformations):
  """Batched transformation of inputs.

  Args:
      inputs: List of [R, S, 3]
      transformations: [R, 4, 4]

  Returns:
      transformed_inputs: List of [R, S, 3]
  """
  transformed_inputs = []
  for x in inputs:
    n_samples = tf.shape(x)[1]
    homog_transformations = tf.expand_dims(
        input=transformations, axis=1)  # [R, 1, 4, 4]
    homog_transformations = tf.tile(homog_transformations,
                                    [1, n_samples, 1, 1])  # [R, S, 4, 4]
    homog_component = tf.ones_like(x)[Ellipsis, 0:1]  # [R, S, 1]
    homog_x = tf.concat([x, homog_component], axis=-1)  # [R, S, 4]
    homog_x = tf.expand_dims(input=homog_x, axis=2)  # [R, S, 1, 4]
    transformed_x = tf.matmul(homog_x,
                              tf.transpose(homog_transformations,
                                           (0, 1, 3, 2)))  # [R, S, 1, 4]
    transformed_x = transformed_x[Ellipsis, 0, :3]  # [R, S, 3]
    transformed_inputs.append(transformed_x)
  return transformed_inputs


def compute_object_inputs(name, ray_batch, pts, scene_info,
                          use_random_lightdirs, **kwargs):
  """Compute inputs to object networks."""
  # Extract viewing and lighting directions.
  # [Ro, S, 3]
  object_viewdirs, light_ray_batch, object_lightdirs = compute_view_light_dirs(
      ray_batch=ray_batch,
      pts=pts,
      scene_info=scene_info,
      use_viewdirs=kwargs['use_viewdirs'],
      use_lightdir_norm=kwargs['use_lightdir_norm'],
      use_random_lightdirs=use_random_lightdirs,
      light_pos=kwargs['light_pos'])

  # Transform points and optionally directions from world to canonical
  # coordinate frame.
  w2o_rt, w2o_r = create_w2o_transformations_tensors(  # [Ro, 4, 4]
      name=name,
      scene_info=scene_info,
      ray_batch=ray_batch,
      use_viewdirs=kwargs['use_viewdirs'],
      box_delta_t=kwargs['box_delta_t'])  # [Ro, 1]
  object_pts = apply_batched_transformations(
      inputs=[pts],
      transformations=w2o_rt,
  )[0]
  if kwargs['use_transform_dirs']:
    # pylint: disable=unbalanced-tuple-unpacking
    [object_viewdirs, object_lightdirs] = apply_batched_transformations(
        inputs=[object_viewdirs, object_lightdirs], transformations=w2o_r)
    # pylint: enable=unbalanced-tuple-unpacking
  return object_pts, object_viewdirs, light_ray_batch, object_lightdirs


def normalize_rgb(raw_rgb, scaled_sigmoid):
  # Extract RGB of each sample position along each ray.
  rgb = tf.math.sigmoid(raw_rgb)  # [n_rays, n_samples, 3]
  if scaled_sigmoid:
    rgb = 1.2 * (rgb - 0.5) + 0.5  # [n_rays, n_samples, 3]
  return rgb


def normalize_raw(raw, z_vals, scaled_sigmoid, raw_noise_std, last_dist_method):
  """Normalize raw outputs of the network."""
  # Compute weight for RGB of each sample along each ray.  A cumprod() is
  # used to express the idea of the ray not having reflected up to this
  # sample yet.
  # [n_rays, n_samples]
  alpha = compute_alpha(
      z_vals=z_vals,
      raw_alpha=raw['alpha'],
      raw_noise_std=raw_noise_std,
      last_dist_method=last_dist_method)
  normalized = {
      'rgb': normalize_rgb(raw_rgb=raw['rgb'], scaled_sigmoid=scaled_sigmoid),
      'alpha': alpha
  }
  return normalized


def run_sparse_network(name, network, intersect_z_vals, intersect_pts,
                       intersect_ray_batch, use_random_lightdirs, **kwargs):
  """Runs a single network."""
  if name.startswith('bkgd'):
    intersect_light_ray_batch, intersect_raw = network_query_fn_helper_nodirs(
        network=network,
        pts=intersect_pts,  # [R, S, 3]
        ray_batch=intersect_ray_batch,  # [R, M]
        use_random_lightdirs=use_random_lightdirs,
        **kwargs)  # [R, 3]
  else:
    (object_intersect_pts, object_intersect_viewdirs, intersect_light_ray_batch,
     object_intersect_lightdirs) = compute_object_inputs(
         name=name,
         ray_batch=intersect_ray_batch,
         pts=intersect_pts,
         use_random_lightdirs=use_random_lightdirs,
         **kwargs)

    # Query the object NeRF.
    intersect_raw = network_query_fn_helper(
        pts=object_intersect_pts,
        ray_batch=intersect_ray_batch,
        network=network,
        network_query_fn=kwargs['network_query_fn'],
        viewdirs=object_intersect_viewdirs,
        lightdirs=object_intersect_lightdirs,
        use_viewdirs=kwargs['use_viewdirs'],
        use_lightdirs=kwargs['use_lightdirs'])

  # Compute weights of the intersecting points on normalized raw values.
  normalized_raw = normalize_raw(  # [Ro, S, 4]
      raw=intersect_raw,  # [Ro, S, 4]
      z_vals=intersect_z_vals,  # [Ro, S]
      scaled_sigmoid=kwargs['scaled_sigmoid'],
      raw_noise_std=kwargs['raw_noise_std'],
      last_dist_method=kwargs['last_dist_method'])
  return intersect_light_ray_batch, normalized_raw


def compute_transmittance(alpha):
  """Computes transmittance from (normalized) alpha values.

  Args:
    alpha: [R, S]

  Returns:
    t: [R, S]
  """
  # Compute the accumulated transmittance along the ray at each point.
  print(f'[compute_transmittance] alpha.shape: {alpha.shape}')
  # TODO(guom): fix this
  t = 1. - alpha
  return t


def compute_weights(normalized_alpha):
  trans = compute_transmittance(normalized_alpha)
  weights = normalized_alpha * trans
  return trans, weights


def run_single_object(name, ray_batch, use_random_lightdirs, **kwargs):
  """Run and generate predictions for a single object.

  Args:
    name: The name of the object to run.
    ray_batch: [R, M] tf.float32. A batch of rays.
    use_random_lightdirs:
    **kwargs: Additional arguments.

  Returns:
    intersect_0: Dict.
    intersect: Dict.
  """
  # Compute intersection rays and indices.
  if name.startswith('bkgd'):
    intersect_ray_batch = ray_batch
    intersect_indices = None
  else:
    intersect_ray_batch, intersect_indices = compute_object_intersect_tensors(
        name=name, ray_batch=ray_batch, **kwargs)
  # Run coarse stage.
  intersect_z_vals_0, intersect_pts_0 = default_ray_sampling(
      ray_batch=intersect_ray_batch,
      n_samples=kwargs['n_samples'],
      perturb=kwargs['perturb'],
      lindisp=kwargs['lindisp'])

  intersect_light_ray_batch_0, normalized_raw_0 = run_sparse_network(
      name=name,
      network=kwargs['name2model'][name],
      intersect_z_vals=intersect_z_vals_0,
      intersect_pts=intersect_pts_0,
      intersect_ray_batch=intersect_ray_batch,
      use_random_lightdirs=use_random_lightdirs,
      **kwargs)

  # Run fine stage.
  if kwargs['N_importance'] > 0:
    # normalized_alpha_0 = normalized_raw_0['alpha']
    _, intersect_weights_0 = compute_weights(
        normalized_alpha=normalized_raw_0['alpha'][Ellipsis, 0])  # [Ro, S]
    # Generate fine samples using weights from the coarse stage.
    intersect_z_vals, _, intersect_pts, _ = default_ray_sampling_fine(
        ray_batch=intersect_ray_batch,  # [Ro, M]
        z_vals=intersect_z_vals_0,  # [Ro, S]
        weights=intersect_weights_0,  # [Ro, S]
        n_samples=kwargs['N_importance'],
        perturb=kwargs['perturb'])

    # Run the networks for all the objects.
    intersect_light_ray_batch, normalized_raw = run_sparse_network(
        name=name,
        network=kwargs['name2model'][name],
        intersect_z_vals=intersect_z_vals,
        intersect_pts=intersect_pts,
        intersect_ray_batch=intersect_ray_batch,
        use_random_lightdirs=use_random_lightdirs,
        **kwargs)

  intersect_0 = {
      'ray_batch': intersect_ray_batch,  # [R, M]
      'light_ray_batch': intersect_light_ray_batch_0,  # [R, S, M]
      'indices': intersect_indices,
      'z_vals': intersect_z_vals_0,
      'pts': intersect_pts_0,
      'normalized_rgb': normalized_raw_0['rgb'],
      'normalized_alpha': normalized_raw_0['alpha'],
  }
  intersect = {
      'ray_batch': intersect_ray_batch,  # [R, M]
      'light_ray_batch': intersect_light_ray_batch,  # [R, S, M]
      'indices': intersect_indices,
      'z_vals': intersect_z_vals,
      'pts': intersect_pts,
      'normalized_rgb': normalized_raw['rgb'],
      'normalized_alpha': normalized_raw['alpha'],
  }
  return intersect_0, intersect


def create_scatter_indices_for_dim(dim, shape, indices=None):
  """Create scatter indieces for a given dimension."""
  dim_size = shape[dim]
  n_dims = len(shape)
  reshape = [1] * n_dims
  reshape[dim] = -1

  if indices is None:
    indices = tf.range(dim_size, dtype=tf.int32)  # [dim_size,]

  indices = tf.reshape(indices, reshape)  # [1, ..., dim_size, ..., 1]

  tf.debugging.assert_equal(tf.shape(indices)[dim], shape[dim])
  indices = tf.broadcast_to(
      indices, shape)  # [Ro, S, 1] or [Ro, S, C, 1]  [0,1,1,1] vs. [512,64,1,1]

  indices = tf.cast(indices, dtype=tf.int32)
  return indices


def create_scatter_indices(updates, dim2known_indices):
  """Create scatter indices."""
  updates_expanded = tf.expand_dims(updates, -1)  # [Ro, S, 1] or [Ro, S, C, 1]
  target_shape = tf.shape(updates_expanded)
  n_dims = len(tf.shape(updates))  # 2 or 3

  dim_indices_list = []
  for dim in range(n_dims):
    indices = None
    if dim in dim2known_indices:
      indices = dim2known_indices[dim]
    dim_indices = create_scatter_indices_for_dim(  # [Ro, S, C, 1]
        dim=dim,
        shape=target_shape,  # [Ro, S, 1] or [Ro, S, C, 1]
        indices=indices)  # [Ro,]
    dim_indices_list.append(dim_indices)
  scatter_indices = tf.concat(dim_indices_list, axis=-1)  # [Ro, S, C, 3]
  return scatter_indices


def scatter_nd(tensor, updates, dim2known_indices):
  scatter_indices = create_scatter_indices(  # [Ro, S, C, 3]
      updates=updates,  # [Ro, S]
      dim2known_indices=dim2known_indices)  # [Ro,]
  scattered_tensor = tf.tensor_scatter_nd_update(
      tensor=tensor,  # [R, S, C]
      indices=scatter_indices,  # [Ro, S, C, 3]
      updates=updates)  # [Ro, S, C]
  return scattered_tensor


def scatter_results(intersect, n_rays, keys):
  """Scatters intersecting ray results into the original set of rays.

  Args:
    intersect: Dict. Intersecting values.
    n_rays: int or tf.int32. Total number of rays.
    keys: [str]. List of keys to scatter.

  Returns:
    scattered_results: Dict. Scattered results.
  """
  # We use `None` to indicate that the intersecting set of rays is equivalent to
  # the full set of rays, so we are done.
  intersect_indices = intersect['indices']
  if intersect_indices is None:
    return {k: intersect[k] for k in keys}

  scattered_results = {}
  n_samples = intersect['z_vals'].shape[1]
  dim2known_indices = {0: intersect_indices}  # [R?, 1]
  for k in keys:
    if k == 'z_vals':
      tensor = tf.random.uniform((n_rays, n_samples),
                                 dtype=tf.float32)  # [R, S]
    elif k == 'pts':
      tensor = tf.cast(  # [R, S, 3]
          tf.fill((n_rays, n_samples, 3), 1000.0),
          dtype=tf.float32)
    elif 'rgb' in k:
      tensor = tf.zeros((n_rays, n_samples, 3), dtype=tf.float32)  # [R, S, 3]
    elif 'alpha' in k:
      tensor = tf.zeros((n_rays, n_samples, 1), dtype=tf.float32)  # [R, S, 1]
    else:
      raise ValueError(f'Invalid key: {k}')
    scattered_v = scatter_nd(  # [R, S, K]
        tensor=tensor,
        updates=intersect[k],  # [Ro, S]
        dim2known_indices=dim2known_indices)
    # Convert the batch dimension to a known dimension.
    # For some reason `scattered_z_vals` becomes [R, ?]. We need to explicitly
    # reshape it with `n_samples`.
    if k == 'z_vals':
      scattered_v = tf.reshape(scattered_v, (n_rays, n_samples))  # [R, S]
    else:
      # scattered_v = tf.reshape(
      #     scattered_v, (n_rays,) + scattered_v.shape[1:])  # [R, S, K]
      # scattered_v = tf.reshape(
      # scattered_v, (-1,) + scattered_v.shape[1:])  # [R, S, K]
      scattered_v = tf.reshape(
          scattered_v, (n_rays, n_samples, tensor.shape[2]))  # [R, S, K]
    scattered_results[k] = scattered_v
  return scattered_results


def combine_results(name2results, keys):
  """Combines network outputs.

  Args:
    name2results: Dict. For each object results, `z_vals` is required.
    keys: [str]. A list of keys to combine results over.

  Returns:
    results: Dict. Combined results.
  """
  # Collect z values across all objects.
  z_vals_list = []
  for _, results in name2results.items():
    z_vals_list.append(results['z_vals'])

  # Concatenate lists of object results into a single tensor.
  z_vals = tf.concat(z_vals_list, axis=-1)  # [R, S*O]

  # Compute the argsort indices.
  z_argsort_indices = tf.argsort(z_vals, -1)  # [R, S*O]
  n_rays, n_samples = tf.shape(z_vals)[0], tf.shape(z_vals)[1]
  gather_indices = tf.range(n_rays)[:, None]  # [R, 1]
  gather_indices = tf.tile(gather_indices, [1, n_samples])  # [R, S]
  gather_indices = tf.concat(
      [gather_indices[Ellipsis, None], z_argsort_indices[Ellipsis, None]], axis=-1)

  results = {}
  for k in keys:
    if k == 'z_vals':
      v_combined = z_vals
    else:
      v_list = [r[k] for r in name2results.values()]
      v_combined = tf.concat(v_list, axis=1)  # [R, S*O, K]

    # Sort the tensors.
    v_sorted = tf.gather_nd(  # [R, S, K]
        params=v_combined,  # [R, S, K]
        indices=gather_indices)  # [R, S, 2]
    results[k] = v_sorted
  return results


def compose_outputs(results, light_rgb, white_bkgd):
  del results
  del light_rgb
  del white_bkgd
  return -1


