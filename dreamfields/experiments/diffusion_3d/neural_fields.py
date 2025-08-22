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

"""Neural rendering utilities and MLP architecture."""

# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
  import tinycudann as tcnn
except ImportError:
  print("Unable to import tinycudann.")
import tqdm.auto as tqdm
import torch
from torch import nn
import torch.nn.functional as F

import helpers
import scene
# pylint: enable=g-bad-import-order
# pylint: enable=g-import-not-at-top


class DreamFieldsMLP(nn.Module):
  """MLP architecture."""

  def __init__(self,
               activation,
               features_early,
               features_residual,
               features_late,
               fourfeat,
               fourfeat_include_axis,
               max_deg,
               ipe,
               num_fourier_features=128):
    super().__init__()

    self.ipe = ipe
    if fourfeat:
      # Random Fourier Feature positional encoding. Fix the matrix used for the
      # fourier feature basis so the encoding does not change over iterations.
      dirs = torch.randn((3, num_fourier_features))
      dirs = F.normalize(dirs, dim=-1)
      rads = 2**(max_deg * torch.rand((num_fourier_features,)))
      feats = (rads * dirs).long()
      if fourfeat_include_axis:
        # Include axis aligned features with scale 1.
        feature_matrix = torch.cat([torch.eye(3), feats], 1).long().T
      else:
        feature_matrix = feats.long().T
    else:
      # Axis-aligned positional encoding.
      feature_matrix = 2**torch.arange(max_deg)[:, None, None] * torch.eye(3)
      feature_matrix = feature_matrix.reshape(-1, 3)
    self.register_buffer("feature_matrix", feature_matrix)

    dim = 2 * feature_matrix.size(0)
    activation = nn.__getattribute__(activation)

    # Early network.
    layers_early = []
    for feat in features_early:
      layers_early.append(nn.Linear(dim, feat))
      layers_early.append(activation())
      dim = feat
    self.layers_early = nn.Sequential(*layers_early)

    # Residual blocks.
    layers_residual = []
    for feat_block in features_residual:
      layers_residual.append(nn.LayerNorm(dim))

      # Create a stack of layers.
      block = []
      for l, feat in enumerate(feat_block):
        block.append(nn.Linear(dim, feat))
        if l < len(feat_block) - 1:  # Don't activate right before the residual.
          block.append(activation())
        dim = feat
      block = nn.Sequential(*block)

      layers_residual.append(block)
    self.layers_residual = nn.ModuleList(layers_residual)

    # Late layers.
    layers_late = []
    for l, feat in enumerate(features_late):
      layers_late.append(nn.Linear(dim, feat))
      if l < len(features_late) - 1:  # Don't activate output.
        layers_late.append(activation())
      dim = feat
    self.layers_late = nn.Sequential(*layers_late)

  def reset_parameters(self):
    """Match the default flax initialization."""
    for layer in self.children():
      if isinstance(layer, nn.Linear):
        torch.nn.init.lecun_normal_(layer.weight)
        torch.nn.init.zero_(layer.bias)

  def forward(self, mean, cov=None, decayscale=1., differentiate_posenc=False):
    """Run MLP. mean is [*batch, 3] and cov is [*batch, 3, 3]."""
    # Positional encoding.
    context_manager = torch.enable_grad if differentiate_posenc else torch.no_grad
    with context_manager():
      fm = self.feature_matrix.type(mean.dtype).T  # [3, dim].
      mean_proj = mean.matmul(fm)  # [*batch, dim].
      if self.ipe and cov is not None:
        # Integrated positional encoding (IPE).
        cov_diag_proj = (cov.matmul(fm) * fm).sum(dim=-2)  # [*batch, dim].
        decay = torch.exp(-.5 * cov_diag_proj * decayscale**2)
      else:
        # Disable IPE.
        decay = 1.
      x = torch.cat(
          [decay * torch.cos(mean_proj), decay * torch.sin(mean_proj)],
          -1).float()

    # Network.
    x = self.layers_early(x)

    for i in range(len(self.layers_residual) // 2):
      norm, block = self.layers_residual[(2 * i):(2 * i + 2)]
      x = x + block(norm(x))

    x = self.layers_late(x)
    return x


def intersect_box(r_o, r_d, box_width):
  # r_o + t * r_d = +/- box_width.
  t0, _ = torch.max((-torch.sign(r_d) * box_width - r_o) / r_d, dim=-1)
  t1, _ = torch.min((torch.sign(r_d) * box_width - r_o) / r_d, dim=-1)
  return t0, t1


def dists_to_samples(rays, t, dtype=torch.float32):
  """Convert mipnerf frustums to gaussians."""
  t_end = t[Ellipsis, 1:]
  t_start = t[Ellipsis, :-1]
  t_mids = .5 * (t_end + t_start)
  mean = rays[0].unsqueeze(-2) + rays[1].unsqueeze(-2) * t_mids.unsqueeze(-1)

  d = rays[1]
  d_mag_sq = torch.sum(d**2, axis=-1, keepdims=True).clamp(1e-10, None)
  t_var = (.5**2) / 3. * (t_end - t_start)**2
  r_var = (rays[2] * t_mids)**2 / 12.

  d_outer = d[Ellipsis, :, None] * d[Ellipsis, None, :]
  eye = torch.eye(d.shape[-1], device=t.device)
  null_outer = eye - d[Ellipsis, :, None] * (d / d_mag_sq)[Ellipsis, None, :]
  t_cov = t_var[Ellipsis, None, None] * d_outer[Ellipsis, None, :, :]
  xy_cov = r_var[Ellipsis, None, None] * null_outer[Ellipsis, None, :, :]
  cov = t_cov + xy_cov

  return mean.type(dtype), cov.type(dtype)


def laplace_cdf(s, beta, noise_std = 0):
  """CDF of the Laplace distribution with zero mean and beta scale."""
  beta_recip = 1. / beta
  x = s * beta_recip
  if noise_std:
    x = x + noise_std * torch.randn(x.shape, dtype=x.dtype, device=x.device)
  return torch.sigmoid(x)


def distance_to_density(d, alpha, beta, sigma_noise_std):
  """VolSDF mapping from signed distance d to differential volume density."""
  return alpha * laplace_cdf(-d, beta, sigma_noise_std)


def render_rays(rays,
                *,
                volume_model,
                deformation_model,
                render_deformation,
                deformation_codes,
                decayscales,
                near,
                far,
                device,
                white_bkgd = True,
                mask_rad = 1.,
                mask_rad_norm="inf",
                jitter = True,
                n_pts_per_ray=192,
                origin=None,
                train = False,
                eps = 1e-6,
                chunksize_per_view=None,
                parameterization="nerf",
                sigma_noise_std = 0.,
                volsdf_beta=0.1,
                **forward_kwargs):
  """Volumetric rendering.

  Args:
    rays: tuple of (ray_origins, ray_directions, ray_diffs). Each is a
      torch.tensor. ray_origins is (..., 3), ray_directions is (..., 3), and
      ray_diffs is (..., 1).
    volume_model (torch.nn.Module): NeRF MLP model.
    deformation_model (torch.nn.Module): Nerfies deformation model.
    render_deformation: If True, deform points with the deformation model.
    deformation_codes (torch.Tensor): Latents describing deformation.
    decayscales: Dictionary of coarse-to-fine schedules.
    near (float): Distance from camera origin to near plane.
    far (float): Distance from camera origin to far plane.
    device: Torch device, e.g. CUDA or CPU. white_bkgd mask_rad mask_rad_norm
      sigma_noise_std jitter.
    white_bkgd (bool):
    mask_rad (float): Norm of scene bounds. Density is set to zero outside this
      radius.
    mask_rad_norm (str): Norm to compute scene bounds, determinining shape. Use
      "inf" for a cube or 2 for a ball.
    jitter (bool): Whether to perturb points along rays.
    n_pts_per_ray (int): Number of samples per ray.
    origin: 3-dimensional origin of the scene.
    train (bool): used to decide whether to add noise to density chunksize.
      (Optional[int])
    eps: Small value for numerical stability.
    chunksize_per_view: Number of rays to render per view per batch.
    parameterization: Either nerf or volsdf.
    sigma_noise_std: Standard deviation of oise to add before softplus for NeRF.
    volsdf_beta: Beta parameter determining sharpness of density transition.
    **forward_kwargs: Keyword arguments passed to the volumetric model.

  Returns:
    (rgb, depth, silhouette), aux where:
      rgb is a (*batch, 3) image, in range [0, 1].
      depth is a (*batch, 1) image, in range [0, 1].
      disparity is a (*batch, 1) image.
      silhouette is a (*batch, 1) grayscale foreground mask image, in
      range [0, 1]. Values closer to 1 indicate foreground. This is sometimes
      called "acc" in NeRF implementations.
  """
  if origin is None:
    rays_shift = rays
  else:
    rays_shift = [rays[0] + origin, rays[1], rays[2]]
  r_o, r_d = rays_shift[:2]

  if mask_rad_norm is not None:
    # Per shifted ray, only sample within the bounding box of scene
    # the computed near and far are [H, W] arrays.
    near_, far_ = intersect_box(r_o, r_d, mask_rad)
    far_ = torch.maximum(far_, near_ + 1e-3)  # Minimum sized interval.
  else:
    near_ = torch.tensor([near], device=device, dtype=r_o.dtype)
    far_ = torch.tensor([far], device=device, dtype=r_o.dtype)

  # Get sample points.
  sh = list(r_o.shape[:-1])
  t = helpers.linspace(near_, far_, n_pts_per_ray + 1)  # [*batch, n_samples+1].
  if jitter:
    delta = (far_ - near_) / n_pts_per_ray  # [*batch],
    jitter_sh = sh + [t.shape[-1]]
    jitter_noise = torch.rand(size=jitter_sh, device=device)
    t = t + (jitter_noise - 0.5) * delta[Ellipsis, None]
  endpts = r_o[Ellipsis, None, :] + r_d[Ellipsis, None, :] * t[Ellipsis, None]
  t_mids = .5 * (t[Ellipsis, 1:] + t[Ellipsis, :-1])
  mean, cov = dists_to_samples(rays_shift, t, dtype=torch.float32)

  deformation_kwargs = dict(
      deformation_codes=deformation_codes,
      decayscale=decayscales["deformation"],
      enabled=render_deformation,
  )

  # Run model.
  if chunksize_per_view:
    raw_outputs = []
    aux = []
    for i in range(0, mean.shape[1], chunksize_per_view):
      mean_chunk = mean[:, i:i + chunksize_per_view]
      mean_transformed, batch_aux = deformation_model(mean_chunk,
                                                      **deformation_kwargs)
      batch_outputs = volume_model(
          mean=mean_transformed,
          cov=cov[:, i:i + chunksize_per_view],
          decayscales=decayscales,
          **forward_kwargs)
      raw_outputs.append(batch_outputs)
      aux.append(batch_aux)
    raw_outputs = torch.cat(raw_outputs, dim=1)
    # Merge auxiliary dictionaries across chunks by concatenating each value.
    # NOTE: assumes the auxiliary outputs per chunk are 0-dimensional (scalars).
    # TODO(jainajay): conat along dim 1 like raw_outputs? These aux outputs
    #   aren't used for full resolution, chunked rendering, though. They contain
    #   metrics and losess during optimization with sparse ray sampling.
    aux = {
        k: torch.tensor([a[k].mean() for a in aux])  # mean across devices
        for k in aux[0].keys()
    }
  else:
    mean_transformed, aux = deformation_model(mean, **deformation_kwargs)
    raw_outputs = volume_model(
        mean=mean_transformed,
        cov=cov,
        decayscales=decayscales,
        **forward_kwargs)

  # Reduce auxiliary dict.
  aux = {k: v.mean() for k, v in aux.items()}

  # Activations to get rgb, sigma.
  # NOTE(jainajay): removed sigmoid for grid encoding, based on ReLU fields.
  # TODO(jainajay): Tune this. Not clear whether color should have sigmoid
  #   or be clamped.
  # rgb = torch.sigmoid(raw_outputs[..., :3])
  rgb = helpers.dclamp(raw_outputs[Ellipsis, :3], 0, 1)

  if parameterization == "nerf":
    if train and sigma_noise_std:  # Don't add noise at test time.
      sigma_noise = sigma_noise_std * torch.randn(
          raw_outputs.shape[:-1], dtype=raw_outputs.dtype)
      sigma = F.softplus(raw_outputs[Ellipsis, 3] + sigma_noise)
    else:
      sigma = F.softplus(raw_outputs[Ellipsis, 3])
  elif parameterization == "volsdf":
    sigma = distance_to_density(raw_outputs[Ellipsis, 3], 1. / volsdf_beta,
                                volsdf_beta, sigma_noise_std)

  sigma = scene.mask_sigma(sigma, mean, mask_rad, mask_rad_norm)

  # Volume rendering.
  delta = torch.linalg.norm(endpts[Ellipsis, 1:, :] - endpts[Ellipsis, :-1, :], dim=-1)
  sigma_delta = sigma * delta
  sigma_delta_shifted = torch.cat(
      [torch.zeros_like(sigma_delta[Ellipsis, :1]), sigma_delta[Ellipsis, :-1]], dim=-1)
  alpha = 1. - torch.exp(-sigma_delta)
  trans = torch.exp(-torch.cumsum(sigma_delta_shifted, dim=-1))
  weights = alpha * trans

  rgb = torch.sum(weights[Ellipsis, None] * rgb, dim=-2)  # [*batch, H, W, 3].

  depth = torch.sum(
      weights.double() * t_mids, dim=-1, keepdim=True)  # [*batch, H, W, 1].
  depth = depth.float()

  # Scale disparity.
  disp_min, disp_max = 1. / (far + eps), 1. / (near + eps)
  disparity = 1. / (depth + eps)  # [*batch, H, W, 1].
  disparity = (disparity - disp_min) / (disp_max - disp_min)

  # Scale depth between [0, 1]. Depth is originally [0, far].
  # Depth should really be [near, far], but isn't actually, perhaps due
  # to rays that don't intersect the clipping box.
  depth = depth / far

  silhouette = 1 - torch.exp(-torch.sum(sigma_delta, dim=-1))
  silhouette = silhouette[Ellipsis, None]  # [*batch, H, W, 1].

  if white_bkgd:
    rgb += 1 - silhouette

  return (rgb, depth, disparity, silhouette), aux


@torch.inference_mode()
def render_rotating_volume(*,
                           deformation_codes,
                           n_frames,
                           video_size,
                           elevation_range,
                           depth_cmap="jet",
                           device,
                           **render_kwargs):
  """Render frames from a camera orbiting the volume."""
  render_azimuths = np.linspace(0., 360., n_frames)
  elevation = (elevation_range[0] + elevation_range[1]) / 2.
  cam2worlds = [
      scene.pose_spherical(azim, phi=elevation, radius=4)
      for azim in render_azimuths
  ]
  frames = []

  height, width, focal = scene.scale_intrinsics(video_size)

  for cam2world in tqdm.tqdm(cam2worlds, desc="Rendering rotating volume"):
    rays = scene.camera_rays(cam2world, height, width, focal)
    rays = [torch.from_numpy(r).to(device).type(torch.float32) for r in rays]
    rendered, _ = render_rays(
        rays,
        deformation_codes=deformation_codes,
        white_bkgd=True,
        device=device,
        **render_kwargs)  # rgb, depth, disparity, silhouette.
    rendered = torch.cat(rendered, dim=-1)  # [H, W, 6].
    frames.append(rendered)

  frames = torch.stack(frames, dim=0)  # [n_frames, H, W, 6].
  frames = frames.cpu()
  rgb, depth, disparity, silhouette = torch.split(frames, [3, 1, 1, 1], dim=-1)
  depth_cmap = plt.get_cmap(depth_cmap)
  return (
      rgb.numpy(),  # [T, H, W, 3].
      depth_cmap(depth.numpy().squeeze(-1))[Ellipsis, :3],  # [T, H, W, 1].
      depth_cmap(disparity.numpy().squeeze(-1))[Ellipsis, :3],  # [T, H, W, 1].
      silhouette.numpy())  # [T, H, W, 1].


def subsample_rays(rays_all_views, n_views, n_rays_per_view):
  """Subsample rays. Keep some rays from each view."""
  device = rays_all_views[0].device
  rays_batched = [
      r.reshape((r.shape[0], -1, r.shape[-1])) for r in rays_all_views
  ]
  max_ray_idx = rays_batched[0].shape[1]
  ray_idx = np.stack([
      np.random.choice(max_ray_idx, size=n_rays_per_view, replace=False)
      for _ in range(n_views)
  ])
  ray_idx = torch.from_numpy(ray_idx).to(device)

  # TODO(jainajay): Optimize this as it is run every iteration.
  rays_batched = [
      torch.stack([rv[idx]
                   for rv, idx in zip(r, ray_idx)])
      for r in rays_batched
  ]  # 3-tuple of [n_views, n_rays_per_view, 3 or 1].
  return rays_batched, ray_idx


@torch.jit.script
def slice_patches(rays, idx, patch_size):
  """Extract patches from rays."""
  patches = []
  n_patches = idx.size(0)
  for v in range(n_patches):
    v_idx = idx[v]
    view = v_idx[0]
    top = v_idx[1]
    left = v_idx[2]
    patches.append(rays[view, top:top + patch_size, left:left + patch_size])
  return torch.stack(patches)


def sample_patches(rays_all_views, patch_size, n_patches_per_view,
                   flatten):
  """Sample random patches from rays."""
  with torch.no_grad():
    device = rays_all_views[0].device
    n_views, height, width, channels = rays_all_views[0].shape
    assert channels == 3  # ray origin is 3 dimensional.
    n_patches = n_views * n_patches_per_view

    view_idx = torch.randint(
        low=0,
        high=n_views,
        size=(n_patches,),
        device=device,
    )
    offset_x = torch.randint(
        low=0,
        high=width - patch_size,
        size=(n_patches,),
        device=device,
    )
    offset_y = torch.randint(
        low=0, high=height - patch_size, size=(n_patches,), device=device)
    idx = torch.stack([view_idx, offset_x, offset_y], dim=1)

  assert idx.dtype == torch.long
  rays_patched = [slice_patches(r, idx, patch_size) for r in rays_all_views]
  assert rays_patched[0].shape == (n_patches, patch_size, patch_size, channels)

  if flatten:
    rays_patched = [
        r.view(n_patches, patch_size * patch_size, -1) for r in rays_patched
    ]

  return rays_patched


class WrappedNetwork(nn.Module):
  """Wrapper for different MLP backends."""

  def __init__(self,
               n_input_dims,
               n_output_dims,
               network_config,
               seed=1337,
               computation_dtype=None,
               output_dtype=None):
    super().__init__()

    self.otype = network_config["otype"]
    self.computation_dtype = computation_dtype
    self.output_dtype = output_dtype

    n_hidden_layers = network_config["n_hidden_layers"]

    if self.otype == "torch":
      # Pure PyTorch network.
      activation = getattr(nn, network_config["activation"])
      width = network_config["n_neurons"]
      if n_hidden_layers == -1:
        # Remove the input layer, so this is linear.
        layers = [nn.Linear(n_input_dims, n_output_dims)]
      else:
        layers = [nn.Linear(n_input_dims, width), activation()]
        for _ in range(n_hidden_layers):
          layers.append(nn.Linear(width, width))
          layers.append(activation())
        layers.append(nn.Linear(width, n_output_dims))
      output_activation = network_config["output_activation"]
      if output_activation and output_activation != "None":
        layers.append(getattr(nn, output_activation)())
      print("layers", layers)
      self.layers = nn.Sequential(*layers)

      if computation_dtype:
        self.layers = self.layers.type(computation_dtype)
    elif self.otype in ["FullyFusedMLP", "CutlassMLP"]:
      assert n_hidden_layers > 0
      self.layers = tcnn.Network(
          n_input_dims, n_output_dims, network_config, seed=seed)

    self.reset_parameters()

  def reset_parameters(self):
    if self.otype == "torch":
      for layer in self.layers.children():
        if isinstance(layer, nn.Linear):
          torch.nn.init.xavier_uniform_(layer.weight)
          # TODO(jainajay): geometric initialization from
          #   https://arxiv.org/pdf/1911.10414.pdf and
          #   https://arxiv.org/pdf/2002.10099.pdf.
          torch.nn.init.zeros_(layer.bias)

  def forward(self, x):
    if self.computation_dtype:
      x = x.type(self.computation_dtype)
    x = self.layers(x)
    if self.output_dtype:
      x = x.type(self.output_dtype)
    return x


## Deformations.
def cross_product_matrix(x1, x2, x3):
  """Construct cross product matrix needed for screw axis parameterization."""
  matrix = torch.zeros(x1.shape + (3, 3), dtype=x1.dtype, device=x1.device)
  matrix[Ellipsis, 1, 2] = -x1
  matrix[Ellipsis, 2, 1] = x1
  matrix[Ellipsis, 0, 2] = x2
  matrix[Ellipsis, 2, 0] = -x2
  matrix[Ellipsis, 0, 1] = -x3
  matrix[Ellipsis, 1, 0] = x3
  return matrix


def compute_elastic_loss(jacobian, eps=1e-6, loss_type="log_svals"):
  """Compute the elastic regularization loss.

  The loss is given by sum(log(S)^2). This penalizes the singular values
  when they deviate from the identity since log(1) = 0.0,
  where D is the diagonal matrix containing the singular values.
  Based on https://github.com/google/nerfies/blob/main/nerfies/training.py#L71.

  Args:
    jacobian: the Jacobian of the point transformation.
    eps: a small value to prevent taking the log of zero.
    loss_type: which elastic loss type to use.

  Returns:
    The elastic regularization loss.
  """
  if loss_type == "log_svals":
    svals = torch.linalg.svdvals(jacobian)
    log_svals = torch.log(torch.clamp(svals, min=eps, max=None))
    sq_residual = torch.sum(log_svals**2, dim=-1)
  else:
    raise NotImplementedError(f"Unknown elastic loss type {loss_type!r}")
  residual = torch.sqrt(sq_residual)
  loss = sq_residual.mean()
  return loss, residual.mean()


class FourierFeatureIPE(nn.Module):
  """Fourier feature positional embedding."""

  def __init__(
      self,
      num_dims_to_encode,
      max_deg,
      ipe,
      fourfeat,
      fourfeat_include_axis = False,
      num_fourier_features = 64,
      dtype=torch.float32,
  ):
    super().__init__()

    self.ipe = ipe
    self.num_dims_to_encode = num_dims_to_encode

    if fourfeat:
      # Random Fourier Feature positional encoding. Fix the matrix used for the
      # fourier feature basis so the encoding does not change over iterations.
      dirs = torch.randn((num_dims_to_encode, num_fourier_features))
      dirs = F.normalize(dirs, dim=-1)
      rads = 2**(max_deg * torch.rand((num_fourier_features,)))
      feature_matrix = (rads * dirs).long()
      if fourfeat_include_axis:
        # Include axis aligned features with scale 1.
        feature_matrix = torch.cat(
            [torch.eye(num_dims_to_encode, dtype=torch.long), feature_matrix],
            1)
    else:
      # Axis-aligned positional encoding.
      feature_matrix = 2**torch.arange(max_deg)[:, None, None] * torch.eye(
          num_dims_to_encode)
      feature_matrix = feature_matrix.reshape(-1, num_dims_to_encode).T

    self.register_buffer(
        "fm", feature_matrix.type(dtype))  # [num_dims_to_encode, dim].
    self.output_dim = self.fm.size(
        1) * 2  # sin and cos. NOTE: does not include identity dims.

  def forward(self, x, cov=None, decayscale = 1.):
    if x.shape[-1] > self.num_dims_to_encode:
      x_extra = x[Ellipsis, self.num_dims_to_encode:]
      x = x[Ellipsis, :self.num_dims_to_encode]
    else:
      x_extra = None

    assert x.dtype == self.fm.dtype
    x_proj = x.matmul(self.fm)  # [*batch, dim].

    # Integrated positional encoding (IPE).
    with torch.no_grad():
      if self.ipe and isinstance(cov, float):
        # Use isotropic covariance so we can still attenuate posenc (just
        # globally). Larger posenc frequencies get attenuated more.
        cov_diag_proj = cov * (self.fm * self.fm).sum(dim=-2)
        decay = torch.exp(-.5 * cov_diag_proj * decayscale**2)
      elif self.ipe and cov is not None:
        # Compute PΣP^T where P=self.fm.T is Dx3, and Σ=cov is *batchx3x3.
        assert cov.ndim == x.ndim + 1
        assert cov.shape[:-2] == x.shape[:-1]
        cov_diag_proj = (cov.matmul(self.fm) * self.fm).sum(
            dim=-2)  # [*batch, dim]
        decay = torch.exp(-.5 * cov_diag_proj * decayscale**2)
      else:
        decay = 1.

    if x_extra is not None:
      # Identity embedding for extra dimensions beyond num_dims_to_encode.
      return torch.cat(
          [decay * torch.cos(x_proj), decay * torch.sin(x_proj), x_extra], -1)

    return torch.cat([decay * torch.cos(x_proj), decay * torch.sin(x_proj)], -1)


def render_validation_views(*,
                            render_size,
                            max_size,
                            thetas=(30,),
                            phis=(-25,),
                            device,
                            depth_cmap="jet",
                            **render_kwargs):
  """Render a frame from a camera at a new perspective."""
  height, width, focal = scene.scale_intrinsics(min(render_size, max_size))

  frames = []
  for theta in thetas:
    for phi in phis:
      cam2world = scene.pose_spherical(theta=theta, phi=phi, radius=4)
      rays = scene.camera_rays(cam2world, height, width, focal)
      rays = [torch.from_numpy(r).to(device).type(torch.float32) for r in rays]
      rendered, _ = render_rays(
          rays, white_bkgd=True, device=device,
          **render_kwargs)  # rgb, depth, disparity, silhouette.
      rendered = torch.cat(rendered, dim=-1)  # [H, W, 6].
      frames.append(rendered)

  frames = torch.stack(frames)
  if height != render_size:
    frames = frames.movedim(-1, 1)  # NHWC to NCHW.
    frames = F.interpolate(frames, render_size, mode="bilinear")
    frames = frames.movedim(1, -1)  # NHWC.

  frames = frames.cpu()
  rgb, depth, disparity, silhouette = torch.split(frames, [3, 1, 1, 1], dim=-1)
  depth_cmap = plt.get_cmap(depth_cmap)
  return (
      rgb.numpy(),  # [T, H, W, 3].
      depth_cmap(depth.numpy().squeeze(-1))[Ellipsis, :3],  # [T, H, W, 1].
      depth_cmap(disparity.numpy().squeeze(-1))[Ellipsis, :3],  # [T, H, W, 1].
      silhouette.numpy())  # [T, H, W, 1].


def mask_grid_features(features, num_levels_to_mask, n_features_per_level):
  """For coarse-to-fine on multi-res grids, zero higher resolution features."""
  num_levels_to_mask = int(num_levels_to_mask)
  if num_levels_to_mask == 0:
    return features

  original_shape = features.shape
  batch_shape = original_shape[:-1]

  # Assuming features are stored as L0, L0, L1, L1, ...
  features = features.view(*batch_shape, -1, n_features_per_level)
  features = F.pad(
      features[Ellipsis, :-num_levels_to_mask, :], (0, 0, 0, num_levels_to_mask),
      mode="constant",
      value=0.)

  features = features.view(*batch_shape, -1)
  assert features.shape == original_shape
  return features
