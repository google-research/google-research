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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
  """Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse
        samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
  """
  N_rays, N_samples_ = weights.shape
  weights = weights + eps  # prevent division by zero (don't do inplace op!)
  pdf = weights / reduce(weights, 'n1 n2 -> n1 1',
                         'sum')  # (N_rays, N_samples_)
  cdf = torch.cumsum(
      pdf, -1)  # (N_rays, N_samples), cumulative distribution function
  cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                  -1)  # (N_rays, N_samples_+1)
  # padded to 0~1 inclusive

  if det:
    u = torch.linspace(0, 1, N_importance, device=bins.device)
    u = u.expand(N_rays, N_importance)
  else:
    u = torch.rand(N_rays, N_importance, device=bins.device)
  u = u.contiguous()

  inds = torch.searchsorted(cdf, u, right=True)
  below = torch.clamp_min(inds - 1, 0)
  above = torch.clamp_max(inds, N_samples_)

  inds_sampled = rearrange(
      torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
  cdf_g = rearrange(
      torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
  bins_g = rearrange(
      torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

  denom = cdf_g[Ellipsis, 1] - cdf_g[Ellipsis, 0]
  denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0,
  # in which case it will not be sampled
  # anyway, therefore any value for it is fine (set to 1 here)

  samples = bins_g[Ellipsis, 0] + (u - cdf_g[Ellipsis, 0]) / denom * (
      bins_g[Ellipsis, 1] - bins_g[Ellipsis, 0])
  return samples


def render_rays(models,
                coverage_models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False,
                with_semantics=False,
                point_transform_func=None,
                topk=0,
                **kwargs):
  """Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in
        nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse
        model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will
        not do inference
                   on coarse rgb to save time
        point_transform_func: transform applied before passing xyz into nerf
        mlps [N, 3] -> [N, 3]
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and
        fine models
  """

  def inference(results,
                model,
                coverage_model,
                typ,
                xyz,
                z_vals,
                test_time=False,
                with_semantics=False,
                bg_model=None,
                **kwargs):
    """Helper function that performs model inference.

        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
    """
    N_nerflets_ = model.n
    N_samples_ = xyz.shape[1]
    xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c')  # (N_rays*N_samples_, 3)

    assert point_transform_func is None
    xyz_normalized = xyz_ if point_transform_func is None \
        else point_transform_func(xyz_)

    rbfs = coverage_model.calc_weights(xyz_)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []

    k = N_nerflets_ if topk == 0 else topk

    if typ == 'coarse' and test_time and 'fine' in models:
      for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_normalized[i:i + chunk])
        out_chunks += [
            model(xyz_embedded, sigma_only=True, rbfs=rbfs[:, i:i + chunk])
        ]

      out = torch.cat(out_chunks, 1)
      sigmas = rearrange(
          out, 'n (n1 n2) 1 -> n n1 n2', n=k, n1=N_rays, n2=N_samples_)

      # TODO: add sigma_only for coverage models
      out = sigmas[Ellipsis, None].expand(-1, -1, -1, 4)
    else:  # infer rgb and sigma and others
      dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
      # (N_rays*N_samples_, embed_dir_channels)
      for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_normalized[i:i + chunk])
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded_[i:i + chunk]],
                                    1)
        out_chunks += [
            model(xyzdir_embedded, sigma_only=False, rbfs=rbfs[:, i:i + chunk])
        ]

      out = torch.cat(out_chunks, 1)
      # out = out.view(N_rays, N_samples_, 4)
      out = rearrange(
          out, 'n (n1 n2) c -> n n1 n2 c', n=k, n1=N_rays, n2=N_samples_, c=4)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    delta_inf = 1e10 * torch.ones_like(
        deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Run BG Model
    # TODO: do we need chunking here?
    if bg_model is not None:
      out_bg = bg_model(dir_embedded)  # (N_rays, 3+C)
      out_bg_rgb = out_bg[Ellipsis, :3]
      out_bg_logits = out_bg[Ellipsis, 3:]

    # Coverage model
    sem_inputs = model.sem_logits if with_semantics else None
    rgba, pen, sem_outputs = coverage_model.blend(
        rbfs, out.view(k, -1, 4), deltas.view(-1, 1), k, sem_logits=sem_inputs)
    rgba = rgba.view(N_rays, N_samples_, -1)
    alphas = rgba[Ellipsis, -1]
    rgbs = rgba[Ellipsis, :-1]

    # # compute alpha by the formula (3)
    # noise = torch.randn_like(sigmas) * noise_std
    # alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
    weights_sum = reduce(
        weights, 'n1 n2 -> n1',
        'sum')  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    if with_semantics:  # Assume bg label 0 here TODO: white bg
      sem_outputs = sem_outputs.view(N_rays, N_samples_, -1)
      sem_logits = reduce(
          rearrange(weights, 'n1 n2 -> n1 n2 1') * sem_outputs,
          'n1 n2 c -> n1 c', 'sum')

      if white_back:
        assert bg_model is None
        bg_logits = torch.zeros(model.sem_logits.size(1))
        bg_logits[0] = 1.
        sem_logits += (1 - weights_sum.unsqueeze(1)) * bg_logits.to(
            sem_logits.device)

      if bg_model is not None:
        sem_logits += (1 - weights_sum.unsqueeze(1)) * out_bg_logits

      results[f'sem_logits_{typ}'] = sem_logits

    results[f'coverage_pen_{typ}'] = pen.view(1)
    results[f'weights_{typ}'] = weights
    results[f'opacity_{typ}'] = weights_sum
    results[f'z_vals_{typ}'] = z_vals
    if test_time and typ == 'coarse' and 'fine' in models:
      return

    rgb_map = reduce(
        rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs, 'n1 n2 c -> n1 c', 'sum')
    depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')

    if white_back:
      assert bg_model is None
      rgb_map += 1 - weights_sum.unsqueeze(1)

    if bg_model is not None:
      rgb_map += (1 - weights_sum.unsqueeze(1)) * out_bg_rgb

    results[f'rgb_{typ}'] = rgb_map
    results[f'depth_{typ}'] = depth_map

    return

  embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

  # Decompose the inputs
  N_rays = rays.shape[0]
  rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
  near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
  # Embed direction
  dir_embedded = embedding_dir(kwargs.get(
      'view_dir', rays_d))  # (N_rays, embed_dir_channels)

  rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
  rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

  # Sample depth points
  z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
  if not use_disp:  # use linear sampling in depth space
    z_vals = near * (1 - z_steps) + far * z_steps
  else:  # use linear sampling in disparity space
    z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

  z_vals = z_vals.expand(N_rays, N_samples)

  if perturb > 0:  # perturb sampling depths (z_vals)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]
                       )  # (N_rays, N_samples-1) interval mid points
    # get intervals between samples
    upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
    lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

    perturb_rand = perturb * torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * perturb_rand

  xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

  bg_model = models['bg'] if 'bg' in models else None

  results = {}
  inference(results, models['coarse'], coverage_models['coarse'], 'coarse',
            xyz_coarse, z_vals, test_time, with_semantics, bg_model, **kwargs)

  if N_importance > 0:  # sample points for fine model
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]
                       )  # (N_rays, N_samples-1) interval mid points
    z_vals_ = sample_pdf(
        z_vals_mid,
        results['weights_coarse'][:, 1:-1].detach(),
        N_importance,
        det=(perturb == 0))
    # detach so that grad doesn't propogate to weights_coarse from here

    z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
    # combine coarse and fine samples

    xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    inference(results, models['fine'], coverage_models['fine'], 'fine',
              xyz_fine, z_vals, test_time, with_semantics, bg_model, **kwargs)

  return results


def render_rays_cls(models,
                    coverage_models,
                    embeddings,
                    rays,
                    N_samples=64,
                    use_disp=False,
                    perturb=0,
                    noise_std=1,
                    N_importance=0,
                    chunk=1024 * 32,
                    white_back=False,
                    test_time=False,
                    with_semantics=False,
                    point_transform_func=None,
                    topk=0,
                    cls_id=0,
                    **kwargs):
  """Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in
        nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse
        model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will
        not do inference
                   on coarse rgb to save time
        point_transform_func: transform applied before passing xyz into nerf
        mlps [N, 3] -> [N, 3]
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and
        fine models
  """

  def inference(results,
                model,
                coverage_model,
                typ,
                xyz,
                z_vals,
                test_time=False,
                with_semantics=False,
                bg_model=None,
                **kwargs):
    """Helper function that performs model inference.

        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
    """
    N_nerflets_ = model.n
    N_samples_ = xyz.shape[1]
    xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c')  # (N_rays*N_samples_, 3)

    assert point_transform_func is None
    xyz_normalized = xyz_ if point_transform_func is None \
        else point_transform_func(xyz_)

    rbfs = coverage_model.calc_weights(xyz_)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []

    k = N_nerflets_ if topk == 0 else topk

    if typ == 'coarse' and test_time and 'fine' in models:
      for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_normalized[i:i + chunk])
        out_chunks += [
            model(xyz_embedded, sigma_only=True, rbfs=rbfs[:, i:i + chunk])
        ]

      out = torch.cat(out_chunks, 1)
      sigmas = rearrange(
          out, 'n (n1 n2) 1 -> n n1 n2', n=k, n1=N_rays, n2=N_samples_)

      # TODO: add sigma_only for coverage models
      out = sigmas[Ellipsis, None].expand(-1, -1, -1, 4)
    else:  # infer rgb and sigma and others
      dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
      # (N_rays*N_samples_, embed_dir_channels)
      for i in range(0, B, chunk):
        xyz_embedded = embedding_xyz(xyz_normalized[i:i + chunk])
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded_[i:i + chunk]],
                                    1)
        out_chunks += [
            model(xyzdir_embedded, sigma_only=False, rbfs=rbfs[:, i:i + chunk])
        ]

      out = torch.cat(out_chunks, 1)
      # out = out.view(N_rays, N_samples_, 4)
      out = rearrange(
          out, 'n (n1 n2) c -> n n1 n2 c', n=k, n1=N_rays, n2=N_samples_, c=4)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    delta_inf = 1e10 * torch.ones_like(
        deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Run BG Model
    # TODO: do we need chunking here?
    if bg_model is not None:
      out_bg = bg_model(dir_embedded)  # (N_rays, 3+C)
      out_bg_rgb = out_bg[Ellipsis, :3]
      out_bg_logits = out_bg[Ellipsis, 3:]

    # Coverage model
    sem_inputs = model.sem_logits if with_semantics else None
    rgba, pen, sem_outputs = coverage_model.blend_cls(
        rbfs,
        out.view(k, -1, 4),
        deltas.view(-1, 1),
        k,
        sem_logits=sem_inputs,
        cls_id=cls_id,
        nerflets=model)
    rgba = rgba.view(N_rays, N_samples_, -1)
    alphas = rgba[Ellipsis, -1]
    rgbs = rgba[Ellipsis, :-1]

    # # compute alpha by the formula (3)
    # noise = torch.randn_like(sigmas) * noise_std
    # alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, 1-a1, 1-a2, ...]
    weights = \
        alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)
    weights_sum = reduce(
        weights, 'n1 n2 -> n1',
        'sum')  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    if with_semantics:  # Assume bg label 0 here TODO: white bg
      sem_outputs = sem_outputs.view(N_rays, N_samples_, -1)
      sem_logits = reduce(
          rearrange(weights, 'n1 n2 -> n1 n2 1') * sem_outputs,
          'n1 n2 c -> n1 c', 'sum')

      if white_back:
        assert bg_model is None
        bg_logits = torch.zeros(model.sem_logits.size(1))
        bg_logits[0] = 1.
        sem_logits += (1 - weights_sum.unsqueeze(1)) * bg_logits.to(
            sem_logits.device)

      if bg_model is not None:
        sem_logits += (1 - weights_sum.unsqueeze(1)) * out_bg_logits

      results[f'sem_logits_{typ}'] = sem_logits

    results[f'coverage_pen_{typ}'] = pen.view(1)
    results[f'weights_{typ}'] = weights
    results[f'opacity_{typ}'] = weights_sum
    results[f'z_vals_{typ}'] = z_vals
    if test_time and typ == 'coarse' and 'fine' in models:
      return

    rgb_map = reduce(
        rearrange(weights, 'n1 n2 -> n1 n2 1') * rgbs, 'n1 n2 c -> n1 c', 'sum')
    depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')

    if white_back:
      assert bg_model is None
      rgb_map += 1 - weights_sum.unsqueeze(1)

    if bg_model is not None:
      rgb_map += (1 - weights_sum.unsqueeze(1)) * out_bg_rgb

    results[f'rgb_{typ}'] = rgb_map
    results[f'depth_{typ}'] = depth_map

    return

  embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

  # Decompose the inputs
  N_rays = rays.shape[0]
  rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
  near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
  # Embed direction
  dir_embedded = embedding_dir(kwargs.get(
      'view_dir', rays_d))  # (N_rays, embed_dir_channels)

  rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
  rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

  # Sample depth points
  z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
  if not use_disp:  # use linear sampling in depth space
    z_vals = near * (1 - z_steps) + far * z_steps
  else:  # use linear sampling in disparity space
    z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

  z_vals = z_vals.expand(N_rays, N_samples)

  if perturb > 0:  # perturb sampling depths (z_vals)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]
                       )  # (N_rays, N_samples-1) interval mid points
    # get intervals between samples
    upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
    lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

    perturb_rand = perturb * torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * perturb_rand

  xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

  bg_model = models['bg'] if 'bg' in models else None

  results = {}
  inference(results, models['coarse'], coverage_models['coarse'], 'coarse',
            xyz_coarse, z_vals, test_time, with_semantics, bg_model, **kwargs)

  if N_importance > 0:  # sample points for fine model
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:]
                       )  # (N_rays, N_samples-1) interval mid points
    z_vals_ = sample_pdf(
        z_vals_mid,
        results['weights_coarse'][:, 1:-1].detach(),
        N_importance,
        det=(perturb == 0))
    # detach so that grad doesn't propogate to weights_coarse from here

    z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
    # combine coarse and fine samples

    xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    inference(results, models['fine'], coverage_models['fine'], 'fine',
              xyz_fine, z_vals, test_time, with_semantics, bg_model, **kwargs)

  return results
