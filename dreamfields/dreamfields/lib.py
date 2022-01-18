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

"""Dream Fields learn a 3D neural radiance field (NeRF) given a textual prompt."""

import collections
import functools
import os
import time
from typing import Optional

from . import augment
from . import helpers
from . import log
from . import scene
from . import schedule

from absl import logging
from clu import metric_writers
import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
from jax import random
import jax.numpy as np
import matplotlib.pyplot as plt
import ml_collections
import numpy as onp
from scipy import stats
import tensorflow.io.gfile as gfile
import tqdm


class DreamField:
  """Trainable Dream Field model."""

  def __init__(self, config):
    self.config = config

  def run_train(self,
                experiment_dir,
                work_unit_dir,
                rng,
                yield_results=False):
    """Train a Dream Field and save results to work_unit_dir."""
    t_start = time.time()
    config = self.config

    logging.info('Local devices: %s', jax.local_devices())
    logging.info('All devices: %s', jax.devices())

    ## Load CLIP
    encode_image, encode_text, preprocess_image, tokenize_fn = (
        helpers.load_image_text_model(config.loss_model))

    ## Pick a prompt
    template = config.get('query_template', '{query}')
    query = template.format(query=config.query)
    z_clip = encode_text(tokenize_fn(query))

    ## Encode retrieval set
    if config.queries_r:
      if config.retrieve_models[0] == config.loss_model:
        # Reuse loss model.
        encode_image_r, preprocess_image_r = encode_image, preprocess_image
        encode_text_r, tokenize_fn_r = encode_text, tokenize_fn
      else:
        # Load new model.
        encode_image_r, encode_text_r, preprocess_image_r, tokenize_fn_r = (
            helpers.load_image_text_model(config.retrieve_models[0]))

      if config.query not in config.queries_r:
        config.queries_r.append(config.query)
      z_clip_r = encode_text_r(tokenize_fn_r(config.queries_r))
      true_idx_r = config.queries_r.index(config.query)
      assert true_idx_r >= 0  # Input query must be set of retrieval queries.

      del encode_text_r, tokenize_fn_r  # Clean up retrieval text encoder.

    del encode_text, tokenize_fn  # Clean up text encoder.

    ## Scene origin manually tracked
    scene_origin = scene.EMA(np.zeros(3, dtype=np.float64), decay=0.999)

    def train_step(state, rays, key, *multistep_constants):
      """Perform a training iteration, optionally composed of multiple substeps.

      Using multiple substeps slightly reduces training time, but only one
      substep per training iteration is used in experiments.

      Args:
        state: Optimizer state.
        rays: Camera rays for rendering, shared across all substeps.
        key: PRNGKey for random number generation (e.g. for augmentations).
        *multistep_constants: Training constants that can vary across substeps.
          7 arrays of constants of length config.substeps are expected:
            (1) lrs: learning rates
            (2) scs: scale factor for integrated positional encoding. Larger
                scales lead to a blurrier appearance. A constant sc=1 is the
                standard mip-NeRF IPE, and used by Dream Fields.
            (3) sns: standard deviation of pre-activation noise for NeRF
                density. Dream Fields use sn=0.
                  density(x) = softplus(s(x) + eps), eps ~ N(0, sn^2)
            (4) mrs: norm of radiance mask, defining scene bounds.
            (5) betas: scale of beta prior loss. Dream Fields use beta=0.
            (6) acct: transmittance loss hyperparameter, defining the target
                average opacity. This is 1 - tau (target transmittance).
            (7) acclam: weight of transmittance loss.

      Returns:
        state: Updated optimizer state.
        last_augs: Augmented views of renderings from the last substep.
        mean_losses: Dictionary of losses averaged over replicas and substeps.
        scene_origin: Updated origin of the scene, based on the center of mass.
      """
      # NOTE(jainajay): rays are shared across all substeps
      pmean = functools.partial(jax.lax.pmean, axis_name='batch')
      psum = functools.partial(jax.lax.psum, axis_name='batch')

      def loss_fn(params, key, sc, sn, mr, beta, acct, acclam):
        render_key, aug_key, key = random.split(key, 3)

        # Render from nerf
        (rgb_est_flat, _, acc_est_flat), aux = render_rays(
            rays=rays,
            variables=params,
            rng=render_key,
            config=config,
            sc=sc,
            sigma_noise_std=sn,
            mask_rad=mr,
            origin=scene_origin.value,
            train=True)
        rgb_est = scene.gather_and_reshape(rgb_est_flat, config.render_width, 3)
        acc_est = scene.gather_and_reshape(acc_est_flat, config.render_width, 1)
        # Make augmentations process specific
        aug_key = random.fold_in(aug_key, pid)
        # Perform augmentations and resize to clip_width
        augs = augment.augment_rendering(config, rgb_est, acc_est, aug_key)

        # Run through CLIP
        z_est = encode_image(preprocess_image(augs))
        clip_loss = -(z_est * z_clip).sum(-1).mean()
        total_loss = clip_loss

        transparency_loss = config.get('transparency_loss', None)
        acc_mean = np.mean(acc_est)
        aux['losses']['acc_mean'] = acc_mean
        if transparency_loss == 'neg_lam_transmittance_clipped':
          # Compute the Dream Fields transmittance loss for scene sparsity.
          trans_mean = 1 - acc_mean
          trans_mean_clipped = np.minimum(1 - acct, trans_mean)
          reg = acclam * trans_mean_clipped
          total_loss -= reg

          aux['losses']['trans_mean_clipped'] = trans_mean_clipped
          aux['losses']['acc_reg_additive'] = reg
        else:
          assert transparency_loss is None

        # Compute a sparsity loss by placing a bimodal beta prior on the
        # per-pixel transmittance. This prior was proposed by Lombardi et al
        # in "Neural Volumes: Learning Dynamic Renderable Volumes from Images"
        # and is used only in ablations.
        beta_loss = np.mean(
            np.log(np.maximum(1e-6, acc_est_flat)) +
            np.log(np.maximum(1e-6, 1. - acc_est_flat)))
        total_loss += beta_loss * beta

        # Compute a weighted mean of each replica's estimated scene origin,
        # since replicas get a different subset of rays
        total_sigma = psum(aux['scene_origin_sigma'])
        aux['scene_origin'] = psum(aux['scene_origin'] *
                                   aux['scene_origin_sigma'] / total_sigma)
        # Compute loss that pushes scene content to 0 origin. We set the loss
        # weight zero_origin_lam = 0 in experiments so the loss is just for
        # logging how far the origin has drifted.
        origin_loss = np.sum(np.square(aux['scene_origin']))
        if config.get('zero_origin_lam', 0.):
          total_loss += config.zero_origin_lam * origin_loss

        aux['losses'].update({
            'clip_loss': clip_loss,
            'beta_loss': beta_loss,
            'origin_loss': origin_loss,
            'loss': total_loss,
        })
        aux['augs'] = augs
        return total_loss, aux

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

      # Scan over substeps
      def body_fn(state, step_constants):
        lr, step_constants = step_constants[0], step_constants[1:]
        grad_fn_key, _ = random.split(key, 2)
        (_, aux), grad = grad_fn(state.target, grad_fn_key, *step_constants)
        grad = pmean(grad)  # all-reduce grad
        aux['losses'] = pmean(aux['losses'])
        aux['losses']['grad_norm'] = helpers.tree_norm(grad)
        state = state.apply_gradient(grad, learning_rate=lr)
        return state, aux

      assert len(multistep_constants) == 7
      multistep_constants = np.array(multistep_constants).T

      if config.substeps == 1:
        state, aux = body_fn(state, np.squeeze(multistep_constants))
        last_augs = aux['augs']
      else:
        state, aux = jax.lax.scan(body_fn, state, multistep_constants)
        # Augmentations from last substep.
        # Shape: [n_local_aug, clip_width, clip_width, 3]
        last_augs = aux['augs'][-1]

      # Average each type of loss over substeps
      mean_losses = jax.tree_map(np.mean, aux['losses'])
      return state, last_augs, mean_losses, aux['scene_origin']

    train_pstep = jax.pmap(
        train_step,
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None, None, None, None))

    onp.random.seed(config.seed)

    n_device = jax.local_device_count()
    pid = jax.process_index()
    logging.info('n_device %d', n_device)
    ## Modified NeRF architecture, with swish, softplus, skips.
    variables, render_rays = helpers.init_nerf_model(rng.advance(1), config)
    state = flax.optim.Adam(config.lr0, eps=config.adam_eps).create(variables)

    ## Try to restore a checkpoint.
    restore_dir = config.get('restore_dir', experiment_dir)
    restore_dir = os.path.join(restore_dir, os.path.basename(work_unit_dir))
    if checkpoints.latest_checkpoint(restore_dir):
      restored = checkpoints.restore_checkpoint(
          restore_dir,
          target={
              'origin': np.zeros(3),
              'state': state,
              'vars': variables
          })
      scene_origin.value = onp.array(restored['origin'])
      state = restored['state']
      variables = restored['vars']
      logging.info('restored checkpoint from step %d', state.state.step)
    else:
      logging.info('did not find checkpoint in %s', restore_dir)

    ## Replicate state.
    step_init = state.state.step
    helpers.defragment()
    state = flax.jax_utils.replicate(state, jax.devices())
    helpers.defragment()

    ## pmap'd rendering for test time evaluation.
    kwargs_test = dict(rng=None, sigma_noise_std=0.)
    config_test = ml_collections.ConfigDict(config)
    config_test.update(config.test)
    config_test_hq = ml_collections.ConfigDict(config_test)
    config_test_hq.update(config.test_hq)

    @functools.partial(jax.pmap, in_axes=(0, None, None, None))
    def render_test_p(rays, variables, sc=1., mr=1.):
      return render_rays(
          rays=rays,
          variables=variables,
          sc=sc,
          mask_rad=mr,
          origin=scene_origin.value,
          config=config_test,
          **kwargs_test)[0]

    @functools.partial(jax.pmap, in_axes=(0, None, None, None))
    def render_test_hq_p(rays, variables, sc=1., mr=1.):
      return render_rays(
          rays=rays,
          variables=variables,
          config=config_test_hq,
          sc=sc,
          mask_rad=mr,
          origin=scene_origin.value,
          **kwargs_test)[0]

    def render_test(rays, variables, sc=1., mr=1., hq=False):
      sh = rays[0].shape
      rays = [x.reshape((jax.device_count(), -1) + x.shape[1:]) for x in rays]
      if hq:
        out = render_test_hq_p(rays, variables, sc, mr)
      else:
        out = render_test_p(rays, variables, sc, mr)
      out = [x.reshape(sh[:-1] + (-1,)) for x in out]
      return out

    def render_loop(rays, variables, sc=1., mr=1., chunk=2**13, hq=False):
      sh = list(rays[0].shape[:-1])
      rays = [x.reshape((-1,) + x.shape[-1:]) for x in rays]
      outs = [
          render_test([x[i:i + chunk]
                       for x in rays], variables, sc, mr, hq=hq)
          for i in range(0, rays[0].shape[0], chunk)
      ]
      outs = [
          np.reshape(np.concatenate([z[i]
                                     for z in outs]), sh + [-1])
          for i in range(3)
      ]
      return outs

    ## Training loop
    t_total = 0.
    logging.info('Experiment dir %s', experiment_dir)
    logging.info('Work unit dir %s', work_unit_dir)
    gfile.makedirs(work_unit_dir)

    # Set up metric writer
    writer = metric_writers.create_default_writer(
        work_unit_dir, asynchronous=True, just_logging=jax.process_index() > 0)
    if jax.process_index() == 0:
      train_config = config.copy_and_resolve_references()
      log.write_config_json(train_config, work_unit_dir)

    # Scale instrinsics to different resolutions.
    hwf_clip_r = scene.scale_intrinsics(config.retrieve_widths[0])
    hwf_base = scene.scale_intrinsics(config.render_width)
    hwf_video = scene.scale_intrinsics(300.)
    hwf_video_hq = scene.scale_intrinsics(400.)

    # JIT compile ray generation
    @jax.jit
    def camera_ray_batch_base(p, focal_mult):
      return scene.camera_ray_batch(p, *hwf_base[:2], hwf_base[2] * focal_mult)

    @jax.jit
    def sample_pose_focal(key):
      return scene.sample_camera(key, config.th_range, config.phi_range,
                                 config.rad_range, config.focal_mult_range)

    shard_rays_jit = jax.jit(functools.partial(scene.shard_rays))

    def sample_iter_data(key, step):
      # Sample pose, focal length multiplier.
      pose, rad, focal_mult = sample_pose_focal(key)

      # Generate rays, shaped for pmap over devices.
      rays = camera_ray_batch_base(pose, focal_mult)
      rays_in = shard_rays_jit(rays)
      # Select rays for this process
      rays_in = jax.tree_map(lambda x: x[pid], rays_in)

      substeps = np.arange(start=step, stop=step + config.substeps, step=1)

      # mip-NeRF scale annealing.
      decays = config.mipnerf.decay_start * (
          1 - substeps / config.mipnerf.decay_iters)
      scs = np.maximum(1., 2**decays)

      # Sigma noise annealing.
      sns = schedule.sigma_noise_std_fn(
          substeps, i_split=config.sn_i_split, sn0=config.sn0, sn1=config.sn1)

      # Scene bounds annealing.
      mrs = schedule.mask_rad_fn(
          substeps, i_split=config.mr_i_split, mr0=config.mr0, mr1=config.mr1)

      # Anneal target opacity (1 - transmittance).
      accts = schedule.anneal_exponentially(substeps, config.acc_target_i_split,
                                            config.acc_target0,
                                            config.acc_target1)
      # The area of an object on the image plane grows with the focal length
      # and shrinks with increasing camera radius. Scale target opacity
      # proportionally with the squared focal multiplier and inversely
      # proportionally with the squared camera radius. For consistency with
      # early experiments that did not use this scaling, we also scale by a
      # constant, 1 / (4^2 * 1.2).
      acct_scaling = focal_mult**2 / ((rad / 4.)**2) / 1.2
      accts = np.minimum(1., acct_scaling * accts)
      acclams = np.where(substeps < config.acc_lam_after, 0., config.acc_lam)

      # Beta prior encourages either 0 or 1 opacity for rays
      betas = np.where(substeps < config.beta_after, .0,
                       config.get('beta_lam', .001))

      # Learning rate schedule.
      # NOTE: vectorized calculation of lrs doesn't work with multiple substeps
      lrs = schedule.lr_fn(
          substeps,
          i_split=config.lr_i_split,
          i_end=config.iters,
          lr0=config.lr0,
          lr1=config.lr1,
          lr2=config.lr2,
          cosine_decay=config.lr_cosine_decay)

      return substeps, rays_in, lrs, scs, sns, mrs, betas, accts, acclams

    pbar = tqdm.trange(
        step_init,
        config.iters + config.substeps,
        config.substeps,
        desc='training')
    for i in pbar:
      t = time.time()

      substeps, rays_in, lrs, scs, sns, mrs, betas, accts, acclams = (
          sample_iter_data(rng.advance(1), i))
      l = substeps[-1]

      keys_pstep = rng.split(n_device)
      # NOTE: loss is averaged across substeps.
      new_state, augs, mean_losses, new_scene_origin = train_pstep(
          state, rays_in, keys_pstep, lrs, scs, sns, mrs, betas, accts, acclams)

      # Reduce across devices
      mean_losses = jax.tree_map(np.mean, mean_losses)

      # Gradient skipping if nan.
      if (helpers.all_finite_tree(mean_losses) and
          helpers.all_finite_tree(new_state)):
        state = new_state
      else:
        logging.warn('Skipping update on step %d. non-finite loss or state', i)
        continue

      # Update scene origin.
      if config.get('ema_scene_origin', False):
        if helpers.all_finite(new_scene_origin):
          scene_origin.update(new_scene_origin[0])
        else:
          logging.warn(
              'Skipping origin update on step %d. '
              'non-finite origin. old: %s skipped update: %s', i,
              scene_origin.value, new_scene_origin)

      ## Yield results, for display in colab.
      augs = augs.reshape(-1, *augs.shape[2:])  # devices, n_localaug, HWC->BHWC
      if yield_results:
        yield mean_losses, augs, scene_origin.value
      else:
        yield None
      pbar.set_description(f'Loss: {mean_losses["loss"]:.4f}')

      ## Logging.
      if i == 0:
        continue

      t_total += time.time() - t

      if i % config.log_scalars_every == 0:
        scalars = {f'losses/{key}': value for key, value in mean_losses.items()}
        scalars.update({
            'schedule/mipnerf_scale': scs[-1],
            'schedule/lr': lrs[-1],
            'schedule/mask_rad': mrs[-1],
            'schedule/sigma_noise_std': sns[-1],
            'schedule/beta': betas[-1],
            'schedule/acc_target': accts[-1],
            'schedule/acc_lam': acclams[-1],
            'origin/x': scene_origin.value[0],
            'origin/y': scene_origin.value[1],
            'origin/z': scene_origin.value[2],
            'origin/norm': np.linalg.norm(scene_origin.value),
        })

        secs_per_iter = t_total / (l - step_init)
        iters_per_sec = (l - step_init) / t_total
        wall = time.time() - t_start
        scalars.update({
            'system/wall': wall,
            'system/secs_per_iter': secs_per_iter,
            'system/iters_per_sec': iters_per_sec,
        })

      if i % config.render_every == 0:
        variables = helpers.state_to_variables(state)
        cam2world = scene.pose_spherical(30., -45., 4.)
        rays = scene.camera_ray_batch(cam2world, *hwf_clip_r)

        # Render with no scale manipulation.
        outs = render_loop(rays, variables, sc=1., mr=mrs[-1], hq=True)
        outs = [np.squeeze(x) for x in outs]
        step_images = {
            'render/rgb': outs[0][None],
            'render/depth': outs[1][None, Ellipsis, None],
            'render/acc': outs[2][None, Ellipsis, None],
        }

        # Compute retrieval metric.
        if config.queries_r:
          z_est = encode_image_r(preprocess_image_r(outs[0][None]))
          cosine_sim = (z_est * z_clip_r).sum(-1)  # 1d, num retrieval queries
          log_prob = nn.log_softmax(cosine_sim)
          prefix = f'val/{config.retrieve_models[0]}/retrieve_'
          scalars.update({
              f'{prefix}cosine_sim':
                  cosine_sim[true_idx_r],
              f'{prefix}loss':
                  -log_prob[true_idx_r],
              f'{prefix}acc':
                  (np.argmax(cosine_sim) == true_idx_r).astype(float)
          })

        augs_tiled = log.make_image_grid(augs[:8])
        step_images['render/augmentations'] = augs_tiled

        fig = plt.figure()
        plt.imshow(1. / np.maximum(config.near, outs[1]))
        plt.colorbar()
        plt.title('disparity')
        disparity = log.plot_to_image(fig)
        step_images['render/disparity'] = disparity

        writer.write_images(step=l, images=step_images)

        if config.render_lq_video and config.video_every and (
            i % config.video_every == 0 or i + 1 == config.iters):

          def rays_theta(th):
            cam2world = scene.pose_spherical(th, -30., 4.)
            return scene.camera_ray_batch(cam2world, *hwf_video)

          th_range = np.linspace(0, 360, 60, endpoint=False)
          frames_all = [
              render_loop(
                  rays_theta(th), variables, scs[-1], mrs[-1], hq=False)
              for th in tqdm.tqdm(th_range, desc='render video')
          ]

          videos = [[np.squeeze(f[i]) for f in frames_all] for i in range(3)]
          for video, label in zip(videos, 'rgb depth acc'.split()):
            scale = (label == 'depth')
            log.log_video(
                None, video, 'frames', label, l, work_unit_dir, scale=scale)

      if i % config.log_scalars_every == 0:
        writer.write_scalars(step=l, scalars=scalars)

      if i % config.flush_every == 0:
        writer.flush()

      defrag_every = config.get('defragment_every', default=0)
      if defrag_every and i % defrag_every == 0:
        helpers.defragment()

      if config.get('checkpoint_every') and i % config.checkpoint_every == 0:
        saved_path = checkpoints.save_checkpoint(
            ckpt_dir=work_unit_dir,
            target={
                'state': flax.jax_utils.unreplicate(state),
                'vars': helpers.state_to_variables(state),
                'origin': np.array(scene_origin.value),
            },
            step=l,
            keep=1,
            overwrite=True,
            keep_every_n_steps=config.get('keep_every_n_steps', None))
        logging.info('saved checkpoint to %s', saved_path)

      # Make a higher res, higher frame rate video.
      if config.render_hq_video and (config.get('hq_video_every', None) and
                                     i % config.hq_video_every == 0 or
                                     i == config.iters):

        my_rays = lambda c2w: scene.camera_ray_batch(c2w, *hwf_video_hq)
        th_range = np.linspace(0, 360, 240, endpoint=False)
        poses = [scene.pose_spherical(th, -30., 4.) for th in th_range]
        variables = helpers.state_to_variables(state)
        frames_all = [
            render_loop(my_rays(pose), variables, 1., config.mr1, hq=True)
            for pose in tqdm.tqdm(poses, 'render hq video')
        ]

        videos = [
            [onp.array(np.squeeze(f[j])) for f in frames_all] for j in range(3)
        ]
        meta_path = os.path.join(work_unit_dir, 'meta_hq.npy')
        with gfile.GFile(meta_path, 'wb') as f:
          logging.info('saving metadata for rendered hq frames to %s',
                       meta_path)
          onp.save(f, dict(poses=onp.array(poses), hwf=onp.array(hwf_video_hq)))
        for video, label in zip(videos, 'rgb depth acc'.split()):
          scale = (label == 'depth')
          log.log_video(
              None, video, 'frames_hq', label, i, work_unit_dir, scale=scale)

    writer.flush()
    writer.close()
    logging.info('%f sec elapsed total', time.time() - t_start)

  def render_from_checkpoint(self,
                             work_unit_dir,
                             widths,
                             render_test_hq_p,
                             step=None):
    """Restore learned radiance field weights and scene origin."""
    zero_outs = {
        width: [np.zeros((width, width, c)).squeeze() for c in [3, 1, 1, 3]
               ] for width in widths
    }
    latest_checkpoint = checkpoints.latest_checkpoint(work_unit_dir)
    if not latest_checkpoint:
      print(f'ERROR: no checkpoint found in {work_unit_dir}')
      return latest_checkpoint, zero_outs

    try:
      restored = checkpoints.restore_checkpoint(
          work_unit_dir, target=None, step=step)
    except ValueError as e:
      print(f'ERROR loading checkpoint from {work_unit_dir} at step {step}:', e)
      return latest_checkpoint, zero_outs
    variables = flax.core.frozen_dict.FrozenDict(restored['vars'])
    origin = restored['origin']
    if not np.all(np.isfinite(origin)):
      print('origin', origin, 'has nan value(s) for wu', work_unit_dir)

    # Render wrapper methods.
    def render_test(rays):
      sh = rays[0].shape
      rays = scene.padded_shard_rays(rays, multihost=False)
      out = render_test_hq_p(rays, variables, origin)
      out = [x.reshape((onp.prod(sh[:-1]), -1)) for x in out]  # gather flat
      out = [x[:sh[0]] for x in out]  # Unpad
      return out

    def render_loop(rays, chunk=2**16):
      sh = list(rays[0].shape[:-1])
      rays = [x.reshape((-1,) + x.shape[-1:]) for x in rays]
      outs = [
          render_test([x[i:i + chunk]
                       for x in rays])
          for i in range(0, rays[0].shape[0], chunk)
      ]
      outs = [
          np.reshape(np.concatenate([z[i]
                                     for z in outs]), sh + [-1])
          for i in range(3)
      ]
      return outs

    # Render validation view.
    renders_by_width = {}
    for width in set(widths):
      logging.info('rendering at width %d', width)
      hwf_clip_r = scene.scale_intrinsics(width)
      cam2world = scene.pose_spherical(30., -45., 4.)
      rays = scene.camera_ray_batch(cam2world, *hwf_clip_r)
      outs = render_loop(rays)
      outs = [np.squeeze(x) for x in outs]
      renders_by_width[width] = outs

    return latest_checkpoint, renders_by_width

  def run_eval(self,
               experiment_dir,
               rng,
               step=None,
               work_units=None,
               model_names_r=None,
               widths_r=None):
    """Evaluate models in experiment_dir for R-Precision."""
    logging.info('Local devices: %s', jax.local_devices())
    logging.info('All devices: %s', jax.devices())

    config = log.load_config_json(os.path.join(experiment_dir, '1'))
    logging.info('Config: %s', config)

    # Load retrieval models.
    if not model_names_r:
      model_names_r = config.retrieve_models
    models_r = [
        helpers.load_image_text_model(name)
        for name in tqdm.tqdm(model_names_r, desc='loading retrieval models')
    ]
    if not widths_r:
      widths_r = config.retrieve_widths

    print('model_names_r', model_names_r)
    print('widths_r', widths_r)

    # Encode retrieval set text descriptions.
    z_clip_rs = []  # text encodings of queries with all retrieval models
    # shape: [n_models, n_queries, d_model for specific model]
    if config.queries_r:
      for _, encode_text, _, tokenize_fn in tqdm.tqdm(
          models_r, desc='embedding queries with retrieval models'):
        z_clip_r = encode_text(tokenize_fn(config.queries_r))
        z_clip_rs.append(z_clip_r)

    # JIT rendering.
    kwargs_test = dict(rng=None, sigma_noise_std=0.)
    config_test_hq = ml_collections.ConfigDict(config)
    config_test_hq.update(config.test_hq)
    _, render_rays = helpers.init_nerf_model(rng.advance(1), config)

    @functools.partial(jax.pmap, in_axes=(0, None, None))
    def render_test_hq_p(rays, variables, origin):
      return render_rays(
          rays=rays,
          variables=variables,
          config=config_test_hq,
          sc=1.,
          mask_rad=config_test_hq.mr1,
          origin=origin,
          **kwargs_test)[0]

    # Render
    if work_units is None:
      work_units = gfile.listdir(experiment_dir)
      work_units = [int(wu) for wu in work_units if wu.isnumeric()]
    work_units.sort()
    work_unit_queries = []
    work_unit_configs = []
    n_wu = len(work_units)
    # create resolution -> n_wu -> 4ximg mapping
    all_renders_by_width = collections.defaultdict(list)
    for work_unit in tqdm.tqdm(work_units, 'Rendering all work units'):
      # Load query used to generate this object
      work_unit_dir = os.path.join(experiment_dir, str(work_unit))
      wu_config = log.load_config_json(work_unit_dir)
      work_unit_configs.append(wu_config)
      work_unit_queries.append(wu_config.query)  # not templated

      # Render the object
      _, renders = self.render_from_checkpoint(work_unit_dir, widths_r,
                                               render_test_hq_p, step)
      for width, render in renders.items():
        all_renders_by_width[width].append(render)

    print('all_renders_by_width keys', list(all_renders_by_width.keys()))

    def aggregate(raw):
      raw = onp.array(raw).astype(onp.float)
      return {
          'mean': onp.mean(raw),
          'sem': stats.sem(raw),
          'raw': raw,
      }

    metrics = {
        'renders_by_width': jax.tree_map(onp.array, dict(all_renders_by_width)),
        'work_unit_configs': work_unit_configs,
        'work_unit_queries': work_unit_queries,
    }

    ## Embed images with all retrieval models
    pbar = tqdm.tqdm(
        zip(model_names_r, widths_r, z_clip_rs, models_r),
        desc='Embedding renderings',
        total=len(model_names_r))
    for model_name, width, z_text, (encode_image, _, preprocess, _) in pbar:
      renders = all_renders_by_width[width]
      rgbs = np.array([rgb for rgb, _, _, _ in renders])
      print('about to encode rgbs with shape', rgbs.shape)
      print('  model_name', model_name)
      print('  width', width)
      z_est = encode_image(preprocess(rgbs))

      assert z_est.shape[0] == n_wu
      assert z_text.shape[0] == len(config.queries_r)
      cosine_sim = (z_est[:, None] * z_text[None]).sum(-1)  # [n_wu, queries_r]
      idx_true = np.array(
          [config.queries_r.index(query) for query in work_unit_queries])
      cosine_sim_true = np.take_along_axis(
          cosine_sim, idx_true[:, None], axis=1).squeeze(1)
      log_prob = nn.log_softmax(cosine_sim, axis=1)  # normalize over captions
      log_likelihood = np.take_along_axis(
          log_prob, idx_true[:, None], axis=1).squeeze(1)
      correct = np.argmax(cosine_sim, axis=1) == idx_true
      metrics[model_name] = {
          'val/retrieve_cosine_sim': aggregate(cosine_sim_true),
          'val/retrieve_loss': aggregate(-log_likelihood),
          'val/retrieve_acc': aggregate(correct),
      }

    metrics_path = os.path.join(experiment_dir, 'metrics.npy')
    with gfile.GFile(metrics_path, 'wb') as f:
      logging.info('Writing metrics to %s', metrics_path)
      onp.save(f, metrics)

    for k, v in metrics.items():
      if k not in ('renders_by_width', 'work_unit_configs'):
        logging.info('Metric %s: %s', k, v)

    return metrics


def run_train(*, config, experiment_dir,
              work_unit_dir, rng):
  for _ in DreamField(config).run_train(
      experiment_dir=experiment_dir,
      work_unit_dir=work_unit_dir,
      rng=rng,
      yield_results=False):
    pass


def run_eval(*,
             experiment_dir,
             rng,
             step = None,
             model_names_r=None,
             widths_r=None):
  return DreamField(None).run_eval(
      experiment_dir=experiment_dir,
      rng=rng,
      step=step,
      model_names_r=model_names_r,
      widths_r=widths_r)
