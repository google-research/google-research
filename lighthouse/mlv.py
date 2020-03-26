# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Functions for training multiscale volumetric lighting prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow.compat.v1 as tf

import lighthouse.geometry.projector as pj
import lighthouse.nets as nets


class MLV(object):
  """Class definition for Multiscale Lighting Volume learning module."""

  def __init__(self):
    pass

  def infer_mpi(self, src_images, ref_image, ref_pose, src_poses, intrinsics,
                psv_planes):
    """Construct the MPI inference graph.

    Args:
      src_images: stack of source images [batch, height, width, 3*#source]
      ref_image: reference image [batch, height, width, 3]
      ref_pose: reference frame pose (camera to world) [batch, 4, 4]
      src_poses: source frame poses (camera to world) [batch, 4, 4, #source]
      intrinsics: camera intrinsics [batch, 3, 3]
      psv_planes: list of depth of PSV planes

    Returns:
      outputs: a collection of output tensors.
    """

    with tf.name_scope('format_network_input'):
      net_input = self.format_network_input(ref_image, src_images, ref_pose,
                                            src_poses, psv_planes, intrinsics)

    with tf.name_scope('layer_prediction'):
      # generate entire MPI (training and inference, but takes more memory)
      rgba_layers = nets.mpi_net(net_input)

    # Collect output tensors
    pred = {}
    pred['rgba_layers'] = rgba_layers
    pred['psv'] = net_input

    # add pred tensors to outputs collection
    for i in pred:
      tf.add_to_collection('outputs', pred[i])

    return pred

  def mpi_render_view(self, input_mpi, ref_pose, tgt_pose, planes, intrinsics):
    """Render a target view from MPI representation.

    Args:
      input_mpi: input MPI [batch, height, width, #planes, 4]
      ref_pose: reference camera pose [batch, 4, 4]
      tgt_pose: target pose to render from [batch, 4, 4]
      planes: list of depths for each plane
      intrinsics: camera intrinsics [batch, 3, 3]

    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose.get_shape().as_list()
    num_planes = tf.shape(planes)[0]
    height = tf.shape(input_mpi)[1]
    width = tf.shape(input_mpi)[2]

    rgba_layers = input_mpi

    # render target viewpoint
    filler = tf.concat(
        [tf.zeros([batch_size, 1, 3]),
         tf.ones([batch_size, 1, 1])], axis=2)
    intrinsics_filler = tf.stack(
        [tf.to_float(height),
         tf.to_float(width), intrinsics[0, 0, 0]], axis=0)[:, tf.newaxis]

    ref_pose_c2w = ref_pose
    ref_pose_c2w = tf.concat([
        tf.concat([
            ref_pose_c2w[:, :3, 0:1], ref_pose_c2w[:, :3, 1:2],
            -1.0 * ref_pose_c2w[:, :3, 2:3], ref_pose_c2w[:, :3, 3:]
        ],
                  axis=2), filler
    ],
                             axis=1)
    ref_pose_c2w = tf.concat([ref_pose_c2w[0, :3, :], intrinsics_filler],
                             axis=1)

    tgt_pose_c2w = tgt_pose
    tgt_pose_c2w = tf.concat([
        tf.concat([
            tgt_pose_c2w[:, :3, 0:1], tgt_pose_c2w[:, :3, 1:2],
            -1.0 * tgt_pose_c2w[:, :3, 2:3], tgt_pose_c2w[:, :3, 3:]
        ],
                  axis=2), filler
    ],
                             axis=1)
    tgt_pose_c2w = tf.concat([tgt_pose_c2w[0, :3, :], intrinsics_filler],
                             axis=1)

    rendering, alpha_acc, accum = pj.render_mpi_homogs(
        rgba_layers,
        ref_pose_c2w,
        tgt_pose_c2w,
        1.0 / planes[0],
        1.0 / planes[-1],
        num_planes,
        debug=False)

    return rendering, alpha_acc, accum

  def img2mpi(self, img, depth, planedepths):
    """Compute ground truth MPI of visible content using depth map."""

    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    num_depths = planedepths.shape[0]
    depth_inds = (tf.to_float(num_depths) - 1) * (
        (1.0 / depth) - (1.0 / planedepths[0])) / ((1.0 / planedepths[-1]) -
                                                   (1.0 / planedepths[0]))
    depth_inds = tf.round(depth_inds)
    depth_inds_tile = tf.to_int32(
        tf.tile(depth_inds[:, :, :, tf.newaxis], [1, 1, 1, num_depths]))
    _, _, d = tf.meshgrid(
        tf.range(height), tf.range(width), tf.range(num_depths), indexing='ij')
    mpi_colors = tf.to_float(
        tf.tile(img[:, :, :, tf.newaxis, :], [1, 1, 1, num_depths, 1]))
    mpi_alphas = tf.to_float(
        tf.where(
            tf.equal(depth_inds_tile, d), tf.ones_like(depth_inds_tile),
            tf.zeros_like(depth_inds_tile)))
    mpi = tf.concat([mpi_colors, mpi_alphas[Ellipsis, tf.newaxis]], axis=4)
    return mpi

  def predict_lighting_vol(self,
                           mpi,
                           planes,
                           intrinsics,
                           cube_res,
                           scale_factors,
                           depth_clip=20.0):
    """Predict lighting volumes from MPI.

    Args:
      mpi: input mpi
      planes: input mpi plane depths
      intrinsics: ref camera intrinsics
      cube_res: resolution of cube volume for lighting prediction
      scale_factors: scales for multiresolution cube sampling
      depth_clip: farthest depth (sets limits of coarsest cube)

    Returns:
      list of completed lighting volumes
    """

    batchsize = tf.shape(mpi)[0]

    max_depth = tf.minimum(planes[0], depth_clip)

    cube_side_lengths = [2.0 * max_depth]
    for i in range(len(scale_factors)):
      cube_side_lengths.append(2.0 * max_depth / scale_factors[i])

    # shape of each cube's footprint within the next coarser volume
    cube_rel_shapes = []
    for i in range(len(scale_factors)):
      if i == 0:
        i_rel_shape = cube_res // scale_factors[0]
      else:
        i_rel_shape = (cube_res * scale_factors[i - 1]) // scale_factors[i]
      cube_rel_shapes.append(i_rel_shape)

    cube_centers = [tf.zeros([batchsize, 3])]
    for i in range(len(scale_factors)):
      i_center_depth = (cube_side_lengths[i] / (cube_res - 1)) * (
          cube_rel_shapes[i] // 2)
      cube_centers.append(
          tf.concat([
              tf.zeros([batchsize, 2]), i_center_depth * tf.ones([batchsize, 1])
          ],
                    axis=1))

    cube_nest_inds = []
    for i in range(len(scale_factors)):
      if i == 0:
        i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2,
                       (cube_res - cube_rel_shapes[i]) // 2,
                       cube_res // 2 - cube_rel_shapes[i]]
      else:
        i_nest_inds = [(cube_res - cube_rel_shapes[i]) // 2,
                       (cube_res - cube_rel_shapes[i]) // 2,
                       cube_res - cube_rel_shapes[i]]
      cube_nest_inds.append(i_nest_inds)

    cube_list = []
    for i in range(len(cube_centers)):
      i_cube, _ = pj.mpi_resample_cube(mpi, cube_centers[i], intrinsics, planes,
                                       cube_side_lengths[i], cube_res)

      cube_list.append(i_cube)
    return cube_list, cube_centers, cube_side_lengths, cube_rel_shapes, cube_nest_inds

  def render_envmap(self, cubes, cube_centers, cube_side_lengths,
                    cube_rel_shapes, cube_nest_inds, ref_pose, env_pose,
                    theta_res, phi_res, r_res):
    """Render environment map from volumetric lights.

    Args:
      cubes: input list of cubes in multiscale volume
      cube_centers: position of cube centers
      cube_side_lengths: side lengths of cubes
      cube_rel_shapes: size of "footprint" of each cube within next coarser cube
      cube_nest_inds: indices for cube "footprints"
      ref_pose: c2w pose of ref camera
      env_pose: c2w pose of environment map camera
      theta_res: resolution of theta (width) for environment map
      phi_res: resolution of phi (height) for environment map
      r_res: number of spherical shells to sample for environment map rendering

    Returns:
      An environment map at the input pose
    """
    num_scales = len(cubes)

    env_c2w = env_pose
    env2ref = tf.matmul(tf.matrix_inverse(ref_pose), env_c2w)

    # cube-->sphere resampling
    all_shells_list = []
    all_rad_list = []
    for i in range(num_scales):
      if i == num_scales - 1:
        # "finest" resolution cube, don't zero out
        cube_removed = cubes[i]
      else:
        # zero out areas covered by finer resolution cubes
        cube_shape = cubes[i].get_shape().as_list()[1]

        zm_y, zm_x, zm_z = tf.meshgrid(
            tf.range(cube_nest_inds[i][0],
                     cube_nest_inds[i][0] + cube_rel_shapes[i]),
            tf.range(cube_nest_inds[i][1],
                     cube_nest_inds[i][1] + cube_rel_shapes[i]),
            tf.range(cube_nest_inds[i][2],
                     cube_nest_inds[i][2] + cube_rel_shapes[i]),
            indexing='ij')
        inds = tf.stack([zm_y, zm_x, zm_z], axis=-1)
        updates = tf.to_float(tf.ones_like(zm_x))
        zero_mask = 1.0 - tf.scatter_nd(
            inds, updates, shape=[cube_shape, cube_shape, cube_shape])
        cube_removed = zero_mask[tf.newaxis, :, :, :, tf.newaxis] * cubes[i]

      spheres_i, rad_i = pj.spherical_cubevol_resample(cube_removed, env2ref,
                                                       cube_centers[i],
                                                       cube_side_lengths[i],
                                                       phi_res, theta_res,
                                                       r_res)
      all_shells_list.append(spheres_i)
      all_rad_list.append(rad_i)

    all_shells = tf.concat(all_shells_list, axis=3)
    all_rad = tf.concat(all_rad_list, axis=0)
    all_shells = pj.interleave_shells(all_shells, all_rad)
    all_shells_envmap = pj.over_composite(all_shells)

    return all_shells_envmap, all_shells_list

  def build_train_graph(self,
                        inputs,
                        min_depth,
                        max_depth,
                        cube_res,
                        theta_res,
                        phi_res,
                        r_res,
                        scale_factors,
                        num_mpi_planes,
                        learning_rate=0.0001,
                        vgg_model_weights=None,
                        global_step=0,
                        depth_clip=20.0):
    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the PSV and MPI planes
      max_depth: maximum depth for the PSV and MPI planes
      cube_res: per-side cube resolution
      theta_res: environment map width
      phi_res: environment map height
      r_res: number of radii to use when sampling spheres for rendering
      scale_factors: downsampling factors of cubes relative to the coarsest
      num_mpi_planes: number of MPI planes to infer
      learning_rate: learning rate
      vgg_model_weights: vgg weights (needed when vgg loss is used)
      global_step: training iteration
      depth_clip: maximum depth for coarsest resampled volumes

    Returns:
      A train_op to be used for training.
    """
    with tf.name_scope('setup'):
      psv_planes = pj.inv_depths(min_depth, max_depth, num_mpi_planes)
      mpi_planes = pj.inv_depths(min_depth, max_depth, num_mpi_planes)

    with tf.name_scope('input_data'):

      tgt_image = inputs['tgt_image']
      ref_image = inputs['ref_image']
      src_images = inputs['src_images']
      env_image = inputs['env_image']

      ref_depth = inputs['ref_depth']

      tgt_pose = inputs['tgt_pose']
      ref_pose = inputs['ref_pose']
      src_poses = inputs['src_poses']
      env_pose = inputs['env_pose']

      intrinsics = inputs['intrinsics']

      _, _, _, num_source = src_poses.get_shape().as_list()

    with tf.name_scope('inference'):
      num_mpi_planes = tf.shape(mpi_planes)[0]
      pred = self.infer_mpi(src_images, ref_image, ref_pose, src_poses,
                            intrinsics, psv_planes)
      rgba_layers = pred['rgba_layers']
      psv = pred['psv']

    with tf.name_scope('synthesis'):
      output_image, output_alpha_acc, _ = self.mpi_render_view(
          rgba_layers, ref_pose, tgt_pose, mpi_planes, intrinsics)
    with tf.name_scope('environment_rendering'):
      mpi_gt = self.img2mpi(ref_image, ref_depth, mpi_planes)
      output_image_gt, _, _ = self.mpi_render_view(mpi_gt, ref_pose, tgt_pose,
                                                   mpi_planes, intrinsics)

      lightvols_gt, _, _, _, _ = self.predict_lighting_vol(
          mpi_gt,
          mpi_planes,
          intrinsics,
          cube_res,
          scale_factors,
          depth_clip=depth_clip)

      lightvols, lightvol_centers, \
      lightvol_side_lengths, \
      cube_rel_shapes, \
      cube_nest_inds = self.predict_lighting_vol(rgba_layers, mpi_planes,
                                                 intrinsics, cube_res,
                                                 scale_factors,
                                                 depth_clip=depth_clip)

      lightvols_out = nets.cube_net_multires(lightvols, cube_rel_shapes,
                                             cube_nest_inds)

      gt_envmap, gt_shells = self.render_envmap(lightvols_gt, lightvol_centers,
                                                lightvol_side_lengths,
                                                cube_rel_shapes, cube_nest_inds,
                                                ref_pose, env_pose, theta_res,
                                                phi_res, r_res)

      prenet_envmap, prenet_shells = self.render_envmap(
          lightvols, lightvol_centers, lightvol_side_lengths, cube_rel_shapes,
          cube_nest_inds, ref_pose, env_pose, theta_res, phi_res, r_res)

      output_envmap, output_shells = self.render_envmap(
          lightvols_out, lightvol_centers, lightvol_side_lengths,
          cube_rel_shapes, cube_nest_inds, ref_pose, env_pose, theta_res,
          phi_res, r_res)

    with tf.name_scope('loss'):
      # mask loss for pixels outside reference frustum
      loss_mask = tf.where(
          tf.equal(output_alpha_acc[Ellipsis, tf.newaxis], 0.0),
          tf.zeros_like(output_image[:, :, :, 0:1]),
          tf.ones_like(output_image[:, :, :, 0:1]))
      loss_mask = tf.stop_gradient(loss_mask)
      tf.summary.image('loss_mask', loss_mask)

      # helper functions for loss
      def compute_error(real, fake, mask):
        mask = tf.ones_like(real) * mask
        return tf.reduce_sum(mask * tf.abs(fake - real)) / (
            tf.reduce_sum(mask) + 1.0e-8)

      # Normalized VGG loss
      def downsample(tensor, ds):
        return tf.nn.avg_pool(tensor, [1, ds, ds, 1], [1, ds, ds, 1], 'SAME')

      def vgg_loss(tgt_image, output_image, loss_mask, vgg_weights):
        """VGG activation loss definition."""

        vgg_real = nets.build_vgg19(tgt_image * 255.0, vgg_weights)
        rescaled_output_image = output_image * 255.0
        vgg_fake = nets.build_vgg19(rescaled_output_image, vgg_weights)
        p0 = compute_error(vgg_real['input'], vgg_fake['input'], loss_mask)
        p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2'],
                           loss_mask) / 2.6
        p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2'],
                           downsample(loss_mask, 2)) / 4.8
        p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2'],
                           downsample(loss_mask, 4)) / 3.7
        p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2'],
                           downsample(loss_mask, 8)) / 5.6
        p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2'],
                           downsample(loss_mask, 16)) * 10 / 1.5
        total_loss = p0 + p1 + p2 + p3 + p4 + p5
        return total_loss

      # rendered image loss
      render_loss = vgg_loss(tgt_image, output_image, loss_mask,
                             vgg_model_weights) / 100.0
      total_loss = render_loss

      # rendered envmap loss
      envmap_loss = vgg_loss(env_image, output_envmap[Ellipsis, :3],
                             tf.ones_like(env_image[Ellipsis, 0:1]),
                             vgg_model_weights) / 100.0

      # set envmap loss to 0 when only training mpi network (see paper)
      envmap_loss = tf.where(tf.greater(global_step, 240000), envmap_loss, 0.0)

      total_loss += envmap_loss

      # adversarial loss for envmap
      real_logit = nets.discriminator(env_image, scope='discriminator')
      fake_logit = nets.discriminator(
          output_envmap[Ellipsis, :3], scope='discriminator')
      adv_loss_list = []
      for i in range(len(fake_logit)):
        adv_loss_list.append(0.1 * -1.0 * tf.reduce_mean(fake_logit[i][-1]))
      adv_loss = tf.reduce_mean(adv_loss_list)
      real_loss_list = []
      fake_loss_list = []
      for i in range(len(fake_logit)):
        real_loss_list.append(
            -1.0 * tf.reduce_mean(tf.minimum(real_logit[i][-1] - 1, 0.0)))
        fake_loss_list.append(
            -1.0 *
            tf.reduce_mean(tf.minimum(-1.0 * fake_logit[i][-1] - 1, 0.0)))
      real_loss = tf.reduce_mean(real_loss_list)
      fake_loss = tf.reduce_mean(fake_loss_list)
      disc_loss = real_loss + fake_loss

      # set adv/disc losses to 0 until end of training
      adv_loss = tf.where(tf.greater(global_step, 690000), adv_loss, 0.0)
      disc_loss = tf.where(tf.greater(global_step, 690000), disc_loss, 0.0)

      tf.summary.scalar('loss_disc', disc_loss)
      tf.summary.scalar('loss_disc_real', real_loss)
      tf.summary.scalar('loss_disc_fake', fake_loss)
      tf.summary.scalar('loss_adv', adv_loss)

      total_loss += adv_loss

    with tf.name_scope('train_op'):
      train_variables = [
          var for var in tf.trainable_variables()
          if 'discriminator' not in var.name
      ]
      optim = tf.train.AdamOptimizer(learning_rate, epsilon=1e-4)
      grads_and_variables = optim.compute_gradients(
          total_loss, var_list=train_variables)
      grads = [gv[0] for gv in grads_and_variables]
      variables = [gv[1] for gv in grads_and_variables]

      def denan(x):
        return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

      grads_clipped = [denan(g) for g in grads]
      grads_clipped, _ = tf.clip_by_global_norm(grads_clipped, 100.0)
      train_op = [optim.apply_gradients(zip(grads_clipped, variables))]
      tf.summary.scalar('gradient global norm', tf.linalg.global_norm(grads))
      tf.summary.scalar('clipped gradient global norm',
                        tf.linalg.global_norm(grads_clipped))

      d_variables = [
          var for var in tf.trainable_variables() if 'discriminator' in var.name
      ]
      optim_d = tf.train.AdamOptimizer(learning_rate, beta1=0.0)
      train_op.append(optim_d.minimize(disc_loss, var_list=d_variables))

    with tf.name_scope('envmap_gt'):
      tf.summary.image('envmap', gt_envmap)
      tf.summary.image('envmap_alpha', gt_envmap[Ellipsis, -1:])
      for i in range(len(gt_shells)):
        i_envmap = pj.over_composite(gt_shells[i])
        tf.summary.image('envmap_level_' + str(i), i_envmap)
    with tf.name_scope('envmap_prenet'):
      tf.summary.image('envmap', prenet_envmap)
      tf.summary.image('envmap_alpha', prenet_envmap[Ellipsis, -1:])
      for i in range(len(prenet_shells)):
        i_envmap = pj.over_composite(prenet_shells[i])
        tf.summary.image('envmap_level_' + str(i), i_envmap)
    with tf.name_scope('envmap_output'):
      tf.summary.image('envmap', output_envmap)
      tf.summary.image('envmap_alpha', output_envmap[Ellipsis, -1:])
      for i in range(len(output_shells)):
        i_envmap = pj.over_composite(output_shells[i])
        tf.summary.image('envmap_level_' + str(i), i_envmap)

    tf.summary.scalar('loss_total', total_loss)
    tf.summary.scalar('loss_render', render_loss)
    tf.summary.scalar('loss_envmap', envmap_loss)
    tf.summary.scalar('min_depth', min_depth)
    tf.summary.scalar('max_depth', max_depth)

    with tf.name_scope('level_stats'):
      for i in range(len(lightvols)):
        tf.summary.scalar('cube_side_length_' + str(i),
                          lightvol_side_lengths[i])
        tf.summary.scalar('cube_center_' + str(i), lightvol_centers[i][0, -1])

    # Source images
    for i in range(num_source):
      src_image = src_images[:, :, :, i * 3:(i + 1) * 3]
      tf.summary.image('image_src_%d' % i, src_image)
    # Output image
    tf.summary.image('image_output', output_image)
    tf.summary.image('image_output_Gt', output_image_gt)
    # Target image
    tf.summary.image('image_tgt', tgt_image)
    tf.summary.image('envmap_tgt', env_image)
    # Ref image
    tf.summary.image('image_ref', ref_image)
    # Predicted color and alpha layers, and PSV
    num_summ = 8  # number of plane summaries to show in tensorboard
    for i in range(num_summ):
      ind = tf.to_int32(i * num_mpi_planes / num_summ)
      rgb = rgba_layers[:, :, :, ind, :3]
      alpha = rgba_layers[:, :, :, ind, -1:]
      ref_plane = psv[:, :, :, ind, :3]
      source_plane = psv[:, :, :, ind, 3:6]
      tf.summary.image('layer_rgb_%d' % i, rgb)
      tf.summary.image('layer_alpha_%d' % i, alpha)
      tf.summary.image('layer_rgba_%d' % i, rgba_layers[:, :, :, ind, :])
      tf.summary.image('psv_avg_%d' % i, 0.5 * ref_plane + 0.5 * source_plane)
      tf.summary.image('psv_ref_%d' % i, ref_plane)
      tf.summary.image('psv_source_%d' % i, source_plane)

    return train_op

  def train(self, train_op, load_dir, checkpoint_dir, summary_dir, summary_freq,
            checkpoint_freq, max_steps, global_step):
    """Runs the training procedure.

    Args:
      train_op: op for training the network
      load_dir: directory to load checkpoint for continuing training
      checkpoint_dir: where to save the model checkpoints
      summary_dir: where to save the tensorboard summaries
      summary_freq: summary frequency
      checkpoint_freq: Frequency of model saving
      max_steps: maximum training steps
      global_step: training iteration placeholder
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    step_start = 1

    with tf.Session(config=config) as sess:

      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
      saver = tf.train.Saver([var for var in tf.trainable_variables()],
                             max_to_keep=None)
      sess.run(tf.global_variables_initializer())

      checkpoint = tf.train.latest_checkpoint(load_dir)
      if checkpoint is not None:
        print('Resume training from previous checkpoint:', checkpoint)
        step_start = int(checkpoint.split('-')[-1])
        saver.restore(sess, checkpoint)

      print('starting training iters')

      for step in range(step_start, max_steps + 1):
        start_time = time.time()

        fetches = {'train': train_op}

        if step % summary_freq == 0:
          fetches['summary'] = merged

        results = sess.run(fetches, feed_dict={global_step: step})

        if step % summary_freq == 0:
          train_writer.add_summary(results['summary'], step)
          print('[Step %.8d] time: %4.4f/it' % (step, time.time() - start_time))

        if step % checkpoint_freq == 0:
          print('Saving checkpoint to %s...' % checkpoint_dir)
          saver.save(
              sess,
              os.path.join(checkpoint_dir, 'model.ckpt'),
              global_step=step)

  def format_network_input(self, ref_image, psv_src_images, ref_pose,
                           psv_src_poses, planes, intrinsics):
    """Format the network input.

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image) [batch,
        height, width, 3*(#source)]
      ref_pose: reference camera-to-world pose (where PSV is constructed)
        [batch, 4, 4]
      psv_src_poses: input poses (camera to world) [batch, 4, 4, #source]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]

    Returns:
      net_input: [batch, height, width, #planes, (#source+1)*3]
    """

    batch_size = tf.shape(psv_src_images)[0]
    height = tf.shape(psv_src_images)[1]
    width = tf.shape(psv_src_images)[2]
    _, _, _, num_psv_source = psv_src_poses.get_shape().as_list()
    num_planes = tf.shape(planes)[0]

    filler = tf.concat(
        [tf.zeros([batch_size, 1, 3]),
         tf.ones([batch_size, 1, 1])], axis=2)
    intrinsics_filler = tf.stack([
        tf.to_float(height),
        tf.to_float(width),
        tf.to_float(intrinsics[0, 0, 0])
    ],
                                 axis=0)[:, tf.newaxis]

    ref_pose_c2w = ref_pose
    ref_pose_c2w = tf.concat([
        tf.concat([
            ref_pose_c2w[:, :3, 0:1], ref_pose_c2w[:, :3, 1:2],
            -1.0 * ref_pose_c2w[:, :3, 2:3], ref_pose_c2w[:, :3, 3:]
        ],
                  axis=2), filler
    ],
                             axis=1)
    ref_pose_c2w = tf.concat([ref_pose_c2w[0, :3, :], intrinsics_filler],
                             axis=1)

    net_input = []
    for i in range(num_psv_source):
      curr_pose_c2w = psv_src_poses[:, :, :, i]
      curr_pose_c2w = tf.concat([
          tf.concat([
              curr_pose_c2w[:, :3, 0:1], curr_pose_c2w[:, :3, 1:2],
              -1.0 * curr_pose_c2w[:, :3, 2:3], curr_pose_c2w[:, :3, 3:]
          ], 2), filler
      ], 1)
      curr_pose_c2w = tf.concat([curr_pose_c2w[0, :3, :], intrinsics_filler],
                                axis=1)
      curr_image = psv_src_images[:, :, :, i * 3:(i + 1) * 3]
      curr_psv = pj.make_psv_homogs(curr_image, curr_pose_c2w, ref_pose_c2w,
                                    1.0 / planes, num_planes)
      net_input.append(curr_psv[tf.newaxis, Ellipsis])

    net_input = tf.concat(net_input, axis=4)
    ref_img_stack = tf.tile(
        tf.expand_dims(ref_image, 3), [1, 1, 1, num_planes, 1])
    net_input = tf.concat([ref_img_stack, net_input], axis=4)
    net_input.set_shape([1, None, None, None, 3 * (num_psv_source + 1)])

    return net_input
