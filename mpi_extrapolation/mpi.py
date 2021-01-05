# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Functions for learning to predict multiplane images (MPI).

For CVPR 2019 paper:
Pushing the Boundaries of View Extrapolation with Multiplane Images
Pratul P. Srinivasan, Richard Tucker, Jonathan T. Barron, Ravi Ramamoorthi, Ren
Ng, Noah Snavely
Modified from code written by Tinghui Zhou
(https://github.com/google/stereo-magnification).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow.compat.v1 as tf

import mpi_extrapolation.geometry.projector as pj
from mpi_extrapolation.nets import build_vgg19
from mpi_extrapolation.nets import ed_3d_net
from mpi_extrapolation.nets import refine_net


class MPI(object):
  """Class definition for MPI learning module.
  """

  def __init__(self):
    pass

  def infer_mpi(self, raw_src_images, raw_ref_image, ref_pose, src_poses,
                intrinsics, num_mpi_planes, mpi_planes,
                run_patched=False,
                patch_ind=np.array([0, 0]),
                patchsize=np.array([256, 256]),
                outsize=np.array([128, 128])):
    """Construct the MPI inference graph.

    Args:
      raw_src_images: stack of source images [batch, height, width, 3*#source]
      raw_ref_image: reference image [batch, height, width, 3]
      ref_pose: reference frame pose (world to camera) [batch, 4, 4]
      src_poses: source frame poses (world to camera) [batch, #source, 4, 4]
      intrinsics: camera intrinsics [batch, 3, 3]
      num_mpi_planes: number of mpi planes to predict
      mpi_planes: list of plane depths
      run_patched: whether to only infer MPI for patches of PSV (inference only)
      patch_ind: patch index for infer MPI inference
      patchsize: spatial patch size for MPI inference
      outsize: size of central portion to keep for patched inference
    Returns:
      outputs: a collection of output tensors.
    """

    with tf.name_scope("preprocessing"):
      src_images = self.preprocess_image(raw_src_images)
      ref_image = self.preprocess_image(raw_ref_image)

    with tf.name_scope("format_network_input"):
      # WARNING: we assume the first src image/pose is the reference
      net_input = self.format_network_input(ref_image, src_images[:, :, :, 3:],
                                            ref_pose, src_poses[:, 1:],
                                            mpi_planes, intrinsics)

    with tf.name_scope("layer_prediction"):
      # The network directly outputs the color image at each MPI plane.

      chout = 4  # Number of output channels, RGBA

      if run_patched:
        # Patch the PSV spatially, with buffer, and generate MPI patch
        # Only for inference (not implemented for training)
        buffersize = (patchsize - outsize) // 2
        padding = [[0, 0], [buffersize[0], buffersize[0]],
                   [buffersize[1], buffersize[1]], [0, 0], [0, 0]]
        net_input_pad = tf.pad(net_input, padding)
        patch_start = patch_ind * outsize
        patch_end = patch_start + patchsize
        net_input_patch = net_input_pad[:, patch_start[0]:patch_end[0],
                                        patch_start[1]:patch_end[1], :, :]
        rgba_layers, _ = ed_3d_net(net_input_patch, chout)
      else:
        # Generate entire MPI (training and inference, but takes more memory)
        print("first step MPI prediction")
        rgba_layers, _ = ed_3d_net(net_input, chout)

      color_layers = rgba_layers[:, :, :, :, :-1]
      alpha_layers = rgba_layers[:, :, :, :, -1:]
      # Rescale alphas to (0, 1)
      alpha_layers = (alpha_layers + 1.)/2.
      rgba_layers = tf.concat([color_layers, alpha_layers], axis=4)

      print("refining MPI")
      transmittance = self.compute_transmittance(alpha_layers)
      refine_input_colors = color_layers * transmittance
      refine_input_alpha = alpha_layers * transmittance
      stuff_behind = tf.cumsum(refine_input_colors, axis=3)
      concat_trans = True  # Concatenate transmittance to second input
      if concat_trans:
        refine_input = tf.concat([tf.stop_gradient(refine_input_colors),
                                  tf.stop_gradient(stuff_behind),
                                  tf.stop_gradient(refine_input_alpha),
                                  tf.stop_gradient(transmittance)], axis=4)

      normalized_disp_inds = tf.reshape(tf.linspace(0.0, 1.0, num_mpi_planes),
                                        [1, 1, 1, num_mpi_planes, 1])
      sh = tf.shape(refine_input)
      normalized_disp_inds_stack = tf.tile(normalized_disp_inds,
                                           [1, sh[1], sh[2], 1, 1])
      refine_input = tf.concat([refine_input, normalized_disp_inds_stack],
                               axis=4)
      print("refine input size:", refine_input.shape)
      rgba_layers_refine = refine_net(refine_input)

      print("predicting flow for occlusions")
      flow_source = tf.stop_gradient(stuff_behind)
      flow_vecs = rgba_layers_refine[:, :, :, :, :2]
      color_layers = pj.flow_gather(flow_source, flow_vecs)
      alpha_layers = rgba_layers_refine[:, :, :, :, -1:]
      # Rescale alphas to (0, 1)
      alpha_layers = (alpha_layers + 1.)/2.
      rgba_layers_refine = tf.concat([color_layers, alpha_layers], axis=4)

    # Collect output tensors
    pred = {}
    pred["rgba_layers"] = rgba_layers
    pred["rgba_layers_refine"] = rgba_layers_refine
    pred["refine_input_mpi"] = tf.concat([refine_input_colors,
                                          refine_input_alpha], axis=-1)
    pred["stuff_behind"] = stuff_behind
    pred["flow_vecs"] = flow_vecs
    pred["psv"] = net_input[:, :, :, :, 0:3]

    # Add pred tensors to outputs collection
    print("adding outputs to collection")
    for i in pred:
      tf.add_to_collection("outputs", pred[i])

    return pred

  def mpi_render_view(self, input_mpi, tgt_pose, planes, intrinsics):
    """Render a target view from MPI representation.

    Args:
      input_mpi: input MPI [batch, height, width, #planes, 4]
      tgt_pose: target pose (relative) to render from [batch, 4, 4]
      planes: list of depth for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      rendered view [batch, height, width, 3]
    """

    batch_size, _, _ = tgt_pose.get_shape().as_list()

    rgba_layers = input_mpi

    # Format for homography code
    depths = tf.tile(planes[:, tf.newaxis], [1, batch_size])
    rgba_layers = tf.transpose(rgba_layers, [3, 0, 1, 2, 4])

    # Render target viewpoint
    proj_images = pj.projective_forward_homography(
        rgba_layers, intrinsics, tgt_pose, depths)
    proj_images = tf.transpose(proj_images, [1, 2, 3, 0, 4])

    output_image = pj.over_composite(proj_images)
    output_image.set_shape([None, None, None, 3])

    return output_image, proj_images

  def build_train_graph(self,
                        inputs,
                        min_depth,
                        max_depth,
                        num_mpi_planes,
                        learning_rate=0.0002,
                        beta1=0.9,
                        vgg_model_file=None,
                        global_step=0):
    """Construct the training computation graph.

    Args:
      inputs: dictionary of tensors (see 'input_data' below) needed for training
      min_depth: minimum depth for the PSV and MPI planes
      max_depth: maximum depth for the PSV and MPI planes
      num_mpi_planes: number of MPI planes to infer
      learning_rate: learning rate
      beta1: hyperparameter for Adam
      vgg_model_file: path to vgg weights (needed when vgg loss is used)
      global_step: current optimization step
    Returns:
      A train_op to be used for training.
    """
    print("starting to build graph")
    with tf.name_scope("input_size_randomization"):
      dim_choices = tf.constant([[1, 16], [2, 32], [4, 32], [4, 64], [4, 128],
                                 [8, 32], [8, 64], [8, 128]],
                                dtype=tf.int32)
      rand_dim = tf.random_shuffle(dim_choices)[0, :]
      height_div = rand_dim[0]
      width_div = rand_dim[0]
      num_mpi_planes = rand_dim[1]
      tf.summary.scalar("num_mpi_planes", num_mpi_planes)

    with tf.name_scope("setup"):
      mpi_planes = self.inv_depths(min_depth, max_depth, num_mpi_planes)

    with tf.name_scope("input_data"):
      raw_tgt_image = inputs["tgt_image"]
      raw_ref_image = inputs["ref_image"]
      raw_src_images = inputs["src_images"]

      _, img_height, img_width, _ = raw_src_images.get_shape().as_list(
      )
      img_height = img_height // height_div
      img_width = img_width // width_div

      raw_tgt_image = tf.image.convert_image_dtype(
          raw_tgt_image, dtype=tf.float32)
      raw_ref_image = tf.image.convert_image_dtype(
          raw_ref_image, dtype=tf.float32)
      raw_src_images = tf.image.convert_image_dtype(
          raw_src_images, dtype=tf.float32)
      raw_tgt_image = tf.image.resize_area(raw_tgt_image,
                                           [img_height, img_width])
      raw_ref_image = tf.image.resize_area(raw_ref_image,
                                           [img_height, img_width])
      raw_src_images = tf.image.resize_area(raw_src_images,
                                            [img_height, img_width])

      tgt_pose = inputs["tgt_pose"]
      ref_pose = inputs["ref_pose"]
      src_poses = inputs["src_poses"]
      intrinsics = inputs["intrinsics"]

      # Scale intrinsics based on size randomization
      intrinsics = tf.concat([
          intrinsics[:, 0:1, :] / tf.to_float(width_div),
          intrinsics[:, 1:2, :] / tf.to_float(height_div), intrinsics[:, 2:3, :]
      ],
                             axis=1)
      inputs["intrinsics"] = intrinsics

      _, num_source, _, _ = src_poses.get_shape().as_list()

    with tf.name_scope("inference"):
      print("setting up MPI inference")
      num_mpi_planes = tf.shape(mpi_planes)[0]
      pred = self.infer_mpi(raw_src_images, raw_ref_image, ref_pose, src_poses,
                            intrinsics, num_mpi_planes,
                            mpi_planes)
      rgba_layers = pred["rgba_layers"]
      rgba_layers_refine = pred["rgba_layers_refine"]
      stuff_behind = pred["stuff_behind"]
      refine_input_mpi = pred["refine_input_mpi"]
      psv = pred["psv"]

    with tf.name_scope("synthesis"):
      print("setting up rendering")
      rel_pose = tf.matmul(tgt_pose, tf.matrix_inverse(ref_pose))
      output_image, output_layers = self.mpi_render_view(
          rgba_layers, rel_pose, mpi_planes, intrinsics)
      output_alpha = output_layers[Ellipsis, -1]
      output_image_refine, _ = self.mpi_render_view(
          rgba_layers_refine, rel_pose, mpi_planes, intrinsics)

    with tf.name_scope("loss"):
      print("computing losses")
      # Mask loss for pixels outside reference frustum
      loss_mask = tf.where(
          tf.equal(
              tf.reduce_min(
                  tf.abs(tf.reduce_sum(output_layers, axis=-1)),
                  axis=3,
                  keep_dims=True), 0.0),
          tf.zeros_like(output_alpha[:, :, :, 0:1]),
          tf.ones_like(output_alpha[:, :, :, 0:1]))
      loss_mask = tf.stop_gradient(loss_mask)
      tf.summary.image("loss_mask", loss_mask)

      # Helper functions for loss
      def compute_error(real, fake, mask):
        return tf.reduce_mean(mask * tf.abs(fake - real))

      # Normalized VGG loss (from
      # https://github.com/CQFIO/PhotographicImageSynthesis)

      downsample = lambda tensor, ds: tf.nn.avg_pool(tensor, [1, ds, ds, 1],
                                                     [1, ds, ds, 1], "SAME")

      def vgg_loss(raw_tgt_image, output_image, loss_mask):
        """Compute VGG loss."""

        vgg_real = build_vgg19(raw_tgt_image * 255.0, vgg_model_file)
        rescaled_output_image = (output_image + 1.)/2. * 255.0
        vgg_fake = build_vgg19(
            rescaled_output_image, vgg_model_file, reuse=True)
        p0 = compute_error(vgg_real["input"], vgg_fake["input"], loss_mask)
        p1 = compute_error(vgg_real["conv1_2"],
                           vgg_fake["conv1_2"],
                           loss_mask)/2.6
        p2 = compute_error(vgg_real["conv2_2"],
                           vgg_fake["conv2_2"],
                           downsample(loss_mask, 2))/4.8
        p3 = compute_error(vgg_real["conv3_2"],
                           vgg_fake["conv3_2"],
                           downsample(loss_mask, 4))/3.7
        p4 = compute_error(vgg_real["conv4_2"],
                           vgg_fake["conv4_2"],
                           downsample(loss_mask, 8))/5.6
        p5 = compute_error(vgg_real["conv5_2"],
                           vgg_fake["conv5_2"],
                           downsample(loss_mask, 16))*10/1.5
        total_loss = p0+p1+p2+p3+p4+p5
        return total_loss, vgg_real, vgg_fake

      vgg_loss_initial, _, _ = vgg_loss(raw_tgt_image, output_image, loss_mask)
      tf.summary.scalar("vgg_loss_initial", vgg_loss_initial)
      total_loss = vgg_loss_initial

      vgg_loss_refine, _, _ = vgg_loss(raw_tgt_image, output_image_refine,
                                       loss_mask)
      tf.summary.scalar("vgg_loss_refine", vgg_loss_refine)
      total_loss += vgg_loss_refine

    with tf.name_scope("train_op"):
      print("setting up train op")
      train_vars = [var for var in tf.trainable_variables()]
      optim = tf.train.AdamOptimizer(learning_rate, beta1)
      grads_and_vars = optim.compute_gradients(total_loss, var_list=train_vars)
      train_op = [optim.apply_gradients(grads_and_vars)]

    # Summaries
    tf.summary.scalar("total_loss", total_loss)
    # Source images
    for i in range(num_source):
      src_image = raw_src_images[:, :, :, i*3:(i+1)*3]
      tf.summary.image("src_image_%d" % i, src_image)
    # Output image
    tf.summary.image("output_image", self.deprocess_image(output_image))
    # Refined output image
    tf.summary.image("output_image_refine",
                     self.deprocess_image(output_image_refine))
    # Target image
    tf.summary.image("tgt_image", raw_tgt_image)
    # Ref image
    tf.summary.image("ref_image", raw_ref_image)
    # Predicted color and alpha layers, and PSV
    num_summ = 16  # Number of plane summaries to show in tensorboard
    for i in range(num_summ):
      ind = tf.to_int32(i * num_mpi_planes/num_summ)
      rgb = rgba_layers[:, :, :, ind, :3]
      alpha = rgba_layers[:, :, :, ind, -1:]
      ref_plane = psv[:, :, :, ind, 3:6]
      source_plane = psv[:, :, :, ind, :3]
      output_rgb = output_layers[:, :, :, ind, :3]
      tf.summary.image("rgb_layer_%d" % i, self.deprocess_image(rgb))
      tf.summary.image("alpha_layer_%d" % i, alpha)
      tf.summary.image("rgba_layer_%d" % i, self.deprocess_image(rgb * alpha))
      tf.summary.image("psv_avg_%d" % i,
                       (self.deprocess_image(0.5*ref_plane + 0.5*source_plane)))
      tf.summary.image("output_rgb_%d" % i,
                       self.deprocess_image(output_rgb))
      tf.summary.image("psv_ref_%d" % i, self.deprocess_image(ref_plane))
      tf.summary.image("psv_source_%d" % i, self.deprocess_image(source_plane))

    # Cumulative rendered images and refined MPI
    for i in range(num_summ):
      ind = tf.to_int32(i * num_mpi_planes/num_summ)
      rgb = rgba_layers_refine[:, :, :, ind, :3]
      alpha = rgba_layers_refine[:, :, :, ind, 3:]
      render = stuff_behind[:, :, :, ind, :3]
      input_colors = refine_input_mpi[:, :, :, ind, :3]
      tf.summary.image("rgb_layer_refine_%d" % i, self.deprocess_image(rgb))
      tf.summary.image("alpha_layer_refine_%d" % i, alpha)
      tf.summary.image("rgba_layer_refine_%d" % i,
                       self.deprocess_image(rgb * alpha))
      tf.summary.image("cumulative_render_%d" % i, self.deprocess_image(render))
      tf.summary.image("input_colors_refine_%d" % i,
                       self.deprocess_image(input_colors))

    return train_op

  def train(self, train_op, load_dir, checkpoint_dir, summary_dir,
            continue_train, summary_freq, save_latest_freq, max_steps,
            global_step):
    """Runs the training procedure.

    Args:
      train_op: op for training the network
      load_dir: where to load pretrained model
      checkpoint_dir: where to save the model checkpoints
      summary_dir: where to save the tensorboard summaries
      continue_train: whether to restore training from previous checkpoint
      summary_freq: summary frequency
      save_latest_freq: Frequency of model saving
      max_steps: maximum training steps
      global_step: tf Variable for current optimization step
    """
    # parameter_count = tf.reduce_sum(
    #     [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
    incr_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver([var for var in tf.trainable_variables()] +
                           [global_step],
                           max_to_keep=None)
    sv = tf.train.Supervisor(logdir=summary_dir, save_summaries_secs=0,
                             saver=None)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with sv.managed_session("local", config=config) as sess:
      if continue_train:
        checkpoint = tf.train.latest_checkpoint(load_dir)
        if checkpoint is not None:
          print("Resume training from previous checkpoint:", checkpoint)
          saver.restore(sess, checkpoint)
      print("starting training iters")
      for step in range(1, max_steps):
        start_time = time.time()
        fetches = {
            "train": train_op,
            "global_step": global_step,
            "incr_global_step": incr_global_step,
        }
        if step % summary_freq == 0:
          fetches["summary"] = sv.summary_op
        results = sess.run(fetches)

        gs = results["global_step"]

        if step % summary_freq == 0:
          sv.summary_writer.add_summary(results["summary"], gs)
          print("[Step %.8d] time: %4.4f/it" % (gs, time.time() - start_time))

        if step % save_latest_freq == 0:
          print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
          saver.save(
              sess, os.path.join(checkpoint_dir, "model.ckpt"), global_step=gs)

  def format_network_input(self, ref_image, psv_src_images, ref_pose,
                           psv_src_poses, planes, intrinsics):
    """Format the network input.

    Args:
      ref_image: reference source image [batch, height, width, 3]
      psv_src_images: stack of source images (excluding the ref image)
                      [batch, height, width, 3*(num_source -1)]
      ref_pose: reference world-to-camera pose (where PSV is constructed)
                [batch, 4, 4]
      psv_src_poses: input poses (world to camera) [batch, num_source-1, 4, 4]
      planes: list of scalar depth values for each plane
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      net_input: [batch, height, width, #planes, num_source*3]
    """
    _, num_psv_source, _, _ = psv_src_poses.get_shape().as_list()
    num_planes = tf.shape(planes)[0]

    net_input = []
    for i in range(num_psv_source):
      curr_pose = tf.matmul(psv_src_poses[:, i], tf.matrix_inverse(ref_pose))
      curr_image = psv_src_images[:, :, :, i*3:(i+1)*3]
      curr_psv = pj.plane_sweep(curr_image, planes, curr_pose, intrinsics)
      net_input.append(curr_psv)

    net_input = tf.concat(net_input, axis=4)
    ref_img_stack = tf.tile(
        tf.expand_dims(ref_image, 3), [1, 1, 1, num_planes, 1])
    net_input = tf.concat([net_input, ref_img_stack], axis=4)

    # Append normalized plane indices
    normalized_disp_inds = tf.reshape(tf.linspace(0.0, 1.0, num_planes),
                                      [1, 1, 1, num_planes, 1])
    sh = tf.shape(net_input)
    normalized_disp_inds_stack = tf.tile(normalized_disp_inds,
                                         [1, sh[1], sh[2], 1, 1])
    net_input = tf.concat([net_input, normalized_disp_inds_stack], axis=4)

    return net_input

  def preprocess_image(self, image):
    """Preprocess the image for CNN input.

    Args:
      image: the input image in either float [0, 1] or uint8 [0, 255]
    Returns:
      A new image converted to float with range [-1, 1]
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image * 2.0 - 1.0

  def deprocess_image(self, image):
    """Undo the preprocessing.

    Args:
      image: the input image in float with range [-1, 1]
    Returns:
      A new image converted to uint8 [0, 255]
    """
    image = (image + 1.)/2.
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)

  def inv_depths(self, start_depth, end_depth, num_depths):
    """Returns reversed, sorted inverse interpolated depths.

    Args:
      start_depth: The first depth.
      end_depth: The last depth.
      num_depths: The total number of depths to create, include start_depth and
          end_depth are always included and other depths are interpolated
          between them, in inverse depth space.
    Returns:
      The depths sorted in descending order (so furthest first). This order is
      useful for back to front compositing.
    """
    depths = 1.0 / tf.linspace(1.0/end_depth, 1.0/start_depth, num_depths)
    return depths

  def compute_transmittance(self, alpha):
    """Returns transmittance of MPI voxels in reference frame.

    Args:
      alpha: MPI alpha values
    Returns:
      Transmittance of each MPI voxel in reference frame.
    """
    transmittance = tf.cumprod(
        1.0 - alpha + 1.0e-8, axis=3, exclusive=True, reverse=True) * alpha
    return transmittance

  def compute_occ_map(self, mpi_planes, rgba_layers, output_alpha,
                      intrinsics, rel_pose):
    """Computes an occlusion map, indicating which pixels are occluded/disoccluded.


    Args:
      mpi_planes: MPI plane depths
      rgba_layers: an MPI
      output_alpha: alphas from MPI that has been warped into target frame
      intrinsics: camera intrinsics [batch, 3, 3]
      rel_pose: relative pose to target camera pose
    Returns:
      One-sided occlusion map (positive diff in transmittance of target vs. ref)
    """
    # compute occlusion map, indicating which pixels are occluded/disoccluded
    # when rendering a novel view
    batch_size = tf.shape(rgba_layers)[0]
    img_height = tf.shape(rgba_layers)[1]
    img_width = tf.shape(rgba_layers)[2]
    num_mpi_planes = tf.shape(rgba_layers)[3]

    depths = tf.tile(mpi_planes[:, tf.newaxis], [1, batch_size])
    # Compute transmittance from reference viewpoint, then warp to tgt viewpoint
    trans_ref = self.compute_transmittance(
        tf.stop_gradient(rgba_layers[Ellipsis, -1]))
    trans_ref = tf.transpose(trans_ref, [3, 0, 1, 2])
    trans_ref = tf.expand_dims(trans_ref, -1)
    trans_ref_reproj = pj.projective_forward_homography(trans_ref, intrinsics,
                                                        rel_pose, depths)
    trans_ref_reproj = tf.reshape(
        trans_ref_reproj,
        [batch_size, num_mpi_planes, img_height, img_width, 1])
    trans_ref_reproj = tf.transpose(trans_ref_reproj, [0, 2, 3, 1, 4])

    # Compute transmittance of alphas that have been warped to tgt viewpoint
    trans_target = self.compute_transmittance(tf.stop_gradient(output_alpha))
    trans_target = tf.expand_dims(trans_target, -1)

    # One-sided occlusion map (positive diff in transmittance of target vs. ref)
    occ_map = tf.reduce_max(tf.nn.relu(trans_target - trans_ref_reproj), axis=3)

    return occ_map
