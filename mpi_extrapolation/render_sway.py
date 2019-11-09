# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Script to run pretrained model and render sway camera path.

For CVPR 2019 paper:
Pushing the Boundaries of View Extrapolation with Multiplane Images
Pratul P. Srinivasan, Richard Tucker, Jonathan T. Barron, Ravi Ramamoorthi, Ren
Ng, Noah Snavely.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import subprocess

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mpi_extrapolation.mpi import MPI

flags.DEFINE_string("input_file", default="", help="Input batch filename")
flags.DEFINE_string(
    "output_dir",
    default="",
    help="Directory to save MPI planes and renderings")
flags.DEFINE_string(
    "model_dir", default="", help="Directory containing pretrained model")

FLAGS = flags.FLAGS

# Model parameters
num_mpi_planes = 128
# RealEstate dataset poses are scaled so depths lie between 1.0 and 100.0 meters
# adjust depending on your data
min_depth = 1.0
max_depth = 100.0
# Image dimensions for RealEstate dataset
image_height = 576
image_width = 1024
# Patched inference parameters
# edit based on GPU memory constraints and your image size
# Recommend using largest patchsize possible
# and decreasing outsize to reduce tiling artifacts
patchsize = np.array([576, 384])  # patch size for inference
outsize = np.array([576, 128])  # central portion of the patch to keep

# Sway path parameters
num_frames = 128
crop = 20  # crop from MPI borders to avoid edgee artifacts in renderings
max_disp = 64.0  # maximum pixel disparity of closest plane (1.0 m) in sway


def main(argv):

  del argv  # Unused.

  if FLAGS.input_file is None:
    raise ValueError("`input_file` must be defined")
  if FLAGS.output_dir is None:
    raise ValueError("`output_dir` must be defined")
  if FLAGS.model_dir is None:
    raise ValueError("`model_dir` must be defined")

  checkpoint = FLAGS.model_dir + "model.ckpt"

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  # Set up model
  model = MPI()

  # Load input batch
  inputs = np.load(FLAGS.input_file)

  # Compute plane depths
  mpi_planes = model.inv_depths(min_depth, max_depth, num_mpi_planes)

  # Format inputs, convert from numpy arrays to tensors
  # Change this if you are training with a dataset iterator
  in_src_images = tf.constant(inputs["src_images"])
  in_ref_image = tf.constant(inputs["ref_image"])
  in_ref_pose = tf.constant(inputs["ref_pose"])
  # in_tgt_pose = tf.constant(inputs["tgt_pose"])  # Unneeded for sway
  in_src_poses = tf.constant(inputs["src_poses"])
  in_intrinsics = tf.constant(inputs["intrinsics"])
  in_tgt_image = tf.constant(inputs["tgt_image"])

  in_ref_image = tf.image.convert_image_dtype(in_ref_image, dtype=tf.float32)
  in_src_images = tf.image.convert_image_dtype(in_src_images, dtype=tf.float32)
  in_tgt_image = tf.image.convert_image_dtype(in_tgt_image, dtype=tf.float32)

  # Patched inference
  patch_ind = tf.placeholder(tf.int32, shape=(2))
  buffersize = (patchsize - outsize)//2

  # Set up graph
  outputs = model.infer_mpi(in_src_images,
                            in_ref_image,
                            in_ref_pose,
                            in_src_poses,
                            in_intrinsics,
                            num_mpi_planes,
                            mpi_planes,
                            run_patched=True,
                            patch_ind=patch_ind,
                            patchsize=patchsize,
                            outsize=outsize)

  # Define shapes to placate tensorflow
  outputs["rgba_layers"].set_shape(
      (1, patchsize[0], patchsize[1], num_mpi_planes, 4))
  outputs["rgba_layers_refine"].set_shape(
      (1, patchsize[0], patchsize[1], num_mpi_planes, 4))
  outputs["refine_input_mpi"].set_shape(
      (1, patchsize[0], patchsize[1], num_mpi_planes, 4))
  outputs["stuff_behind"].set_shape(
      (1, patchsize[0], patchsize[1], num_mpi_planes, 3))
  outputs["flow_vecs"].set_shape(
      (1, patchsize[0], patchsize[1], num_mpi_planes, 2))

  # Patched inference for MPI (128 planes at 0.5MP res likely won't fit on GPU)
  saver = tf.train.Saver()
  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    if checkpoint is not None:
      print("Loading from checkpoint:", checkpoint)
      saver.restore(sess, checkpoint)

    num_patches = [image_height // outsize[0], image_width // outsize[1]]
    print("patched inference with:", num_patches, "patches,", "buffersize:",
          buffersize)
    out_rgba = None
    for r in range(num_patches[0]):
      out_row_rgba = None
      for c in range(num_patches[1]):
        patch_num = r*num_patches[1]+c
        print("running patch:", patch_num)
        patch_ind_rc = np.array([r, c])
        patch_start = patch_ind_rc * outsize
        patch_end = patch_start + patchsize
        print("patch ind:", patch_ind_rc, "patch_start", patch_start,
              "patch_end", patch_end)
        feed_dict = {
            patch_ind: patch_ind_rc,
            in_src_images: inputs["src_images"],
            in_ref_image: inputs["ref_image"],
            in_ref_pose: inputs["ref_pose"],
            in_src_poses: inputs["src_poses"],
            in_intrinsics: inputs["intrinsics"]
        }
        outs = sess.run(outputs, feed_dict=feed_dict)
        outs_rgba_patch = outs["rgba_layers"][:, buffersize[0]:buffersize[0] +
                                              outsize[0],
                                              buffersize[1]:buffersize[1] +
                                              outsize[1], :, :]
        outs_rgba_patch_refine = outs[
            "rgba_layers_refine"][:, buffersize[0]:buffersize[0] + outsize[0],
                                  buffersize[1]:buffersize[1] +
                                  outsize[1], :, :]
        outs_refine_input_mpi_patch = outs[
            "refine_input_mpi"][:, buffersize[0]:buffersize[0] + outsize[0],
                                buffersize[1]:buffersize[1] + outsize[1], :, :]
        outs_stuff_behind_patch = outs[
            "stuff_behind"][:, buffersize[0]:buffersize[0] + outsize[0],
                            buffersize[1]:buffersize[1] + outsize[1], :, :]
        outs_flow_vecs = outs["flow_vecs"][:, buffersize[0]:buffersize[0] +
                                           outsize[0],
                                           buffersize[1]:buffersize[1] +
                                           outsize[1], :, :]

        if out_row_rgba is None:
          out_row_rgba = outs_rgba_patch
          out_row_rgba_refine = outs_rgba_patch_refine
          out_row_refine_input_mpi = outs_refine_input_mpi_patch
          out_row_stuff_behind = outs_stuff_behind_patch
          out_row_flow_vecs = outs_flow_vecs
        else:
          out_row_rgba = np.concatenate([out_row_rgba, outs_rgba_patch], 2)
          out_row_rgba_refine = np.concatenate(
              [out_row_rgba_refine, outs_rgba_patch_refine], 2)
          out_row_refine_input_mpi = np.concatenate(
              [out_row_refine_input_mpi, outs_refine_input_mpi_patch], 2)
          out_row_stuff_behind = np.concatenate(
              [out_row_stuff_behind, outs_stuff_behind_patch], 2)
          out_row_flow_vecs = np.concatenate(
              [out_row_flow_vecs, outs_flow_vecs], 2)

      if out_rgba is None:
        out_rgba = out_row_rgba
        out_rgba_refine = out_row_rgba_refine
        out_refine_input_mpi = out_row_refine_input_mpi
        out_stuff_behind = out_row_stuff_behind
        out_flow_vecs = out_row_flow_vecs
      else:
        out_rgba = np.concatenate([out_rgba, out_row_rgba], 1)
        out_rgba_refine = np.concatenate([out_rgba_refine, out_row_rgba_refine],
                                         1)
        out_refine_input_mpi = np.concatenate(
            [out_refine_input_mpi, out_row_refine_input_mpi], 1)
        out_stuff_behind = np.concatenate(
            [out_stuff_behind, out_row_stuff_behind], 1)
        out_flow_vecs = np.concatenate([out_flow_vecs, out_row_flow_vecs], 1)

    outs["rgba_layers"] = np.concatenate(
        [out_rgba[Ellipsis, :3] / 2.0 + 0.5, out_rgba[Ellipsis, 3:]], axis=-1)
    outs["rgba_layers_refine"] = np.concatenate(
        [out_rgba_refine[Ellipsis, :3] / 2.0 + 0.5, out_rgba_refine[Ellipsis, 3:]],
        axis=-1)
    outs["refine_input_mpi"] = np.concatenate([
        out_refine_input_mpi[Ellipsis, :3] / 2.0 + 0.5, out_refine_input_mpi[Ellipsis, 3:]
    ],
                                              axis=-1)
    outs["stuff_behind"] = out_stuff_behind / 2.0 + 0.5
    outs["flow_vecs"] = out_flow_vecs

  # Save MPI layers
  layers = outs["rgba_layers_refine"]
  for i in range(layers.shape[3]):
    i_filename = FLAGS.output_dir + "mpi_rgba_{:04d}.png".format(i)
    plt.imsave(i_filename, layers[0, :, :, i, :])
    print("wrote layer:", i)

  # Render example sway camera path
  mpi_placeholder = tf.placeholder(
      dtype=tf.float32,
      shape=[
          1, layers.shape[1] - 2 * crop, layers.shape[2] - 2 * crop,
          layers.shape[3], 4
      ])
  tgt_pose_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 4, 4])
  intrinsics_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, 3, 3])
  output_render, _ = model.mpi_render_view(mpi_placeholder,
                                           tgt_pose_placeholder, mpi_planes,
                                           intrinsics_placeholder)

  # Compute sway path poses
  max_trans = max_disp / inputs["intrinsics"][
      0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
  output_poses = []
  for i in range(num_frames):
    i_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
    i_pose = np.concatenate([
        np.concatenate(
            [np.eye(3), np.array([i_trans, 0.0, 0.0])[:, np.newaxis]], axis=1),
        np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
    ],
                            axis=0)[np.newaxis, :, :]
    output_poses.append(i_pose)

  # Render sway path
  output_render_list = []
  with tf.Session() as sess:
    for i in range(num_frames):
      print("Rendering pose:", i, "of:", num_frames)
      i_output = sess.run(
          output_render,
          feed_dict={
              mpi_placeholder:
                  outs["rgba_layers_refine"][:, crop:-crop, crop:-crop, :, :],
              tgt_pose_placeholder:
                  output_poses[i],
              intrinsics_placeholder:
                  inputs["intrinsics"]
          })
      output_render_list.append(i_output)

  for i in range(len(output_render_list)):
    plt.imsave(FLAGS.output_dir + "tmp_{:03d}.png".format(i),
               output_render_list[i][0, :, :, :])

  # Save sway path to video (requires FFMPEG)
  subprocess.call([
      "ffmpeg", "-i", FLAGS.output_dir + "tmp_%03d.png",
      FLAGS.output_dir + "sway.mp4"
  ])
  for f in glob.glob(FLAGS.output_dir + "tmp*.png"):
    os.remove(f)


if __name__ == "__main__":
  app.run(main)
