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

"""Test pretrained multiscale lighting volume prediction on our InteriorNet test set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

import lighthouse.geometry.projector as pj
from lighthouse.mlv import MLV
import lighthouse.nets as nets

flags.DEFINE_string(
    "checkpoint_dir", default="", help="Directory for loading checkpoint")
flags.DEFINE_string(
    "data_dir", default="", help="InteriorNet test dataset directory")
flags.DEFINE_string(
    "output_dir", default="", help="Output directory to save images")

FLAGS = flags.FLAGS

# Model parameters.
batch_size = 1  # implementation only works for batch size 1 currently.
height = 240  # px
width = 320  # px
env_height = 120  # px
env_width = 240  # px
cube_res = 64  # px
theta_res = 240  # px
phi_res = 120  # px
r_res = 128  # px
scale_factors = [2, 4, 8, 16]
num_planes = 32
depth_clip = 20.0


def main(argv):

  del argv  # unused

  if FLAGS.checkpoint_dir is None:
    raise ValueError("`checkpoint_dir` must be defined")
  if FLAGS.data_dir is None:
    raise ValueError("`data_dir` must be defined")
  if FLAGS.output_dir is None:
    raise ValueError("`output_dir` must be defined")

  # Set up placeholders
  ref_image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
  ref_depth = tf.placeholder(dtype=tf.float32, shape=[None, height, width])
  intrinsics = tf.placeholder(dtype=tf.float32, shape=[None, 3, 3])
  ref_pose = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4])
  src_images = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
  src_poses = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4, 1])
  env_pose = tf.placeholder(dtype=tf.float32, shape=[None, 4, 4])

  # Set up model
  model = MLV()

  # We use the true depth bounds for testing
  # Adjust to estimated bounds for your dataset
  min_depth = tf.reduce_min(ref_depth)
  max_depth = tf.reduce_max(ref_depth)

  # Set up graph
  mpi_planes = pj.inv_depths(min_depth, max_depth, num_planes)

  pred = model.infer_mpi(src_images, ref_image, ref_pose, src_poses, intrinsics,
                         mpi_planes)
  rgba_layers = pred["rgba_layers"]

  lightvols, lightvol_centers, \
  lightvol_side_lengths, \
  cube_rel_shapes, \
  cube_nest_inds = model.predict_lighting_vol(rgba_layers, mpi_planes,
                                              intrinsics, cube_res,
                                              scale_factors,
                                              depth_clip=depth_clip)
  lightvols_out = nets.cube_net_multires(lightvols, cube_rel_shapes,
                                         cube_nest_inds)
  output_envmap, _ = model.render_envmap(lightvols_out, lightvol_centers,
                                         lightvol_side_lengths, cube_rel_shapes,
                                         cube_nest_inds, ref_pose, env_pose,
                                         theta_res, phi_res, r_res)

  if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  input_files = sorted(
      [f for f in os.listdir(FLAGS.data_dir) if f.endswith(".npz")])
  print("found {:05d} input files".format(len(input_files)))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, "model.ckpt"))

    for i in range(0, len(input_files)):
      print("running example:", i)

      # Load inputs
      batch = np.load(FLAGS.data_dir + input_files[i])

      output_envmap_eval, = sess.run(
          [output_envmap],
          feed_dict={
              ref_image: batch["ref_image"],
              ref_depth: batch["ref_depth"],
              intrinsics: batch["intrinsics"],
              ref_pose: batch["ref_pose"],
              src_images: batch["src_images"],
              src_poses: batch["src_poses"],
              env_pose: batch["env_pose"]
          })

      # Write environment map image
      plt.imsave(
          os.path.join(FLAGS.output_dir, "{:05d}.png".format(i)),
          output_envmap_eval[0, :, :, :3])


if __name__ == "__main__":
  app.run(main)
