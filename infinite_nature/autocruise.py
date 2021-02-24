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

"""Uses a heuristic to automatically navigate generated scenes.

fly_camera.fly_dynamic will generate poses using disparity maps that avoid
crashing into nearby terrain.
"""
import pickle
import time

import config
import fly_camera
import imageio
import infinite_nature_lib
import numpy as np
import tensorflow as tf


tf.compat.v1.flags.DEFINE_string(
    "output_folder", "autocruise_output",
    "Folder to save autocruise results")
tf.compat.v1.flags.DEFINE_integer(
    "num_steps", 500,
    "Number of steps to fly.")

FLAGS = tf.compat.v1.flags.FLAGS


def generate_autocruise(np_input_rgbd, checkpoint,
                        save_directory, num_steps, np_input_intrinsics=None):
  """Saves num_steps frames of infinite nature using an autocruise algorithm.

  Args:
    np_input_rgbd: [H, W, 4] numpy image and disparity to start
      Infinite Nature with values ranging in [0, 1]
    checkpoint: (str) path to the pre-trained checkpoint
    save_directory: (str) the directory to save RGB images to
    num_steps: (int) the number of steps to generate
    np_input_intrinsics: [4] estimated intrinsics. If not provided,
      makes assumptions on the FOV.
  """
  render_refine, style_encoding = infinite_nature_lib.load_model(checkpoint)
  if np_input_intrinsics is None:
    # 0.8 focal_x corresponds to a FOV of ~64 degrees. This can be
    # manually changed if more assumptions about the input image is given.
    h, w, unused_channel = np_input_rgbd.shape
    ratio = w / float(h)
    np_input_intrinsics = np.array([0.8, 0.8 * ratio, .5, .5], dtype=np.float32)

  np_input_rgbd = tf.image.resize(np_input_rgbd, [160, 256])
  style_noise = style_encoding(np_input_rgbd)

  meander_x_period = 100
  meander_y_period = 100
  meander_x_magnitude = 0.0
  meander_y_magnitude = 0.0
  fly_speed = 0.2
  horizon = 0.3
  near_fraction = 0.2

  starting_pose = np.array(
      [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
      dtype=np.float32)

  # autocruise heuristic funciton
  fly_next_pose_function = fly_camera.fly_dynamic(
      np_input_intrinsics, starting_pose,
      speed=fly_speed,
      meander_x_period=meander_x_period,
      meander_x_magnitude=meander_x_magnitude,
      meander_y_period=meander_y_period,
      meander_y_magnitude=meander_y_magnitude,
      horizon=horizon,
      near_fraction=near_fraction)

  if not tf.io.gfile.exists(save_directory):
    tf.io.gfile.makedirs(save_directory)

  curr_pose = starting_pose
  curr_rgbd = np_input_rgbd
  t0 = time.time()
  for i in range(num_steps - 1):
    next_pose = fly_next_pose_function(curr_rgbd)
    curr_rgbd = render_refine(
        curr_rgbd, style_noise, curr_pose, np_input_intrinsics,
        next_pose, np_input_intrinsics)

    # Update pose information for view.
    curr_pose = next_pose
    imageio.imsave("%s/%04d.png" % (save_directory, i),
                   (255 * curr_rgbd[:, :, :3]).astype(np.uint8))
    if i % 100 == 0:
      print("%d / %d frames generated" % (i, num_steps))
      print("time / step: %04f" % ((time.time() - t0) / (i + 1)))
      print()


def main(unused_arg):
  if len(unused_arg) > 1:
    raise tf.app.UsageError(
        "Too many command-line arguments.")
  config.set_training(False)
  model_path = "ckpt/model.ckpt-6935893"
  input_pkl = pickle.load(open("autocruise_input1.pkl", "rb"))
  generate_autocruise(input_pkl["input_rgbd"],
                      model_path,
                      FLAGS.output_folder,
                      FLAGS.num_steps)

if __name__ == "__main__":
  tf.compat.v1.enable_eager_execution()
  tf.compat.v1.app.run(main)
