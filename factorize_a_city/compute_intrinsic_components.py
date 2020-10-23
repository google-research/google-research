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

"""Computes the intrinsic image decomposition of a stack."""

import tensorflow.compat.v1 as tf
from factorize_a_city.libs import image_alignment
from factorize_a_city.libs import network
from factorize_a_city.libs import stack_io
from factorize_a_city.libs import utils

tf.flags.DEFINE_string(
    "stack_folder", "", "Folder containing input panoramas of the same scene. "
    "Produces reflectance and shadings of the stack.")
tf.flags.DEFINE_string("output_dir", "intrinsic_image_results",
                       "Where to save the results.")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
  if not FLAGS.stack_folder:
    raise ValueError("stack_folder was not defined")

  (pano_stack, alignment_params) = stack_io.read_stack(FLAGS.stack_folder)
  # [0, 1]-ranged panoramas of shape [384, 960].
  pano_stack = tf.constant(pano_stack, dtype=tf.float32)
  alignment_params = tf.constant(alignment_params, dtype=tf.float32)

  # Align images using parameters.
  alignment_module = image_alignment.ImageAlignment(regularization=0.3)
  aligned_stack = alignment_module.align_images(pano_stack, alignment_params)

  factorize_model = network.FactorizeEncoderDecoder(
      {
          "lighting_dim": 32,
          "permanent_dim": 16
      }, is_training=False)

  stack_factors = factorize_model.compute_decomposition(
      aligned_stack, single_image_decomposition=False, average_stack=True)

  # Restore factorization network weights from ckpt.
  tf.train.init_from_checkpoint("./factorize_a_city/ckpt/factorize_model.ckpt",
                                {"decomp_internal/": "decomp_internal/"})
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  log_shading, log_reflectance = sess.run(
      [stack_factors["log_shading"], stack_factors["log_reflectance"]])

  stack_io.write_stack_images(
      FLAGS.output_dir,
      utils.outlier_normalization(log_shading),
      prefix="log_shading")
  stack_io.write_stack_images(
      FLAGS.output_dir,
      utils.outlier_normalization(log_reflectance),
      prefix="log_reflectance")


if __name__ == "__main__":
  # spectral normalization does not work with TF2.0
  tf.disable_eager_execution()
  tf.app.run(main)
