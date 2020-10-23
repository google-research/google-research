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

"""Rotates the sun azimuth angle of a given panorama (or panorama stack)."""

import tensorflow.compat.v1 as tf

from factorize_a_city.libs import image_alignment
from factorize_a_city.libs import network
from factorize_a_city.libs import stack_io

# The following flags describe input panoramas to use.
tf.flags.DEFINE_string(
    "stack_folder", "", "Folder containing input panoramas of the same scene. "
    "Produces sequence of images with the input scene and "
    "rotated sun positions.")
tf.flags.DEFINE_integer(
    "lighting_context_index", 0,
    "Which lighting context from data/lighting_context.npy to use as the "
    "lighting_context representation when generating the rotation.")
tf.flags.DEFINE_string("output_dir", "rotate_results",
                       "Where to save the results.")
tf.flags.DEFINE_integer("azimuth_frame_rate", 10,
                        "Frame rate of the azimuth gif.")

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  if not FLAGS.stack_folder:
    raise ValueError("stack_folder was not defined")

  (pano_stack, alignment_params) = stack_io.read_stack(FLAGS.stack_folder)

  unused_azimuth, lighting_context = stack_io.load_sample_illuminations()

  lighting_context_factors = tf.constant(lighting_context, dtype=tf.float32)

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
  recon = network.recomposite_from_log_components(
      stack_factors["log_reflectance"], stack_factors["log_shading"])
  rotate_shading_image = factorize_model.generate_sun_rotation(
      stack_factors["permanent_factor"][:1],
      lighting_context_factors[FLAGS.lighting_context_index:FLAGS
                               .lighting_context_index + 1],
      FLAGS.azimuth_frame_rate)

  results = network.recomposite_from_log_components(
      stack_factors["log_reflectance"], rotate_shading_image)

  # Restore factorization network weights from ckpt.
  tf.train.init_from_checkpoint("./factorize_a_city/ckpt/factorize_model.ckpt",
                                {"decomp_internal/": "decomp_internal/"})
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  out = sess.run(results)
  stack_io.write_stack_images(FLAGS.output_dir, out / 255.)


if __name__ == "__main__":
  # spectral normalization does not work with TF2.0
  tf.disable_eager_execution()
  tf.app.run(main)
