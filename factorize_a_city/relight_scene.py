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

"""Relights a given scene from stack_folder under sample illuminations."""

import tensorflow.compat.v1 as tf

from factorize_a_city.libs import image_alignment
from factorize_a_city.libs import network
from factorize_a_city.libs import stack_io

tf.flags.DEFINE_string(
    "stack_folder", "", "Folder containing input panoramas of the same scene. "
    "Relights the input scene with sample illuminations.")
tf.flags.DEFINE_string("output_dir", "relit_results",
                       "Where to save the results.")
FLAGS = tf.flags.FLAGS


def main(unused_argv):
  if not FLAGS.stack_folder:
    raise ValueError("stack_folder was not defined")

  # Load example stacks. Each panorama has shape [384, 960, 3] and has values
  # between [0, 1].
  (permanent_stack, alignment_params) = stack_io.read_stack(FLAGS.stack_folder)

  # Load example azimuth and illumination samples.
  azimuth, lighting_context = stack_io.load_sample_illuminations()

  permanent_stack = tf.constant(permanent_stack, dtype=tf.float32)
  azimuth_factors = tf.constant(azimuth, dtype=tf.float32)
  lighting_context_factors = tf.constant(lighting_context, dtype=tf.float32)

  # Align images using learnt parameters.
  alignment_module = image_alignment.ImageAlignment(regularization=0.3)
  aligned_stack = alignment_module.align_images(permanent_stack,
                                                alignment_params)

  factorize_model = network.FactorizeEncoderDecoder(
      {
          "lighting_dim": 32,
          "permanent_dim": 16
      }, is_training=False)
  stack_factors = factorize_model.compute_decomposition(aligned_stack)
  permanent_factor = stack_factors["permanent_factor"]
  permanent_factor = tf.tile(permanent_factor[:1], [azimuth.shape[0], 1, 1, 1])
  shading_image = factorize_model.generate_shading_image(
      permanent_factor, lighting_context_factors, azimuth_factors)
  relit_results = network.recomposite_from_log_components(
      stack_factors["log_reflectance"], shading_image)
  # Restore factorization network weights from ckpt.
  tf.train.init_from_checkpoint("./factorize_a_city/ckpt/factorize_model.ckpt",
                                {"decomp_internal/": "decomp_internal/"})
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  out = sess.run(relit_results)
  stack_io.write_stack_images(FLAGS.output_dir, out / 255.)


if __name__ == "__main__":
  # spectral normalization does not work with TF2.0
  tf.disable_eager_execution()
  tf.app.run(main)
