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

"""Iteratively align test stacks using gradient descent.

Given a stack of slightly misaligned test panoramas saved in
misaligned_stack_folder, iteratively apply our alignment procedure using the
frozen pre-trained factorization weights and the reflectance consistency
loss. After finishing, alignment parameters are saved in
<misaligned_stack_folder>/alignment.npy pickles which will be loaded by later
scripts.
"""
import os

import tensorflow.compat.v1 as tf

from factorize_a_city.libs import image_alignment
from factorize_a_city.libs import network
from factorize_a_city.libs import stack_io

tf.flags.DEFINE_string("misaligned_stack_folder", "",
                       "Folder containing panorama(s) to align.")
tf.flags.DEFINE_integer(
    "num_alignment_steps", 20, "Number of gradient steps to take when fitting "
    "alignment.")
tf.flags.DEFINE_bool(
    "overwrite", False, "If true, skip checking existing alignment.npy before "
    "saving.")

FLAGS = tf.flags.FLAGS


def main(unused_argv):
  if not FLAGS.misaligned_stack_folder:
    raise ValueError("misaligned_stack_folder was not defined")

  alignment_save_location = os.path.join(FLAGS.misaligned_stack_folder,
                                         "alignment.npy")
  if not FLAGS.overwrite:
    if os.path.exists(alignment_save_location):
      raise ValueError("Existing alignment found. "
                       "Pass --overwrite flag to overwrite it.")

  misaligned_stack = stack_io.read_stack(
      FLAGS.misaligned_stack_folder, require_alignment=False)

  # [0, 1]-ranged panoramas of shape [384, 960].
  misaligned_stack = tf.constant(misaligned_stack, dtype=tf.float32)

  # Randomly initialized warp parameters.
  alignment_params = tf.get_variable(
      "alignment_params",
      [8, 8, 32, 2],
      trainable=True,
      initializer=tf.random_normal_initializer(0, 1e-3),
  )
  tanh_alignment_params = tf.nn.tanh(alignment_params)

  # Align images using parameters.
  alignment_module = image_alignment.ImageAlignment(regularization=0.3)
  aligned_stack = alignment_module.align_images(misaligned_stack,
                                                tanh_alignment_params)

  # Freeze weight during the decomposition.
  factorize_model = network.FactorizeEncoderDecoder(
      {
          "lighting_dim": 32,
          "permanent_dim": 16
      }, is_training=False)

  stack_factors = factorize_model.compute_decomposition(
      aligned_stack, single_image_decomposition=False, average_stack=True)

  individual_ref = stack_factors["individual_log_reflectance"]
  stack_size = individual_ref.shape.as_list()[0]
  ref_consistency_loss = tf.zeros([])
  for i in range(stack_size):
    for j in range(i + 1, stack_size):
      ref_consistency_loss += tf.reduce_mean(
          tf.abs(individual_ref[i] - individual_ref[j]))

  warp_opt = tf.train.AdamOptimizer(1e-4, name="Adam", beta1=0., epsilon=1e-4)
  warp_train_step = warp_opt.minimize(
      ref_consistency_loss, var_list=[alignment_params])

  # Restore factorization network weights from ckpt.
  tf.train.init_from_checkpoint("./factorize_a_city/ckpt/factorize_model.ckpt",
                                {"decomp_internal/": "decomp_internal/"})
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for i in range(FLAGS.num_alignment_steps):
    loss, _ = sess.run([ref_consistency_loss, warp_train_step])
    if i % 10 == 0:
      print("[%d / %d] Steps, Loss: %5f" % (i, FLAGS.num_alignment_steps, loss))

  print("[%d / %d] Steps, Loss: %5f" %
        (FLAGS.num_alignment_steps, FLAGS.num_alignment_steps, loss))
  np_alignment_weights = sess.run(tanh_alignment_params)

  with open(alignment_save_location, "wb") as f:
    np_alignment_weights.dump(f)


if __name__ == "__main__":
  # spectral normalization does not work with TF2.0
  tf.disable_eager_execution()
  tf.app.run(main)
