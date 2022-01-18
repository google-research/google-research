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

"""Runs Infinite Nature on an input image + disparity.
"""
import config
import networks
import render
import tensorflow as tf


def load_model(checkpoint):
  """Load a trained model and return functions to run it.

  This code gives an "eager-like" interface to the underlying computation
  graph.

  Args:
    checkpoint: The checkpoint name to load.

  Returns:
    pvsm_from_image: A function which takes a [160, 256, 4] RGBD images
                     and a [160, 256, 4] encoding image,
                     and camera parameters:
                     pose, pose_next [3, 4], intrinsics, intrinsics_next [4]
    and returns a list of 3 images of size [H, W, 4], [predicted, render, mask]
    style_embedding_from_encoding: A function which takes [160, 256, 4] RGBD
                                   image and returns a style embedding [256]
  """
  sess = tf.compat.v1.Session()
  with sess.graph.as_default():
    image_placeholder = tf.compat.v1.placeholder(tf.float32, [160, 256, 4])
    # Initial RGB_D to set the latent
    encoding_placeholder = tf.compat.v1.placeholder(tf.float32, [160, 256, 4])
    style_noise_placeholder = tf.compat.v1.placeholder(
        tf.float32, [config.DIM_OF_STYLE_EMBEDDING])
    intrinsic_placeholder = tf.compat.v1.placeholder(tf.float32, [4])
    intrinsic_next_placeholder = tf.compat.v1.placeholder(tf.float32, [4])
    pose_placeholder = tf.compat.v1.placeholder(tf.float32, [3, 4])
    pose_next_placeholder = tf.compat.v1.placeholder(tf.float32, [3, 4])

    # Add batch dimensions.
    image = image_placeholder[tf.newaxis]
    encoding = encoding_placeholder[tf.newaxis]
    style_noise = style_noise_placeholder[tf.newaxis]
    intrinsic = intrinsic_placeholder[tf.newaxis]
    intrinsic_next = intrinsic_next_placeholder[tf.newaxis]
    pose = pose_placeholder[tf.newaxis]
    pose_next = pose_next_placeholder[tf.newaxis]

    mulogvar = get_encoding_mu_logvar(encoding)
    if config.is_training():
      z = networks.reparameterize(mulogvar[0], mulogvar[1])
    else:
      z = mulogvar[0]

    z = z[0]

    refine_fn = create_refinement_network(style_noise)
    render_rgbd, mask = render.render(
        image, pose, intrinsic, pose_next, intrinsic_next)

    generated_image = refine_fn(render_rgbd, mask)

    refined_disparity = rescale_refined_disparity(render_rgbd[Ellipsis, 3:], mask,
                                                  generated_image[Ellipsis, 3:])
    generated_image = tf.concat([generated_image[Ellipsis, :3], refined_disparity],
                                axis=-1)[0]

    saver = tf.compat.v1.train.Saver()
    print("Restoring from %s" % checkpoint)
    saver.restore(sess, checkpoint)
    print("Model restored.")

  def as_numpy(x):
    if tf.is_tensor(x):
      return x.numpy()
    else:
      return x

  def render_refine(image, style_noise, pose, intrinsic,
                    pose_next, intrinsic_next):
    return sess.run(generated_image, feed_dict={
        image_placeholder: as_numpy(image),
        style_noise_placeholder: as_numpy(style_noise),
        pose_placeholder: as_numpy(pose),
        intrinsic_placeholder: as_numpy(intrinsic),
        pose_next_placeholder: as_numpy(pose_next),
        intrinsic_next_placeholder: as_numpy(intrinsic_next),
    })

  def encoding_fn(encoding_image):
    return sess.run(z, feed_dict={
        encoding_placeholder: as_numpy(encoding_image)})

  return render_refine, encoding_fn


def rescale_refined_disparity(rendered_disparity, input_mask,
                              refined_disparity):
  """Rescales the refined disparity to match the input's scale.

  This is done to prevent drifting in the disparity. We match the scale
  by solving a least squares optimization.

  Args:
    rendered_disparity: [B, H, W, 1] disparity produced by the render step
    input_mask: [B, H, W, 1] a mask with 1's denoting regions that were
      visible through the rendering.
    refined_disparity: [B, H, W, 1] disparity of the refinement network output
  Returns:
    refined_disparity that has been scale and shifted to match the statistics
    of rendered_disparity.
  """
  log_refined = tf.math.log(tf.clip_by_value(refined_disparity, 0.01, 1))
  log_rendered = tf.math.log(tf.clip_by_value(rendered_disparity, 0.01, 1))
  log_scale = tf.reduce_sum(input_mask * (log_rendered - log_refined)) / (
      tf.reduce_sum(input_mask) + 1e-4)
  scale = tf.exp(log_scale)
  scaled_refined_disparity = tf.clip_by_value(scale * refined_disparity,
                                              0, 1)
  return scaled_refined_disparity


def create_refinement_network(noise_input):
  """Creates a refinement network with the encoding_image's style noise.

  Args:
    noise_input: [1, z_dim] a noise sampled from a normal distribution
      parameterized by an encoder.
  Returns:
    A refinement function that uses the encoding_image to seed the noise.
  """
  def fn(rgbd, mask):
    with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
      with tf.compat.v1.variable_scope("spade_network_noenc"):
        return networks.refinement_network(rgbd, mask, noise_input)
  return fn


def get_encoding_mu_logvar(encoding_image):
  """Computes the encoding_image's style noise parameters.

  Args:
    encoding_image: [B, H, W, 4] input RGBD image to run Infinite Nature on
  Returns:
    tuple of tensors ([B, z_dim], [B, z_dim]) corresponding to mu and logvar
    normal parameters.
  """

  with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
    mu_logvar = networks.encoder(encoding_image)
  return mu_logvar
