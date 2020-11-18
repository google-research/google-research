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

#!/usr/bin/env python
"""Goal-conditioned transport Module."""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ravens import utils
from ravens.models.resnet import ResNet43_8s

import tensorflow as tf
import tensorflow_addons as tfa


class TransportGoal:
  """Goal-conditioned transport Module."""

  def __init__(self, input_shape, num_rotations, crop_size, preprocess):  # pylint: disable=g-doc-args
    """Inits transport module with separate goal FCN.

    Assumes the presence of a goal image, that cropping is done after the
    query, that per-pixel loss is not used, and SE(2) grasping.
    """
    self.num_rotations = num_rotations
    self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
    self.preprocess = preprocess
    self.lr = 1e-5

    self.pad_size = int(self.crop_size / 2)
    self.padding = np.zeros((3, 2), dtype=int)
    self.padding[:2, :] = self.pad_size

    input_shape = np.array(input_shape)
    input_shape[0:2] += self.pad_size * 2
    input_shape = tuple(input_shape)

    # Output dimension (i.e., number of channels) of 3.
    self.odim = output_dim = 3

    # 3 fully convolutional ResNets. Third one is for the goal.
    in0, out0 = ResNet43_8s(input_shape, output_dim, prefix='s0_')
    in1, out1 = ResNet43_8s(input_shape, output_dim, prefix='s1_')
    in2, out2 = ResNet43_8s(input_shape, output_dim, prefix='s2_')

    self.model = tf.keras.Model(
        inputs=[in0, in1, in2], outputs=[out0, out1, out2])
    self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
    self.metric = tf.keras.metrics.Mean(name='transport_loss')

  def forward(self, in_img, goal_img, p, apply_softmax=True):  # pylint: disable=g-doc-args
    """Forward pass of goal-conditioned Transporters.

    Runs input through all three networks, to get output of the same
    shape, except the last channel is 3 (output_dim). Then, the output
    for one stream has the convolutional kernels for another. Call
    tf.nn.convolution, and the operation is be differentiable, so that
    gradients apply to all the FCNs.

    Cropping after passing the input image to the query network is
    easier, because otherwise we need to do a forward pass, then call
    tf.multiply, then do a second forward pass after that.

    Returns:
      ouput tensor
    """
    assert in_img.shape == goal_img.shape, f'{in_img.shape}, {goal_img.shape}'

    # input image --> TF tensor, shape (384,224,6) --> (1,384,224,6)
    input_unproc = np.pad(in_img, self.padding, mode='constant')
    input_data = self.preprocess(input_unproc.copy())
    input_shape = (1,) + input_data.shape
    input_data = input_data.reshape(input_shape)
    in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # goal image --> TF tensor, shape (384,224,6) --> (1,384,224,6)
    goal_unproc = np.pad(goal_img, self.padding, mode='constant')
    goal_data = self.preprocess(goal_unproc.copy())
    goal_shape = (1,) + goal_data.shape
    goal_data = goal_data.reshape(goal_shape)
    goal_tensor = tf.convert_to_tensor(goal_data, dtype=tf.float32)

    # Get SE2 rotation vectors for cropping.
    pivot = np.array([p[1], p[0]]) + self.pad_size
    rvecs = self.get_se2(self.num_rotations, pivot)

    # Forward pass through three separate FCNs. All logits: (1,384,224,3).
    in_logits, kernel_nocrop_logits, goal_logits = self.model(
        [in_tensor, in_tensor, goal_tensor])

    # Use features from goal logits and combine with input and kernel.
    goal_x_in_logits = tf.multiply(goal_logits, in_logits)
    goal_x_kernel_logits = tf.multiply(goal_logits, kernel_nocrop_logits)

    # Crop the kernel_logits about the picking point and get rotations.
    crop = tf.identity(goal_x_kernel_logits)  # (1,384,224,3)
    crop = tf.repeat(crop, repeats=self.num_rotations, axis=0)  # (24,384,224,3)
    crop = tfa.image.transform(crop, rvecs, interpolation='NEAREST')
    kernel = crop[:, p[0]:(p[0] + self.crop_size),
                  p[1]:(p[1] + self.crop_size), :]
    assert kernel.shape == (self.num_rotations, self.crop_size, self.crop_size,
                            self.odim)

    # Cross-convolve `in_x_goal_logits`. Padding kernel: (24,64,64,3) -->
    # (65,65,3,24).
    kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    kernel = tf.pad(kernel, kernel_paddings, mode='CONSTANT')
    kernel = tf.transpose(kernel, [1, 2, 3, 0])
    output = tf.nn.convolution(goal_x_in_logits, kernel, data_format='NHWC')
    output = (1 / (self.crop_size**2)) * output

    if apply_softmax:
      output_shape = output.shape
      output = tf.reshape(output, (1, np.prod(output.shape)))
      output = tf.nn.softmax(output)
      output = np.float32(output).reshape(output_shape[1:])

    # Daniel: visualize crops and kernels, for Transporter-Goal figure.
    # self.visualize_images(p, in_img, input_data, crop)
    # self.visualize_transport(p, in_img, input_data, crop, kernel)
    # self.visualize_logits(in_logits,            name='input')
    # self.visualize_logits(goal_logits,          name='goal')
    # self.visualize_logits(kernel_nocrop_logits, name='kernel')
    # self.visualize_logits(goal_x_in_logits,     name='goal_x_in')
    # self.visualize_logits(goal_x_kernel_logits, name='goal_x_kernel')

    return output

  def train(self, in_img, goal_img, p, q, theta):
    """Transport Goal training.

    Both `in_img` and `goal_img` have the color and depth. Much is
    similar to the attention model: (a) forward pass, (b) get angle
    discretizations, (c) make the label consider rotations in the last
    axis, but only provide the label to one single (pixel,rotation).

    Args:
      in_img:
      goal_img:
      p:
      q:
      theta:

    Returns:
      Transport loss as a numpy float32.
    """
    self.metric.reset_states()

    with tf.GradientTape() as tape:
      output = self.forward(in_img, goal_img, p, apply_softmax=False)

      # Compute label
      itheta = theta / (2 * np.pi / self.num_rotations)
      itheta = np.int32(np.round(itheta)) % self.num_rotations
      label_size = in_img.shape[:2] + (self.num_rotations,)
      label = np.zeros(label_size)
      label[q[0], q[1], itheta] = 1
      label = label.reshape(1, np.prod(label.shape))
      label = tf.convert_to_tensor(label, dtype=tf.float32)

      # Compute loss after re-shaping the output.
      output = tf.reshape(output, (1, np.prod(output.shape)))
      loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
      loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
    self.metric(loss)
    return np.float32(loss)

  def get_se2(self, num_rotations, pivot):
    """Get SE2 rotations discretized into num_rotations angles counter-clockwise."""
    rvecs = []
    for i in range(num_rotations):
      theta = i * 2 * np.pi / num_rotations
      rmat = utils.get_image_transform(theta, (0, 0), pivot)
      rvec = rmat.reshape(-1)[:-1]
      rvecs.append(rvec)
    return np.array(rvecs, dtype=np.float32)

  def save(self, fname):
    self.model.save(fname)

  def load(self, fname):
    self.model.load_weights(fname)

  #-------------------------------------------------------------------------
  # Visualization.
  #-------------------------------------------------------------------------

  def visualize_images(self, p, in_img, input_data, crop):
    """Visualize images."""

    def get_itheta(theta):
      itheta = theta / (2 * np.pi / self.num_rotations)
      return np.int32(np.round(itheta)) % self.num_rotations

    plt.subplot(1, 3, 1)
    plt.title('Perturbed', fontsize=15)
    plt.imshow(np.array(in_img[:, :, :3]).astype(np.uint8))
    plt.subplot(1, 3, 2)
    plt.title('Process/Pad', fontsize=15)
    plt.imshow(input_data[0, :, :, :3])
    plt.subplot(1, 3, 3)
    # Let's stack two crops together.
    theta1 = 0.0
    theta2 = 90.0
    itheta1 = get_itheta(theta1)
    itheta2 = get_itheta(theta2)
    crop1 = crop[itheta1, :, :, :3]
    crop2 = crop[itheta2, :, :, :3]
    barrier = np.ones_like(crop1)
    barrier = barrier[:4, :, :]  # white barrier of 4 pixels
    stacked = np.concatenate((crop1, barrier, crop2), axis=0)
    plt.imshow(stacked)
    plt.title(f'{theta1}, {theta2}', fontsize=15)
    plt.suptitle(f'pick: {p}', fontsize=15)
    plt.tight_layout()
    plt.show()
    # plt.savefig('viz.png')

  def visualize_transport(self, p, in_img, input_data, crop, kernel):  # pylint: disable=g-doc-args
    """Like the attention map visualize the transport data from a trained model.

    https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
    In my normal usage, the attention is already softmax-ed but just be
    aware in case it's not. Also be aware of RGB vs BGR mode. We should
    ensure we're in BGR mode before saving. Also with RAINBOW mode,
    red=hottest (highest attention values), green=medium, blue=lowest.

    See also:
    https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.subplot.html

    crop.shape: (24,64,64,6)
    kernel.shape = (65,65,3,24)
    """
    del p
    del in_img
    del input_data

    def colorize(img):
      # I don't think we have to convert to BGR here...
      img = img - np.min(img)
      img = 255 * img / np.max(img)
      img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
      # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      return img

    kernel = (tf.transpose(kernel, [3, 0, 1, 2])).numpy()

    # Top two rows: crops from processed RGBD. Bottom two: output from FCN.
    nrows = 4
    ncols = 12
    assert self.num_rotations == nrows * (ncols / 2)
    idx = 0
    _, _ = plt.subplots(nrows, ncols, figsize=(12, 6))
    for _ in range(nrows):
      for _ in range(ncols):
        plt.subplot(nrows, ncols, idx + 1)
        plt.axis('off')  # Ah, you need to put this here ...
        if idx < self.num_rotations:
          plt.imshow(crop[idx, :, :, :3])
        else:
          # Offset because idx goes from 0 to (rotations * 2) - 1.
          idx_ = idx - self.num_rotations
          processed = colorize(img=kernel[idx_, :, :, :])
          plt.imshow(processed)
        idx += 1
    plt.tight_layout()
    plt.show()

  def visualize_logits(self, logits, name):  # pylint: disable=g-doc-args
    """Given logits (BEFORE tf.nn.convolution), get a heatmap.

    Here we apply a softmax to make it more human-readable. However, the
    tf.nn.convolution with the learned kernels happens without a softmax
    on the logits. [Update: wait, then why should we have a softmax,
    then? I forgot why we did this ...]
    """
    original_shape = logits.shape
    logits = tf.reshape(logits, (1, np.prod(original_shape)))
    # logits = tf.nn.softmax(logits)  # Is this necessary?
    vis_transport = np.float32(logits).reshape(original_shape)
    vis_transport = vis_transport[0]
    vis_transport = vis_transport - np.min(vis_transport)
    vis_transport = 255 * vis_transport / np.max(vis_transport)
    vis_transport = cv2.applyColorMap(
        np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)

    # Only if we're saving with cv2.imwrite()
    vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'tmp/logits_{name}.png', vis_transport)

    plt.subplot(1, 1, 1)
    plt.title(f'Logits: {name}', fontsize=15)
    plt.imshow(vis_transport)
    plt.tight_layout()
    plt.show()
