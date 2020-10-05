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

"""Transport Module."""

import os
import shutil

import imageio
import matplotlib.pyplot as plt
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np
from ravens import utils
from ravens.models import ResNet43_8s
import tensorflow as tf
import tensorflow_addons as tfa


class Transport:
  """Transport Module."""

  def __init__(self,
               input_shape,
               num_rotations,
               crop_size,
               preprocess,
               per_pixel_loss=False,
               six_dof=False,
               crop_bef_q=True,
               use_goal_image=False):
    """Defines the transport module for determining placing.

    Args:
      input_shape: Shape of original, stacked color and heightmap images
        before padding, usually (320,160,6).
      num_rotations: Number of rotations considered, which determines the
        number of challens of the output.
      crop_size: Size of crop for Transporter Network.
      preprocess: Method to subtract mean and divide by standard dev.
      per_pixel_loss: True if training using a per-pixel loss. Default is
        False (which performs better in CoRL submission experiments) and
        means that a softmax is applied on all pixels, and the designated
        pixel (from demonstration data) is used as the correct class,
        passing gradients via every pixel, and avoiding the need for
        negative data samples.
      six_dof: True if using the 6 DoF extension of Transporters.
      crop_bef_q: True if cropping the input before passing it through the
        query FCN, meaning that the FCN takes input of shape (H,W) which
        is `kernel_shape`, and has output of the same `kernel_shape` (H,W)
        dims.
      use_goal_image: True only if we're using goal-conditioning, and if
        so, this is the naive version where we stack the current and goal
        images and pass them through this Transport module.
    """
    self.num_rotations = num_rotations
    self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
    self.preprocess = preprocess
    self.per_pixel_loss = per_pixel_loss
    self.six_dof = six_dof
    self.crop_bef_q = crop_bef_q
    self.lr = 1e-5

    self.pad_size = int(self.crop_size / 2)
    self.padding = np.zeros((3, 2), dtype=int)
    self.padding[:2, :] = self.pad_size

    input_shape = np.array(input_shape)
    input_shape[0:2] += self.pad_size * 2
    input_shape = tuple(input_shape)

    kernel_shape = (self.crop_size, self.crop_size, input_shape[2])

    if not self.per_pixel_loss and not self.six_dof:
      output_dim = 3
      kernel_dim = 3
    elif self.per_pixel_loss and not self.six_dof:
      output_dim = 6
      kernel_dim = 3
    elif self.six_dof:
      output_dim = 24
      kernel_dim = 24
      self.regress_loss = tf.keras.losses.Huber()
      self.z_regressor = Regressor()
      self.roll_regressor = Regressor()
      self.pitch_regressor = Regressor()
    else:
      raise ValueError("I don't support this config!")

    # 2 fully convolutional ResNets with 57 layers and 16-stride
    if not self.six_dof:
      in0, out0 = ResNet43_8s(input_shape, output_dim, prefix="s0_")
      if self.crop_bef_q:
        # Passing in kernels: (64,64,6) --> (64,64,3)
        in1, out1 = ResNet43_8s(kernel_shape, kernel_dim, prefix="s1_")
      else:
        # Passing in original images: (384,224,6) --> (394,224,3)
        in1, out1 = ResNet43_8s(input_shape, output_dim, prefix="s1_")
    else:
      in0, out0 = ResNet43_8s(input_shape, output_dim, prefix="s0_")
      # early cutoff just so it all fits on GPU.
      in1, out1 = ResNet43_8s(
          kernel_shape, kernel_dim, prefix="s1_", cutoff_early=True)

    self.model = tf.keras.Model(inputs=[in0, in1], outputs=[out0, out1])
    self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)

    self.metric = tf.keras.metrics.Mean(name="transport_loss")

    if self.six_dof:
      self.z_metric = tf.keras.metrics.Mean(name="z_loss")
      self.roll_metric = tf.keras.metrics.Mean(name="roll_loss")
      self.pitch_metric = tf.keras.metrics.Mean(name="pitch_loss")

    # For visualization
    self.feature_visualize = False
    if self.feature_visualize:
      self.fig, self.ax = plt.subplots(5, 1)
    self.write_visualize = False
    self.plot_interval = 20
    self.iters = 0

  def set_bounds_pixel_size(self, bounds, pixel_size):
    self.bounds = bounds
    self.pixel_size = pixel_size

  def forward(self, in_img, p, apply_softmax=True, theta=None):
    """Forward pass."""
    img_unprocessed = np.pad(in_img, self.padding, mode="constant")
    input_data = self.preprocess(img_unprocessed.copy())
    input_shape = (1,) + input_data.shape
    input_data = input_data.reshape(input_shape)
    in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Rotate crop
    pivot = np.array([p[1], p[0]]) + self.pad_size
    rvecs = self.get_se2(self.num_rotations, pivot)

    if self.crop_bef_q:
      # Defaults for Transporters in the CoRL submission.
      crop = tf.convert_to_tensor(input_data.copy(), dtype=tf.float32)
      crop = tf.repeat(crop, repeats=self.num_rotations, axis=0)
      crop = tfa.image.transform(crop, rvecs, interpolation="NEAREST")
      crop = crop[:, p[0]:(p[0] + self.crop_size),
                  p[1]:(p[1] + self.crop_size), :]
      logits, kernel_raw = self.model([in_tensor, crop])
    else:
      # Pass `in_tensor` twice, crop from `kernel_bef_crop`, not `input_data`.
      logits, kernel_bef_crop = self.model([in_tensor, in_tensor])
      crop = tf.identity(kernel_bef_crop)
      crop = tf.repeat(crop, repeats=self.num_rotations, axis=0)
      crop = tfa.image.transform(crop, rvecs, interpolation="NEAREST")
      kernel_raw = crop[:, p[0]:(p[0] + self.crop_size),
                        p[1]:(p[1] + self.crop_size), :]

    # Obtain kernels for cross-convolution.
    kernel_paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
    kernel = tf.pad(kernel_raw, kernel_paddings, mode="CONSTANT")
    kernel = tf.transpose(kernel, [1, 2, 3, 0])

    if not self.per_pixel_loss and not self.six_dof:
      output = tf.nn.convolution(logits, kernel, data_format="NHWC")
    elif self.per_pixel_loss:
      output0 = tf.nn.convolution(logits[Ellipsis, :3], kernel, data_format="NHWC")
      output1 = tf.nn.convolution(logits[Ellipsis, 3:], kernel, data_format="NHWC")
      output = tf.concat((output0, output1), axis=0)
      output = tf.transpose(output, [1, 2, 3, 0])
    elif self.six_dof:
      # TODO(peteflorence): output not used with separate regression model
      output = tf.nn.convolution(
          logits[Ellipsis, :3], kernel[:, :, :3, :], data_format="NHWC")
      z_tensor = tf.nn.convolution(
          logits[Ellipsis, :8], kernel[:, :, :8, :], data_format="NHWC")
      roll_tensor = tf.nn.convolution(
          logits[Ellipsis, 8:16], kernel[:, :, 16:24, :], data_format="NHWC")
      pitch_tensor = tf.nn.convolution(
          logits[Ellipsis, 16:24], kernel[:, :, 16:24, :], data_format="NHWC")
    else:
      raise ValueError("Unintended config!")

    output = (1 / (self.crop_size**2)) * output

    if apply_softmax:
      output_shape = output.shape
      if self.per_pixel_loss:
        output = tf.reshape(output, (np.prod(output.shape[:-1]), 2))
      else:
        output = tf.reshape(output, (1, np.prod(output.shape)))
      output = tf.nn.softmax(output)
      if self.per_pixel_loss:
        output = np.float32(output[:, 1]).reshape(output_shape[:-1])
      else:
        output = np.float32(output).reshape(output_shape[1:])

    # Need to pass in theta to use this visualization.
    self.iters += 1
    if theta is not None and self.feature_visualize and self.iters % self.plot_interval == 0:
      self.visualize_introspection(img_unprocessed, p, rvecs, input_shape,
                                   theta, logits, kernel_raw, output)

    if not self.six_dof:
      return output
    else:
      return output, z_tensor, roll_tensor, pitch_tensor

  def train(self,
            in_img,
            p,
            q,
            theta,
            z=None,
            roll=None,
            pitch=None,
            validate=False):
    """Transport pixel p to pixel q.

    Args:
      in_img:
      p: pixel (y, x)
      q: pixel (y, x)
      theta:
      z:
      roll:
      pitch:
      validate:

    Returns:
      A `Tensor`. Has the same type as `input`.
    """
    visualize_input = False
    if visualize_input and self.six_dof:  # only supported for six dof model
      self.visualize_train_input(in_img, p, q, theta, z, roll, pitch)

    self.metric.reset_states()
    if self.six_dof:
      self.z_metric.reset_states()
      self.roll_metric.reset_states()
      self.pitch_metric.reset_states()

    with tf.GradientTape() as tape:
      output = self.forward(in_img, p, apply_softmax=False, theta=theta)

      if self.six_dof:
        z_label, roll_label, pitch_label = z, roll, pitch
        output, z_tensor, roll_tensor, pitch_tensor = output

      itheta = theta / (2 * np.pi / self.num_rotations)
      itheta = np.int32(np.round(itheta)) % self.num_rotations

      label_size = in_img.shape[:2] + (self.num_rotations,)
      label = np.zeros(label_size)
      label[q[0], q[1], itheta] = 1

      if self.per_pixel_loss:
        sampling = True  # sampling negatives seems to converge faster
        if sampling:
          num_samples = 100
          inegative = utils.sample_distribution(1 - label, num_samples)
          inegative = [np.ravel_multi_index(i, label.shape) for i in inegative]
          ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
          output = tf.reshape(output, (-1, 2))
          output_samples = ()
          for i in inegative:
            output_samples += (tf.reshape(output[i, :], (1, 2)),)
          output_samples += (tf.reshape(output[ipositive, :], (1, 2)),)
          output = tf.concat(output_samples, axis=0)
          label = np.int32([0] * num_samples + [1])[Ellipsis, None]
          label = np.hstack((1 - label, label))
          weights = np.ones(label.shape[0])
          weights[:num_samples] = 1. / num_samples
          weights = weights / np.sum(weights)

        else:
          ipositive = np.ravel_multi_index([q[0], q[1], itheta], label.shape)
          output = tf.reshape(output, (-1, 2))
          label = np.int32(np.reshape(label, (int(np.prod(label.shape)), 1)))
          label = np.hstack((1 - label, label))
          weights = np.ones(label.shape[0]) * 0.0025  # magic constant
          weights[ipositive] = 1

        label = tf.convert_to_tensor(label, dtype=tf.int32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
        loss = tf.reduce_mean(loss * weights)
        transport_loss = loss

      elif not self.six_dof:
        label = label.reshape(1, np.prod(label.shape))
        label = tf.convert_to_tensor(label, dtype=tf.float32)
        output = tf.reshape(output, (1, np.prod(output.shape)))
        loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
        loss = tf.reduce_mean(loss)
        transport_loss = loss

      if self.six_dof:

        # Use a window for regression, rather than only exact
        u_window = 7
        v_window = 7
        theta_window = 1
        u_min = max(q[0] - u_window, 0)
        u_max = min(q[0] + u_window + 1, z_tensor.shape[1])
        v_min = max(q[1] - v_window, 0)
        v_max = min(q[1] + v_window + 1, z_tensor.shape[2])
        theta_min = max(itheta - theta_window, 0)
        theta_max = min(itheta + theta_window + 1, z_tensor.shape[3])

        z_est_at_xytheta = z_tensor[0, u_min:u_max, v_min:v_max,
                                    theta_min:theta_max]
        roll_est_at_xytheta = roll_tensor[0, u_min:u_max, v_min:v_max,
                                          theta_min:theta_max]
        pitch_est_at_xytheta = pitch_tensor[0, u_min:u_max, v_min:v_max,
                                            theta_min:theta_max]

        z_est_at_xytheta = tf.reshape(z_est_at_xytheta, (-1, 1))
        roll_est_at_xytheta = tf.reshape(roll_est_at_xytheta, (-1, 1))
        pitch_est_at_xytheta = tf.reshape(pitch_est_at_xytheta, (-1, 1))

        z_est_at_xytheta = self.z_regressor(z_est_at_xytheta)
        roll_est_at_xytheta = self.roll_regressor(roll_est_at_xytheta)
        pitch_est_at_xytheta = self.pitch_regressor(pitch_est_at_xytheta)

        z_weight = 10.0
        roll_weight = 10.0
        pitch_weight = 10.0

        z_label = tf.convert_to_tensor(z_label)[None, Ellipsis]
        roll_label = tf.convert_to_tensor(roll_label)[None, Ellipsis]
        pitch_label = tf.convert_to_tensor(pitch_label)[None, Ellipsis]

        z_loss = z_weight * self.regress_loss(z_label, z_est_at_xytheta)
        roll_loss = roll_weight * self.regress_loss(roll_label,
                                                    roll_est_at_xytheta)
        pitch_loss = pitch_weight * self.regress_loss(pitch_label,
                                                      pitch_est_at_xytheta)

        loss = z_loss + roll_loss + pitch_loss

    if self.six_dof:
      train_vars = self.model.trainable_variables + \
                   self.z_regressor.trainable_variables + \
                   self.roll_regressor.trainable_variables + \
                   self.pitch_regressor.trainable_variables
    else:
      train_vars = self.model.trainable_variables

    if not validate:
      grad = tape.gradient(loss, train_vars)
      self.optim.apply_gradients(zip(grad, train_vars))

    if not self.six_dof:
      self.metric(transport_loss)

    if self.six_dof:
      self.z_metric(z_loss)
      self.roll_metric(roll_loss)
      self.pitch_metric(pitch_loss)

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
  # Visualization methods.
  #-------------------------------------------------------------------------

  def visualize_introspection(self, img_unprocessed, p, rvecs, input_shape,
                              theta, logits, kernel, output):
    """Utils for visualizing features at the end of the Background and Foreground networks."""

    # Do this again, for visualization
    crop_rgb = tf.convert_to_tensor(
        img_unprocessed.copy().reshape(input_shape), dtype=tf.float32)
    crop_rgb = tf.repeat(crop_rgb, repeats=self.num_rotations, axis=0)
    crop_rgb = tfa.image.transform(crop_rgb, rvecs, interpolation="NEAREST")
    crop_rgb = crop_rgb[:, p[0]:(p[0] + self.crop_size),
                        p[1]:(p[1] + self.crop_size), :]
    crop_rgb = crop_rgb.numpy()

    self.ax[0].cla()
    self.ax[1].cla()
    self.ax[2].cla()
    self.ax[3].cla()
    itheta = theta / (2 * np.pi / self.num_rotations)
    itheta = np.int32(np.round(itheta)) % self.num_rotations

    self.ax[0].imshow(crop_rgb[itheta, :, :, :3].transpose(1, 0, 2) / 255.)

    if self.write_visualize:
      # delete first:
      try:
        shutil.rmtree("vis/crop_rgb")
        shutil.rmtree("vis/crop_kernel")
      except:  # pylint: disable=bare-except
        print("Warning: couldn't delete folder for visualization.")

      os.system("mkdir -p vis/crop_rgb")
      os.system("mkdir -p vis/crop_kernel")

      for theta_idx in range(self.num_rotations):
        filename = "itheta_" + str(theta_idx).zfill(4) + ".png"
        if itheta == theta_idx:
          filename = "label-" + filename
        imageio.imwrite(
            os.path.join("vis/crop_rgb/", filename),
            crop_rgb[theta_idx, :, :, :3].transpose(1, 0, 2))

    self.ax[1].imshow(img_unprocessed[:, :, :3].transpose(1, 0, 2) / 255.)
    if self.write_visualize:
      filename = "img_rgb.png"
      imageio.imwrite(
          os.path.join("vis/", filename),
          img_unprocessed[:, :, :3].transpose(1, 0, 2))

    logits_numpy = logits.numpy()
    kernel_numpy = kernel.numpy()

    for c in range(3):
      channel_mean = np.mean(logits_numpy[:, :, :, c])
      channel_std = np.std(logits_numpy[:, :, :, c])
      channel_1std_max = channel_mean + channel_std
      # channel_1std_max = np.max(logits_numpy[:, :, :, c])
      channel_1std_min = channel_mean - channel_std
      # channel_1std_min = np.min(logits_numpy[:, :, :, c])
      logits_numpy[:, :, :, c] -= channel_1std_min
      logits_numpy[:, :, :, c] /= (channel_1std_max - channel_1std_min)
      for theta_idx in range(self.num_rotations):
        channel_mean = np.mean(kernel_numpy[theta_idx, :, :, c])
        channel_std = np.std(kernel_numpy[theta_idx, :, :, c])
        channel_1std_max = channel_mean + channel_std
        # channel_1std_max = np.max(kernel_numpy[itheta, :, :, c])
        channel_1std_min = channel_mean - channel_std
        # channel_1std_min = np.min(kernel_numpy[itheta, :, :, c])
        kernel_numpy[theta_idx, :, :, c] -= channel_1std_min
        kernel_numpy[theta_idx, :, :, c] /= (
            channel_1std_max - channel_1std_min)

    self.ax[2].imshow(logits_numpy[0, :, :, :3].transpose(1, 0, 2))
    self.ax[3].imshow(kernel_numpy[itheta, :, :, :3].transpose(1, 0, 2))

    if self.write_visualize:
      imageio.imwrite(
          os.path.join("vis", "img_features.png"),
          logits_numpy[0, :, :, :3].transpose(1, 0, 2))

      for theta_idx in range(self.num_rotations):
        filename = "itheta_" + str(theta_idx).zfill(4) + ".png"
        if itheta == theta_idx:
          filename = "label-" + filename
        imageio.imwrite(
            os.path.join("vis/crop_kernel/", filename),
            kernel_numpy[theta_idx, :, :, :3].transpose(1, 0, 2))

    heatmap = output[0, :, :, itheta].numpy().transpose()
    # variance = 0.1
    heatmap = -np.exp(-heatmap / 0.1)

    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    heatmap = cmap(norm(heatmap))

    self.ax[4].imshow(heatmap)
    if self.write_visualize:
      imageio.imwrite("vis/heatmap.png", heatmap)

    # non-blocking
    plt.draw()
    plt.pause(0.001)

    # blocking
    # plt.show()

  def visualize_train_input(self, in_img, p, q, theta, z, roll, pitch):
    """Visualize the training input."""
    points = []
    colors = []
    height = in_img[:, :, 3]

    for i in range(in_img.shape[0]):
      for j in range(in_img.shape[1]):
        pixel = (i, j)
        position = utils.pixel_to_position(pixel, height, self.bounds,
                                           self.pixel_size)
        points.append(position)
        colors.append(in_img[i, j, :3])

    points = np.array(points).T  # shape (3, N)
    colors = np.array(colors).T / 255.0  # shape (3, N)

    self.vis["pointclouds/scene"].set_object(
        g.PointCloud(position=points, color=colors))

    pick_position = utils.pixel_to_position(p, height, self.bounds,
                                            self.pixel_size)
    label = "pick"
    utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=0.1)

    pick_transform = np.eye(4)
    pick_transform[0:3, 3] = pick_position
    self.vis[label].set_transform(pick_transform)

    place_position = utils.pixel_to_position(q, height, self.bounds,
                                             self.pixel_size)
    label = "place"
    utils.make_frame(self.vis, label, h=0.05, radius=0.0012, o=0.1)

    place_transform = np.eye(4)
    place_transform[0:3, 3] = place_position
    place_transform[2, 3] = z

    rotation = utils.get_pybullet_quaternion_from_rot((roll, pitch, -theta))
    quaternion_wxyz = np.asarray(
        [rotation[3], rotation[0], rotation[1], rotation[2]])

    place_transform[0:3, 0:3] = mtf.quaternion_matrix(quaternion_wxyz)[0:3, 0:3]
    self.vis[label].set_transform(place_transform)

    _, ax = plt.subplots(2, 1)
    ax[0].imshow(in_img.transpose(1, 0, 2)[:, :, :3] / 255.0)
    ax[0].scatter(p[0], p[1])
    ax[0].scatter(q[0], q[1])
    ax[1].imshow(in_img.transpose(1, 0, 2)[:, :, 3])
    ax[1].scatter(p[0], p[1])
    ax[1].scatter(q[0], q[1])
    plt.show()


class Regressor(tf.keras.Model):
  """Regressor module."""

  def __init__(self):
    super(Regressor, self).__init__()
    activation = "relu"
    self.fc1 = tf.keras.layers.Dense(
        32,
        input_shape=(None, 1),
        kernel_initializer="normal",
        bias_initializer="normal",
        activation=activation)

    self.fc2 = tf.keras.layers.Dense(
        32,
        kernel_initializer="normal",
        bias_initializer="normal",
        activation=activation)

    self.fc3 = tf.keras.layers.Dense(
        1, kernel_initializer="normal", bias_initializer="normal")

  def __call__(self, x):
    return self.fc3(self.fc2(self.fc1(x)))
