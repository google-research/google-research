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

"""Tests for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from saccader import utils


def _construct_images(batch_size):
  image_shape = (50, 50, 3)
  images = tf.convert_to_tensor(
      np.random.randn(*((batch_size,) + image_shape)), dtype=tf.float32)
  return images


def _construct_locations_list(batch_size, num_times):
  locations_list = [
      tf.convert_to_tensor(
          np.random.rand(batch_size, 2) * 2 - 1, dtype=tf.float32)
      for _ in range(num_times)
  ]
  return locations_list


def _count_parameters(vars_list):
  count = 0
  for v in vars_list:
    count += np.prod(v.get_shape().as_list())
  return count


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_draw_bounding_boxes(self):
    batch_size = 5
    images = _construct_images(batch_size)
    image_shape = tuple(images.get_shape().as_list()[1:])
    num_times = 2
    box_shape = (12, 12)
    locations_list = _construct_locations_list(batch_size, num_times)

    normalized_box_size = box_shape[0] / float(image_shape[0])
    images_with_boxes = utils.draw_bounding_boxes(images, locations_list,
                                                  normalized_box_size)

    self.assertEqual((batch_size,) + image_shape,
                     self.evaluate(images_with_boxes).shape)

  def test_location_diversity(self):
    # Zero distance and sqrt(2) distance cases.
    locations_list = [
        tf.convert_to_tensor([[1, 1]], tf.float32),
        tf.convert_to_tensor([[1, 1]], tf.float32),
        tf.convert_to_tensor([[0, 0]], tf.float32)
    ]
    diversity = utils.location_diversity(locations_list)

    expected_result = np.mean([np.sqrt(2.), np.sqrt(2.), 0])
    self.assertAlmostEqual(self.evaluate(diversity), expected_result, 5)

  def test_vectors_alignment(self):
    # Aligned and orthogonal case
    locations_list = [
        tf.convert_to_tensor([[1, 1], [1, 1], [-1, 1]], tf.float32)
    ]
    alignment = utils.vectors_alignment(locations_list)

    expected_result = np.mean([1, 0, 0])
    self.assertAlmostEqual(self.evaluate(alignment), expected_result, 5)

  def test_normalize_range(self):
    min_value = -2
    max_value = 5
    x = tf.convert_to_tensor(np.random.randn(100), dtype=tf.float32)
    x = utils.normalize_range(x, min_value=min_value, max_value=max_value)
    x = self.evaluate(x)
    self.assertEqual(x.min(), min_value)
    self.assertEqual(x.max(), max_value)

  def test_position_channels(self):
    corner_locations = [
        (-1, -1),  # Upper left.
        (-1, 1),  # Upper right.
        (1, -1),  # Lower left.
        (1, 1),  # Lower right.
    ]
    batch_size = len(corner_locations)
    images = _construct_images(batch_size)
    channels = utils.position_channels(images)
    # Corner positions.
    upper_left = channels[0][0, 0]  # Should be position [-1, -1].
    upper_right = channels[1][0, -1]  # Should be position [-1, 1].
    lower_left = channels[2][-1, 0]  # Should be position [1, -1].
    lower_right = channels[3][-1, -1]  # Should be position [1, 1].

    corners = (upper_left, upper_right, lower_left, lower_right)

    corner_locations = tf.convert_to_tensor(corner_locations, dtype=tf.float32)
    glimpses = tf.image.extract_glimpse(
        channels,
        size=(1, 1),
        offsets=corner_locations,
        centered=True,
        normalized=True)

    # Check shape.
    self.assertEqual(channels.shape.as_list(),
                     images.shape.as_list()[:-1] + [
                         2,
                     ])
    corners, glimpses, corner_locations = self.evaluate((corners, glimpses,
                                                         corner_locations))
    glimpses = np.squeeze(glimpses)

    # Check correct corners
    self.assertEqual(tuple(corners[0]), tuple(corner_locations[0]))
    self.assertEqual(tuple(corners[1]), tuple(corner_locations[1]))
    self.assertEqual(tuple(corners[2]), tuple(corner_locations[2]))
    self.assertEqual(tuple(corners[3]), tuple(corner_locations[3]))
    # Check match with extract_glimpse function.
    self.assertEqual(tuple(corners[0]), tuple(glimpses[0]))
    self.assertEqual(tuple(corners[1]), tuple(glimpses[1]))
    self.assertEqual(tuple(corners[2]), tuple(glimpses[2]))
    self.assertEqual(tuple(corners[3]), tuple(glimpses[3]))

  def test_index_to_normalized_location(self):
    image_size = 40
    ground_truth = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]],
                            dtype="float32")
    corner_ixs = tf.constant(
        [[0, 0], [0, image_size], [image_size, 0], [image_size, image_size]],
        dtype=tf.int32)
    normalized_locations = utils.index_to_normalized_location(
        corner_ixs, image_size)
    normalized_locations = self.evaluate(normalized_locations)
    self.assertEqual(np.sum(np.abs(normalized_locations - ground_truth)), 0.)

  @parameterized.named_parameters(
      ("uniform_noise_cover", True),
      ("zero_cover", False),
  )
  def test_location_guide(self, uniform_noise):
    image_shape = (20, 20, 3)
    image = tf.constant(np.random.randn(*image_shape), dtype=tf.float32)
    image, location, blocked_indicator = utils.location_guide(
        image,
        image_size=20,
        open_fraction=0.2,
        uniform_noise=uniform_noise,
        block_probability=0.5)
    image, location, blocked_indicator = self.evaluate((image, location,
                                                        blocked_indicator))
    self.assertEqual(image.shape, image_shape)
    self.assertEqual(location.shape, (2,))

  def test_extract_glimpse(self):
    batch_size = 50
    glimpse_shape = (8, 8)
    images = _construct_images(batch_size)
    location_scale = 1. - float(glimpse_shape[0]) / float(
        images.shape.as_list()[1])
    locations = tf.convert_to_tensor(
        2 * np.random.rand(batch_size, 2) - 1, dtype=tf.float32)
    locations = tf.clip_by_value(locations, -location_scale, location_scale)
    images_glimpse1 = utils.extract_glimpse(
        images, size=glimpse_shape, offsets=locations)
    images_glimpse2 = tf.image.extract_glimpse(
        images,
        size=glimpse_shape,
        offsets=locations,
        centered=True,
        normalized=True)
    diff = tf.reduce_sum(tf.abs(images_glimpse1 - images_glimpse2))
    self.assertEqual(self.evaluate(diff), 0)

  def test_extract_glimpses_at_boundaries(self):
    glimpse_shape = (8, 8)
    locations = tf.constant([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ], dtype=tf.float32)
    images = _construct_images(4)

    glimpse = utils.extract_glimpse(
        images, size=glimpse_shape, offsets=locations)
    glimpses_np, images_np = self.evaluate((glimpse, images))

    true_glimpse = np.stack(
        (np.pad(images_np[0, :4, :4, :], [[4, 0], [4, 0], [0, 0]], "edge"),
         np.pad(images_np[1, :4, -4:, :], [[4, 0], [0, 4], [0, 0]], "edge"),
         np.pad(images_np[2, -4:, :4, :], [[0, 4], [4, 0], [0, 0]], "edge"),
         np.pad(images_np[3, -4:, -4:, :], [[0, 4], [0, 4], [0, 0]], "edge")
        ), 0)

    self.assertAllEqual(glimpses_np, true_glimpse)

  def test_cosine_decay_with_warmup(self):
    global_step = tf.train.get_or_create_global_step()
    inc_global_step = global_step.assign(global_step + 1)

    init_lr = 1.
    total_steps = 20
    lr = utils.cosine_decay_with_warmup(global_step, init_lr, total_steps)
    self.evaluate(tf.global_variables_initializer())
    # Test initial learning rate.
    self.assertEqual(self.evaluate(lr), init_lr)
    for _ in range(total_steps):
      self.evaluate(inc_global_step)

    # Test final learning rate.
    self.assertEqual(self.evaluate(lr), 0.)

  @parameterized.named_parameters(
      ("axis1", 1),
      ("axis2", 2),
      ("axis3", 3),
      ("axis-1", -1),
  )
  def test_batch_gather_nd(self, axis):
    batch_size = 10
    axis_dims = [50, 40, 30]
    x = tf.random.uniform([batch_size,]+axis_dims, minval=0, maxval=1)
    indices = tf.random.uniform(
        (batch_size,),
        minval=0,
        maxval=axis_dims[axis-1 if axis > 0 else axis], dtype=tf.int32)
    x_gathered = utils.batch_gather_nd(x, indices, axis)
    if axis == 1:
      identity = tf.eye(batch_size, batch_size)[:, :, tf.newaxis, tf.newaxis]
    if axis == 2:
      identity = tf.eye(batch_size, batch_size)[:, tf.newaxis, :, tf.newaxis]
    if axis == 3 or axis == -1:
      identity = tf.eye(batch_size, batch_size)[:, tf.newaxis, tf.newaxis, :]
    x_gathered2 = tf.reduce_sum(
        identity * tf.gather(x, indices, axis=axis), axis=axis)
    diff = tf.reduce_sum(tf.math.abs(x_gathered - x_gathered2))
    self.assertAlmostEqual(self.evaluate(diff), 0., 9)

  def test_softmax2d(self):
    batch_size = 10
    axis_dims = [50, 40, 30]
    x = tf.random.uniform([batch_size,]+axis_dims, minval=-1, maxval=1)
    # Sum probability should be equal to 1.
    sum_prob = tf.reduce_sum(utils.softmax2d(x), axis=[1, 2])
    diff = tf.reduce_mean(
        tf.math.abs(sum_prob - tf.ones_like(sum_prob, dtype=tf.float32)))
    self.assertAlmostEqual(self.evaluate(diff), 0., 5)

  def test_onehot2d(self):
    locations = [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]
    ]
    batch_size = len(locations)
    tensor = _construct_images(batch_size)

    onehot = self.evaluate(utils.onehot2d(
        tensor, offsets=tf.constant(locations, dtype=tf.float32)))

    h, w = tensor.shape.as_list()[1:3]
    for i, loc in enumerate(locations):
      # Check edges equal to 1.
      ix_h = 0 if loc[0] == -1 else (h - 1)
      ix_w = 0 if loc[1] == -1 else (w - 1)
      self.assertEqual(onehot[i, ix_h, ix_w, 0], 1)

      # Chech sum2d equal to 1.
      self.assertEqual(onehot[i].sum(), 1)

  @parameterized.named_parameters(
      ("case0", "ASCENDING"),
      ("case1", "DESCENDING"),
  )
  def test_sort2d(self, direction):
    elements = range(0, 9)
    x = tf.convert_to_tensor(
        np.array(
            [
                np.reshape(elements, (3, 3)),
                np.reshape(elements[::-1], (3, 3))
            ]),
        dtype=tf.float32
        )

    ref_indices = utils.position_channels(x)
    sorted_x, argsorted_x = utils.sort2d(x, ref_indices, direction=direction)
    sorted_x = self.evaluate(sorted_x)
    argsorted_x = self.evaluate(argsorted_x)
    # Examples include same elements. So sorted examples should be equal.
    self.assertAllEqual(sorted_x[:, 0], sorted_x[:, 1])

    # Examples are in reverse order. So, indices should be reversed.
    ndims = 2
    for i in range(ndims):
      self.assertAllEqual(argsorted_x[:, 0, i], argsorted_x[:, 1, i][::-1])

  def test_patches_masks(self):
    batch_size = 16
    patch_size = 8
    images = _construct_images(batch_size)
    image_size = images.shape.as_list()[1]
    location_scale = 1. - float(patch_size) / float(image_size)

    locations = tf.convert_to_tensor(
        2 * np.random.rand(batch_size, 2) - 1, dtype=tf.float32)
    locations = tf.clip_by_value(locations, -location_scale, location_scale)
    locations_t = [locations, locations]

    # Construct masks.
    masks = utils.patches_masks(locations_t, image_size, patch_size=patch_size)
    patches_masks = utils.extract_glimpse(
        masks, size=(patch_size, patch_size), offsets=locations)

    # Check the mask value at the patches location is 0.
    self.assertEqual(self.evaluate(tf.reduce_sum(patches_masks)), 0)

    # Check the area of mask is equal the specified patch area.
    patches_area = self.evaluate(
        tf.reduce_sum(1. - masks, axis=[1, 2, 3]))
    self.assertAllEqual(patches_area, np.array([patch_size**2] * batch_size))

  def test_metric_fn(self):
    num_classes = 10
    labels = tf.range(num_classes)
    logits = tf.one_hot(labels, num_classes)
    mask = tf.ones(num_classes)
    metrics = utils.metric_fn(logits, labels, mask)
    with tf.Session() as sess:
      sess.run(tf.local_variables_initializer())
      # Run the update op.
      sess.run(metrics["accuracy/top1"][1])
      sess.run(metrics["accuracy/top5"][1])
      # Read the value.
      top1 = sess.run(metrics["accuracy/top1"][0])
      top5 = sess.run(metrics["accuracy/top5"][0])
    self.assertEqual(top1, 1.)
    self.assertEqual(top5, 1.)

if __name__ == "__main__":
  tf.test.main()
