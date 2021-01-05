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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from single_view_mpi.libs import mpi


class MPITest(tf.test.TestCase):

  def test_layer_weights_and_visibility(self):
    # A 4-layer 2x2 set of alphas. The back (first) layer is opaque, as usual.
    alphas = [
        [[[1.0], [1.0]], [[1.0], [1.0]]],
        [[[0.0], [0.8]], [[0.5], [0.5]]],
        [[[1.0], [0.5]], [[0.5], [0.2]]],
        [[[0.0], [1.0]], [[0.0], [0.5]]]]  # pyformat: disable
    visibility = mpi.layer_visibility(tf.constant(alphas))
    self.assertAllClose(visibility, [
        [[[0.0], [0.0]], [[0.25], [0.2]]],
        [[[0.0], [0.0]], [[0.5], [0.4]]],
        [[[1.0], [0.0]], [[1.0], [0.5]]],
        [[[1.0], [1.0]], [[1.0], [1.0]]]])  # pyformat: disable
    weights = mpi.layer_weights(tf.constant(alphas))
    self.assertAllClose(weights, [
        [[[0.0], [0.0]], [[0.25], [0.2]]],
        [[[0.0], [0.0]], [[0.25], [0.2]]],
        [[[1.0], [0.0]], [[0.5], [0.1]]],
        [[[0.0], [1.0]], [[0.0], [0.5]]]])  # pyformat: disable
    total_weight = tf.reduce_sum(weights, axis=-4)
    self.assertAllClose(total_weight, tf.ones_like(total_weight))

  def test_compose_back_to_front(self):
    # A 3-layer 2x2 MPI.
    red = [1.0, 0.0, 0.0, 1.0]
    black = [0.0, 0.0, 0.0, 1.0]
    half_green = [0.0, 1.0, 0.0, 0.5]
    quarter_blue = [0.0, 0.0, 1.0, 0.25]
    none = [0.0, 0.0, 0.0, 0.0]
    white = [1.0, 1.0, 1.0, 1.0]
    layers = tf.constant([
        [[black, black], [red, white]],
        [[red, half_green], [none, red]],
        [[none, none], [quarter_blue, black]]])  # pyformat: disable
    composed = mpi.compose_back_to_front(layers)
    self.assertAllClose(composed, [
        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        [[0.75, 0.0, 0.25], [0.0, 0.0, 0.0]]])  # pyformat: disable

  def test_disparity_from_layers(self):
    # A 3-layer 2x2 MPI. Only the alpha channel is relevant.
    one = [0.0, 0.0, 0.0, 1.0]
    zero = [0.0, 0.0, 0.0, 0.0]
    half = [0.0, 0.0, 0.0, 0.5]
    quarter = [0.0, 0.0, 0.0, 0.25]
    layers = tf.constant([
        [[one, one], [one, one]],
        [[one, zero], [half, quarter]],
        [[one, half], [zero, quarter]]])  # pyformat: disable
    # Layer disparities will be 1, 2 and 3.
    depths = mpi.make_depths(1.0 / 3.0, 1.0, 3)
    disparity = mpi.disparity_from_layers(layers, depths)
    self.assertAllClose(disparity, [
        [[3.0], [2.0]], [[1.5], [1.6875]]])  # pyformat: disable

  def test_make_depths(self):
    self.assertAllClose(mpi.make_depths(2.0, 18.0, 2), [18.0, 2.0])
    self.assertAllClose(
        mpi.make_depths(1.0, 10.0, 10), [10.0 / (x + 1.0) for x in range(10)])

  def test_render(self):
    red = [1.0, 0.0, 0.0, 1.0]
    blue = [0.0, 0.0, 1.0, 1.0]
    green = [0.0, 1.0, 0.0, 1.0]
    none = [0.0, 0.0, 0.0, 0.0]
    # An example 8x8 MPI with 3 layers:
    layers = tf.constant([
        # * a solid red background layer (8x8)
        [[red, red, red, red, red, red, red, red]] * 8,
        # * a 4x4 blue square in the center
        [[none, none, none, none, none, none, none, none]] * 2
        + [[none, none, blue, blue, blue, blue, none, none]] * 4
        + [[none, none, none, none, none, none, none, none]] * 2,
        # * a 2x2 green square in the center
        [[none, none, none, none, none, none, none, none]] * 3
        + [[none, none, none, green, green, none, none, none]] * 2
        + [[none, none, none, none, none, none, none, none]] * 3
    ])  # pyformat: disable
    # Disparities will be 1/6, 1/3, 1/2.
    depths = mpi.make_depths(2.0, 6.0, 3)
    # 90-degree field of view
    intrinsics = tf.constant([0.5, 0.5, 0.5, 0.5])
    identity_pose = tf.constant([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]])  # pyformat: disable

    # Rendering at the reference position gives three concentric squares.
    reference_image = mpi.render(layers, depths, identity_pose, intrinsics,
                                 identity_pose, intrinsics)
    r = [1.0, 0.0, 0.0]
    g = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0]
    self.assertAllClose(
        reference_image,
        [[r, r, r, r, r, r, r, r]] * 2
        + [[r, r, b, b, b, b, r, r]]
        + [[r, r, b, g, g, b, r, r]] * 2
        + [[r, r, b, b, b, b, r, r]]
        + [[r, r, r, r, r, r, r, r]] * 2)  # pyformat: disable

    # At the reference position with narrower field of view, we only see the
    # inner two squares:
    narrow_intrinsics = tf.constant([1.0, 1.0, 0.5, 0.5])
    narrow_image = mpi.render(
        layers, depths,
        identity_pose, intrinsics,
        identity_pose, narrow_intrinsics,
        height=4, width=4)  # pyformat: disable
    self.assertAllClose(
        narrow_image,
        [[b, b, b, b]]
        + [[b, g, g, b]] * 2
        + [[b, b, b, b]])  # pyformat: disable

    # Back to wider field of view. If we move forward enough, the central
    # green square fills the image:
    close_pose = tf.constant([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.75]])  # pyformat: disable
    close_image = mpi.render(
        layers, depths,
        identity_pose, intrinsics,
        close_pose, intrinsics,
        height=4, width=4)  # pyformat: disable
    self.assertAllClose(close_image, [[g, g, g, g]] * 4)

    # If we move far enough sideways, only the background layer is visible:
    side_pose = tf.constant([
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]])  # pyformat: disable
    side_image = mpi.render(
        layers, depths,
        identity_pose, intrinsics,
        side_pose, intrinsics,
        height=5, width=7)  # pyformat: disable
    self.assertAllClose(side_image, [[r, r, r, r, r, r, r]] * 5)


if __name__ == '__main__':
  tf.test.main()
