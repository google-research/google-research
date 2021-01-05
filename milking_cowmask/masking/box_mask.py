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

"""Box mask generation."""
import jax
import jax.numpy as jnp


def generate_boxes(n_boxes, mask_size, scale, random_aspect_ratio, rng):
  """Generate boxes for box masking.

  Used for CutOut or CutMix regularizers.

  Args:
    n_boxes: number of boxes to generate
    mask_size: image size as a `(height, width)` tuple
    scale: either a float in the range (0,1) for a box whose area is a fixed
      proportion of that of the mask
     or 'random_area' for area proportion drawn from U(0,1)
     or 'random_size' for edge length proportion drawn from U(0,1)
    random_aspect_ratio: if True, randomly select aspect ratio, if False fix
      to same AR as image
    rng: a PRNGKey

  Returns:
    boxes as a [n_boxes, [y0, x0, y1, x1]] jnp.array
  """
  k1, k2, k3 = jax.random.split(rng, num=3)
  # Draw area scales
  if isinstance(scale, float):
    area_scales = jnp.ones((n_boxes,), dtype=jnp.float32) * scale
  elif scale == 'random_size':
    area_scales = jax.random.uniform(k1, (n_boxes,), dtype=jnp.float32,
                                     minval=0.0, maxval=1.0) ** 2
  elif scale == 'random_area':
    area_scales = jax.random.uniform(k1, (n_boxes,), dtype=jnp.float32,
                                     minval=0.0, maxval=1.0)
  else:
    raise TypeError('Invalid scale {}'.format(scale))

  j_mask_size = jnp.array(mask_size, dtype=jnp.float32)

  if random_aspect_ratio:
    log_scale = jnp.log(jnp.maximum(area_scales, 1e-8))
    log_aspect_ratios = (jax.random.uniform(
        k2, (n_boxes,), dtype=jnp.float32) * 2 - 1) * log_scale
    aspect_ratios = jnp.exp(log_aspect_ratios)
    root_scale = jnp.sqrt(area_scales)
    root_aspect = jnp.sqrt(aspect_ratios)
    box_props = jnp.stack([root_scale * root_aspect, root_scale / root_aspect],
                          axis=1)
    box_sizes = j_mask_size[None, :] * box_props
  else:
    box_sizes = j_mask_size[None, :] * jnp.sqrt(area_scales)[:, None]

  box_pos = jax.random.uniform(k3, (n_boxes, 2), dtype=jnp.float32) * \
      (j_mask_size[None, :] - box_sizes)

  boxes = jnp.concatenate([box_pos, box_pos + box_sizes], axis=1)

  return boxes


def box_masks(boxes, mask_size):
  """Generate box masks, given boxes from `generate_boxes`.

  Used for CutOut or CutMix regularizers.

  Args:
      boxes: bounding boxes as a [n_boxes, [y0, x0, y1, x1]] tf.Tensor,
      mask_size: image size as a `(height, width)` tuple

  Returns:
      Cut Masks as a [n_boxes, height, width, 1] jnp.array
  """
  y = jnp.arange(0, mask_size[0], dtype=jnp.float32) + 0.5
  x = jnp.arange(0, mask_size[1], dtype=jnp.float32) + 0.5

  boxes = boxes.astype(jnp.float32)

  y_mask = (y[None, :] >= boxes[:, 0:1]) & \
           (y[None, :] <= boxes[:, 2:3])
  x_mask = (x[None, :] >= boxes[:, 1:2]) & \
           (x[None, :] <= boxes[:, 3:4])

  masks = y_mask.astype(jnp.float32)[:, :, None, None] * \
          x_mask.astype(jnp.float32)[:, None, :, None]

  return 1.0 - masks
