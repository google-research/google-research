# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""File containing utility functions for rendering."""

import functools

from diffren.jax import render
from einops import rearrange
from einops import repeat
import jax
import jax.numpy as jnp


def hp_matmul(x, y):
  return jnp.matmul(x, y, precision=jax.lax.Precision.HIGHEST)


vm_hp_matmul = jax.vmap(hp_matmul)


def invert_camera(pose):
  rot, t = pose[Ellipsis, :3, :3], pose[Ellipsis, :3, -1:]
  rot_inv = rearrange(rot, "b x y -> b y x")
  t_inv = -vm_hp_matmul(rot_inv, t)
  bottom = jnp.array([0, 0, 0, 1], dtype=jnp.float32)
  bottom = repeat(bottom, "x -> b 1 x", b=rot_inv.shape[0])
  pose = jnp.concatenate([rot_inv, t_inv], axis=-1)
  return jnp.concatenate([pose, bottom], axis=-2)


def splat_layers_to_target(
    src_image_layers,
    src_disp_layers,
    src_mask_layers,
    src_camtoworld,
    src_intrinsic,
    tgt_camtoworld,
    tgt_intrinsic,
    use_rts,  # pylint: disable=unused-argument
):
  """Function to project source ldi layers to target.

  Args:
    src_image_layers: [b, l, h, w, c]
    src_disp_layers: [b, l, h, w, 1]
    src_mask_layers: [b, l, h, w, 1]
    src_camtoworld: [b, 4, 4]
    src_intrinsic: [b, 4]
    tgt_camtoworld: [b, 4, 4]
    tgt_intrinsic: [b, 4]
    use_rts: bool

  Returns:
    out: projected image.
  """

  # Create Vectices for each layer.
  vertex_fn = jax.vmap(
      create_vertices_intrinsics, in_axes=(1, None), out_axes=1
  )
  vertices = vertex_fn(src_disp_layers, src_intrinsic)  # b l (h w) c

  batch_size, num_layers, image_height, image_width = src_image_layers.shape[
      :-1
  ]

  triangles = create_triangles(image_height, image_width)

  num_vertices = image_height * image_width
  offsets = jnp.array(
      [i * num_vertices for i in range(num_layers)], dtype=triangles.dtype
  )
  #
  offsets = repeat(offsets, "l -> b l 1 1", b=batch_size)

  triangles = repeat(triangles, "x c -> b l x c", b=batch_size, l=num_layers)
  triangles = triangles + offsets

  # Compute the perspective projection matrix.
  target_perspective = perspective_from_intrinsics(tgt_intrinsic)

  # relative_pose = jnp.linalg.inv(tgt_camtoworld) @ src_camtoworld
  relative_pose = hp_matmul(invert_camera(tgt_camtoworld), src_camtoworld)

  # Compute the view projection matrix.
  proj_matrix = hp_matmul(target_perspective, relative_pose)

  # Make the src uv image.
  src_uv = make_uv_image(image_height, image_width)
  src_uv_layers = repeat(
      src_uv, "h w c -> b l h w c", b=batch_size, l=num_layers
  )

  # Reshape to concatenate all the mesh from different layers.
  rgb_attributes = rearrange(src_image_layers, "b l h w c -> b (l h w) c")
  alpha_attributes = rearrange(src_mask_layers, "b l h w 1 -> b (l h w) 1")
  uv_attributes = rearrange(src_uv_layers, "b l h w c -> b (l h w) c")

  rgba = jnp.concatenate([rgb_attributes, alpha_attributes], axis=-1)

  vertices = rearrange(vertices, "b l x c -> b (l x) c")
  triangles = rearrange(triangles, "b l x c -> b (l x) c")
  splat_fn = jax.vmap(
      functools.partial(
          render.render_triangles,
          image_height=image_height,
          image_width=image_width,
          shading_function=shader,
          num_layers=2,
      ),
      in_axes=0,
  )
  out = splat_fn(
      vertices=vertices,
      attributes={"vertex_color": rgba, "vertex_uv": uv_attributes},
      triangles=triangles,
      camera_matrices=proj_matrix,
  )

  return out


def shader(x):
  """Custom shader function.

  Args:
    x: dictionary contaning attributes. Attributes will have shape [h, w, c].

  Returns:
    rgba: shaded output.
  """
  target_rgba = x["vertex_color"]
  target_uvs = x["vertex_uv"]
  target_weights = compute_stretch_weights(target_uvs)
  target_weights = rearrange(target_weights, "h w -> h w 1")
  target_mask = target_rgba[Ellipsis, -1:] > 0.98
  target_weights = target_weights * target_mask

  return jnp.concatenate([target_rgba, target_weights], axis=-1)


def make_uv_image(image_height, image_width):
  u_grid, v_grid = jnp.meshgrid(
      jnp.arange(0, image_width), jnp.arange(0, image_height)
  )
  u_grid = u_grid.astype(jnp.float32)
  v_grid = v_grid.astype(jnp.float32)
  return jnp.stack((u_grid, v_grid), axis=-1)


def sobel_filter(images):
  """Sobel edge detection.

  Args:
   images: shape [h, w, c].

  Returns:
    outputs: Generted sobel filters.
  """
  # There are two 3x3 kernels
  kernels = [
      [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  ]
  kernels = rearrange(
      jnp.asarray(kernels, dtype=jnp.float32) / 8.0, "n x y -> x y 1 n"
  )
  kernels = repeat(kernels, "x y 1 n -> x y c n", c=images.shape[-1])
  # Pad image with reflect.
  images = jnp.pad(images, ((1, 1), (1, 1), (0, 0)), mode="reflect")
  # import ipdb; ipdb.set_trace()
  output = jax.lax.conv(
      rearrange(images, "x y c -> 1 c x y"),
      rearrange(kernels, "x y i o -> o i x y"),
      window_strides=(1, 1),
      padding="VALID",
  )
  return rearrange(output, "1 n x y -> x y n")


def structure_tensor(image):
  image_grads = sobel_filter(image)
  dx, dy = image_grads[Ellipsis, 0], image_grads[Ellipsis, 1]
  dxx = dx * dx
  dxy = dx * dy
  dyy = dy * dy
  # Store only the three unique values of the symmetric structure tensor.
  return jnp.stack((dxx, dxy, dyy), axis=-1)


def compute_stretch_weights(rendered_uvs):
  st_u = structure_tensor(rendered_uvs[Ellipsis, 0:1])
  st_v = structure_tensor(rendered_uvs[Ellipsis, 1:2])
  st = st_u + st_v
  det = jnp.abs(st[Ellipsis, 0] * st[Ellipsis, 2] - st[Ellipsis, 1] * st[Ellipsis, 1]) * 2
  weights = jnp.minimum(det, jnp.ones_like(det))
  return weights


def create_vertices_intrinsics(disparity, intrinsics):
  """3D mesh vertices from a given disparity and intrinsics.

  Args:
     disparity: [B, H, W, 1] inverse depth
     intrinsics: [B, 4] reference intrinsics

  Returns:
     [B, H*W, 3] vertex coordinates.
  """
  # Focal lengths
  fx = intrinsics[:, 0]
  fy = intrinsics[:, 1]
  fx = rearrange(fx, "b -> b 1 1 1")
  fy = rearrange(fy, "b -> b 1 1 1")

  # Centers
  cx = intrinsics[:, 2]
  cy = intrinsics[:, 3]
  cx = rearrange(cx, "b -> b 1 1 1")
  cy = rearrange(cy, "b -> b 1 1 1")

  _, height, width, _ = disparity.shape

  i, j = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
  i = i.astype(jnp.float32)
  j = j.astype(jnp.float32)

  # 0.5 is added to get the position of the pixel centers.
  i = (i + 0.5) / width
  j = (j + 0.5) / height

  i = rearrange(i, "h w -> 1 h w 1")
  j = rearrange(j, "h w -> 1 h w 1")

  depths = 1.0 / jnp.clip(disparity, 0.001, 200.0)
  mx = depths / fx
  my = depths / fy
  px = (i - cx) * mx
  py = (j - cy) * my

  vertices = jnp.concatenate([px, py, depths], axis=-1)
  vertices = rearrange(vertices, "b h w c -> b (h w) c")
  return vertices


def create_triangles(h, w):
  """Creates mesh triangle indices from a given pixel grid size.

     This function is not and need not be differentiable as triangle indices are
     fixed.

  Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.

  Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
  """
  x, y = jnp.meshgrid(jnp.arange(w - 1), jnp.arange(h - 1))
  tl = y * w + x
  tr = y * w + x + 1
  bl = (y + 1) * w + x
  br = (y + 1) * w + x + 1

  triangles = jnp.array([tl, tr, bl, br, bl, tr])
  triangles = rearrange(triangles, "x h w -> h w x")
  triangles = rearrange(triangles, "h w (x y) -> (h w x) y", x=2, y=3)
  return triangles


def perspective_from_intrinsics(intrinsics):
  """Computes a perspective matrix from camera intrinsics.

  The matrix maps camera-space to clip-space (x, y, z, w) where (x/w, y/w, z/w)
  ranges from -1 to 1 in each axis. It's a standard OpenGL-stye perspective
  matrix, except that we use positive Z for the viewing direction (instead of
  negative) so there are sign differences.
  Args:
    intrinsics: [B, 4] Source camera intrinsics tensor (f_x, f_y, c_x, c_y)

  Returns:
    A [B, 4, 4] float32 Tensor that maps from right-handed camera space
    to left-handed clip space.
  """

  focal_x = intrinsics[:, 0]
  focal_y = intrinsics[:, 1]
  principal_x = intrinsics[:, 2]
  principal_y = intrinsics[:, 3]

  zero = jnp.zeros_like(focal_x)
  one = jnp.ones_like(focal_x)
  near_z = 0.001 * one
  far_z = 10000.0 * one

  a = (near_z + far_z) / (far_z - near_z)
  b = -2.0 * near_z * far_z / (far_z - near_z)

  matrix = [
      [2.0 * focal_x, zero, 2.0 * principal_x - 1.0, zero],
      [zero, 2.0 * focal_y, 2.0 * principal_y - 1.0, zero],
      [zero, zero, a, b],
      [zero, zero, one, zero],
  ]
  return jnp.stack([jnp.stack(row, axis=-1) for row in matrix], axis=-2)


def rgba_to_rgb(images, white_bkgd):
  if white_bkgd:
    images = images[Ellipsis, :3] * images[Ellipsis, 3:] + (1.0 - images[Ellipsis, 3:])
  else:
    images = images[Ellipsis, :3]
  return images
