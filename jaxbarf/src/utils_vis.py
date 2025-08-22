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

"""Visualization utilities."""
import io

import jax.numpy as jnp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jaxbarf.src import camera


def get_camera_mesh(pose,depth=0.5):
  """Get camera mesh."""
  vertices = jnp.array([[-0.5, -0.5, 1],
                        [0.5, -0.5, 1],
                        [0.5, 0.5, 1],
                        [-0.5, 0.5, 1],
                        [0, 0, 0]])*depth
  faces = jnp.array([[0, 1, 2],
                     [0, 2, 3],
                     [0, 1, 4],
                     [1, 2, 4],
                     [2, 3, 4],
                     [3, 0, 4]])
  vertices = camera.cam2world(vertices[None], pose)
  wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
  return vertices, faces, wireframe


def plot_poses(pose, pose_ref=None, step=None):
  """Plot poses."""
  # set up plot window(s)
  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111, projection="3d")
  ax.set_title("step {}".format(step), pad=0)
  setup_3d_plot(
      ax, elev=45, azim=35, x_lim=(-3, 3), y_lim=(-3, 3), z_lim=(-3, 2.4))
  plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0, hspace=0)
  plt.margins(tight=True, x=0, y=0)
  # plot reference camera
  if pose_ref is not None:
    num = len(pose_ref)
    ref_color = (0.7, 0.2, 0.7)
    _, _, cam_ref = get_camera_mesh(pose_ref)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],
                                         alpha=0.2, facecolor=ref_color))
    for i in range(num):
      ax.plot(cam_ref[i, :, 0], cam_ref[i, :, 1], cam_ref[i, :, 2],
              color=ref_color, linewidth=0.5)
      ax.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1], cam_ref[i, 5, 2],
                 color=ref_color, s=20)
  # plot reference camera
  if pose is not None:
    num = len(pose)
    pred_color = (0, 0.6, 0.7)
    _, _, cam = get_camera_mesh(pose)
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],
                                         alpha=0.2, facecolor=pred_color))
    for i in range(num):
      ax.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2],
              color=pred_color, linewidth=1)
      ax.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2],
                 color=pred_color, s=20)
  # plot links between pred camera and reference camera
  if (pose_ref is not None) and (pose is not None):
    for i in range(num):
      ax.plot([cam[i, 5, 0], cam_ref[i, 5, 0]],
              [cam[i, 5, 1], cam_ref[i, 5, 1]],
              [cam[i, 5, 2], cam_ref[i, 5, 2]], color=(1, 0, 0),
              linewidth=3)
  # return ploted figure for tensorboard
  with io.BytesIO() as buff:
    fig.savefig(buff, format='png')
    buff.seek(0)
    im = plt.imread(buff)
  return im


def setup_3d_plot(ax, elev, azim, x_lim, y_lim, z_lim):
  """Setup 3D plot."""
  ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
  ax.xaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)  # pylint:disable=protected-access
  ax.yaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)  # pylint:disable=protected-access
  ax.zaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)  # pylint:disable=protected-access
  ax.xaxis.set_tick_params(labelsize=8)
  ax.yaxis.set_tick_params(labelsize=8)
  ax.zaxis.set_tick_params(labelsize=8)
  ax.set_xlabel("X", fontsize=16)
  ax.set_ylabel("Y", fontsize=16)
  ax.set_zlabel("Z", fontsize=16)
  ax.set_xlim(x_lim[0], x_lim[1])
  ax.set_ylim(y_lim[0], y_lim[1])
  ax.set_zlim(z_lim[0], z_lim[1])
  ax.view_init(elev=elev, azim=azim)
