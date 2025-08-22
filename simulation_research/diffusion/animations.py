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

"""Classes for animating the dynamical systems produced in ode_datasets."""
from typing import Optional, Tuple, List

import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


class Animation:
  """Base class for generic animations of points in 2d and 3d."""

  def __init__(self,
               qt,
               lims = None,
               traj_lw = 1,
               **figkwargs):
    """Init method for Animation.

    Args:
      qt: spatial data to plot of shape (T,n,d) or (T,d)
      lims: tuple[tuple] representing x,y(,z) figure limits
      traj_lw: line width to use for the trajectory
      **figkwargs: additional figure keyword arguments
    """
    self.qt = qt
    if len(self.qt.shape) == 2:
      self.qt = self.qt[:, None, :]
    _, n, d = self.qt.shape  # pylint: disable=invalid-name
    assert d in (2, 3), "too many dimensions for animation"
    self.fig = plt.figure(**figkwargs)
    if d == 3:
      self.ax = self.fig.add_axes([0, 0, 1, 1], projection="3d")
    else:
      self.ax = self.fig.add_axes([0, 0, 1, 1])
    # self.ax.axis('equal')
    xyzmin = self.qt.min(0).min(0)
    xyzmax = self.qt.max(0).max(0)
    delta = xyzmax - xyzmin
    lower = xyzmin - .1 * delta
    upper = xyzmax + .1 * delta
    if lims is None:
      lims = 3 * [(min(lower), max(upper))]
    self.ax.set_xlim(lims[0])
    self.ax.set_ylim(lims[1])
    if d == 3:
      self.ax.set_zlim(lims[2])
    if d != 3:
      self.ax.set_aspect("equal")
    empty = d * [[]]
    colors = [f"C{i}" for i in range(10)]
    self.colors = np.random.choice(colors, size=n, replace=False)
    pts = [self.ax.plot(*empty, "o", ms=6, color=c) for c in self.colors]
    lines = [
        self.ax.plot(*empty, "-", color=c, lw=traj_lw) for c in self.colors
    ]
    self.objects = {
        "pts": sum(pts, []),
        "traj_lines": sum(lines, []),
    }

  def init(self):
    empty = 2 * [[]]
    for obj in self.objects.values():
      for elem in obj:
        elem.set_data(*empty)
        # if self.qt.shape[-1]==3: elem.set_3d_properties([])
    return sum(self.objects.values(), [])

  def update(self, i = 0):
    """Update the figure to timestep i."""
    _, n, d = self.qt.shape  # pylint: disable=invalid-name
    trail_len = 150
    for j in range(n):
      # trails
      xyz = self.qt[max(i - trail_len, 0):i + 1, j, :]
      self.objects["traj_lines"][j].set_data(*xyz[Ellipsis, :2].T)
      if d == 3:
        self.objects["traj_lines"][j].set_3d_properties(xyz[Ellipsis, 2].T)
      self.objects["pts"][j].set_data(*xyz[-1:, Ellipsis, :2].T)
      if d == 3:
        self.objects["pts"][j].set_3d_properties(xyz[-1:, Ellipsis, 2].T)
    # self.fig.canvas.draw()
    return sum(self.objects.values(), [])

  def animate(self):
    return animation.FuncAnimation(
        self.fig,
        self.update,
        frames=self.qt.shape[0],
        interval=33,
        init_func=self.init,
        blit=True)


class PendulumAnimation(Animation):
  """Animation specialized to the N-link pendulum.

  In addition to drawing the
    bobs, also draws the links between them.
  Takes as input the angles with respect to the vertical of each bob
  """

  def __init__(self, thetas, *args, **kwargs):
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    ys = -np.cumsum(cos, -1)
    xs = np.cumsum(sin, -1)
    qt = np.stack([xs, ys], -1)
    super().__init__(qt, *args, **kwargs)
    n = self.qt.shape[1]
    empty = n * [[]]
    self.objects["pts"] = sum(
        [self.ax.plot(*empty, "o", ms=10, c=self.colors[i]) for i in range(n)],
        [])
    self.objects["beams"] = sum(
        [self.ax.plot(*empty, "-", color="k") for _ in range(n)], [])

  def update(self, i = 0):
    beams = [
        np.stack([self.qt[i, k, :], self.qt[i, l, :]], axis=1)
        for (k,
             l) in zip(range(self.qt.shape[-1]), range(1, self.qt.shape[-1]))
    ] + [np.stack([np.zeros(2), self.qt[i, 0, :]], axis=1)]
    for beam, line in zip(beams, self.objects["beams"]):
      line.set_data(*beam[:2])
      if self.qt.shape[-1] == 3:
        line.set_3d_properties(beam[2])
    return super().update(i)
