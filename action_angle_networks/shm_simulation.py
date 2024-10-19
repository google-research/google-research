# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Simulation of simple harmonic motion."""

from typing import Mapping, Sequence, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from matplotlib import animation

import matplotlib.pyplot as plt
import numpy as np


def generate_canonical_coordinates(
    t,
    simulation_parameters,
):
  """Returns q (position) and p (momentum) coordinates at instant t."""
  phi, a, m, w = (simulation_parameters["phi"], simulation_parameters["A"],
                  simulation_parameters["m"], simulation_parameters["w"])
  position = a * jnp.cos(w * t + phi)
  momentum = -m * w * a * jnp.sin(w * t + phi)
  return position, momentum


def compute_hamiltonian(
    position,
    momentum,
    simulation_parameters,
):
  """Computes the Hamiltonian at the given coordinates."""
  m, w = simulation_parameters["m"], simulation_parameters["w"]
  q, p = position, momentum
  hamiltonian = ((p**2) / (2 * m)).sum()
  hamiltonian += (m * (w**2) * (q**2)).sum() / 2
  return hamiltonian


def plot_coordinates(positions, momentums,
                     simulation_parameters,
                     title):
  """Plots coordinates in the canonical basis."""
  assert len(positions) == len(momentums)

  qs, ps = positions, momentums
  qs, ps = np.asarray(qs), np.asarray(ps)
  if qs.ndim == 1:
    qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

  assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
  assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

  # Create new Figure with black background
  fig = plt.figure(figsize=(8, 6), facecolor="black")

  # Add a subplot with no frame
  ax = plt.subplot(frameon=False)

  # Compute Hamiltonians.
  num_steps = qs.shape[0]
  q_max = np.max(np.abs(qs))
  p_max = np.max(np.abs(ps))
  p_scale = (q_max / p_max) / 5
  hs = jax.vmap(  # pytype: disable=wrong-arg-types  # numpy-scalars
      compute_hamiltonian, in_axes=(0, 0, None))(qs, ps, simulation_parameters)
  hs_formatted = np.round(hs.squeeze(), 5)

  def update(t):
    # Update data
    ax.clear()

    # 2 part titles to get different font weights
    ax.text(0.5, 1.0, title + " ", transform=ax.transAxes,
            ha="center", va="bottom", color="w",
            family="sans-serif", fontweight="light", fontsize=16)
    ax.text(0.5, 0.93, "VISUALIZED", transform=ax.transAxes,
            ha="center", va="bottom", color="w",
            family="sans-serif", fontweight="bold", fontsize=16)

    for qs_series, ps_series in zip(qs.T, ps.T):
      ax.scatter(qs_series[t], 10, marker="o", s=40, color="white")
      ax.annotate(
          r"$q$",
          xy=(qs_series[t], 8),
          ha="center",
          va="center",
          size=12,
          color="white")
      ax.annotate(
          r"$p$",
          xy=(qs_series[t], 10 - 0.15),
          xytext=(qs_series[t] + ps_series[t] * p_scale, 10 - 0.15),
          arrowprops=dict(arrowstyle="<-", color="white"),
          ha="center", va="center", size=12,
          color="white")

    ax.plot([0, 0], [5, 15], linestyle="dashed", color="white")

    ax.annotate(
        r"$H = \dfrac{p^2}{2m} + \dfrac{m\omega^2q^2}{2}$ = %0.5f" %
        hs_formatted[t],
        xy=(0, 40),
        ha="center",
        va="center",
        size=14,
        color="white")

    ax.set_xlim(-(q_max * 1.1), (q_max * 1.1))
    ax.set_ylim(-1, 50)

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])

  # Construct the animation with the update function as the animation director.
  anim = animation.FuncAnimation(
      fig, update, frames=num_steps, interval=100, blit=False)
  plt.close()
  return anim


def plot_coordinates_in_phase_space(
    positions, momentums,
    simulation_parameters, title,
):
  """Plots a phase space diagram of the given coordinates."""
  assert len(positions) == len(momentums)

  qs, ps = positions, momentums
  qs, ps = np.asarray(qs), np.asarray(ps)
  if qs.ndim == 1:
    qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

  assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
  assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

  # Create new Figure with black background
  fig = plt.figure(figsize=(8, 6), facecolor="black")

  # Add a subplot.
  ax = plt.subplot(facecolor="black")
  pos = ax.get_position()
  pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]
  ax.set_position(pos)

  # Compute Hamiltonians.
  num_steps = qs.shape[0]
  q_max = np.max(np.abs(qs))
  p_max = np.max(np.abs(ps))
  hs = jax.vmap(  # pytype: disable=wrong-arg-types  # numpy-scalars
      compute_hamiltonian, in_axes=(0, 0, None))(qs, ps, simulation_parameters)
  hs_formatted = np.round(hs.squeeze(), 5)

  def update(t):
    # Update data
    ax.clear()

    # 2 part titles to get different font weights
    ax.text(
        0.5,
        0.83,
        title + " ",
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        color="w",
        family="sans-serif",
        fontweight="light",
        fontsize=16)
    ax.text(
        0.5,
        0.78,
        "PHASE SPACE VISUALIZED",
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        color="w",
        family="sans-serif",
        fontweight="bold",
        fontsize=16)

    for qs_series, ps_series in zip(qs.T, ps.T):
      ax.plot(
          qs_series,
          ps_series,
          marker="o",
          markersize=2,
          linestyle="None",
          color="white")
      ax.scatter(qs_series[t], ps_series[t], marker="o", s=40, color="white")

    ax.text(
        0,
        p_max * 1.7,
        r"$p$",
        ha="center",
        va="center",
        size=14,
        color="white")
    ax.text(
        q_max * 1.7,
        0,
        r"$q$",
        ha="center",
        va="center",
        size=14,
        color="white")

    ax.plot([-q_max * 1.5, q_max * 1.5], [0, 0],
            linestyle="dashed",
            color="white")
    ax.plot([0, 0], [-p_max * 1.5, p_max * 1.5],
            linestyle="dashed",
            color="white")

    ax.annotate(
        r"$H = \dfrac{p^2}{2m} + \dfrac{m\omega^2q^2}{2}$ = %0.5f" %
        hs_formatted[t],
        xy=(0, p_max * 2.4),
        ha="center",
        va="center",
        size=14,
        color="white")

    ax.set_xlim(-(q_max * 2), (q_max * 2))
    ax.set_ylim(-(p_max * 2.5), (p_max * 2.5))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])

  # Construct the animation with the update function as the animation director.
  anim = animation.FuncAnimation(
      fig, update, frames=num_steps, interval=100, blit=False)
  plt.close()
  return anim


def static_plot_coordinates_in_phase_space(
    positions,
    momentums,
    title,
    fig = None,
    ax = None,
    max_position = None,
    max_momentum = None):
  """Plots a static phase space diagram of the given coordinates."""
  assert len(positions) == len(momentums)

  qs, ps = positions, momentums
  qs, ps = np.asarray(qs), np.asarray(ps)
  if qs.ndim == 1:
    qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

  assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
  assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

  if fig is None:
    # Create new Figure with black background
    fig = plt.figure(figsize=(8, 6), facecolor="black")
  else:
    fig.set_facecolor("black")

  if ax is None:
    # Add a subplot.
    ax = plt.subplot(facecolor="black", frameon=False)
  else:
    ax.set_facecolor("black")
    ax.set_frame_on(False)

  # Two part titles to get different font weights
  fig.text(
      x=0.5,
      y=0.83,
      s=title + " ",
      ha="center",
      va="bottom",
      color="w",
      family="sans-serif",
      fontweight="light",
      fontsize=16)
  fig.text(
      x=0.5,
      y=0.78,
      s="PHASE SPACE VISUALIZED",
      ha="center",
      va="bottom",
      color="w",
      family="sans-serif",
      fontweight="bold",
      fontsize=16)

  for qs_series, ps_series in zip(qs.T, ps.T):
    ax.plot(
        qs_series,
        ps_series,
        marker="o",
        markersize=2,
        linestyle="None",
        color="white")
    ax.scatter(qs_series[0], ps_series[0], marker="o", s=40, color="white")

  if max_position is None:
    q_max = np.max(np.abs(qs))
  else:
    q_max = max_position

  if max_momentum is None:
    p_max = np.max(np.abs(ps))
  else:
    p_max = max_momentum

  ax.text(
      0, p_max * 1.7, r"$p$", ha="center", va="center", size=14, color="white")
  ax.text(
      q_max * 1.7, 0, r"$q$", ha="center", va="center", size=14, color="white")

  ax.plot(
      [-q_max * 1.5, q_max * 1.5],  # pylint: disable=invalid-unary-operand-type
      [0, 0],
      linestyle="dashed",
      color="white")
  ax.plot(
      [0, 0],
      [-p_max * 1.5, p_max * 1.5],  # pylint: disable=invalid-unary-operand-type
      linestyle="dashed",
      color="white")

  ax.set_xlim(-(q_max * 2), (q_max * 2))
  ax.set_ylim(-(p_max * 2.5), (p_max * 2.5))

  # No ticks
  ax.set_xticks([])
  ax.set_yticks([])
  plt.close()
  return fig  # pytype: disable=bad-return-type
