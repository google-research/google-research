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

"""Code to visualize the agent's behaviour.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import gin
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
import scipy.misc


@gin.configurable
class LinePlotHelper(object):
  """A helper class for generating line plots."""

  def __init__(self, fontsize=None, max_xwidth=500,
               plot_value_differences=True):
    """Constructor for LinePlotHelper.

    Args:
      fontsize: None or int, size of font to use.
      max_xwidth: int, max width of the x-axis to display, for better
        legibility.
      plot_value_differences: bool, whether to plot the value differences
        with the bisimulation distances.
    """
    self.fontsize = fontsize
    self.plot_value_differences = plot_value_differences
    # Colours. All of these are in BGR format, which is what PyGame eventually
    # expects.
    self.bg_color = '#f8f7f2'  #  '#f1f0ec'  # Clouds
    self.face_color = '#ffffff'
    self.colors = [
        '#71cc2e',  # Emerald
        '#b6599b',  # Amethyst
        '#0fc4f1',  # Sunflower
        '#0054d3',  # Pumpkin
        '#a6a595',  # Concrete
        '#9cbc1a',  # Turquoise
        '#2b39c0',  # Pomegranate
        '#b98029',  # Belize Hole
        '#5e4934',  # Wet Asphalt
    ]
    self.max_xwidth = max_xwidth
    # Create figure, set axes.
    self.fig = plt.figure(frameon=False, figsize=(12, 9))
    self.plot = self.fig.add_subplot(111)
    self.plot_surface = None
    self.distances = []
    self.start_frames = []
    font = {'family': 'Bitstream Vera Sans',
            'weight': 'regular',
            'size': 26}
    matplotlib.rc('font', **font)

  def set_axes_style(self, xlabel='Timestep', ylabel='Distance'):
    """Set up the figure properties."""
    self.fig.patch.set_facecolor(self.face_color)
    ax = self.plot
    ax.set_facecolor(self.bg_color)
    ax.set_xlabel(xlabel, fontsize=self.fontsize - 2)
    ax.set_ylabel(ylabel, fontsize=self.fontsize - 2)
    if self.fontsize is not None:
      ax.tick_params(labelsize=self.fontsize)

  def set_data(self, distances, start_frames):
    """Populate the data arrays for plotting."""
    self.distances = distances.bisimulation
    if self.plot_value_differences:
      self.value_differences = distances.value
    self.start_frames = start_frames

  def set_sorted_data(self, distances):
    self.distances = distances

  def draw_distances(self):
    """Draw the line plot for distances."""
    self.plot.cla()  # Clear current figure.
    self.set_axes_style()
    self.plot.plot(self.distances, color='black')
    if self.plot_value_differences:
      self.plot.plot(self.value_differences, color='red')
    max_xlim = len(self.distances)
    min_xlim = max(0, max_xlim - self.max_xwidth)
    self.plot.set_xlim(min_xlim, max_xlim)
    for sf in self.start_frames:
      self.plot.axvline(x=sf, color='red')
    self.fig.canvas.draw()
    # Now transfer to surface.
    width, height = self.fig.canvas.get_width_height()
    if self.plot_surface is None:
      self.plot_surface = pygame.Surface((width, height))
    plot_buffer = np.frombuffer(self.fig.canvas.buffer_rgba(), np.uint32)
    surf_buffer = np.frombuffer(self.plot_surface.get_buffer(),
                                dtype=np.int32)
    np.copyto(surf_buffer, plot_buffer)
    return self.plot_surface

  def draw_distribution(self):
    """Draw the distribution of distances."""
    self.plot.cla()  # Clear current figure.
    self.set_axes_style(xlabel='Distance', ylabel='Frequency')
    self.plot.hist(self.distances, bins=51)
    self.fig.canvas.draw()
    # Now transfer to surface.
    width, height = self.fig.canvas.get_width_height()
    if self.plot_surface is None:
      self.plot_surface = pygame.Surface((width, height))
    plot_buffer = np.frombuffer(self.fig.canvas.buffer_rgba(), np.uint32)
    surf_buffer = np.frombuffer(self.plot_surface.get_buffer(),
                                dtype=np.int32)
    np.copyto(surf_buffer, plot_buffer)
    return self.plot_surface


@gin.configurable
class AgentVisualizer(object):
  """Code to visualize an agent's behaviour."""

  def __init__(self,
               record_path,
               render_rate=1,
               viz_scale=1,
               fontsize=30,
               generate_videos=True):
    """Constructor for the AgentVisualizer class.

    This class can generate a series of .png files consisting of a number of
    frames. In one of the frames the agent will be visualized interacting
    with the environment. In the other frames other visualizations (such as
    line plots, etc.) can be added).

    Args:
      record_path: str, path where to save png files.
      render_rate: int, frame frequency at which to generate .png files.
      viz_scale: int, scale of visualization.
      fontsize: int, size of fonts to use in plots.
      generate_videos: bool, whether to generate videos.
    """
    assert record_path
    # Dimensions.
    self.original_image_width = 160
    self.original_image_height = 210
    self.viz_image_width = self.original_image_height * viz_scale * 4 // 3
    self.viz_image_height = self.original_image_height * viz_scale
    self.line_plot_width = 4 * self.viz_image_height // 3
    self.line_plot_height = self.viz_image_height
    self.screen_width = self.viz_image_width + self.line_plot_width
    self.screen_height = self.viz_image_height
    self.screen_height += self.viz_image_height

    # Parameters.
    self.render_rate = render_rate
    self.generate_videos = generate_videos
    # Step counter.
    self.step = 0
    # Frame buffer for recording pngs.
    self.record_frame = np.zeros((self.screen_height, self.screen_width, 3),
                                 dtype=np.uint8)
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    self.record_path = record_path
    # Construct distances visualizer.
    self.line_plot = LinePlotHelper(fontsize=fontsize)
    # Initialize pyGame engine.
    self._pygame_init()

    self.source_observation = None
    self.closest_observation = None
    self.closest_distance = np.inf
    self.start_frames = []
    self.sorted_distances_and_frames = []

  def _pygame_init(self):
    """Initalize pygame.
    """
    pygame.init()
    self.screen = pygame.display.set_mode((self.screen_width,
                                           self.screen_height),
                                          0, 32)
    self.game_surface = pygame.Surface((self.original_image_width,
                                        self.original_image_height))

  def visualize(self, environment, agent, start_state, source_state):
    if self.step % self.render_rate == 0:
      self.screen.fill((0, 0, 0))
      distances = agent.get_distances()
      self.render_observation(environment, distances.bisimulation, source_state)
      self.render_distances(distances, start_state, self.viz_image_width, 0)
      self.save_frame()
    self.step += 1

  def render_observation(self, environment, distances, source_state):
    """Render the agent interacting with the environment.

    Args:
      environment: A Gym environment.
      distances: floats, the distances returned by agent.
      source_state: bool, whether we need to record the source state.
    """
    numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                  dtype=np.int32)
    obs = environment.render(mode='rgb_array').astype(np.int32)
    obs = np.transpose(obs)
    obs = np.swapaxes(obs, 1, 2)
    obs = obs[2] | (obs[1] << 8) | (obs[0] << 16)
    np.copyto(numpy_surface, obs.ravel())
    self.screen.blit(pygame.transform.scale(self.game_surface,
                                            (self.viz_image_width,
                                             self.viz_image_height)),
                     (0, 0))
    if source_state and self.source_observation is None:
      self.source_observation = np.copy(obs)
    if self.source_observation is not None:
      numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                    dtype=np.int32)
      np.copyto(numpy_surface, self.source_observation.ravel())
      self.screen.blit(pygame.transform.scale(self.game_surface,
                                              (self.viz_image_width,
                                               self.viz_image_height)),
                       (0, self.viz_image_height))
    if distances:
      current_distance = distances[-1]
      if current_distance < self.closest_distance:
        self.closest_distance = current_distance
        self.closest_observation = np.copy(obs)
      self.sorted_distances_and_frames.append((current_distance, obs))
    if self.closest_observation is not None:
      numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                    dtype=np.int32)
      np.copyto(numpy_surface, self.closest_observation.ravel())
      self.screen.blit(pygame.transform.scale(self.game_surface,
                                              (self.viz_image_width,
                                               self.viz_image_height)),
                       (self.viz_image_width, self.viz_image_height))

  def render_distances(self, distances, start_state, x, y):
    """Draw a plot of distances (line or bar graph).

    Args:
      distances: SequentialDistances, consisting of a list of bisimulation
        distances and a list of value function distances (both floats).
      start_state: bool, whether the current state is the start state.
      x: int, x-position in canvas where to place plot.
      y: int, y-position in canvas where to place plot.
    """
    if not distances.bisimulation:
      return
    if start_state:
      self.start_frames.append(len(distances.bisimulation) - 1)
    self.line_plot.set_data(distances, self.start_frames)
    drawn_plot = self.line_plot.draw_distances()
    scaled_surface = pygame.transform.smoothscale(
        drawn_plot,
        (self.line_plot_width, self.line_plot_height))
    self.screen.blit(scaled_surface, (x, y))

  def render_sorted_distances(self, distances, x, y, distribution=False):
    """Draw a plot of distances (line or bar graph).

    Args:
      distances: list of pairs of distances and frames to render.
      x: int, x-position in canvas where to place plot.
      y: int, y-position in canvas where to place plot.
      distribution: bool, whether to plot the distances distribution (bar
        graph).
    """
    self.line_plot.set_sorted_data(distances)
    if distribution:
      drawn_plot = self.line_plot.draw_distribution()
    else:
      drawn_plot = self.line_plot.draw_distances()
    scaled_surface = pygame.transform.smoothscale(
        drawn_plot,
        (self.line_plot_width, self.line_plot_height))
    self.screen.blit(scaled_surface, (x, y))

  def save_frame(self):
    """Save a frame to disk and generate a video, if enabled."""
    screen_buffer = (
        np.frombuffer(self.screen.get_buffer(), dtype=np.int32)
        .reshape(self.screen_height, self.screen_width))
    sb = screen_buffer[:, 0:self.screen_width]
    self.record_frame[Ellipsis, 2] = sb % 256
    self.record_frame[Ellipsis, 1] = (sb >> 8) % 256
    self.record_frame[Ellipsis, 0] = (sb >> 16) % 256
    frame_number = self.step // self.render_rate
    for file_format in ['png', 'pdf']:
      scipy.misc.imsave('{}/frame_{:06d}.{}'.format(
          self.record_path, frame_number, file_format), self.record_frame)

  def generate_video(self, filename='distances.mp4'):
    if not self.generate_videos:
      return
    os.chdir(self.record_path)
    subprocess.call(['ffmpeg', '-r', '30', '-f', 'image2', '-s', '1920x1080',
                     '-i', 'frame_%06d.png', '-vcodec', 'libx264', '-crf', '25',
                     '-pix_fmt', 'yuv420p', filename])

  def print_sorted_frames(self, sorted_dir):
    """Generate images and video of the frames sorted by distance.

    Args:
      sorted_dir: list of pairs, where first element is distance, second is the
        frame array to be rendered.
    """
    assert self.source_observation is not None
    self.sorted_distances_and_frames.sort(key=lambda x: x[0])
    distances = []
    all_distances = [x[0] for x in self.sorted_distances_and_frames]
    self.start_frames = []
    self.screen.fill((0, 0, 0))
    for i in range(len(self.sorted_distances_and_frames)):
      # Render source state.
      numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                    dtype=np.int32)
      np.copyto(numpy_surface, self.source_observation.ravel())
      self.screen = self.screen.copy()
      self.screen.blit(pygame.transform.scale(self.game_surface,
                                              (self.viz_image_width,
                                               self.viz_image_height)),
                       (0, self.viz_image_height))
      # Render target states.
      distance, obs = self.sorted_distances_and_frames[i]
      numpy_surface = np.frombuffer(self.game_surface.get_buffer(),
                                    dtype=np.int32)
      np.copyto(numpy_surface, obs.ravel())
      self.screen = self.screen.copy()
      self.screen.blit(pygame.transform.scale(self.game_surface,
                                              (self.viz_image_width,
                                               self.viz_image_height)),
                       (self.viz_image_width, self.viz_image_height))
      # Plot the overall distribution of distances.
      self.render_sorted_distances(all_distances, 0, 0, distribution=True)
      # Render a line plot of the increasing distances.
      distances.append(distance)
      self.render_sorted_distances(distances, self.viz_image_width, 0)
      # Save the final image to file.
      screen_buffer = (
          np.frombuffer(self.screen.get_buffer(), dtype=np.int32)
          .reshape(self.screen_height, self.screen_width))
      sb = screen_buffer[:, 0:self.screen_width]
      self.record_frame[Ellipsis, 2] = sb % 256
      self.record_frame[Ellipsis, 1] = (sb >> 8) % 256
      self.record_frame[Ellipsis, 0] = (sb >> 16) % 256
      for file_format in ['png', 'pdf']:
        scipy.misc.imsave('{}/dist_{:06d}.{}'.format(
            sorted_dir, i, file_format), self.record_frame)
    if self.generate_videos:
      os.chdir(sorted_dir)
      subprocess.call(['ffmpeg', '-r', '5', '-f', 'image2', '-s', '1920x1080',
                       '-i', 'dist_%06d.png', '-vcodec', 'libx264', '-crf',
                       '25', '-pix_fmt', 'yuv420p', 'sorted_frames.mp4'])
