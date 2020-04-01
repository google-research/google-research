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

"""Session hooks for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path as osp

from absl import logging
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest
from tensorflow.compat.v1.io import gfile

from stacked_capsule_autoencoders.capsules.train import tools as _tools


def make_grid(batch, grid_height=None, zoom=1, old_buffer=None, border_size=1):
  """Creates a grid out an image batch.

  Args:
    batch: numpy array of shape [batch_size, height, width, n_channels]. The
      data can either be float in [0, 1] or int in [0, 255]. If the data has
      only 1 channel it will be converted to a grey 3 channel image.
    grid_height: optional int, number of rows to have. If not given, it is
      set so that the output is a square. If -1, then tiling will only be
      vertical.
    zoom: optional int, how much to zoom the input. Default is no zoom.
    old_buffer: Buffer to write grid into if possible. If not set, or if shape
      doesn't match, we create a new buffer.
    border_size: int specifying the white spacing between the images.

  Returns:
    A numpy array corresponding to the full grid, with 3 channels and values
    in the [0, 255] range.

  Raises:
    ValueError: if the n_channels is not one of [1, 3].
  """

  batch_size, height, width, n_channels = batch.shape

  if grid_height is None:
    n = int(math.ceil(math.sqrt(batch_size)))
    grid_height = n
    grid_width = n
  elif grid_height == -1:
    grid_height = batch_size
    grid_width = 1
  else:
    grid_width = int(math.ceil(batch_size/grid_height))

  if n_channels == 1:
    batch = np.tile(batch, (1, 1, 1, 3))
    n_channels = 3

  if n_channels != 3:
    raise ValueError('Image batch must have either 1 or 3 channels, but '
                     'was {}'.format(n_channels))

  # We create the numpy buffer if we don't have an old buffer or if the size has
  # changed.
  shape = (height * grid_height + border_size * (grid_height - 1),
           width * grid_width + border_size * (grid_width - 1),
           n_channels)
  if old_buffer is not None and old_buffer.shape == shape:
    buf = old_buffer
  else:
    buf = np.full(shape, 255, dtype=np.uint8)

  multiplier = 1 if batch.dtype in (np.int32, np.int64) else 255

  for k in range(batch_size):
    i = k // grid_width
    j = k % grid_width
    arr = batch[k]
    x, y = i * (height + border_size), j * (width + border_size)
    buf[x:x + height, y:y + width, :] = np.clip(multiplier * arr,
                                                0, 255).astype(np.uint8)

  if zoom > 1:
    buf = buf.repeat(zoom, axis=0).repeat(zoom, axis=1)
  return buf


class PlottingHook(tf.train.SessionRunHook):
  """Hook for saving numpy arrays as images."""

  def __init__(self,
               output_dir,
               data_dict,
               save_secs=None,
               save_steps=None,
               basename=None,
               zoom=1.,
               write_current=False,
               write_last=True,
               matplotlib_plot_func=None,
               param_dict=None):
    """Builds the object.

    Args:
      output_dir: local or cns path.
      data_dict: dictionary of image tensors.
      save_secs: number of seconds to wait until saving.
      save_steps: number of steps to wait until saving.
      basename: name added to files as prefix.
      zoom: image size is increased by this factor.
      write_current: if True, images are written as {name}_{step}.png
        and images for all steps are kept.
      write_last: if True, images are written as {name}_last.png and are
        overwritten every time the hook is invoked.
      matplotlib_plot_func: a callable that takes the evaluated `data_dict` and
        returns another dict of key: np.ndarray pairs; it can be used to
        plot stuff with matplotlib.
      param_dict: dict of parameters for plotting.
    """

    logging.info('Create PlottingHook.')
    self._logdir = output_dir
    self._data_dict = data_dict
    self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
                                             every_steps=save_steps)
    self._basename = basename
    self._zoom = zoom
    self._write_current = write_current
    self._write_last = write_last
    self._matplotlib_plot_func = matplotlib_plot_func
    self._param_dict = param_dict if param_dict else dict()
    self._steps_per_run = 1
    self._last_run = -1

  def _set_steps_per_run(self, steps_per_run):
    self._steps_per_run = steps_per_run

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError('Global step should be created to use PlottingHook.')

    if not gfile.exists(self._logdir):
      gfile.makedirs(self._logdir)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    run_args = [self._global_step_tensor]

    global_step = self._last_run + self._steps_per_run
    should_trigger = self._timer.should_trigger_for_step(global_step)
    if should_trigger:
      run_args.append(self._data_dict)

    return tf.train.SessionRunArgs(run_args)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results[0]
    self._last_run = stale_global_step + self._steps_per_run

    if self._timer.should_trigger_for_step(self._last_run):
      self._plot(self._last_run, run_values.results[1])
      self._timer.update_last_triggered_step(self._last_run)

  def end(self, session):
    last_step = session.run(self._global_step_tensor)
    if last_step != self._timer.last_triggered_step():
      self._plot(last_step, session.run(self._data_dict))

  def _plot(self, global_step, data_dict):

    if self._matplotlib_plot_func is not None:
      data_dict = self._matplotlib_plot_func(data_dict)

    filename = '{key}_{which}.png'
    if self._basename:
      filename = self._basename + '_' + filename

    filename = osp.join(self._logdir, filename)

    for k, v in data_dict.items():

      zoom = self._param_dict.get(k, dict()).get('zoom', self._zoom)
      grid_height = self._param_dict.get(k, dict()).get('grid_height', None)

      if self._write_last:
        self._savefile(filename.format(key=k, which='last'), v,
                       zoom, grid_height)

      if self._write_current:
        global_step = '{:05}'.format(global_step)
        self._savefile(filename.format(key=k, which=global_step), v,
                       zoom, grid_height)

  def _savefile(self, path, img, zoom, grid_height):

    data = make_grid(img, zoom=zoom, grid_height=grid_height)

    # Writing takes time, and opening a file for writing erases its contents,
    # so it's better to write to a temporary file and then copy the results.
    dirname, basename = osp.dirname(path), osp.basename(path)
    temp_file = osp.join(dirname, '.' + basename)
    with gfile.GFile(temp_file, 'wb') as f:
      img = Image.fromarray(data)
      img.save(f, format='PNG')

    gfile.copy(temp_file, path, overwrite=True)
    gfile.remove(temp_file)


class UpdateOpsHook(tf.train.SessionRunHook):
  """Hook for running custom update hooks."""

  def __init__(self,
               update_ops=None,
               collections=_tools.GraphKeys.CUSTOM_UPDATE_OPS,
               save_secs=None,
               save_steps=None):
    """Builds the object.

    Args:
      update_ops: an op or a list of ops.
      collections: tensorflow collection key or a list of collections keys.
      save_secs: number of seconds to wait until saving.
      save_steps: number of steps to wait until saving.
    """

    logging.info('Create UpdateOpsHook.')

    update_ops = nest.flatten(update_ops) if update_ops else []
    if collections:
      for collection in nest.flatten(collections):
        update_ops.extend(tf.get_collection(collection))

    self._update_op = tf.group(update_ops)
    self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
                                             every_steps=save_steps)

    self._steps_per_run = 1
    self._last_run = -1

  def _set_steps_per_run(self, steps_per_run):
    self._steps_per_run = steps_per_run

  def begin(self):
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError('Global step should be created to use UpdateOpsHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    run_args = [self._global_step_tensor]

    global_step = self._last_run + self._steps_per_run
    should_trigger = self._timer.should_trigger_for_step(global_step)
    if should_trigger:
      run_args.append(self._update_op)
      self._timer.update_last_triggered_step(global_step)

    return tf.train.SessionRunArgs(run_args)

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results[0]
    self._last_run = stale_global_step + self._steps_per_run


def create_hooks(parsed_flags, plot_dict=None, plot_params=None):
  """Creates session hooks."""
  config = parsed_flags.config
  logdir = parsed_flags.logdir

  hooks = dict(
      step_counter=tf.train.StepCounterHook(
          config.report_loss_steps,
          output_dir=logdir
      ),
      summary=tf.train.SummarySaverHook(
          save_steps=config.summary_steps,
          output_dir=logdir,
          summary_op=tf.summary.merge_all()
      ),
  )

  if config.snapshot_secs or config.snapshot_steps:
    hooks['saver'] = tf.train.CheckpointSaverHook(
        logdir,
        save_secs=config.snapshot_secs if config.snapshot_secs else None,
        save_steps=config.snapshot_steps if config.snapshot_steps else None,
        saver=tf.train.Saver(max_to_keep=config.snapshots_to_keep)
    )

  if plot_dict is not None:
    hooks['plot'] = PlottingHook(
        save_steps=config.plot_steps,
        output_dir=logdir,
        data_dict=plot_dict,
        zoom=5.,
        param_dict=plot_params,
    )

  if config.global_ema_update:
    hooks['update'] = UpdateOpsHook(
        save_steps=config.run_updates_every
    )

  return hooks.values()
