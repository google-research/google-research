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

"""A SDSimulator using the PySD package as the simulation backend."""

from typing import Any, Dict, Optional

import pathlib
import pandas as pd
import numpy as np

from pysd import read_xmile

from sd_gym import core


class PySDSimulator(core.SDSimulator):
  """A SDSimulator using the PySD package as the simulation backend."""

  def __init__(self, params):
    """Compiles SD model file and initializes class from a generated file."""
    self._params = params
    sd_model_file_path = pathlib.Path(params.sd_file)
    self.sd_model = read_xmile(sd_model_file_path)

    vars_df = self.sd_model.doc[['Py Name',
                                 'Type',
                                 'Subtype',
                                 'Limits',
                                 'Units']].dropna(subset=['Type', 'Subtype'])
    exclude = list(self.sd_model.components._control_vars.keys())
    vars_df = vars_df[~vars_df['Py Name'].isin(exclude)]

    self._vars = vars_df['Py Name'].to_list()
    self._stocks = vars_df[
        (vars_df['Type'] == 'Stateful') &
        (vars_df['Subtype'] == 'Integ')]['Py Name'].to_list()
    self._constants = vars_df[
        (vars_df['Type'] == 'Constant') &
        (vars_df['Subtype'] == 'Normal')]['Py Name'].to_list()

    # Set min,max,units for variables.
    self._limits = {}
    self._units = {}
    for _, row in vars_df.iterrows():
      min_, max_ = None, None
      if row['Limits']:
        min_, max_ = row['Limits']
        if np.isnan(min_) and np.isnan(max_):
          min_, max_ = None, None

      units = row['Units']

      self._limits[row['Py Name']] = (min_, max_)
      self._units[row['Py Name']] = units

    if params.starttime:
      self._start_time = params.starttime
      # overwrite start time if set in params.
      self.sd_model.time.set_control_vars(initial_time=params.starttime)
    else:
      self._start_time = self.sd_model.components.initial_time()

    self._stop_time = params.stoptime or self.sd_model.components.final_time()
    self._dt = params.sd_dt or self.sd_model.components.time_step()

    # Get initial conditions by running a timestep, then reset
    self._init_cond = self._run_initial(self._start_time + self._dt,
                                        params.initial_conditions_override,
                                        {}).iloc[:1]
    self.sd_model = read_xmile(sd_model_file_path)
    self.sd_model.time.set_control_vars(initial_time=self._start_time)

  def _run_initial(self, stop_time, initial_conditions, var_actions):
    return self.sd_model.run(params=var_actions,
                             initial_condition=(self._start_time,
                                                initial_conditions),
                             final_time=stop_time,
                             time_step=self._dt,
                             return_columns=self._vars)

  def run(self, stop_time, initial_conditions, var_actions):
    # Only set stock initial conditions before or at start.
    if (self.get_current_time() <= self._start_time or
        np.isclose(self.get_current_time(), self._start_time)):
      return self._run_initial(stop_time, initial_conditions, var_actions)

    return self.sd_model.run(params=var_actions,
                             initial_condition='c',
                             final_time=stop_time,
                             time_step=self._dt,
                             return_columns=self._vars)

  def reset(self):
    self.__init__(self._params)

  def list_variables(self):
    return self._vars

  def list_stocks(self):
    return self._stocks

  def list_constants(self):
    return self._constants

  def list_limits(self):
    return self._limits

  def list_units(self):
    return self._units

  def get_start_time(self):
    return self._start_time

  def get_stop_time(self):
    return self._stop_time

  def get_timestep(self):
    return self._dt

  def get_current_time(self):
    return self.sd_model.time()

  def get_initial_conditions(self):
    return self._init_cond
