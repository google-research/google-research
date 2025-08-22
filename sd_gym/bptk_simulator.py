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

"""A SDSimulator using the BPTK_Py package as the simulation backend."""

import pathlib
import types

from BPTK_Py.sdcompiler.compile import compile_xmile
from BPTK_Py.sdcompiler.parsers.xmile.xmile import parse_xmile
from BPTK_Py.sdsimulation import SdSimulation
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from sd_gym import core


class BPTKSDSimulator(core.SDSimulator):
  """A SDSimulator using the BPTK_Py package as the simulation backend."""

  _SYSTEM_DYNAMICS_DIR = "."
  _BPTK_GEN_MODEL_CLASS_NAME = "simulation_model"
  _GEN_MODEL_LOC = "generated_sd_models"
  _BPTK_GEN_FILE_END = "_BPTK"

  def __init__(self, params):
    """Compiles SD model file and initializes class from a generated file."""
    self._params = params

    # Sets generated file name to be the same as the sd model file +
    # _BPTK_GEN_FILE_END
    sd_model_file_path = pathlib.Path(params.sd_file)
    gen_file_name = sd_model_file_path.stem + self._BPTK_GEN_FILE_END
    gen_file_loc = pathlib.Path(
        f"{sd_model_file_path.parent}/{self._GEN_MODEL_LOC}/{gen_file_name}.py"
    )
    gen_file_loc = self._SYSTEM_DYNAMICS_DIR / gen_file_loc
    # Uses BPTK to compile the sd model into a python file
    compile_xmile(params.sd_file, gen_file_loc, "py")
    parsed_model = parse_xmile(params.sd_file)

    with open(f"{gen_file_loc}", "r") as fp:
      code = fp.read()

    module = types.ModuleType(gen_file_name)
    exec(code, module.__dict__)
    # Load and initialize generated sd model class
    self.sd_model = SdSimulation(
        getattr(module, self._BPTK_GEN_MODEL_CLASS_NAME)()
    )

    self._vars = list(self.sd_model.mod.equations.keys())
    self._stocks = list(self.sd_model.mod.stocks)
    # Find all constant non-stock components in the model
    self._constants = []
    self._limits = {}
    self._units = {}
    entities = list(parsed_model["models"].values())[0]["entities"]

    # Set min,max,units for variables.
    for item in sum(list(entities.values()), []):
      min_, max_ = (None, None)
      units = ""
      if "scale" in item and item["scale"]:
        min_, max_ = tuple(item["scale"])
      if "units" in item and item["units"]:
        units = item["units"]

      # If stock is non-negative, override min if negative and add if not set.
      # BPTK does no enforcement.
      if item["name"] in self._stocks:
        if item["non_negative"] and (min_ is None or min_ < 0):
          min_ = 0

      self._limits[item["name"]] = (min_, max_)
      self._units[item["name"]] = units

    for item in (entities["aux"] + entities["flow"] + entities["group"]):
      try:
        eval(item["equation"][0])
        self._constants.append(item["name"])
      except:
        continue

    self._start_time = params.starttime or self.sd_model.mod.starttime
    self._stop_time = params.stoptime or self.sd_model.mod.stoptime
    self._dt = params.sd_dt or self.sd_model.mod.dt
    self._init_input_handler()

    # Get initial conditions by running a timestep, then reset
    self._init_cond = self._run_initial(
        params.initial_conditions_override).iloc[:1]
    self.sd_model = SdSimulation(
        getattr(module, self._BPTK_GEN_MODEL_CLASS_NAME)()
    )
    self._init_input_handler()

  def _init_input_handler(self):
    self.input_handler = InputHandler(self._stocks, self._constants,
                                      self._start_time)
    for var_name in (self._stocks + self._constants):
      var_equ = self.sd_model.mod.equations[var_name]
      equ_w_input = lambda t, name=var_name, func=var_equ: self.input_handler(
          name, func, t
      )
      self._update_equation(var_name, equ_w_input)

  def _run_initial(self, initial_conditions):
    self.input_handler.set_initial_conditions(initial_conditions)
    return self.sd_model.start(
        start=self._start_time,
        until=self._start_time + self._dt,
        dt=self._dt,
        output="frame",
        equations=self._vars,
    )

  def run(self, stop_time, initial_conditions, var_actions):
    # Make stocks non-actionable if we're past the start time.
    if initial_conditions and (self.get_current_time() <= self._start_time or
                               np.isclose(self.get_current_time(),
                                          self._start_time)):
      self.input_handler.set_initial_conditions(initial_conditions)

    # Take action by updating internal input time series
    if var_actions:
      self.input_handler.take_action(var_actions,
                                     self.get_current_time(),
                                     stop_time)

    output = self.sd_model.start(
        start=self.get_current_time(),
        until=stop_time,
        dt=self._dt,
        output="frame",
        equations=self._vars,
    )
    # Clear last time memoization. It was run for convenience.
    # Args may change in next run.
    for var_name in self.sd_model.mod.memo:
      if self.get_current_time() in self.sd_model.mod.memo[var_name]:
        del self.sd_model.mod.memo[var_name][self.get_current_time()]
    return output

  def _update_equation(self, equ_name, new_equ):
    self.sd_model.change_equation(equ_name, new_equ)

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
    if self.sd_model.results:
      return max(list(self.sd_model.results.values())[0].keys())
    return self.get_start_time()

  def get_initial_conditions(self):
    return self._init_cond


class InputHandler:
  """Class for injecting input into to SD models.

  The InputHandler class manages injecting input into the model
  and also collects the input signals as they arrive.
  """

  def __init__(self, stocks, constants, start_time):
    self._stocks = stocks
    self._constants = constants
    self._start_time = start_time
    self._var_input_sigs = {}
    self._initial_conditions = {}

  def __call__(self, var_name, func, t):
    """Evaluates the equation corresponding to `var_name` and adds input."""
    # Only set stock initial conditions before or at start
    if var_name in self._stocks:
      if (var_name in self._initial_conditions and
          (t <= self._start_time or np.isclose(t, self._start_time))):
        return self._initial_conditions[var_name]
      else:
        return func(t)

    if var_name in self._constants and var_name not in self._var_input_sigs:
      return func(t)

    # Process non-initial condition inputs
    input_sig = self._var_input_sigs[var_name]
    # series needing interpolation
    if isinstance(input_sig, pd.Series):
      return np.interp(t, input_sig.index, input_sig.values)
    # function
    elif callable(input_sig):
      return input_sig(t)
    # constant
    return input_sig

  def set_initial_conditions(self, initial_conditions):
    """Initializes stocks using signal injection at a fixed start time."""
    self._initial_conditions = initial_conditions

  def take_action(self, var_actions, current_time, next_end_time):
    """Turns an action into signals to inject into SD model."""
    for var_name, input_sig in var_actions.items():
      if var_name not in self._constants:
        continue

      if isinstance(input_sig, (np.ndarray, list)):
        # We assume that the input represents equally spaced points across the
        # next environment timestep.
        timestep_idx = np.linspace(current_time, next_end_time, len(input_sig))
        self._var_input_sigs[var_name] = pd.Series(input_sig,
                                                   index=timestep_idx)
      else:
        self._var_input_sigs[var_name] = input_sig
