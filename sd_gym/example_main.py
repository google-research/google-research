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

"""A quick example showing how to create environments."""

import os
from urllib import request

from absl import app

from sd_gym import core
from sd_gym import env as env_lib


SD_MODEL_URL = 'https://exchange.iseesystems.com/model/mindsproject/electric-vehicles-in-norway'
SD_MODEL_FILE_PATH = '/tmp/electric_vehicles_norway.stmx'
GENERATED_MODEL_PATH = '/tmp/generated_sd_models'


def main(argv):
  del argv  # unused

  sd_model_filename, _ = request.urlretrieve(SD_MODEL_URL, SD_MODEL_FILE_PATH)
  if not os.path.exists(GENERATED_MODEL_PATH):
    os.mkdir(GENERATED_MODEL_PATH)

  # PySD doesn't support random, so replace with a fixed value
  with open(sd_model_filename, 'r') as f:
    data = f.read()
    data = data.replace('RANDOM(0.98, 1.02)', '0.99')

  with open(sd_model_filename, 'w') as f:
    f.write(data)

  bptk_params = core.Params(sd_model_filename,
                            env_dt=1.0,
                            sd_dt=.1,
                            simulator='BPTK_Py')
  bptk_sd_env = env_lib.SDEnv(bptk_params)
  bptk_obs = bptk_sd_env.reset()

  print('\nBPTK environment')
  print('Action space: ', bptk_sd_env.action_space)
  print('Observation space: ', bptk_sd_env.observation_space)
  print('Initial conditions: ', bptk_obs)

  pysd_params = core.Params(sd_model_filename,
                            env_dt=1.0,
                            sd_dt=.1,
                            simulator='PySD')
  pysd_sd_env = env_lib.SDEnv(pysd_params)
  pysd_obs = pysd_sd_env.reset()

  print('\nPySD environment')
  print('Action space: ', pysd_sd_env.action_space)
  print('Observation space: ', pysd_sd_env.observation_space)
  print('Initial conditions: ', pysd_obs)


if __name__ == '__main__':
  app.run(main)
