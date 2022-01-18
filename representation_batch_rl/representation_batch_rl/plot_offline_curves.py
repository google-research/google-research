# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Plots curves from offline algorithms overlaid with those of online demonstrations.

"""

import glob
from typing import Sequence

from absl import app


def main(argv):
  del argv
  path_online = ('experiments/'
                 '20210607_2023.policy_weights_dmc_1M_SAC_pixel/tb/events.out*')

  online_files = glob.glob(path_online)
  print(online_files)

if __name__ == '__main__':
  app.run(main)
