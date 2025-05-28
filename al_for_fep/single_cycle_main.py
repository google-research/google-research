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

"""Entry point for running a single cycle of active learning."""

from typing import Sequence

from absl import app
from ml_collections.config_flags import config_flags

from al_for_fep import single_cycle_lib

_CYCLE_CONFIG = config_flags.DEFINE_config_file(
    name='cycle_config',
    default=None,
    help_string='Location of the ConfigDict file containing experiment specifications.',
    lock_config=True)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  single_cycle_lib.MakitaCycle(_CYCLE_CONFIG.value).run_cycle()


if __name__ == '__main__':
  app.run(main)
