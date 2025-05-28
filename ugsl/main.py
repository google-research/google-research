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

# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Runner of the UGSL framework."""
import os

from absl import app
from ml_collections import config_flags

from ugsl import trainer


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(__file__), "config.py"),
    "Path to file containing configuration hyperparameters. "
    "File must define method `get_config()` to return an instance of "
    "`config_dict.ConfigDict`",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  trainer.train(_CONFIG.value)


if __name__ == "__main__":
  app.run(main)
