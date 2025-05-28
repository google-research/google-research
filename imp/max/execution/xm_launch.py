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

"""XM launcher (API v2) for MAX.

This launch script is largely based on t5x.
"""

import getpass
import os
from typing import Any, Sequence

from absl import app
from absl import flags
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc
# Add your xmanager launch scripts below


if __name__ == '__main__':
  app.run(main)
