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

"""Helper methods to evaluate performance of algorithms."""

import contextlib
import time
from absl import logging


@contextlib.contextmanager
def timer(message, enable=True):
  """Context manager for timing snippets of code."""
  tick = time.time()
  yield
  tock = time.time()
  if enable:
    logging.info("%s: %.5f seconds", message, (tock - tick))
