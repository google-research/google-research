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

"""Script to monitor current memory usage.

Useful for long-running experiments which consume large amounts of memory.
"""

import datetime
import time

import psutil


prev_pct = None
prev_used = None
while True:
  now = datetime.datetime.now()
  vmem = psutil.virtual_memory()
  pct = vmem[2]
  used = round(vmem[3] / 1000000000, 1)

  if prev_used != used:
    print(f'[{now}]\t{used} GB used ({pct}%)')
    prev_pct = pct
    prev_used = used

  time.sleep(1)

