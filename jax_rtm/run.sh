#!/bin/bash
# Copyright 2026 The Google Research Authors.
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


set -e

# Install dependencies
pip install -r requirements.txt

# Automatically download Zenodo assets if missing
if [ ! -f "data/ping_yang_multi_habit.npz" ] || [ ! -f "data/weather_85x85.npz" ] || [ ! -f "data/weather_339x339.npz" ]; then
  echo "Downloading required large data assets from Zenodo..."
  python3 download_data.py
fi

# Run the test suite
pytest
