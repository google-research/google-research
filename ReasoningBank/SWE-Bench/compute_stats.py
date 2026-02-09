# coding=utf-8
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

import json

# get all folders in ./results
import os
folders = [f.path for f in os.scandir('./results_memory') if f.is_dir()]
print(f"Found {len(folders)} folders.")

# read the json file for each folder
all_results = []
for folder in folders:
    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    if len(json_files) != 1:
        print(f"Skipping {folder}, found {len(json_files)} json files.")
        continue
    json_file = json_files[0]
    with open(os.path.join(folder, json_file), 'r') as f:
        data = json.load(f)
        all_results.append(data)

step = 0
for item in all_results:
    step += item['info']['model_stats']['api_calls']

print(step/500)