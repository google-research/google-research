# Copyright 2024 The Google Research Authors.
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

#!/bin/bash
./scripts/zipnerf/blender_train.sh
./scripts/zipnerf/blender_eval.sh
# There's no blender_render.sh script because the Blender test set is already
# a nice spiral path that you can turn into a video.
python scripts/zipnerf/generate_tables_blender.py 