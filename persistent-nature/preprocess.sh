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

# prepare the dataset -- filter LHQ 
python -m preprocessing.filter_lhq # adjust source directory as necessary
# at the end of this step, there should be a folder structure like this:
# dataset/lhq_processed/
# 	img/
# 	dpt_sky/
# 	dpt_depth/


# generate camera poses
python -m preprocessing.generate_poses 
# at the end of this step, there should be a file like this:
# poses/width38.4_far16_noisy_height.pth

# prepare dataset for triplane variation
# need to run the previous two steps first
python -m preprocessing.prepare_triplane_data
# at the end of this step, there should be a folder structure like this:
# dataset/lhq_processed_for_triplane_cam050/
# 	img/
# 	disp/
# 	sky/
# 	dataset.json
