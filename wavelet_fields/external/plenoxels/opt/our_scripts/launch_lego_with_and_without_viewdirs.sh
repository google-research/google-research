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

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate plenoxels

export COMMON_ARGS="--debug False --use-pytorch False"
# python opt.py -t checkpoints/lego data/nerf_synthetic/lego -c configs/syn.json $COMMON_ARGS
# python opt.py -t checkpoints/lego_no_view_dirs data/nerf_synthetic/lego -c configs/syn_no_view_dir.json $COMMON_ARGS
python opt.py -t checkpoints/lego_test_low_res data/nerf_synthetic/lego -c configs/syn_testing.json $COMMON_ARGS
# python opt.py -t checkpoints/lego_test_low_res_no_viewdir data/nerf_synthetic/lego -c configs/syn_testing_no_view_dir.json $COMMON_ARGS
