#!/bin/bash
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



export TZ="America/Los_Angeles"
JOBID=$(date +"%Y_%m_%d_%H_%M_%S")
CONFIG="vg_blip2_instruct_vicuna7b_ckd"
EPOCH=1


MPORT=$(shuf -i 6000-9999 -n 1); CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m \
torch.distributed.run --nproc_per_node=8 --master_port $MPORT train_ckd.py \
 --cfg-path configs/${CONFIG}.yaml \
 --job_id $JOBID
