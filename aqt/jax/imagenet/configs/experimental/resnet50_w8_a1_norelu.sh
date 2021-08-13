# Copyright 2021 The Google Research Authors.
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
gxm ../../google/xm_launch.py \
  --platform=df --tpu_topology=8x8 \
  --batch_size=8192 \
  --hparams_config_filename=experimental/resnet50_w8_a1_norelu.py \
  --name=resnet50-w8a1-baseline \
  --xm_resource_pool=peace --xm_resource_alloc=user:peace/catalyst
