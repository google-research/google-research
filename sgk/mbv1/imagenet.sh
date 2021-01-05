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

dir=$1
width=$2
sparsity=$3

python3 main.py \
  --runmode "imagenet" \
  --ckpt_dir "$dir" \
  --width "$width" \
  --sparsity "$sparsity" \
  --fuse_bnbr \
  --imagenet_glob "/mount/data/imagenet/raw/ILSVRC2012*.JPEG" \
  --imagenet_label "/mount/data/imagenet/raw/val.txt" \
  --num_images -1 \
  --logtostderr

