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



GPUS="device=${1:-'0'}"
EXPNAME=${2:-'docker_run'}

docker build -t diffusion_3d:latest .

echo "GPUS" $GPUS
docker run \
    --gpus $GPUS \
    --mount type=bind,source="$HOME"/diffusion_3d_logs,target=/gcs/xcloud-shared/$USER/diffusion_3d \
    -it diffusion_3d:latest \
    nvidia-smi

# Run job.
docker run \
    --gpus $GPUS \
    --mount type=bind,source="$HOME"/diffusion_3d_logs,target=/gcs/xcloud-shared/$USER/diffusion_3d \
    -it diffusion_3d:latest \
    python diffusion_3d/main.py \
        --config.view_specific_templates='("{query}. on a white background.",)' \
        --config.seed=0 \
        --config.output_dir="/gcs/xcloud-shared/$USER/diffusion_3d/$(date +"%m-%d-%s")_$EXPNAME" \

# Run an interactive bash session.
# docker run \
#     --gpus $GPUS \
#     --mount type=bind,source="$HOME"/diffusion_3d_logs,target=/gcs/xcloud-shared/jainajay/diffusion_3d \
#     -it diffusion_3d:latest bash
