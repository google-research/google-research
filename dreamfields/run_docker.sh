# Copyright 2022 The Google Research Authors.
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

GPUS=$1
QUERY=$2

echo "GPUS $GPUS"
echo "QUERY $QUERY"

DOCKER_BUILDKIT=1 docker build -t dreamfields:v1 .

mkdir -p results

docker run --gpus $GPUS \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -v $HOME/.cache/scenic:/root/.cache/scenic \
  -v `pwd`/results:/dreamfields/results \
  --ipc=host dreamfields:v1 \
  python run.py --config=dreamfields/config/config_lq.py --query="$QUERY"
