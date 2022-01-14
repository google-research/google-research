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

PORT=$1
docker run --rm -it -v `pwd`/results:/results \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -w /results -p $PORT:$PORT \
        --ipc=host nvcr.io/nvidia/tensorflow:21.12-tf2-py3 \
        tensorboard --logdir . --port $PORT
