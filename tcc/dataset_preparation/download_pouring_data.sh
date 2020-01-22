# Copyright 2020 The Google Research Authors.
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

OUTPUT_DIR=/tmp/pouring_tfrecords/
mkdir -p "${OUTPUT_DIR}"
wget -P "${OUTPUT_DIR}" https://storage.googleapis.com/tcc-pouring/pouring_train-0-of-2.tfrecord
wget -P "${OUTPUT_DIR}" https://storage.googleapis.com/tcc-pouring/pouring_train-1-of-2.tfrecord
wget -P "${OUTPUT_DIR}" https://storage.googleapis.com/tcc-pouring/pouring_val-0-of-1.tfrecord
