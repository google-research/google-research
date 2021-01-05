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

pip install tensorflow_datasets tensorflow sklearn numpy
# If the task_pb2.py doesn't work, you will need to install protobuf compiler
# (https://github.com/protocolbuffers/protobuf/blob/master/README.md#protocol-compiler-installation)
# and run the line below to compite the protobuf for your environment.
# protoc --python_out=. task.proto
