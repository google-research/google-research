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

JOB_NAME="norbcapsule_`date +"%b%d_%H%M%S"`"
TAG="norbcapsule_local_gpu"

docker build -f DockerfileGPU -t $TAG $PWD
docker  run -v \
  $HOME/datasets/smallNORB:/root/datasets/smallNORB \
  --runtime=nvidia $TAG \
  --job_name $JOB_NAME
