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
# use DATA_NAME from one of {pure, eps1, eps3, gaussian1, gaussian3}
# to use already saved policies on CNS.
# use DATA_NAME=example to use locally saved policies
DATA_NAME=example
python collect_data.py \
  --alsologtostderr \
  --sub_dir=0 \
  --env_name=HalfCheetah-v2 \
  --data_name=$DATA_NAME \
  --config_file=dcfg_$DATA_NAME \
  --n_samples=1000000 \
