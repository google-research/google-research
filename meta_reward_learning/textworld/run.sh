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
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r meta_reward_learning/textworld/requirements.txt

TEXTWORLD_LOCATION="/tmp/textworld"
mkdir -p $TEXTWORLD_LOCATION
rm -rf $TEXTWORLD_LOCATION/*

wget -P $TEXTWORLD_LOCATION https://storage.googleapis.com/merl/textworld/datasets.tar.gz
wget -P $TEXTWORLD_LOCATION https://storage.googleapis.com/merl/textworld/merl_ckpt.tar.gz

tar -xvzf $TEXTWORLD_LOCATION/datasets.tar.gz -C $TEXTWORLD_LOCATION/
tar -xvzf $TEXTWORLD_LOCATION/merl_ckpt.tar.gz -C $TEXTWORLD_LOCATION/

python -m meta_reward_learning.textworld.experiment\
            --eval_dir=$TEXTWORLD_LOCATION/merl_ckpt\
            --test_file=$TEXTWORLD_LOCATION/datasets/textworld-test.pkl\
            --log_summaries --eval_only
