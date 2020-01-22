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
set -e
set -x

# Install TensorFlow with GPU support before running this script.

CAPSULE_LOCATION="/tmp/capsule"
rm -rf $CAPSULE_LOCATION
mkdir $CAPSULE_LOCATION

wget -P $CAPSULE_LOCATION/ https://storage.googleapis.com/capsule_toronto/norb_em_checkpoints.tar.gz
wget -P $CAPSULE_LOCATION/ https://storage.googleapis.com/capsule_toronto/smallNORB_data.tar.gz

tar -xvzf $CAPSULE_LOCATION/smallNORB_data.tar.gz -C $CAPSULE_LOCATION/smallNORB/
tar -xvzf $CAPSULE_LOCATION/norb_em_checkpoints.tar.gz -C $CAPSULE_LOCATION/


python -m capsule_em.experiment --train=0 --eval_once=1 --eval_size=24300\
            --ckpnt=$CAPSULE_LOCATION/model.ckpt-1\
            --norb_data_dir=$CAPSULE_LOCATION/smallNORB/

