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
set -e
set -x

virtualenv -p python3.7 .
source ./bin/activate

pip install tensorflow
pip install -r seq2act/requirements.txt
python -m seq2act.bin.setup_test --train_file_list "seq2act/data/android_howto/tfexample/*.tfrecord,seq2act/data/rico_sca/tfexample/*.tfrecord" \
                                 --train_source_list "android_howto,rico_sca" \
                                 --train_batch_sizes "2,2" \
                                 --train_steps 2 \
                                 --batch_size 2 \
                                 --experiment_dir "/tmp/seq2act" \
                                 --logtostderr
