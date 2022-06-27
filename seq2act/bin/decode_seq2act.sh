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
source gbash.sh || exit

DEFINE_string output_dir --required "" "Specify the output directory"
DEFINE_string data_files "./seq2act/data/pixel_help/*.tfrecord" \
                         "Specify the test data files"
DEFINE_string checkpoint_path "./seq2act/ckpt_hparams/grounding" \
                              "Specify the checkpoint file"
DEFINE_string problem "pixel_help" "Specify the dataset to decode"

gbash::init_google "$@"

set -e
set -x

virtualenv -p python3.7 .
source ./bin/activate

pip install tensorflow
pip install -r seq2act/requirements.txt

python -m seq2act.bin.seq2act_decode --problem ${FLAGS_problem} \
                                     --data_files "${FLAGS_data_files}" \
                                     --checkpoint_path "${FLAGS_checkpoint_path}" \
                                     --output_dir "${FLAGS_output_dir}"
