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

virtualenv -p python3.7 .
source ./bin/activate

pip install -r seq2act/data_generation/requirements.txt

mkdir -p ${PWD}"/seq2act/data/android_howto/tfexample/"

python -m seq2act.data_generation.create_commoncrawl_dataset \
--input_instruction_json_file=${PWD}"/seq2act/data/android_howto/crawled_instructions.json" \
--input_csv_file=${PWD}"/seq2act/data/android_howto/common_crawl_annotation.csv" \
--vocab_file=${PWD}"/seq2act/data_generation/commoncrawl_rico_vocab_subtoken_44462" \
--output_dir=${PWD}"/seq2act/data/android_howto/tfexample/"

