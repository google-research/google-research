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

virtualenv -p python3 bam_py
source bam_py/bin/activate

pip install tf-nightly
pip install -r bam/requirements.txt

BAM_DIR="/tmp/bam"
rm -rf $BAM_DIR
mkdir -p $BAM_DIR/glue_data
mkdir -p $BAM_DIR/pretrained_models/uncased_L-12_H-768_A-12

wget -P $BAM_DIR "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb"
unzip -d $BAM_DIR "${BAM_DIR}/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb"
mv $BAM_DIR/RTE $BAM_DIR/glue_data/rte

wget -P $BAM_DIR/pretrained_models/uncased_L-12_H-768_A-12 "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12/vocab.txt"
wget -P $BAM_DIR/pretrained_models/uncased_L-12_H-768_A-12 "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12/bert_config.json"

python -m bam.run_classifier debug-model $BAM_DIR '{"debug": true, "task_names": ["rte"]}'

