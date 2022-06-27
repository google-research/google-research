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
# Generate the training data from rocstories raw files.
#
# Example:
#   generate_data.sh $ROCSTORIES_PATH $VOCAB_FILE /tmp/roc_processed
set -x
set -e

RAW_ROCSTORIES_PATH=$1
TESTDATA=$2
VOCAB_FILE=$TESTDATA/wikitext103_32768.subword_vocab
OUTPUT_DIR=$3
BIN="python -m summae.process_rocstories"

function gen_data() {
  $BIN \
    --raw_dir=$RAW_ROCSTORIES_PATH \
    --output_base=${OUTPUT_DIR}/rocstories_springwintertrain \
    --vocab_file=${VOCAB_FILE} \
    --alsologtostderr
  # Wrote 98161 records to 20 shards.
}
cp $VOCAB_FILE $OUTPUT_DIR

gen_data

# Rename test/validation to prevent training on them.
mv $OUTPUT_DIR/rocstories_springwintertrain.all.0018.tfrecord $OUTPUT_DIR/unlabeled.valid.tfrecord
mv $OUTPUT_DIR/rocstories_springwintertrain.all.0019.tfrecord $OUTPUT_DIR/unlabeled.test.tfrecord

cp $TESTDATA/rocstories_gt.valid.tfrecord $OUTPUT_DIR
cp $TESTDATA/rocstories_gt.test.tfrecord $OUTPUT_DIR
