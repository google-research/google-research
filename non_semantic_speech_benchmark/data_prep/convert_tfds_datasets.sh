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
#
# bash convert_tfds_datasets.sh
#

OUTPUT_BASE="~/tmp/noss"

function convert_tfds() {
  local dataset_name=$1
  local split_name=$2

  python3 -m audio_to_embeddings_beam \
--alsologtostderr \
--tfds_dataset="${dataset_name}:${split_name}" \
--output_filename="${OUTPUT_BASE}/${dataset_name}/${split_name}" \
--embedding_names=trill,trill-distilled \
--embedding_modules="https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/1,https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/1" \
--module_output_keys=layer19,embedding \
--audio_key=audio
}

convert_tfds "crema_d" "train"
convert_tfds "crema_d" "validation"
convert_tfds "crema_d" "test"

convert_tfds "savee" "train"
convert_tfds "savee" "validation"
convert_tfds "savee" "test"

convert_tfds "speech_commands" "train"
convert_tfds "speech_commands" "validation"
convert_tfds "speech_commands" "test"
