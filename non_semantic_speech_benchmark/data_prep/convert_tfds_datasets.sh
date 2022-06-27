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
#
# bash convert_tfds_datasets.sh
#
# The TF-Hub models are located at:
# https://tfhub.dev/s?q=nonsemantic-speech-benchmark
#
# You can also use a TensorFlow Lite version of TRILL to compute embeddings.
# Provide the path to the `.tflite` flatbuffer as `embedding_module`, and the
# embedding tensor index in TFLite outputs as `output_key` (usually 0).
# NOTE: Only the distilled version of TRILL is suited for mobile conversion.

OUTPUT_BASE="~/tmp/noss"

function convert_tfds() {
  local dataset_name=$1
  local has_speaker_id=$2

  if [ "$has_speaker_id" = true ] ; then
    extra_args="--speaker_id_key=speaker_id"
  else
    extra_args=""
  fi

  python3 -m audio_to_embeddings_beam_main \
--alsologtostderr \
--tfds_dataset="${dataset_name}" \
--output_filename="${OUTPUT_BASE}/${dataset_name}" \
--embedding_names=trill,trill-distilled \
--embedding_modules="https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3,https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3" \
--module_output_keys=layer19,embedding \
--audio_key=audio \
--label_key=label ${extra_args} &
}

## For TFLite
## Note that this *only* outputs one embedding name.

# function convert_tfds() {
#   local dataset_name=$1
#   local has_speaker_id=$2

#   if [ "$has_speaker_id" = true ] ; then
#     extra_args="--speaker_id_key=speaker_id"
#   else
#     extra_args=""
#   fi

#   wget "https://tfhub.dev/google/lite-model/nonsemantic-speech-benchmark/trill-distilled/1?lite-format=tflite" -O ${OUTPUT_BASE}/model.tflite

#   python3 -m audio_to_embeddings_beam_main \
# --alsologtostderr \
# --tfds_dataset="${dataset_name}" \
# --output_filename="${OUTPUT_BASE}/${dataset_name}" \
# --embedding_names=trill-distilled \
# --embedding_modules=${OUTPUT_BASE}/model.tflite \
# --module_output_keys=0 \
# --audio_key=audio \
# --label_key=label ${extra_args} &
# }

convert_tfds "crema_d" true
convert_tfds "savee" true
convert_tfds "dementiabank" true
#convert_tfds "voxforge" true  # This dataset requires manual download.
convert_tfds "speech_commands" false
convert_tfds "voxceleb" false
