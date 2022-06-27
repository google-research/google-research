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
CONFIG=$1
DATA_ROOT=$2
ROOT_DIR=/tmp/jaxnerf/"$CONFIG"
if [ $CONFIG == "llff" ]
then
  SCENES="room fern leaves fortress orchids flower trex horns"
  DATA_FOLDER="nerf_llff_data"
else
  SCENES="lego chair drums ficus hotdog materials mic ship"
  DATA_FOLDER="nerf_synthetic"
fi

# launch evaluation jobs for all scenes.
for scene in $SCENES; do
  python -m jaxnerf.eval \
    --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$scene" \
    --train_dir="$ROOT_DIR"/"$scene" \
    --chunk=4096 \
    --config=configs/"$CONFIG"
done

# collect PSNR of all scenes.
touch "$ROOT_DIR"/psnr.txt
for scene in $SCENES; do
  printf "${scene}: " >> "$ROOT_DIR"/psnr.txt
  cat "$ROOT_DIR"/"$scene"/test_preds/psnr.txt >> \
    "$ROOT_DIR"/psnr.txt
  printf $'\n' >> "$ROOT_DIR"/psnr.txt
done
