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
#
# Creates the three plots used in the paper to show the problems with Penn &
# Choma's approach.  Assumes you have created the data (see run.sh) and that the
# outputs reside in /var/tmp/penn_choma

DIR=/var/tmp/penn_choma
COLOR_MAP=Greys_r

python3 pennchoma/plot_cooccurrence.py \
  --color_map="${COLOR_MAP}" \
  --input_file="${DIR}/Chinese_ungrouped_0.txt" \
  --output_file="${DIR}/Chinese_ungrouped_0.png"
python3 pennchoma/plot_cooccurrence.py \
  --color_map="${COLOR_MAP}" \
  --input_file="${DIR}/English_ungrouped_0.txt" \
  --output_file="${DIR}/English_ungrouped_0.png"
python3 pennchoma/plot_cooccurrence.py \
  --color_map="${COLOR_MAP}" \
  --input_file="${DIR}/English_grouped_0.txt" \
  --output_file="${DIR}/English_grouped_0.png"
