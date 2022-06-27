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

images="\
  https://upload.wikimedia.org/wikipedia/commons/9/90/Texture_P7220210.JPG \
  https://upload.wikimedia.org/wikipedia/commons/b/be/Surface_wood_chipboard.jpg \
  https://upload.wikimedia.org/wikipedia/commons/7/7f/Blue_Gray_knit_texture.jpg \
  https://upload.wikimedia.org/wikipedia/commons/b/b8/Fabric_length_(AM_842431-1).jpg \
  https://upload.wikimedia.org/wikipedia/commons/2/20/Coloured,_textured_craft_card_edit.jpg \
  "

for image in $images;
do
  echo "Downloading $image"
  wget $image
  basename=`basename $image`
  echo "Resizing $basename"
  convert $basename -resize 512x512\! $basename.png
  rm $basename
done
