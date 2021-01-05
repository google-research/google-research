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
# Converts *.pdf files to *.jpg files, then compile to gif animation.
# Example usage: pdf_to_images.sh --output test_dir --input_dir test_dir

set -e
source gbash.sh || exit 1
DEFINE_string --required output_dir "./" \
  "Output directory. E.g. ./"
DEFINE_string --required input_dir "./" \
  "Input directory. E.g. ./"
DEFINE_string jpg_density 300 \
  "The density of the jpg figure. E.g. 300"
DEFINE_string gif_frame_delay 80 \
  "Time delay between adjacent frames in the gif. 100/delay = FPS. E.g. 80"


gbash::init_google "$@"
set -u

while true; do
    read -p "Do you need to convert pdf to jpg? [Y/n]" yn
    case $yn in
        [Yy]* | '' ) echo "Converting pdf to jpg.";
                for f in "${FLAGS_input_dir}"/*.pdf; do
                    filename="${f%.*}"
                    convert -density "${FLAGS_jpg_density}" "${f}" \
                        "${filename}.jpg"
                    echo "Save file ${filename}.jpg"
                done;
                break;;
        [Nn]* ) echo "Continue to gif compilation.";
                break;;
        * ) echo "Please answer yes or no. [Y/n]";;
    esac
done

# Compiles images into the gif video. Note that by default, all the images in
# the input folder are sorted by their names in the alphabetical order.
all_images=$(ls -v "${FLAGS_input_dir}"/*.jpg)
declare -a args; args=($(ls -v "${FLAGS_input_dir}"/*.jpg))
echo "${all_images}"
echo " "

while true; do
    read -p "Are these images in the right order for the animation? [Y/n]" yn
    case $yn in
        [Yy]* | '') echo "Compiling jpg files into gif animation.";
                convert -delay "${FLAGS_gif_frame_delay}" "${args[@]}" \
                    "evacuation_traffic_flow.gif";
                echo "Save file evacuation_traffic_flow.gif.";
                break;;
        [Nn]* ) echo "Quit. You can use the following command by yourself.";
                echo "$ convert -delay 60 file1 file2 output_file";
                exit;;
        * ) echo "Please answer yes or no. [Y/n]";;
    esac
done
