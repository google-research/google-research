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
# Split large SUMO FCD file into small pieces by the timestamp.

# The script performs `csplit` iteratively and cut the file at time="xxx.00".
# It first cut the file into two pieces, then cut the second piece at another
# timestamp. It can cut a 10GB file into 20 pieces in 3 min on a gLinux desktop.
# Example usage:
# ./fcd_file_cutter.sh -i output/traffic.fcd.xml -o output/fcd_segments -s 1800
# The output files are: traffic.segment_0.fcd.xml, traffic.segment_1.fcd.xml, ..

# Parses the arguments. Sets default values. The input file is required.
output_folder_arg='.'
step_size_arg=1800
while getopts ":i:o:s:" opt; do
    case $opt in
        i) cut_file_arg="$OPTARG"
        ;;
        o) output_folder_arg="$OPTARG"
        ;;
        s) step_size_arg="$OPTARG"
        ;;
        \?) echo "Invalid option -$OPTARG" >&2
        ;;
    esac
done

step_size=$step_size_arg
timestamp=$step_size
cut_file=$cut_file_arg
output_folder=$output_folder_arg

exit_code=0
file_counter=0

printf "Input file: %s  \n" "$cut_file"
printf "Time step size %s\n" "$step_size"
printf "Output folder in %s\n" "$output_folder"

while [[ "$exit_code" -eq 0 ]]; do
    echo "Cutting at timestamp: $timestamp.00"

    # `csplit` splits the file at pattern like 'time="xxx.00"'
    # The `csplit` outputs two temporary files, `tmp.n.0.fcd.xml` and
    # `tmp.n.1.fcd.xml`. The code will keep cutting the second file.
    csplit "${cut_file}" \
        /time=\""${timestamp}".00\"/ \
        -f "tmp.${file_counter}." \
        -b "%01d.fcd.xml"
    # Catches the return code of `csplit`.
    exit_code=$?
    echo "Exit with ${exit_code}"

    # Breaks the loop if the timestamp is not in the file. No more cutting work.
    if [[ "$exit_code" -ne 0 ]]; then
        break
    fi

    # Add header and tail for the split xml file. No header for the first xml.
    if [[ "$file_counter" -ge 1 ]]; then
        sed -i '1i <fcd-export>' "tmp.${file_counter}.0.fcd.xml"
    fi
    echo "</fcd-export>" >> "tmp.${file_counter}.0.fcd.xml"

    # The first part contains the information for one step_size.
    mv "tmp.${file_counter}.0.fcd.xml" \
        "${output_folder}/traffic.segment_${file_counter}.fcd.xml"
    echo "Save file " "${output_folder}/traffic.segment_${file_counter}.fcd.xml"
    # In the next iteration, `cut_file` will be further split.
    cut_file="tmp.${file_counter}.1.fcd.xml"

    # Remove the second half file from the last iteration.
    if [[ "$file_counter" -ge 1 ]]; then
        rm "tmp.$((file_counter-1)).1.fcd.xml"
        echo "Remove" "tmp.$((file_counter-1)).1.fcd.xml"
    fi

    let timestamp=timestamp+step_size
    let file_counter=file_counter+1

done

# Move the rest piece to the last segment.
# let file_counter=file_counter-1
# Add header for the last xml. No need for the tail.
sed -i '1i <fcd-export>' "tmp.$((file_counter-1)).1.fcd.xml"
mv "tmp.$((file_counter-1)).1.fcd.xml" \
    "${output_folder}/traffic.segment_${file_counter}.fcd.xml"
echo "Remove" "tmp.$((file_counter-1)).1.fcd.xml"
echo "Save file " "${output_folder}/traffic.segment_${file_counter}.fcd.xml"
