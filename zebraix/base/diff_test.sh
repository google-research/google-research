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

set -u

#
# Process options.
#

USAGE_MESSAGE="Usage: $0 [options...]
  Options:
    --out_file (Generated output file)
    --golden_file (Golden file)
    --allow_updates (Optional. If provided update goldens instead of testing)
    --update_dir (Optional. Update goldens instead of testing)
    --diff_lines (Optional. If provided overrides default number of diff lines)

    Both update args must be set for updating of goldens"

LONG_OPTS="out_file:,golden_file:,allow_updates:,update_dir:,diff_lines:"

args=`getopt -n $0 -o v -l "${LONG_OPTS}" -- $*`

if [ $? != 0 ]
then
  echo "${USAGE_MESSAGE}"
  exit 2
fi

set -- $args

FLAGS_out_file=""
FLAGS_golden_file=""
FLAGS_update_dir=""
FLAGS_allow_updates=false
FLAGS_diff_lines=40

stripquotes() {
  local temp="${1%\'}"
  local temp="${temp#\'}"
  echo "$temp"
}

for i
do
  case "$i"
  in
    --out_file)
      FLAGS_out_file=`stripquotes $2`; shift;
      shift;;
    --golden_file)
      FLAGS_golden_file=`stripquotes $2`; shift;
      shift;;
    --update_dir)
      FLAGS_update_dir=`stripquotes $2`; shift;
      shift;;
    --allow_updates)
      FLAGS_allow_updates=`stripquotes $2`; shift;
      shift;;
    --diff_lines)
      FLAGS_diff_lines=`stripquotes $2`; shift;
      shift;;
    --)
      shift; break;;
  esac
done

#
# Demand filenames.
#

if [ -z "$FLAGS_out_file"  ] || [ -z "$FLAGS_golden_file"  ]
then
  echo -ne "\nMissing file name.\n\n"
  echo "${USAGE_MESSAGE}"
  exit 2
fi

#
# Update goldens.
#

if [ ! -z "$FLAGS_update_dir" ] && [ "$FLAGS_allow_updates" = "true" ]
then
  echo "Updating golden $FLAGS_update_dir/$FLAGS_golden_file from output file $FLAGS_out_file."

  cp $FLAGS_out_file $FLAGS_update_dir/$FLAGS_golden_file
  if [ $? != 0 ]
  then
    echo "Error updating goldens. Ensure that you are using local \"test strategy\"."
    exit 1
  fi
fi

#
# Test diff.
#

diff -q --unidirectional-new-file $FLAGS_golden_file $FLAGS_out_file >/dev/null
if [ $? != 0 ]
then
   echo -e "ERROR: Files $FLAGS_golden_file and $FLAGS_out_file differ.\n"
   diff -u $FLAGS_golden_file $FLAGS_out_file | head -$FLAGS_diff_lines
   echo -e "\n(Diff output limited to $FLAGS_diff_lines lines.)"
   exit 1
else
   exit 0
fi
