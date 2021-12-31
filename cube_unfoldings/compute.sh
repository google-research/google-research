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
set -e

[ -z "$1" ] && echo "Usage: $0 N"
[ -z "$1" ] && exit 1

N=$1

cd "$(dirname "$0")"

DIR="/tmp/unfoldings-$N"

[ -e $DIR ] && echo "Directory $DIR already exists, please remove it"
[ -e $DIR ] && exit 1

mkdir $DIR

nauty-gentreeg $((N*2)) | shuf > $DIR/trees

split -l 128 $DIR/trees $DIR/trees.part.

parallel build/unfoldings ::: $DIR/trees.part.* > $DIR/counts

paste -sd + $DIR/counts | bc
