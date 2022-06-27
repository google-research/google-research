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

#!/bin/sh

# This script regenerates all the .out files for the examples.
# It should be run in the directory where the database files are

set -eux

EXAMPLEDIR=$(dirname $0)

for FN in $(ls ${EXAMPLEDIR}/*.py)
do
    BASE=$(basename ${FN})
    STEM=${BASE%.*}

    python -m smu.examples.${STEM} > ${EXAMPLEDIR}/${STEM}.out
done
