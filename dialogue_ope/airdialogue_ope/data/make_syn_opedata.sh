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

for size in 10 50 100 500 5K
do
    orig=syn_opedata/orig/syn_ope_data_$size
    savedir=syn_opedata/syn_ope_data_$size
    for subdir in $orig/*/
    do
        echo $subdir
        newdir=${subdir%*/}
        newdir=${newdir##*/}
        newdir=$savedir/$newdir
        echo "Saving to $newdir"

        cat $subdir/data_ref_*.json > $subdir/data_all.json
        cat $subdir/kb_ref_*.json > $subdir/kb_all.json
        echo "Number of lines $(wc -l $subdir/data_all.json)"
        python make_data_syn.py \
            --ref_file $subdir/data_all.json \
            --ref_kb $subdir/kb_all.json \
            --output_dir $newdir
    done
done
