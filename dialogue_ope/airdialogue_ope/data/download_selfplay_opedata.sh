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

# Internal Use Download Data from GCS

#!/bin/bash
FNAME=air_selfchat_ope_data.tar.gz
rm -rf selfplay_opedata
mkdir -p selfplay_opedata
cd selfplay_opedata
gsutil cp gs://airdialogue_share/$FNAME ./
tar -xvzf $FNAME
rm $FNAME

rm -rf orig
mv selfchat_ope_data orig

orig=orig
savedir=opedata

rm -rf $savedir
for subdir in $orig/*/
do
    echo $subdir
    model_name=${subdir%*/}
    model_name=${model_name##*/}

    rm $subdir/${model_name}.jsonl

    newdir=$savedir/$model_name
    echo "Saving to $newdir Model name $model_name"
    mkdir -p $newdir

    cat $subdir/*.jsonl > $newdir/data.json
    echo "Number of lines $(wc -l $newdir/data.json)"
done
