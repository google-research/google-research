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
FNAME=convai2_opedata.tar.gz
gsutil cp gs://airdialogue_share/$FNAME ./
rm -rf cleaned_logs
tar -xvzf $FNAME
rm $FNAME

rm -rf orig
mv cleaned_logs orig

orig=orig
savedir=opedata_small

rm -rf $savedir
for subdir in $orig/*/
do
    echo $subdir
    model_name=${subdir%*/}
    model_name=${model_name##*/}

    newdir=$savedir/$model_name
    echo "Saving to $newdir Model name $model_name"
    mkdir -p $newdir

    sed -n 1~10p $subdir/*.jsonl > $newdir/data.json
    echo "Number of lines $(wc -l $newdir/data.json)"
done
