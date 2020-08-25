# Copyright 2020 The Google Research Authors.
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
mkdir syn_opedata
cd syn_opedata
FNAME=syn_opedata.tar.gz
gsutil  cp gs://airdialogue_share/$FNAME ./
tar -xvzf $FNAME
rm $FNAME
cd ..
sh make_syn_opedata.sh
