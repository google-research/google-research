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

# Make a local copy of CFQ.
svn export https://github.com/google-research/google-research/trunk/cfq
# Update requirements to be able to reference *-CFQ dataset.
REQ=cfq/requirements.txt
OLD_REQ=cfq/requirements.old.txt
rm -f ${OLD_REQ}
mv ${REQ} ${OLD_REQ}
cat ${OLD_REQ} | grep -v 'tensorflow-datasets' > ${REQ}
echo 'tensorflow-datasets==4.3.0' >> ${REQ}
# Process requirements, adding all dependencies locally.
python3.7 -m pip install -r ${REQ} --user
# Allow TFDS to work with TF 1.x (tested with v1.15.3)
SRC=~/.local/lib/python3.7/site-packages/tensorflow_datasets/core/tf_compat.py
sed -i 's/^MIN_TF_VERSION[ ]=[ ].*/MIN_TF_VERSION = "1.15.3"/' "${SRC}"
