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
#
# Run the wikinews parser on the Jan 1, 2019 Wikinews archive from archive.org
# to generate the 2018 Wikinews dataset.
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

# Download the archive if it does not already exist in the current directory.
FILENAME="enwikinews-20190101-pages-articles.xml.bz2"
if [ ! -f "${FILENAME}" ]; then
   wget https://archive.org/download/enwikinews-20190101/${FILENAME}
fi

# Verify the checksum of the archive.
md5sum --check <(echo 60f05eeb668a6c5f663c4c193df78811  "${FILENAME}")

# Run the parser on the archive, outputting the result to the current directory.
pip install -r dense_representations_for_entity_retrieval/requirements.txt
python -m dense_representations_for_entity_retrieval.parse_wikinews \
   --wikinews_archive=enwikinews-20190101-pages-articles.xml.bz2 \
   --output_dir=. \
   --logtostderr
