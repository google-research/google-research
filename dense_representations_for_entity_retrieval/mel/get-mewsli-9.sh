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

# End-to-end script to reconstruct clean text for Mewsli-9 dataset from publicly
# available data sources.

# Run this from the mel/ directory:
#   bash get-mewsli-9.sh

set -eux

# Final output location.
DATASET_DIR="./mewsli-9/output/dataset"
mkdir -p ${DATASET_DIR}

# Download the dataset descriptors.
wget https://storage.googleapis.com/gresearch/mewsli/mewsli-9.zip

# Extract dataset descriptors archive
unzip -d ${DATASET_DIR} mewsli-9.zip

# Download WikiNews dumps for 9 languages from archive.org.
bash mewsli-9/get_wikinews_dumps.sh

# Download the external wikiextractor tool and patch it.
bash tools/get_wikiextractor.sh

# Process the WikiNews dumps into lightly marked-up JSON format.
bash mewsli-9/run_wikiextractor.sh

# Install dependencies into a virtual environment.
virtualenv -p python3 ./env
source ./env/bin/activate
pip install -r wikinews_extractor/requirements.txt

# Parse clean text from the processed dumps according to the Mewsli-9 dataset
# descriptors.
bash mewsli-9/run_parse_wikinews_i18n.sh

# Summary.
tail -n4 ${DATASET_DIR}/??/log
