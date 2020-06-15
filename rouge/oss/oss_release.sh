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

#!/bin/bash

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

GIT_COMMIT_ID=${1:-""}
[[ -z $GIT_COMMIT_ID ]] && echo "Must provide a commit" && exit 1

TMP_DIR=$(mktemp -d)
pushd $TMP_DIR

echo "Cloning trax and checking out commit $GIT_COMMIT_ID"
git clone https://github.com/google-research/google-research
cd google-research/rouge
git checkout $GIT_COMMIT_ID
sed  -i 's/from rouge/from rouge_score/' *.py

python -m pip install wheel twine pyopenssl

# Build the distribution
echo "Building distribution"
python setup.py sdist
python setup.py bdist_wheel --universal

# Publish to PyPI
echo "Publishing to PyPI"
twine upload dist/*

# Cleanup
popd
rm -rf $TMP_DIR
