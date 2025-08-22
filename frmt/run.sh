#!/bin/bash
# Copyright 2025 The Google Research Authors.
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


# Testing script. Should be run from the parent directory of 'frmt/', e.g.
# bash frmt/run.sh

set -eux

pip install --upgrade pip  # Ensures that pip is current.

# If you run into installation errors upon repeat runs of this script, try
# removing `frmt_env` and `bleurt`.
pip install virtualenv
virtualenv -p python3 frmt_env
source frmt_env/bin/activate

pip install -r frmt/requirements.txt

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

python -m frmt.lexical_accuracy_test
python -m frmt.evaluation_test

# ==============================================================================
# Run integration test for evaluate.
bash frmt/evaluate_itest.sh
HAS_ERROR="$?"

# ==============================================================================
# Simple integration test for lexical_accuracy_eval using the gold test set.
MSG="[lexical_accuracy_eval] Failure for"
BUCKET_DIR=frmt/dataset/lexical_bucket
RESULT_ZH=$(python -m frmt.lexical_accuracy_eval \
  --corpus_cn=<(cut -f2 ${BUCKET_DIR}/zh_lexical_test_en_zh-CN.tsv) \
  --corpus_tw=<(cut -f2 ${BUCKET_DIR}/zh_lexical_test_en_zh-TW.tsv)  )
EXPECTED_ZH="0.9444"
if [[ "${RESULT_ZH}" != "${EXPECTED_ZH}" ]]; then
  echo "${MSG} ZH: Obtained ${RESULT_ZH} != expected ${EXPECTED_ZH}."
  HAS_ERROR=1
fi

RESULT_PT=$(python -m frmt.lexical_accuracy_eval \
  --corpus_br=<(cut -f2 ${BUCKET_DIR}/pt_lexical_test_en_pt-BR.tsv) \
  --corpus_pt=<(cut -f2 ${BUCKET_DIR}/pt_lexical_test_en_pt-PT.tsv) )
EXPECTED_PT="0.9858"
if [[ "${RESULT_PT}" != "${EXPECTED_PT}" ]]; then
  echo "${MSG} PT: Obtained ${RESULT_PT} != expected ${EXPECTED_PT}."
  HAS_ERROR=1
fi

# ==============================================================================
if [[ $HAS_ERROR -eq 1 ]]; then
  echo "Failure."
  exit 1
fi

set +x

echo "Success!"
echo
echo 'The Python virtual environment `frmt_env` has all the necessary '
echo 'requirements to run evaluation installed. You can enter the environment '
echo 'later by running the command `source frmt_env/bin/activate` in the '
echo 'current directory and leave the environment with the command '
echo '`deactivate`.'
