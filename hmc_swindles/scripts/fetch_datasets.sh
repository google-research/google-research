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
# Downloads the datasets used by this repository and places them into /tmp.
# Requires rpy2 (thus r-base and the Python C headers).
# `sudo apt-get install r-base libpython3.7-dev; pip install rpy2`
# Usage:
#
#  fetch_datasets.sh

# Exit if any process returns non-zero status.
set -e
# Display the commands being run in logs.
set -x

curl -o /tmp/german.data-numeric https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric
echo "German credit dataset saved to /tmp/german.data-numeric"
echo "If you move it, pass --german_credit_numeric_path=<new path> to use the German credit targets."

curl -o /tmp/irt.data.R https://raw.githubusercontent.com/stan-dev/example-models/master/misc/irt/irt.data.R
python3 -m hmc_swindles.scripts.r_to_pkl /tmp/irt.data.R /tmp/irt.data.pkl "jj->student_ids,kk->question_ids,y->correct"
echo "Stan item response theory dataset saved to /tmp/irt.data.pkl"
echo "If you move it, pass --stan_item_response_theory_path=<new path> to use the Stan item response theory targets."
