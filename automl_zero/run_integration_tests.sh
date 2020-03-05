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

# Run this file as an integrated test that will generate outputs to copy to
# critique.

# Usage:
# ./run_integration_tests.sh

echo "Running integrated tests. You need to copy the output below to the 'Tested' section of the CL."
echo
echo "LOCAL_LINEAR_TEST:"
echo "output is"
./run_integration_test_linear.sh 2>&1 | grep "indivs=[2|1]1000\|Final evaluation fitness"
echo "expected is
indivs=11000, elapsed_secs=0, mean=0.240699, stdev=0.090653, best fit=0.305231,
indivs=21000, elapsed_secs=0, mean=0.481513, stdev=0.401085, best fit=0.999813,
Final evaluation fitness (on unseen data) = 1.000000"
echo
echo "LOCAL_NONLINEAR_TEST:"
echo "output is"
./run_integration_test_nonlinear.sh 2>&1 | grep "indivs=[1|9]1000\|Final evaluation fitness"
echo "expected is
indivs=11000, elapsed_secs=X, mean=0.855631, stdev=0.277759, best fit=0.999835,
indivs=91000, elapsed_secs=X, mean=0.724830, stdev=0.423565, best fit=1.000000,
Final evaluation fitness (on unseen data) = 0.999680"
