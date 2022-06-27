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

# for i in "sim_sparsity" "sim_sparsity_alternate" "sim_sparsity_regression"\
#          "sim_sparsity_regression_alternate"  "sim_cardinality"\
#          "sim_cardinality_alternate" "sim_cardinality_regression"\
#          "sim_cardinality_regression_alternate" "sim_linear" "sim_linear_regression"\
#          "sim_multiplicative" "sim_multiplicative_regression"; do
for i in "sim_cardinality_regression_inform"\
         "sim_cardinality_regression_uninform"\
         "sim_sparsity_regression_inform" "sim_sparsity_regression_uninform"\
         "sim_cardinality_inform" "sim_cardinality_uninform"\
         "sim_sparsity_inform" "sim_sparsity_uninform"; do
  	printf "$i\n---\n"
	for j in "l1_gbdt" "l2_gbdt" "random_forest" "l1_linear" "l2_linear" "l1_dnn" "l2_dnn" \
           "oracle"; do
			if [ "$j" = "l1_gbdt" ]; then
        printf "GBDT (L1)"
      elif [ "$j" = "l2_gbdt" ]; then
        printf "GBDT (L2)"
      elif [ "$j" = "random_forest" ]; then
        printf "Random forest"
      elif [ "$j" = "l1_linear" ]; then
        if [[ "$i" = *"regression"* ]]; then
          printf "Linear regression (L1)"
        else
          printf "Logistic regression (L1)"
        fi
      elif [ "$j" = "l2_linear" ]; then
        if [[ "$i" = *"regression"* ]]; then
          printf "Linear regression (L2)"
        else
          printf "Logistic regression (L2)"
        fi
      elif [ "$j" = "l1_dnn" ]; then
        printf "DNN (L1)"
      elif [ "$j" = "l2_dnn" ]; then
        printf "DNN (L1)"
      else
        printf "Oracle"
      fi

      printf " & "

      if [ "$j" = "oracle" ] || [[ "$i" = *"regression"* ]]; then
        cat "trees/out/logs/$i/$j.log" | sed -e "1!d" -e "s/.*=//" | tr -d "\n"
        printf " +/- "
        cat "trees/out/logs/$i/$j.log" | sed -e "4!d" -e "s/.*=//" | tr -d "\n"
      else
        cat "trees/out/logs/$i/$j.log" | sed -e "3!d" -e "s/.*=//" | tr -d "\n"
        printf " +/- "
        cat "trees/out/logs/$i/$j.log" | sed -e "10!d" -e "s/.*=//" | tr -d "\n"
      fi

      if [[ "$i" != *"inform"* ]]; then
        if [[ "$i" = *"sparsity"* ]] || [[ "$i" = *"cardinality"* ]] && [[ "$j" != *"dnn"* ]] && [ "$j" != "oracle" ]; then
          printf " & "
          cat "trees/out/logs/$i/$j.log" | tail -1 | sed -e "s/.*=//" | tr -d "\n"
        else
          printf " & -"
        fi
      fi

			printf " \\"
      printf "\\"
      printf "\n"
	done
	printf "\n"
done
