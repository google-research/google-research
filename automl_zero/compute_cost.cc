// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compute_cost.h"

#include "instruction.pb.h"
#include "algorithm.h"

namespace automl_zero {

double ComputeCost(
    const std::vector<std::shared_ptr<const Instruction>>& component_function) {
  double cost = 0.0;
  for (const std::shared_ptr<const Instruction>& instruction :
       component_function) {
    cost += ComputeCost(*instruction);
  }
  return cost;
}

// To add a new op compute cost here, first run ops_benchmark.cc. Then,
// calculate the normalization ratio for your new costs:
//
// ratio = MATRIX_BETA_SET_OP cost in this file / your MATRIX_BETA_SET_OP cost
//
// Multiply each of your ops costs by this ratio to determine their new
// normalized costs, which can be added here.
double ComputeCost(const Instruction& instruction) {
  switch (instruction.op_) {
    case NO_OP:
      return 3013921.0;
    case SCALAR_SUM_OP:
      return 4703759.0;
    case SCALAR_DIFF_OP:
      return 5102027.0;
    case SCALAR_PRODUCT_OP:
      return 4418347.0;
    case SCALAR_DIVISION_OP:
      return 3963916.0;
    case SCALAR_MIN_OP:
      return 3677265.0;
    case SCALAR_MAX_OP:
      return 4573695.0;
    case SCALAR_ABS_OP:
      return 3684352.0;
    case SCALAR_HEAVYSIDE_OP:
      return 4057075.0;
    case SCALAR_CONST_SET_OP:
      return 3691108.0;
    case SCALAR_SIN_OP:
      return 7538523.0;
    case SCALAR_COS_OP:
      return 7452807.0;
    case SCALAR_TAN_OP:
      return 7583203.0;
    case SCALAR_ARCSIN_OP:
      return 9145616.0;
    case SCALAR_ARCCOS_OP:
      return 7673039.0;
    case SCALAR_ARCTAN_OP:
      return 9273978.0;
    case SCALAR_EXP_OP:
      return 11755929.0;
    case SCALAR_LOG_OP:
      return 11627725.0;
    case VECTOR_SUM_OP:
      return 4701519.0;
    case VECTOR_DIFF_OP:
      return 4165747.0;
    case VECTOR_PRODUCT_OP:
      return 4504555.0;
    case VECTOR_DIVISION_OP:
      return 4556237.0;
    case VECTOR_MIN_OP:
      return 3989956.0;
    case VECTOR_MAX_OP:
      return 3974229.0;
    case VECTOR_ABS_OP:
      return 3713533.0;
    case VECTOR_HEAVYSIDE_OP:
      return 3965743.0;
    case VECTOR_CONST_SET_OP:
      return 3991421.0;
    case MATRIX_SUM_OP:
      return 5579381.0;
    case MATRIX_DIFF_OP:
      return 5429452.0;
    case MATRIX_PRODUCT_OP:
      return 5475227.0;
    case MATRIX_DIVISION_OP:
      return 18234378.0;
    case MATRIX_MIN_OP:
      return 5457728.0;
    case MATRIX_MAX_OP:
      return 5840952.0;
    case MATRIX_ABS_OP:
      return 4630926.0;
    case MATRIX_HEAVYSIDE_OP:
      return 6728311.0;
    case MATRIX_CONST_SET_OP:
      return 3693035.0;
    case SCALAR_VECTOR_PRODUCT_OP:
      return 3705015.0;
    case VECTOR_INNER_PRODUCT_OP:
      return 4294617.0;
    case VECTOR_OUTER_PRODUCT_OP:
      return 9833707.0;
    case SCALAR_MATRIX_PRODUCT_OP:
      return 5986302.0;
    case MATRIX_VECTOR_PRODUCT_OP:
      return 9310002.0;
    case VECTOR_NORM_OP:
      return 4810562.0;
    case MATRIX_NORM_OP:
      return 6422746.0;
    case MATRIX_TRANSPOSE_OP:
      return 9394756.0;
    case MATRIX_MATRIX_PRODUCT_OP:
      return 14516296.0;
    case VECTOR_MEAN_OP:
      return 4280443.0;
    case VECTOR_ST_DEV_OP:
      return 5585730.0;
    case MATRIX_MEAN_OP:
      return 4595196.0;
    case MATRIX_ST_DEV_OP:
      return 10640216.0;
    case MATRIX_ROW_MEAN_OP:
      return 9763956.0;
    case MATRIX_ROW_ST_DEV_OP:
      return 16766477.0;
    case SCALAR_GAUSSIAN_SET_OP:
      return 20775635.0;
    case VECTOR_GAUSSIAN_SET_OP:
      return 77483018.0;
    case MATRIX_GAUSSIAN_SET_OP:
      return 277690384.0;
    case SCALAR_UNIFORM_SET_OP:
      return 26610704.0;
    case VECTOR_UNIFORM_SET_OP:
      return 57817372.0;
    case MATRIX_UNIFORM_SET_OP:
      return 208339446.0;
    case SCALAR_RECIPROCAL_OP:
      return 4212086.0;
    case SCALAR_BROADCAST_OP:
      return 4165814.0;
    case VECTOR_RECIPROCAL_OP:
      return 6311851.0;
    case MATRIX_RECIPROCAL_OP:
      return 24336141.0;
    case MATRIX_ROW_NORM_OP:
      return 13351123.0;
    case MATRIX_COLUMN_NORM_OP:
      return 6682187.0;
    case VECTOR_COLUMN_BROADCAST_OP:
      return 10993088.0;
    case VECTOR_ROW_BROADCAST_OP:
      return 11087322.0;
    // Do not add default clause. All ops should be supported here.
  }
}

}  // namespace automl_zero
