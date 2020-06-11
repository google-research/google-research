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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/data_format/gfv_properties.h"

#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

string_view GfvFeatureTypeName(int gfv_feature_type) {
  switch (gfv_feature_type) {
    case GenericFeatureVector::INT64:
      return "INT64";
    case GenericFeatureVector::FLOAT:
      return "FLOAT";
    case GenericFeatureVector::DOUBLE:
      return "DOUBLE";
    case GenericFeatureVector::STRING:
      return "STRING";
    default:
      return "INVALID_GFV_FEATURE_TYPE";
  }
}

StatusOr<size_t> GetGfvVectorSize(const GenericFeatureVector& gfv) {
  switch (gfv.feature_type()) {
    case GenericFeatureVector::INT64:
    case GenericFeatureVector::BINARY:
      return gfv.feature_value_int64_size();
    case GenericFeatureVector::FLOAT:
      return gfv.feature_value_float_size();
    case GenericFeatureVector::DOUBLE:
      return gfv.feature_value_double_size();
    case GenericFeatureVector::STRING:
      return 1;
    default:
      return InvalidArgumentError("Unknown feature type:  %d",
                                  gfv.feature_type());
  }
}

StatusOr<DimensionIndex> GetGfvDimensionality(const GenericFeatureVector& gfv) {
  if (gfv.feature_dim() == 0) {
    return InvalidArgumentError(
        "GenericFeatureVector dimensionality cannot be == 0.");
  }

  TF_ASSIGN_OR_RETURN(bool is_sparse, IsGfvSparse(gfv));
  if (is_sparse) {
    return gfv.feature_dim();
  } else {
    return GetGfvVectorSize(gfv);
  }
}

StatusOr<bool> IsGfvSparse(const GenericFeatureVector& gfv) {
  if (gfv.feature_type() == GenericFeatureVector::STRING) {
    return false;
  }

  if (gfv.feature_index_size() > 0) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(DimensionIndex vector_size, GetGfvVectorSize(gfv));
  return vector_size == 0;
}

StatusOr<bool> IsGfvDense(const GenericFeatureVector& gfv) {
  if (gfv.feature_type() == GenericFeatureVector::STRING) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(bool is_sparse, IsGfvSparse(gfv));
  return !is_sparse;
}

Status GetGfvVectorSize(const GenericFeatureVector& gfv,
                        DimensionIndex* result) {
  DCHECK(result);
  TF_ASSIGN_OR_RETURN(*result, GetGfvVectorSize(gfv));
  return OkStatus();
}

Status GetGfvDimensionality(const GenericFeatureVector& gfv,
                            DimensionIndex* result) {
  DCHECK(result);
  TF_ASSIGN_OR_RETURN(*result, GetGfvDimensionality(gfv));
  return OkStatus();
}

Status IsGfvSparse(const GenericFeatureVector& gfv, bool* result) {
  DCHECK(result);
  TF_ASSIGN_OR_RETURN(*result, IsGfvSparse(gfv));
  return OkStatus();
}

Status IsGfvDense(const GenericFeatureVector& gfv, bool* result) {
  DCHECK(result);
  TF_ASSIGN_OR_RETURN(*result, IsGfvDense(gfv));
  return OkStatus();
}

size_t GetGfvDimensionalityOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(GetGfvDimensionality(gfv));
}

bool IsGfvSparseOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(IsGfvSparse(gfv));
}

bool IsGfvDenseOrDie(const GenericFeatureVector& gfv) {
  return ValueOrDie(IsGfvDense(gfv));
}

}  // namespace scann_ops
}  // namespace tensorflow
