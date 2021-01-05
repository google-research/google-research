// Copyright 2021 The Google Research Authors.
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

#include "scann/scann_ops/cc/kernels/scann_ops_utils.h"

namespace tensorflow {
namespace scann_ops {

Status TensorFromProto(OpKernelContext* context, absl::string_view name,
                       const protobuf::MessageLite* proto) {
  if (proto == nullptr) return EmptyTensor(context, name);

  Tensor* tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(name, TensorShape({1}), &tensor));

  if (!SerializeToTString(*proto, &tensor->scalar<tstring>()()))
    return errors::Internal(
        absl::StrCat("Failed to create string tensor ", name));
  return Status::OK();
}

void TensorFromProtoRequireOk(OpKernelContext* context, absl::string_view name,
                              const protobuf::MessageLite* proto) {
  OP_REQUIRES_OK(context, TensorFromProto(context, name, proto));
}

Status EmptyTensor(OpKernelContext* context, absl::string_view name) {
  Tensor* tensor;
  return context->allocate_output(name, TensorShape({}), &tensor);
}

void EmptyTensorRequireOk(OpKernelContext* context, absl::string_view name) {
  OP_REQUIRES_OK(context, EmptyTensor(context, name));
}

Status ConvertStatus(const Status& status) { return status; }

}  // namespace scann_ops
}  // namespace tensorflow
