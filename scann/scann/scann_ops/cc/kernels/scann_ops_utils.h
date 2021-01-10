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

#ifndef SCANN_SCANN_OPS_CC_KERNELS_SCANN_OPS_UTILS_H_
#define SCANN_SCANN_OPS_CC_KERNELS_SCANN_OPS_UTILS_H_

#include "absl/types/span.h"
#include "scann/data_format/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace scann_ops {

Status TensorFromProto(OpKernelContext* context, absl::string_view name,
                       const protobuf::MessageLite* proto);
void TensorFromProtoRequireOk(OpKernelContext* context, absl::string_view name,
                              const protobuf::MessageLite* proto);

Status EmptyTensor(OpKernelContext* context, absl::string_view name);

void EmptyTensorRequireOk(OpKernelContext* context, absl::string_view name);

Status ConvertStatus(const Status& status);

template <typename T>
Status PopulateDatapointFromTensor(const Tensor& tensor,
                                   research_scann::DatapointPtr<T>* datapoint);

template <typename DstType, typename SrcType = DstType>
Status PopulateDenseDatasetFromTensor(
    const Tensor& tensor, research_scann::DenseDataset<DstType>* dataset);

template <typename T>
Status PopulateDatapointFromTensor(const Tensor& tensor,
                                   research_scann::DatapointPtr<T>* datapoint) {
  if (tensor.dims() != 1) {
    return errors::InvalidArgument("Dataset must be 1-dimensional",
                                   tensor.DebugString());
  }
  auto tensor_flat = tensor.flat<T>();
  int dims = tensor.NumElements();
  *datapoint =
      research_scann::DatapointPtr<T>(nullptr, tensor_flat.data(), dims, dims);
  return Status::OK();
}

template <typename DstType, typename SrcType>
Status PopulateDenseDatasetFromTensor(
    const Tensor& tensor, research_scann::DenseDataset<DstType>* dataset) {
  if (tensor.dims() != 2) {
    return errors::InvalidArgument("Dataset must be 2-dimensional",
                                   tensor.DebugString());
  }
  auto tensor_t = tensor.matrix<SrcType>();
  int num_dim = tensor_t.dimension(1);
  int num_datapoint = tensor_t.dimension(0);

  if (!num_dim) return Status::OK();

  dataset->clear();
  dataset->set_dimensionality(num_dim);
  dataset->Reserve(num_datapoint);

  for (int i = 0; i < num_datapoint; ++i) {
    const research_scann::DatapointPtr<DstType> dptr(
        nullptr, reinterpret_cast<const DstType*>(&tensor_t(i, 0)), num_dim,
        num_dim);
    TF_RETURN_IF_ERROR(ConvertStatus(dataset->Append(dptr, "")));
  }
  return Status::OK();
}

template <typename T>
Status TensorFromDenseDataset(OpKernelContext* context, absl::string_view name,
                              const research_scann::DenseDataset<T>* dataset) {
  if (dataset == nullptr) return EmptyTensor(context, name);
  Tensor* tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      name,
      TensorShape(
          {dataset->size(), static_cast<int64_t>(dataset->dimensionality())}),
      &tensor));
  auto tensor_flat = tensor->flat<T>();
  std::copy(dataset->data().begin(), dataset->data().end(), tensor_flat.data());
  return Status::OK();
}

template <typename T>
void TensorFromDenseDatasetRequireOk(
    OpKernelContext* context, absl::string_view name,
    const research_scann::DenseDataset<T>* dataset) {
  OP_REQUIRES_OK(context, TensorFromDenseDataset(context, name, dataset));
}

template <typename T>
Status TensorFromSpan(OpKernelContext* context, absl::string_view name,
                      research_scann::ConstSpan<T> span) {
  if (span.empty()) return EmptyTensor(context, name);
  Tensor* tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      name, TensorShape({static_cast<int64_t>(span.size())}), &tensor));
  auto tensor_flat = tensor->flat<T>();
  std::copy(span.begin(), span.end(), tensor_flat.data());
  return Status::OK();
}

template <typename T>
void TensorFromSpanRequireOk(OpKernelContext* context, absl::string_view name,
                             research_scann::ConstSpan<T> span) {
  OP_REQUIRES_OK(context, TensorFromSpan(context, name, span));
}

template <typename T>
research_scann::ConstSpan<T> TensorToConstSpan(const Tensor* t) {
  return absl::MakeConstSpan(t->flat<T>().data(), t->NumElements());
}

template <typename T>
research_scann::MutableSpan<T> TensorToMutableSpan(const Tensor* t) {
  return absl::MakeSpan(t->flat<T>().data(), t->NumElements());
}

}  // namespace scann_ops
}  // namespace tensorflow

#endif
