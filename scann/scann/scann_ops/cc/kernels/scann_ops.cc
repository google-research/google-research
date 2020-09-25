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



#include <limits>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/kernels/scann_ops_utils.h"
#include "scann/scann_ops/cc/scann.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace scann_ops {

class ScannResource : public ResourceBase {
 public:
  explicit ScannResource()
      : scann_(std::make_unique<tensorflow::scann_ops::ScannInterface>()) {}

  string DebugString() const override { return "I am the one who knocks."; }

  bool is_initialized() const { return initialized_; }

  void Initialize() { initialized_ = true; }

  std::unique_ptr<tensorflow::scann_ops::ScannInterface> scann_;

 private:
  bool initialized_ = false;
};

class ScannCreateSearcherOp : public ResourceOpKernel<ScannResource> {
 public:
  explicit ScannCreateSearcherOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ResourceOpKernel::Compute(context);

    mutex_lock l(mu_);
    if (resource_ && resource_->is_initialized()) return;

    const Tensor* config_tensor;
    const Tensor* db_tensor;
    const Tensor* threads_tensor;
    OP_REQUIRES_OK(context, context->input("scann_config", &config_tensor));
    OP_REQUIRES_OK(context, context->input("x", &db_tensor));
    OP_REQUIRES_OK(context,
                   context->input("training_threads", &threads_tensor));

    OP_REQUIRES(context, db_tensor->dims() == 2,
                errors::InvalidArgument("Dataset must be two-dimensional"));

    std::string config = config_tensor->scalar<tstring>()();
    auto db_span = TensorToConstSpan<float>(db_tensor);

    OP_REQUIRES_OK(context, ConvertStatus(resource_->scann_->Initialize(
                                db_span, db_tensor->dim_size(0), config,
                                threads_tensor->scalar<int>()())));
    resource_->Initialize();
  }

 private:
  Status CreateResource(ScannResource** ret)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *ret = new ScannResource();
    return Status::OK();
  }
};

class ScannSearchOp : public OpKernel {
 public:
  explicit ScannSearchOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ScannResource* scann_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &scann_resource));

    const Tensor* query_tensor;
    const Tensor* final_nn_tensor;
    const Tensor* reorder_nn_tensor;
    const Tensor* leaves_tensor;
    OP_REQUIRES_OK(context, context->input("queries", &query_tensor));
    OP_REQUIRES_OK(context,
                   context->input("final_num_neighbors", &final_nn_tensor));
    OP_REQUIRES_OK(context, context->input("pre_reordering_num_neighbors",
                                           &reorder_nn_tensor));
    OP_REQUIRES_OK(context, context->input("leaves_to_search", &leaves_tensor));

    OP_REQUIRES(context, query_tensor->dims() == 1,
                errors::InvalidArgument("Query must be one-dimensional. Use "
                                        "ScannSearchBatched for batching"));

    int leaves = leaves_tensor->scalar<int>()();
    int final_nn = final_nn_tensor->scalar<int>()();
    int pre_reorder_nn = reorder_nn_tensor->scalar<int>()();

    auto query_span = TensorToConstSpan<float>(query_tensor);
    tensorflow::scann_ops::DatapointPtr<float> query_ptr(
        nullptr, query_span.data(), query_span.size(), query_span.size());
    tensorflow::scann_ops::NNResultsVector res;
    OP_REQUIRES_OK(context,
                   ConvertStatus(scann_resource->scann_->Search(
                       query_ptr, &res, final_nn, pre_reorder_nn, leaves)));
    Tensor *index_t, *distance_t;

    int64_t res_size = static_cast<int64_t>(res.size());
    OP_REQUIRES_OK(context, context->allocate_output(
                                "index", TensorShape({res_size}), &index_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output("distance", TensorShape({res_size}),
                                            &distance_t));
    scann_resource->scann_->ReshapeNNResult(
        res, index_t->flat<int32_t>().data(), distance_t->flat<float>().data());
  }
};

class ScannSearchBatchedOp : public OpKernel {
 public:
  explicit ScannSearchBatchedOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ScannResource* scann_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &scann_resource));

    const Tensor* query_tensor;
    const Tensor* final_nn_tensor;
    const Tensor* reorder_nn_tensor;
    const Tensor* leaves_tensor;
    const Tensor* parallel_tensor;
    OP_REQUIRES_OK(context, context->input("queries", &query_tensor));
    OP_REQUIRES_OK(context,
                   context->input("final_num_neighbors", &final_nn_tensor));
    OP_REQUIRES_OK(context, context->input("pre_reordering_num_neighbors",
                                           &reorder_nn_tensor));
    OP_REQUIRES_OK(context, context->input("leaves_to_search", &leaves_tensor));
    OP_REQUIRES_OK(context, context->input("parallel", &parallel_tensor));

    OP_REQUIRES(context, query_tensor->dims() == 2,
                errors::InvalidArgument(
                    "Expected 2-dimensional input for query batch."));

    int leaves = leaves_tensor->scalar<int>()();
    int final_nn = final_nn_tensor->scalar<int>()();
    int pre_reorder_nn = reorder_nn_tensor->scalar<int>()();

    tensorflow::scann_ops::DenseDataset<float> queries;
    OP_REQUIRES_OK(context, scann_ops::PopulateDenseDatasetFromTensor(
                                *query_tensor, &queries));
    std::vector<tensorflow::scann_ops::NNResultsVector> res(queries.size());
    auto res_span = tensorflow::scann_ops::MakeMutableSpan(res);
    if (parallel_tensor->scalar<bool>()())
      OP_REQUIRES_OK(
          context, ConvertStatus(scann_resource->scann_->SearchBatchedParallel(
                       queries, res_span, final_nn, pre_reorder_nn, leaves)));
    else
      OP_REQUIRES_OK(context,
                     ConvertStatus(scann_resource->scann_->SearchBatched(
                         queries, res_span, final_nn, pre_reorder_nn, leaves)));
    Tensor *index_t, *distance_t;

    int64_t num_queries = static_cast<int64_t>(res.size()),
            num_neighbors = res.empty() ? 0 : res.front().size();
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            "indices", TensorShape({num_queries, num_neighbors}), &index_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "distances", TensorShape({num_queries, num_neighbors}),
                       &distance_t));
    scann_resource->scann_->ReshapeBatchedNNResult(
        res_span, index_t->flat<int32_t>().data(),
        distance_t->flat<float>().data());
  }
};

class ScannToTensorsOp : public OpKernel {
 public:
  explicit ScannToTensorsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    using tensorflow::scann_ops::ConstSpan;

    ScannResource* scann_resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &scann_resource));

    auto options_or_status = scann_resource->scann_->ExtractOptions();
    OP_REQUIRES_OK(context, ConvertStatus(options_or_status.status()));
    auto opts = options_or_status.ValueOrDie();

    TensorFromProtoRequireOk(context, "scann_config",
                             scann_resource->scann_->config());
    TensorFromProtoRequireOk(context, "serialized_partitioner",
                             opts.serialized_partitioner.get());
    if (opts.datapoints_by_token != nullptr) {
      Tensor* tensor;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  "datapoint_to_token",
                                  TensorShape({static_cast<int64_t>(
                                      scann_resource->scann_->n_points())}),
                                  &tensor));
      auto tensor_flat = tensor->flat<int32_t>();
      for (int i = 0; i < opts.datapoints_by_token->size(); i++) {
        for (auto dp_idx : opts.datapoints_by_token->at(i))
          tensor_flat(dp_idx) = i;
      }
    } else {
      scann_ops::EmptyTensorRequireOk(context, "datapoint_to_token");
    }

    TensorFromProtoRequireOk(context, "ah_codebook", opts.ah_codebook.get());
    TensorFromDenseDatasetRequireOk(context, "hashed_dataset",
                                    opts.hashed_dataset.get());

    tensorflow::scann_ops::DenseDataset<int8_t>* int8_dataset = nullptr;
    ConstSpan<float> int8_mults, dp_norms;
    auto int8_struct = opts.pre_quantized_fixed_point;
    if (int8_struct != nullptr) {
      if (int8_struct->fixed_point_dataset != nullptr)
        int8_dataset = int8_struct->fixed_point_dataset.get();
      auto mults = int8_struct->multiplier_by_dimension;
      if (mults != nullptr)
        int8_mults = ConstSpan<float>(mults->data(), mults->size());
      auto norms = int8_struct->squared_l2_norm_by_datapoint;
      if (norms != nullptr)
        dp_norms = ConstSpan<float>(norms->data(), norms->size());
    }
    TensorFromDenseDatasetRequireOk(context, "int8_dataset", int8_dataset);
    TensorFromSpanRequireOk(context, "int8_multipliers", int8_mults);
    TensorFromSpanRequireOk(context, "dp_norms", dp_norms);
    const tensorflow::scann_ops::DenseDataset<float>* dataset = nullptr;
    if (scann_resource->scann_->needs_dataset()) {
      auto dataset_untyped = scann_resource->scann_->dataset();
      OP_REQUIRES(context, dataset_untyped != nullptr,
                  ConvertStatus(tensorflow::scann_ops::FailedPreconditionError(
                      "Searcher needs original dataset but none is present.")));
      dataset = dynamic_cast<const tensorflow::scann_ops::DenseDataset<float>*>(
          dataset_untyped);
      OP_REQUIRES(context, dataset != nullptr,
                  ConvertStatus(tensorflow::scann_ops::InternalError(
                      "Failed to cast dataset to DenseDataset<float>.")));
    }
    TensorFromDenseDatasetRequireOk(context, "dataset", dataset);
  }
};

class TensorsToScannOp : public ResourceOpKernel<ScannResource> {
 public:
  explicit TensorsToScannOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ResourceOpKernel::Compute(context);

    mutex_lock l(mu_);
    if (resource_ && resource_->is_initialized()) return;

    const Tensor* db_tensor;
    const Tensor* config_tensor;
    const Tensor* serialized_partitioner;
    const Tensor* dp_to_token;
    const Tensor* ah_codebook;
    const Tensor* hashed_dataset;
    const Tensor* int8_dataset;
    const Tensor* int8_multipliers;
    const Tensor* dp_norms;

    OP_REQUIRES_OK(context, context->input("x", &db_tensor));
    OP_REQUIRES_OK(context, context->input("scann_config", &config_tensor));
    OP_REQUIRES_OK(context, context->input("serialized_partitioner",
                                           &serialized_partitioner));
    OP_REQUIRES_OK(context, context->input("datapoint_to_token", &dp_to_token));
    OP_REQUIRES_OK(context, context->input("ah_codebook", &ah_codebook));
    OP_REQUIRES_OK(context, context->input("hashed_dataset", &hashed_dataset));
    OP_REQUIRES_OK(context, context->input("int8_dataset", &int8_dataset));
    OP_REQUIRES_OK(context,
                   context->input("int8_multipliers", &int8_multipliers));
    OP_REQUIRES_OK(context, context->input("dp_norms", &dp_norms));

    uint32_t n_points = tensorflow::scann_ops::kInvalidDatapointIndex;
    tensorflow::scann_ops::ConstSpan<float> dataset;
    if (db_tensor->dims() != 0) {
      OP_REQUIRES(context, db_tensor->dims() == 2,
                  errors::InvalidArgument("Dataset must be two-dimensional"));
      n_points = db_tensor->dim_size(0);
      dataset = scann_ops::TensorToConstSpan<float>(db_tensor);
    }

    const tstring& config_tstr = config_tensor->scalar<tstring>()();
    tensorflow::scann_ops::ScannConfig config;
    config.ParseFromArray(config_tstr.data(), config_tstr.size());

    tensorflow::scann_ops::SingleMachineFactoryOptions opts;
    if (serialized_partitioner->dims() != 0) {
      opts.serialized_partitioner =
          std::make_shared<tensorflow::scann_ops::SerializedPartitioner>();
      const tstring& partitioner_tstr =
          serialized_partitioner->scalar<tstring>()();
      opts.serialized_partitioner->ParseFromArray(partitioner_tstr.data(),
                                                  partitioner_tstr.size());
    }
    if (ah_codebook->dims() != 0) {
      opts.ah_codebook =
          std::make_shared<tensorflow::scann_ops::CentersForAllSubspaces>();
      const tstring& codebook_str = ah_codebook->scalar<tstring>()();
      opts.ah_codebook->ParseFromArray(codebook_str.data(),
                                       codebook_str.size());
    }
    tensorflow::scann_ops::ConstSpan<int32_t> tokenization;
    tensorflow::scann_ops::ConstSpan<uint8_t> hashed_span;
    if (dp_to_token->dims() != 0) {
      n_points = dp_to_token->dim_size(0);
      tokenization = scann_ops::TensorToConstSpan<int32_t>(dp_to_token);
    }
    if (hashed_dataset->dims() != 0) {
      n_points = hashed_dataset->dim_size(0);
      hashed_span = scann_ops::TensorToConstSpan<uint8_t>(hashed_dataset);
    }

    tensorflow::scann_ops::ConstSpan<int8_t> int8_span;
    tensorflow::scann_ops::ConstSpan<float> int8_multiplier_span, norm_span;
    if (int8_dataset->dims() != 0) {
      n_points = int8_dataset->dim_size(0);
      int8_span = scann_ops::TensorToConstSpan<int8_t>(int8_dataset);
    }
    if (int8_multipliers->dims() != 0)
      int8_multiplier_span =
          scann_ops::TensorToConstSpan<float>(int8_multipliers);
    if (dp_norms->dims() != 0)
      norm_span = scann_ops::TensorToConstSpan<float>(dp_norms);

    OP_REQUIRES_OK(context,
                   ConvertStatus(resource_->scann_->Initialize(
                       config, opts, dataset, tokenization, hashed_span,
                       int8_span, int8_multiplier_span, norm_span, n_points)));
    resource_->Initialize();
  }

 private:
  Status CreateResource(ScannResource** ret)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) override {
    *ret = new ScannResource();
    return Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(Name("Scann>ScannCreateSearcher").Device(DEVICE_CPU),
                        ScannCreateSearcherOp);
REGISTER_KERNEL_BUILDER(Name("Scann>ScannSearch").Device(DEVICE_CPU),
                        ScannSearchOp);
REGISTER_KERNEL_BUILDER(Name("Scann>ScannSearchBatched").Device(DEVICE_CPU),
                        ScannSearchBatchedOp);
REGISTER_KERNEL_BUILDER(Name("Scann>ScannToTensors").Device(DEVICE_CPU),
                        ScannToTensorsOp);
REGISTER_KERNEL_BUILDER(Name("Scann>TensorsToScann").Device(DEVICE_CPU),
                        TensorsToScannOp);

}  // namespace scann_ops
}  // namespace tensorflow
