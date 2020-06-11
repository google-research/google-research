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

#include "scann/scann_ops/cc/scann_npy.h"

namespace tensorflow {
namespace scann_ops {

template <typename T>
inline pybind11::array_t<T> VectorToNumpy2D(const std::vector<T>& v,
                                            size_t dim1) {
  size_t dim2 = v.size() / dim1;
  return pybind11::array_t<T>({dim1, dim2}, v.data());
}

void RuntimeErrorIfNotOk(const char* prefix, const Status& status) {
  if (!status.ok()) {
    std::string msg = prefix + std::string(status.error_message());
    throw std::runtime_error(msg);
  }
}

ScannNumpy::ScannNumpy(
    const np_row_major_arr<float>& np_dataset,
    std::optional<const np_row_major_arr<int32_t>> datapoint_to_token,
    std::optional<const np_row_major_arr<uint8_t>> hashed_dataset,
    const std::string& artifacts_dir) {
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());

  ConstSpan<int32_t> tokenization;
  ConstSpan<uint8_t> hashed_span;
  if (datapoint_to_token) {
    if (datapoint_to_token->ndim() != 1)
      throw std::invalid_argument(
          "Dataset tokenization must be one-dimensional");
    tokenization = ConstSpan<int32_t>(datapoint_to_token->data(),
                                      datapoint_to_token->size());
  }
  if (hashed_dataset) {
    if (hashed_dataset->ndim() != 2)
      throw std::invalid_argument(
          "Hashed dataset input must be two-dimensional");
    hashed_span =
        ConstSpan<uint8_t>(hashed_dataset->data(), hashed_dataset->size());
  }
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, tokenization, hashed_span,
                                        np_dataset.shape()[1], artifacts_dir));
}

ScannNumpy::ScannNumpy(const np_row_major_arr<float>& np_dataset,
                       const std::string& config, int training_threads) {
  if (np_dataset.ndim() != 2)
    throw std::invalid_argument("Dataset input must be two-dimensional");
  ConstSpan<float> dataset(np_dataset.data(), np_dataset.size());
  RuntimeErrorIfNotOk("Error initializing searcher: ",
                      scann_.Initialize(dataset, np_dataset.shape()[1], config,
                                        training_threads));
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::Search(const np_row_major_arr<float>& query, int final_nn,
                   int pre_reorder_nn, int leaves) {
  if (query.ndim() != 1)
    throw std::invalid_argument("Query must be one-dimensional");

  DatapointPtr<float> ptr(nullptr, query.data(), query.size(), query.size());
  NNResultsVector res;
  auto status = scann_.Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
  RuntimeErrorIfNotOk("Error during search: ", status);

  pybind11::array_t<DatapointIndex> indices(res.size());
  pybind11::array_t<float> distances(res.size());
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeNNResult(res, idx_ptr, dis_ptr);
  return {indices, distances};
}

std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
ScannNumpy::SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                          int pre_reorder_nn, int leaves, bool parallel) {
  if (queries.ndim() != 2)
    throw std::invalid_argument("Queries must be in two-dimensional array");

  vector<float> queries_vec(queries.data(), queries.data() + queries.size());
  auto query_dataset = DenseDataset<float>(queries_vec, queries.shape()[0]);

  std::vector<NNResultsVector> res(query_dataset.size());
  Status status;
  if (parallel)
    status = scann_.SearchBatchedParallel(query_dataset, MakeMutableSpan(res),
                                          final_nn, pre_reorder_nn, leaves);
  else
    status = scann_.SearchBatched(query_dataset, MakeMutableSpan(res), final_nn,
                                  pre_reorder_nn, leaves);
  RuntimeErrorIfNotOk("Error during search: ", status);

  if (!res.empty()) final_nn = res.front().size();
  pybind11::array_t<DatapointIndex> indices(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  pybind11::array_t<float> distances(
      {static_cast<long>(query_dataset.size()), static_cast<long>(final_nn)});
  auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
  auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
  scann_.ReshapeBatchedNNResult(MakeConstSpan(res), idx_ptr, dis_ptr);
  return {indices, distances};
}

void ScannNumpy::Serialize(std::string path) {
  Status status = scann_.Serialize(path);
  RuntimeErrorIfNotOk("Failed to extract SingleMachineFactoryOptions: ",
                      status);
}

}  // namespace scann_ops
}  // namespace tensorflow
