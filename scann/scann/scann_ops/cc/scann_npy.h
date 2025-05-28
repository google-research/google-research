// Copyright 2025 The Google Research Authors.
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

#ifndef SCANN_SCANN_OPS_CC_SCANN_NPY_H_
#define SCANN_SCANN_OPS_CC_SCANN_NPY_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "scann/base/single_machine_factory_options.h"
#include "scann/data_format/dataset.h"
#include "scann/scann_ops/cc/scann.h"
#include "scann/utils/types.h"

namespace research_scann {

template <typename T>
using np_row_major_arr =
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

class ScannNumpy {
 public:
  ScannNumpy(const std::string& artifacts_dir,
             const std::string& scann_assets_pbtxt);
  ScannNumpy(const np_row_major_arr<float>& np_dataset,
             const std::string& config, int training_threads);
  std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>> Search(
      const np_row_major_arr<float>& query, int final_nn, int pre_reorder_nn,
      int leaves);
  std::pair<pybind11::array_t<DatapointIndex>, pybind11::array_t<float>>
  SearchBatched(const np_row_major_arr<float>& queries, int final_nn,
                int pre_reorder_nn, int leaves, bool parallel = false,
                int batch_size = 256);
  void Serialize(std::string path, bool relative_path = false);

  vector<DatapointIndex> Upsert(
      std::vector<std::optional<DatapointIndex>> indices,
      std::vector<np_row_major_arr<float>>& vecs, int batch_size = 256);
  vector<DatapointIndex> Delete(std::vector<DatapointIndex> indices);

  int Rebalance(const string& config = "");

  size_t Size() const;

  void SetNumThreads(int num_threads);

  void Reserve(size_t num_datapoints);

  static string SuggestAutopilot(const std::string& config, DatapointIndex n,
                                 DimensionIndex dim);

  string Config();

  pybind11::dict GetHealthStats() const;
  void InitializeHealthStats();

 private:
  ScannInterface scann_;
};

}  // namespace research_scann

#endif
