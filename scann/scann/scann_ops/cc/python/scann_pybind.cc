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

#include <cstdint>
#include <string>

#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "scann/scann_ops/cc/scann_npy.h"

PYBIND11_MODULE(scann_pybind, py_module) {
  py_module.doc() = "pybind11 wrapper for ScaNN";
  pybind11::class_<research_scann::ScannNumpy>(py_module, "ScannNumpy")
      .def(pybind11::init<const std::string&, const std::string&>())
      .def(pybind11::init<const research_scann::np_row_major_arr<float>&,
                          const std::string&, int>())
      .def("search", &research_scann::ScannNumpy::Search)
      .def("search_batched", &research_scann::ScannNumpy::SearchBatched)

      .def("upsert", &research_scann::ScannNumpy::Upsert)
      .def("delete", &research_scann::ScannNumpy::Delete)
      .def("rebalance", &research_scann::ScannNumpy::Rebalance)
      .def_static("suggest_autopilot",
                  &research_scann::ScannNumpy::SuggestAutopilot)
      .def("size", &research_scann::ScannNumpy::Size)
      .def("reserve", &research_scann::ScannNumpy::Reserve)
      .def("set_num_threads", &research_scann::ScannNumpy::SetNumThreads)
      .def("config", &research_scann::ScannNumpy::Config)
      .def("serialize", &research_scann::ScannNumpy::Serialize)
      .def("get_health_stats", &research_scann::ScannNumpy::GetHealthStats)
      .def("initialize_health_stats",
           &research_scann::ScannNumpy::InitializeHealthStats);
}
