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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "scann/scann_ops/cc/scann_npy.h"

#include "absl/types/optional.h"

PYBIND11_MODULE(scann_pybind, py_module) {
  py_module.doc() = "pybind11 wrapper for ScaNN";
  pybind11::class_<tensorflow::scann_ops::ScannNumpy>(py_module, "ScannNumpy")
      .def(pybind11::init<
           std::optional<const tensorflow::scann_ops::np_row_major_arr<float>>,
           std::optional<
               const tensorflow::scann_ops::np_row_major_arr<uint32_t>>,
           std::optional<
               const tensorflow::scann_ops::np_row_major_arr<uint8_t>>,
           std::optional<const tensorflow::scann_ops::np_row_major_arr<int8_t>>,
           std::optional<const tensorflow::scann_ops::np_row_major_arr<float>>,
           std::optional<const tensorflow::scann_ops::np_row_major_arr<float>>,
           const std::string&>())
      .def(pybind11::init<const tensorflow::scann_ops::np_row_major_arr<float>&,
                          const std::string&, int>())
      .def("search", &tensorflow::scann_ops::ScannNumpy::Search)
      .def("search_batched", &tensorflow::scann_ops::ScannNumpy::SearchBatched)
      .def("serialize", &tensorflow::scann_ops::ScannNumpy::Serialize);
}
