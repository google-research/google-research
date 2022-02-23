// Copyright 2022 The Google Research Authors.
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

#ifndef ALIAS_METHOD_H
#define ALIAS_METHOD_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <vector>

class AliasMethod
{
 public:
    AliasMethod();

    void setup_from_numpy(
        uint64_t num_choices,
        py::array_t<double,
                    py::array::c_style | py::array::forcecast> sample_weights);
    void setup(std::vector<double>& sample_weights);
    uint64_t draw_sample();

 protected:
    void clear_table();
    void build_table(const double* weights, uint64_t _num_choices);
    uint64_t num_choices;
    std::vector<double> prob_small;
    std::vector<uint64_t> choice_small, choice_large;
};

#endif
