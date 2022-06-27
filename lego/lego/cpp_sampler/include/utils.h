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

#ifndef UTILS_H
#define UTILS_H


#include <vector>
#include <string>

void split_str(std::string s, std::string delim,
               std::vector<std::string>& result);

double randu();

template<typename Dtype>
Dtype rand_int(Dtype st, Dtype ed);

void hash_combine(long long& seed, long long key);  // NOLINT

template<typename Dtype>
std::string dtype2string();

#endif
