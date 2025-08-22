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

#ifndef LAMP_IO_H
#define LAMP_IO_H

#include <string>
#include <vector>

#include "common.h"  // NOLINT(build/include)

// string location ids are replaced with ints in [0..num_locations).
// location 0 represents all places with # distinct users (=trails) <
// min_user_count.
void ToTrails(const SplitStringTrails& string_trails, int min_user_count,
              bool split_by_time, Trails* train_trails, Trails* test_trails,
              int* num_items);

SplitStringTrails ReadBrightkite(const std::string& file_name,
                                 const std::string& max_train_time);
SplitStringTrails ReadWiki(const std::string& file_name);
SplitStringTrails ReadLastfm(const std::string& file_name, int max_users,
                             const std::string& max_train_time);
SplitStringTrails ReadText(const std::string& file_name, bool verbose = true);
SplitStringTrails ReadReuters();

#endif
