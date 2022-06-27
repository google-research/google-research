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

#ifndef SCANN_OSS_WRAPPERS_SCANN_SERIALIZE_H_
#define SCANN_OSS_WRAPPERS_SCANN_SERIALIZE_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace research_scann {
namespace strings {

std::string Uint32ToKey(uint32_t u32);
std::string Uint64ToKey(uint64_t u64);
void KeyFromUint32(uint32_t u32, std::string* key);
void KeyFromUint64(uint64_t u64, std::string* key);
float KeyToFloat(absl::string_view key);
uint32_t KeyToUint32(absl::string_view key);
uint64_t KeyToUint64(absl::string_view key);
void KeyFromFloat(float x, std::string* key);
std::string FloatToKey(float x);

}  // namespace strings
}  // namespace research_scann

#endif
