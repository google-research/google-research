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

#include "scann/oss_wrappers/scann_serialize.h"

#include "absl/base/casts.h"
#include "absl/base/internal/endian.h"

namespace tensorflow {
namespace scann_ops {
namespace strings {
namespace {

template <typename UintType, typename FloatType>
UintType UintFromIEEE754(FloatType f) {
  const UintType n = absl::bit_cast<UintType>(f);
  const UintType sign_bit = ~(~static_cast<UintType>(0) >> 1);
  if ((n & sign_bit) == 0) return n + sign_bit;
  return 0 - n;
}

template <typename FloatType, typename UintType>
FloatType IEEE754FromUint(UintType n) {
  const UintType sign_bit = ~(~static_cast<UintType>(0) >> 1);
  if (n & sign_bit) {
    n -= sign_bit;
  } else {
    n = 0 - n;
  }
  return absl::bit_cast<FloatType>(n);
}
}  // namespace

inline std::string Uint32ToKey(uint32_t u32) {
  std::string key;
  KeyFromUint32(u32, &key);
  return key;
}

inline std::string Uint64ToKey(uint64_t u64) {
  std::string key;
  KeyFromUint64(u64, &key);
  return key;
}

inline void KeyFromUint32(uint32_t u32, std::string* key) {
  uint32_t norder = absl::ghtonl(u32);
  key->assign(reinterpret_cast<const char*>(&norder), sizeof(norder));
}

inline void KeyFromUint64(uint64_t u64, std::string* key) {
  uint64_t norder = absl::ghtonll(u64);
  key->assign(reinterpret_cast<const char*>(&norder), sizeof(norder));
}

inline uint32_t KeyToUint32(absl::string_view key) {
  uint32_t value;
  memcpy(&value, key.data(), sizeof(value));
  return absl::gntohl(value);
}

inline uint64_t KeyToUint64(absl::string_view key) {
  uint64_t value;
  memcpy(&value, key.data(), sizeof(value));
  return absl::gntohll(value);
}

void KeyFromFloat(float x, std::string* key) {
  const uint32_t n = UintFromIEEE754<uint32_t>(x);
  KeyFromUint32(n, key);
}

std::string FloatToKey(float x) {
  std::string key;
  KeyFromFloat(x, &key);
  return key;
}

float KeyToFloat(absl::string_view key) {
  const uint32_t n = KeyToUint32(key);
  return IEEE754FromUint<float>(n);
}

}  // namespace strings
}  // namespace scann_ops
}  // namespace tensorflow
