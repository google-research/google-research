// Copyright 2023 The Google Research Authors.
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

#include <iostream>
#include <optional>
#include <utility>


#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "third_party/australis/australis.h"
#include "third_party/australis/petri.h"
#include "xmap_transformer_exporter.h"

namespace {

// absl::StatusOr<std::tuple<aux::PTree, aux::PTree>> Unpack2Tuple(
//     absl::StatusOr<aux::PTree> input) {
//   ASSIGN_OR_RETURN(auto tmp, aux::PTree::DestructureTuple(std::move(input)));
//   if (tmp.size() != 2) {
//     return absl::InvalidArgumentError(absl::StrCat("Wrong size: ",
//     tmp.size()));
//   }
//   return std::tuple<aux::PTree, aux::PTree>(std::move(tmp[0]),
//                                             std::move(tmp[1]));
// }

// TEST(InferenceTest, BasicTest) {
//   ASSERT_OK_AND_ASSIGN(auto client, aux::Client::GetDefault());
//   ASSERT_OK_AND_ASSIGN(auto init_fn,
//                        xmap_transformer::XmapTransformerInit::Load(client));

//   ASSERT_OK_AND_ASSIGN(auto fwd_fn,
//                        xmap_transformer::XmapTransformerFwd::Load(client));

//   ASSERT_OK_AND_ASSIGN((auto [params, token_chunk]),
//   Unpack2Tuple(init_fn())); ASSERT_OK_AND_ASSIGN(auto result, fwd_fn(params,
//   token_chunk));

//   EXPECT_EQ(
//       "(Buffer(f32[4,32,256]), (Buffer(s32[1]), Buffer(bf16[32,8,1,4]), "
//       "Buffer(bf16[32,8,1,4])))",
//       result.ToString());
// }

}  // namespace
