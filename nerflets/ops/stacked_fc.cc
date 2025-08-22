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

// Copyright 2022 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <torch/extension.h>
#include "stacked_fc.h"


torch::Tensor stacked_fc_forward(
    torch::Tensor input_feat,       // (B, Cin)
    torch::Tensor input_idx,        // (B, K)
    torch::Tensor weights,          // (B, Cin, Cout)
    torch::Tensor bias) {           // (B, 1, Cout)

  auto n = weights.size(0);
  auto in_c = weights.size(1);
  auto out_c = weights.size(2);
  auto k = input_idx.size(1);

  auto w = weights[input_idx].view({-1, in_c, out_c});
  auto b = bias[input_idx].view({-1, 1, out_c});
  auto out = input_feat.repeat({1, k}).view({-1, 1, in_c});
  out = torch::bmm(out, w) + b;
  out = out.view({-1, k, out_c})

  return out;                       // (B, K, Cout)
}

std::vector<torch::Tensor> stacked_fc_backward(
    torch::Tensor out,              // (B, K, Cout)
    torch::Tensor input_feat,       // (B, Cin)
    torch::Tensor input_idx,        // (B, K)
    torch::Tensor weights,          // (B, Cin, Cout)
    torch::Tensor bias,             // (B, 1, Cout)
    torch::Tensor d_out) {          // (B, K, Cout)

  auto w = weights[input_idx].view({-1, in_c, out_c});

  auto d_input_feat = torch::matmul(d_out.t(), w).t();
  auto d_input_idx = torch::zeros_like(input_idx);
  auto d_weights = torch::matmul();
  auto d_bias = torch::matmul();

}


