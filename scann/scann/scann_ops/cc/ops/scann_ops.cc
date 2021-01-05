// Copyright 2021 The Google Research Authors.
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace addons {

REGISTER_OP("Scann>ScannCreateSearcher")
    .Input("x: float32")
    .Input("scann_config: string")
    .Input("training_threads: int32")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("searcher_handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("Scann>ScannSearch")
    .Input("scann_handle: resource")
    .Input("queries: float32")
    .Input("final_num_neighbors: int32")
    .Input("pre_reordering_num_neighbors: int32")
    .Input("leaves_to_search: int32")
    .Output("index: int32")
    .Output("distance: float32")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("Scann>ScannSearchBatched")
    .Input("scann_handle: resource")
    .Input("queries: float32")
    .Input("final_num_neighbors: int32")
    .Input("pre_reordering_num_neighbors: int32")
    .Input("leaves_to_search: int32")
    .Input("parallel: bool")
    .Output("indices: int32")
    .Output("distances: float32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("Scann>ScannToTensors")
    .Input("scann_handle: resource")
    .Output("scann_config: string")

    .Output("serialized_partitioner: string")
    .Output("datapoint_to_token: int32")

    .Output("ah_codebook: string")
    .Output("hashed_dataset: uint8")

    .Output("int8_dataset: int8")
    .Output("int8_multipliers: float")
    .Output("dp_norms: float")

    .Output("dataset: float");

REGISTER_OP("Scann>TensorsToScann")
    .Input("x: float32")
    .Input("scann_config: string")

    .Input("serialized_partitioner: string")
    .Input("datapoint_to_token: int32")

    .Input("ah_codebook: string")
    .Input("hashed_dataset: uint8")

    .Input("int8_dataset: int8")
    .Input("int8_multipliers: float")
    .Input("dp_norms: float")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("searcher_handle: resource")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace addons
}  // namespace tensorflow
