# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definitions for found TVNs."""

TVN1 = {
    "num_frames": 2,
    "num_blocks": 4,
    "frame_stride": 4,
    "blocks": [{
        "temporal_kernel": 2,
        "current_size": [192, 2],
        "skip": 0,
        "context_gate": 1,
        "non_local": 0,
        "spatial_kernel": 4,
        "temporal_type": "maxpool",
        "repeats": 2,
        "temporal_stride": 2,
        "filters": 64,
        "spatial_type": "std",
        "spatial_stride": 2,
        "squeeze": 0.0,
        "expand": 4
    }, {
        "temporal_kernel": 1,
        "current_size": [64.0, 1.0],
        "skip": 0,
        "context_gate": 0,
        "non_local": 0,
        "spatial_kernel": 3,
        "temporal_type": "1d",
        "repeats": 1,
        "temporal_stride": 1,
        "filters": 128,
        "spatial_type": "maxpool",
        "spatial_stride": 2,
        "squeeze": 0.9067600752736579,
        "expand": 4
    }, {
        "temporal_kernel": 1,
        "current_size": [22.0, 1.0],
        "skip": 1,
        "context_gate": 1,
        "non_local": 0,  # We removed the non-local layer from this module
                         # to further improve runtime.
        "spatial_kernel": 5,
        "temporal_type": "avgpool",
        "repeats": 2,
        "temporal_stride": 1,
        "filters": 128,
        "spatial_type": "depth",
        "spatial_stride": 3,
        "squeeze": 0.0,
        "expand": 4
    }, {
        "temporal_kernel": 1,
        "current_size": [8.0, 1.0],
        "skip": 0,
        "context_gate": 1,
        "non_local": 0,
        "spatial_kernel": 3,
        "temporal_type": "avgpool",
        "repeats": 2,
        "temporal_stride": 1,
        "filters": 256,
        "spatial_type": "std",
        "spatial_stride": 2,
        "squeeze": 0.7265534208526172,
        "expand": 8
    }],
    "image_size": 224,
    "total_layers": 0
}
TVN2 = {
    "num_frames":
        2,
    "image_size":
        256,
    "total_layers":
        0,
    "num_blocks":
        7,
    "frame_stride":
        7,
    "blocks": [{
        "spatial_type": "depth",
        "repeats": 3,
        "spatial_kernel": 5,
        "context_gate": 1,
        "spatial_stride": 3,
        "expand": 4,
        "temporal_type": "1d",
        "filters": 64,
        "non_local": 0,
        "temporal_kernel": 2,
        "skip": 1,
        "temporal_stride": 1,
        "current_size": [256, 2],
        "squeeze": 0.06667482492426025
    }, {
        "spatial_type": "std",
        "repeats": 5,
        "spatial_kernel": 4,
        "context_gate": 1,
        "spatial_stride": 2,
        "expand": 4,
        "temporal_type": "maxpool",
        "filters": 128,
        "non_local": 0,
        "temporal_kernel": 2,
        "skip": 1,
        "temporal_stride": 2,
        "current_size": [52.0, 2],
        "squeeze": 0.693452618624749
    }, {
        "spatial_type": "depth",
        "repeats": 6,
        "spatial_kernel": 4,
        "context_gate": 0,
        "spatial_stride": 1,
        "expand": 1,
        "temporal_type": "1d",
        "filters": 1024,
        "non_local": 0,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "current_size": [26.0, 1.0],
        "squeeze": 0.0
    }, {
        "spatial_type": "avgpool",
        "repeats": 1,
        "spatial_kernel": 3,
        "context_gate": 1,
        "spatial_stride": 2,
        "expand": 4,
        "temporal_type": "1d",
        "filters": 256,
        "non_local": 0,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "current_size": [26.0, 1.0],
        "squeeze": 0.0
    }, {
        "spatial_type": "avgpool",
        "repeats": 6,
        "spatial_kernel": 4,
        "context_gate": 0,
        "spatial_stride": 3,
        "expand": 4,
        "temporal_type": "1d",
        "filters": 512,
        "non_local": 0,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "current_size": [26.0, 1.0],
        "squeeze": 0.9472192794783585
    }, {
        "spatial_type": "depth",
        "repeats": 4,
        "spatial_kernel": 3,
        "context_gate": 0,
        "spatial_stride": 3,
        "expand": 4,
        "temporal_type": "maxpool",
        "filters": 512,
        "non_local": 0,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "current_size": [9.0, 1.0],
        "squeeze": 0.0
    }, {
        "spatial_type": "maxpool",
        "repeats": 1,
        "spatial_kernel": 2,
        "context_gate": 1,
        "spatial_stride": 1,
        "expand": 6,
        "temporal_type": "maxpool",
        "filters": 348,
        "non_local": 0,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "current_size": [3.0, 1.0],
        "squeeze": 0.0
    }]
}
TVN3 = {
    "num_frames":
        8,
    "image_size":
        160,
    "total_layers":
        0,
    "num_blocks":
        4,
    "frame_stride":
        2,
    "blocks": [{
        "temporal_type": "1d",
        "repeats": 2,
        "spatial_kernel": 3,
        "context_gate": 1,
        "spatial_stride": 2,
        "expand": 3,
        "non_local": 0,
        "spatial_type": "depth",
        "filters": 64,
        "temporal_kernel": 5,
        "skip": 0,
        "temporal_stride": 2,
        "current_size": [64, 16],
        "squeeze": 0.6457257964198263
    }, {
        "temporal_type": "1d",
        "repeats": 4,
        "spatial_kernel": 3,
        "context_gate": 0,
        "spatial_stride": 2,
        "expand": 5,
        "non_local": 0,
        "spatial_type": "depth",
        "filters": 256,
        "temporal_kernel": 1,
        "skip": 1,
        "temporal_stride": 1,
        "current_size": [64.0, 6.0],
        "squeeze": 0.9061418818635367
    }, {
        "temporal_type": "maxpool",
        "repeats": 4,
        "spatial_kernel": 5,
        "context_gate": 0,
        "spatial_stride": 4,
        "expand": 2,
        "non_local": 0,
        "spatial_type": "depth",
        "filters": 256,
        "temporal_kernel": 3,
        "skip": 1,
        "temporal_stride": 2,
        "current_size": [64.0, 6.0],
        "squeeze": 0.0
    }, {
        "temporal_type": "maxpool",
        "repeats": 4,
        "spatial_kernel": 3,
        "context_gate": 1,
        "spatial_stride": 2,
        "expand": 5,
        "non_local": 0,
        "spatial_type": "std",
        "filters": 512,
        "temporal_kernel": 3,
        "skip": 0,
        "temporal_stride": 3,
        "current_size": [16.0, 3.0],
        "squeeze": 0.0
    }]
}

TVN4 = {
    "num_frames":
        8,
    "image_size":
        128,
    "total_layers":
        0,
    "num_blocks":
        3,
    "frame_stride":
        4,
    "blocks": [{
        "temporal_type": "1d",
        "repeats": 5,
        "spatial_kernel": 5,
        "context_gate": 0,
        "spatial_stride": 2,
        "expand": 4,
        "non_local": 0,
        "spatial_type": "std",
        "filters": 64,
        "temporal_kernel": 3,
        "skip": 1,
        "temporal_stride": 2,
        "squeeze": 0.4775900391242449,
        "current_size": [64, 8]
    }, {
        "temporal_type": "maxpool",
        "repeats": 4,
        "spatial_kernel": 5,
        "context_gate": 1,
        "spatial_stride": 2,
        "expand": 6,
        "non_local": 0,
        "spatial_type": "depth",
        "filters": 256,
        "temporal_kernel": 2,
        "skip": 0,
        "temporal_stride": 2,
        "squeeze": 0.0,
        "current_size": [64.0, 8.0]
    }, {
        "temporal_type": "1d",
        "repeats": 4,
        "spatial_kernel": 5,
        "context_gate": 1,
        "spatial_stride": 3,
        "expand": 8,
        "non_local": 0,
        "spatial_type": "depth",
        "filters": 256,
        "temporal_kernel": 1,
        "skip": 0,
        "temporal_stride": 1,
        "squeeze": 0.25309102981899967,
        "current_size": [32.0, 4.0]
    }]
}

TVN_MOBILE_1 = {
    "blocks": [{
        "non_local": 0,
        "context_gate": 0,
        "spatial_kernel": 5,
        "temporal_act": "relu",
        "temporal_kernel": 2,
        "spatial_act": "relu",
        "temporal_type": "1d",
        "output_size": [56.0, 2.0],
        "squeeze": 0.004598785768716973,
        "inputs": [0],
        "inv-bottle": 0,
        "current_size": [224, 2],
        "spatial_stride": 4,
        "input_size": [224, 2],
        "spatial_type": "depth",
        "temporal_stride": 1,
        "filters": 32,
        "act": "relu",
        "repeats": 4,
        "expand": 3,
        "skip": 0
    }, {
        "non_local": 0,
        "context_gate": 0,
        "spatial_kernel": 5,
        "temporal_act": "hswish",
        "temporal_kernel": 2,
        "spatial_act": "hswish",
        "temporal_type": "maxpool",
        "output_size": [56.0, 2.0],
        "squeeze": 0.8589741047143076,
        "inputs": [1],
        "inv-bottle": 1,
        "current_size": [224, 2],
        "spatial_stride": 4,
        "input_size": [224, 2],
        "spatial_type": "depth",
        "temporal_stride": 1,
        "filters": 128,
        "act": "relu",
        "repeats": 4,
        "expand": 6,
        "skip": 0
    }],
    "input_streams": [{
        "frame_stride": 1,
        "num_frames": 2,
        "image_size": 224,
        "current_size": [224, 2]
    }],
}

TVN_MOBILE_2 = {
    "blocks": [{
        "non_local": 0,
        "context_gate": 0,
        "spatial_kernel": 5,
        "temporal_act": "relu",
        "temporal_kernel": 2,
        "spatial_act": "hswish",
        "output_size": [56.0, 2.0],
        "spatial_type": "depth",
        "inputs": [0],
        "temporal_type": "1d",
        "spatial_stride": 4,
        "skip": 0,
        "inv-bottle": 0,
        "input_size": [224, 2],
        "squeeze": 0.004598785768716973,
        "temporal_stride": 1,
        "filters": 32,
        "act": "relu",
        "repeats": 4,
        "expand": 3,
        "current_size": [224, 2]
    }, {
        "non_local": 0,
        "context_gate": 0,
        "spatial_kernel": 5,
        "temporal_act": "hswish",
        "temporal_kernel": 2,
        "spatial_act": "hswish",
        "output_size": [56.0, 2.0],
        "spatial_type": "depth",
        "inputs": [1],
        "temporal_type": "maxpool",
        "spatial_stride": 4,
        "skip": 0,
        "inv-bottle": 1,
        "input_size": [224, 2],
        "squeeze": 0.8589741047143076,
        "temporal_stride": 1,
        "filters": 128,
        "act": "relu",
        "repeats": 3,
        "expand": 6,
        "current_size": [224, 2]
    }],
    "input_streams": [{
        "frame_stride": 1,
        "num_frames": 2,
        "image_size": 224,
        "current_size": [224, 2]
    }],
}
