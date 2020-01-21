# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Supported models."""
import kws_streaming.models.cnn as cnn
import kws_streaming.models.dnn as dnn
import kws_streaming.models.dnn_raw as dnn_raw
import kws_streaming.models.ds_cnn as ds_cnn
import kws_streaming.models.gru as gru
import kws_streaming.models.lstm as lstm
import kws_streaming.models.svdf as svdf
# dict with supported models
MODELS = {
    'dnn': dnn.model,
    'dnn_raw': dnn_raw.model,
    'ds_cnn': ds_cnn.model,
    'cnn': cnn.model,
    'gru': gru.model,
    'lstm': lstm.model,
    'svdf': svdf.model,
}
