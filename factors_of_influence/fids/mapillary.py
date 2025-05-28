# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Defines MapillaryVistas Public dataset, including the MSeg version.

URL: https://www.mapillary.com/dataset/vistas

Paper:
The Mapillary Vistas Dataset for semantic understanding of street scenes.
G. Neuhold, T. Ollmann, S. Rota Bulo, and P. Kontschieder. In ICCV, 2017.
"""


from factors_of_influence.fids import mseg_base

MapillaryVistasPublic = mseg_base.MSegBase(
    mseg_name='Mapillary Vistas Public (MVD)',
    mseg_original_name='mapillary-public66',
    mseg_base_name='mapillary-public65',
    mseg_dirname='MapillaryVistasPublic/',
    mseg_train_dataset=True,
    )
