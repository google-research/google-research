#!/bin/bash
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


gdown --fuzzy https://drive.google.com/file/d/1alE33S1GmP5wACMXaLu50rDIoVzBM4ik/view?usp=share_link
unzip all_six_datasets.zip
mv all_six_datasets/* .
rm -rf all_six_datasets*
mkdir ../results