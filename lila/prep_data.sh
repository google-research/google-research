#!/bin/bash
# Copyright 2026 The Google Research Authors.
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



DATA_DIR=/workdir/data
mkdir -p $DATA_DIR

### Checkpoints ###
CHECK_DIR=/workdir/checkpoints
mkdir $CHECK_DIR
gcloud storage cp --recursive gs://araslanov-71167094c93f/checkpoints/*.pth $CHECK_DIR/


### COCOStuff ###
COCO_DIR=/workdir/data/COCOStuff
mkdir -p $COCO_DIR
gcloud storage cp --recursive gs://araslanov-71167094c93f/data/COCOStuff/* $COCO_DIR/
cd $COCO_DIR && unzip -q train2017.zip && unzip -q val2017.zip && unzip -q stuffthingmaps_trainval2017.zip


### NYU ###
NYU_DIR=/workdir/data/NYUv2
mkdir -p $NYU_DIR
gcloud storage cp --recursive gs://xcloud-shared/datasets/nyuv2_dataset/* $NYU_DIR/


### DAVIS ###
mkdir -p $COCO_DIR
gcloud storage cp --recursive gs://xcloud-shared/datasets/DAVIS2017* /workdir/data/
#gcloud storage cp --recursive gs://araslanov-71167094c93f/data/DAVIS2017/Depth* /workdir/data/DAVIS2017/


### SSv2 ###
#SSv2_DIR=/workdir/data/SSv2
#mkdir -p $SSv2_DIR
#gcloud storage cp --recursive gs://araslanov-71167094c93f/data/SSv2/20bn-something-something-v2/* $SSv2_DIR/


### RefYTVOS ###
YT_DIR=/workdir/data/RefYTVOS
mkdir -p $YT_DIR
gcloud storage cp --recursive gs://araslanov-71167094c93f/data/RefYTVOS/*.zip $YT_DIR/
unzip $YT_DIR/train.zip -d $YT_DIR/
unzip $YT_DIR/valid.zip -d $YT_DIR/
unzip $YT_DIR/train_all_frames.zip -d $YT_DIR/

# removing broken file
rm /workdir/data/RefYTVOS/train_all_frames/JPEGImages/ef45ce3035/00099.jpg


### Veo3 ###
#VEO_DIR=/workdir/data/Veo3/train
#mkdir -p $VEO_DIR
#gcloud storage cp --recursive gs://araslanov-71167094c93f/data/Veo3Frames/* $VEO_DIR/
