Copyright 2022 The Domain-Agnostic Contrastive Representations for Learning from Label Proportions Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Domain-Agnostic Contrastive Representations for Learning from Label Proportions

Authors: Jay Nandy, Rishi Saket, Prateek Jain, Jatin Chauhan, Aravindan Raghuveer, Balaraman Ravindran

To Appear in CIKM'22.

## Criteo-Kaggle dataset::
1. Download Criteo Kaggle dataset from http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/.
2. We download and keep the original Criteo dataset (train.txt) in the same directory as the codes. `preprocessing.py' creates './data/train/', './data/test/' and './data/val/' for training set, test set and validation set respectively. We divide the dataset as training:test:val :: 75:20:5. 
3. Train SelfCLR-LLP, DLLP and supervised classification models using selfclr_llp_main.py, dllp_main.py and supervised_main.py respectively. This will run the code with default sets of hyper-parameters. For changing the hyper-parameters, please update inside the code.


## MovieLens-1M dataset::
1. Download MovieLens-1M dataset from https://grouplens.org/datasets/movielens/1m/.
2. We download and keep the original Movielens-1M dataset in the same directory as this code. `preprocessing.py' randomly split the dataset as training:test :: 80:20 ratio.
3. Train SelfCLR-LLP, DLLP and supervised classification models using selfclr_llp_main.py, dllp_main.py and supervised_main.py respectively. This will run the code with default sets of hyper-parameters. For changing the hyper-parameters, please update inside the code.