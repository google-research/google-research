Copyright 2021 The On Combining Bags to Better Learn from Label Proportions Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# On Combining Bags to Better Learn from Label Proportions

Authors: Rishi Saket, Aravindan Raghuveer, Balaraman Ravindran

To Appear in AISTATS'22.

# For Pseudo-synthetic datasets experiments
1. Collect code for previous work:
      * Git pull `https://github.com/giorgiop/almostnolabel` (commit `4de5f54`) as folder `almostnolabel-master` in directory containing this README.
      * Git pull `https://github.com/Z-Jianxin/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework` (commit `adb57fa`) and copy the contents of the LMMCM directory to ./Code/PythCode/.
2. Create virtual environment `pythenv` with python 3.9.2.
      * `cd ./Code`, `python -m venv pythenv` 
      * `source pythenv/bin/activate`, `pip install -r requirements.txt`, `deactivate`.
3. Create virtual environment `LLP` using conda. See instructions in `https://github.com/Z-Jianxin/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework` (commit `adb57fa`).
4. Download, Preprocess original datasets, create bag-level training and test-data for scenarios.
      * `mkdir OrigData` in top dir (containing this README.md)
      * In `OrigData/`
          * `wget http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data`
          * `wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat`
          * `wget https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat`
      * From top dir (containing README.md): `mkdir -p Data/FullDatasets Data/Australian Data/Ionosphere Data/Heart`.
      * From top dir (containing README.md): `source Code/pythenv/bin/activate`, `./dataCreation.sh`, `deactivate`.
5. Run LR and Mean-Map methods.
      * Install `R`.
      * From top dir (containing README.md): `./RCode.sh`.
6. Run Generalized Bag baselines.
      * From top dir (containing README.md): `source Code/pythenv/bin/activate`, `./GenBagCode.sh`, `deactivate`.
7. Run LMMCM baseline.
      * From top dir (containing README.md): `conda activate LLP`, `./PythCode.sh`, `conda deactivate`.
8. Collect the results and process them.
      * From top dir (containing README.md): `mkdir -p Results/Raw_Results`.
      * From top dir (containing README.md): `source Code/pythenv/bin/activate`, `./resultsProcessing.sh`, `deactivate`.
9. Results available in ./Results/.

# For Large dataset experiments
1. Download Criteo Kaggle dataset from http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/ . Create directory `./Large_dataset/Dataset_Preprocessing/Dataset/` and and place `train.txt` in it.
2. Create directories `Split_0`, `Split_1`, `Split_2`, `Split_3` and `Split_4` in `./Large_dataset/Dataset_Preprocessing/Dataset/`.
3. Create virtual environment `pythenv_large` with python 3.9.7.
      * `cd ./Large_dataset`, `python -m venv pythenv_large` 
      * `source pythenv_large/bin/activate`, `pip install -r requirements.txt`
4. From `pythenv_large` run script from top dir (containing README.md): `./Large_Dataset_expts.sh`.
5. Obtain results in directories `Split_0`, `Split_1`, `Split_2`, `Split_3` and `Split_4` in `./Large_dataset/Dataset_Preprocessing/Dataset/`. In each `Split_i`, `results_lin_i` has the results for the event level linear logistic reg (`val_auc` column), and `results_multi_genbags_i` has the results (in the last 4 columns) for (i) Generalized Bags with SDP weights and sq-Euclidean loss, (ii) same with L1-loss, (iii) Single bags and sq-Euclidean loss, (iv) Single bags and KL-div loss. One row per epoch.

# For Large dataset-2 experiments
1. Download Movielens 20m dataset from https://grouplens.org/datasets/movielens/20m/ . Create directory `./Large_dataset_2/Dataset_Preprocessing/Dataset/` and and place all the dataset files in it.
2. Create directories `Split_0`, `Split_1`, `Split_2`, `Split_3` and `Split_4` in `./Large_dataset_2/Dataset_Preprocessing/Dataset/`.
3. Create virtual environment `pythenv_large_2` with python 3.9.7.
      * `cd ./Large_dataset_2`, `python -m venv pythenv_large_2`
      * `source pythenv_large_2/bin/activate`, `pip install -r requirements.txt`
4. From `pythenv_large_2` run script from top dir (containing README.md): `./Large_Dataset_2_expts.sh`.
5. Obtain results in directories `Split_0`, `Split_1`, `Split_2`, `Split_3` and `Split_4` in `./Large_dataset_2/Dataset_Preprocessing/Dataset/`. In each `Split_i`, `results_lin_i` has the results for the event level linear logistic reg (`val_auc` column), and `result_multi_genbags_movielensi` has the results (in the last 4 columns) for (i) Generalized Bags with SDP weights and sq-Euclidean loss, (ii) same with L1-loss, (iii) Single bags and sq-Euclidean loss, (iv) Single bags and KL-div loss. One row per epoch.

