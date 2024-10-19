## LLP-Bench: A Large Scale Tabular Benchmark for Learning from Label Proportions

Inside the ```data/bag_ds``` directory create the following directory structure.

```
├── data
│   ├── bag_ds
│   │   ├── split_0
│   │   │   ├── test
│   │   │   └── train
│   │   ├── split_1
│   │   │   ├── test
│   │   │   └── train
│   │   ├── split_2
│   │   │   ├── test
│   │   │   └── train
│   │   ├── split_3
│   │   │   ├── test
│   │   │   └── train
│   │   └── split_4
│   │       ├── test
│   │       └── train
│   ├── preprocessed_dataset
│   └── raw_dataset
```

Inside the ```results``` directory create the following directory structure.

```
├── results
│   ├── autoint_embeddings
│   ├── dist_dicts
│   ├── mean_map_vectors
│   ├── metrics_dicts
│   └── training_dicts
│       ├── feature_bags_ds
│       │   ├── dllp_bce
│       │   ├── dllp_mse
│       │   ├── easy_llp
│       │   ├── genbags
│       │   ├── hard_erot_llp
│       │   ├── mean_map
│       │   ├── ot_llp
│       │   ├── sim_llp
│       │   └── soft_erot_llp
│       ├── fixed_size_feature_bags_ds
│       │   ├── dllp_bce
│       │   ├── dllp_mse
│       │   ├── easy_llp
│       │   ├── genbags
│       │   ├── hard_erot_llp
│       │   ├── mean_map
│       │   ├── ot_llp
│       │   ├── sim_llp
│       │   └── soft_erot_llp
│       └── random_bags_ds
│           ├── dllp_bce
│           ├── dllp_mse
│           ├── easy_llp
│           ├── genbags
│           ├── hard_erot_llp
│           ├── mean_map
│           ├── ot_llp
│           ├── sim_llp
│           └── soft_erot_llp
```


The following code implements the creation and analysis of LLP-Bench discussed in "LLP-Bench: A Large Scale Tabular Benchmark for Learning from Label Proportions". The paper is currently under review in NeurIPS'23.
The authors highly recommend parallelising the for loops in all shell scripts. To run the code in a sequential manner, 
1. Run [preprocess.py](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/featureRec/data/Dataprocess/Criteo/preprocess.py) in this [github repository](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/featureRec/). Copy the files "train_x.txt", "train_y.txt" and "train_i.txt" to data/raw_dataset.
2. Run the command ```source run.sh ```