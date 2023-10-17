# Pretraining for irregular time series classification

## Requirements
Install requirements from requirements.txt:

```
pip install -r requirements.txt
```

Pull smart_cond function from tensorflow version 2.12 - this resolves versioning issues with the modules imported from the STraTS authors.

```
wget https://raw.githubusercontent.com/tensorflow/tensorflow/v2.12.0/tensorflow/python/framework/smart_cond.py -P imported_code/
```

## Description of folders

1. **data/** - save one folder for each dataset containing all training, validation, and test data
    -  **physionet_data_procssed/** - [source](https://physionet.org/content/challenge-2012/1.0.0/). We describe the expected data format below.
    - Other datasets require registration or data use agreements:
        - MIMIC III - [apply here](https://physionet.org/content/mimiciii/1.4/)
        - EICU - [apply here](https://eicu-crd.mit.edu/gettingstarted/access/)
        - H&M Retail - [kaggle dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
    - See details of data processing below

2. **geometric_masks_cached/** - For geometric masking augmentations, we found that generating new masks at each iteration was extremely slow. Our solution was to pre-generate a large set of possible masks, and randomly select from those during training time. The pre-computed masks are saved in this folder.

3. **imported_code/** - Modules that we load from outside sources. In particular, we use some code from [STraTS](https://github.com/sindhura97/STraTS/blob/main/strats_notebook.ipynb).

4. **modules/**
    - **data_generators.py** - functions to generate augmentations of input data, and a DataGenerator class to use during model training (loads batches of data and applies augmentations)
    - **experiment_hepers.py** - contains many helper functions for experiments, including  loss functions, callbacks, data loading, etc.
    - **masking_utils.py** - helper functions for sampling masks (for input data augmentations)



5. **strategy_config.py** - hyperparameter settings, including which hyperparameters options to sweep over.

6. **save_geometric_masks.py** - code to save a set of geometric masks (much faster than re-sampling for every batch)

6. **run.sh** - runs end-to-end training and evaluation (calls the four python scripts below, in order).
7. **generate_random_model_order.py** - creates a random ordering of pretraining and finetuning settings to try.

8. **train_models.py** - runs pretraining and finetuning for a specified list of hyperparameter settings.

9. **get_top_methods.py** - after initial training runs, orders hyperparameter settings by validation performance.

10. **get_test_res.py** - for selected hyperparameter settings, runs finetuning and evaluation procedure across different labeled dataset sizes.


## Instructions

### Expected data

Our training procedure expects the following files:

- fore_train_ip.json & fore_valid_ip.json: forecasting input data for pretraining
- train_ip.json, valid_ip.json & test_ip.json: input data for finetuning classification task
- fore_train_op.json & fore_valid_op.json: forecasting labels for pretraining
- train_op.json, valid_op.json & test_op.json: classification labels

Each ***Input data (pretraining and finetuning)*** json file should contain a list of four components: `[static features, times, values, features]`. Times, values, and features, are used to represent a sequence of events in a time series for each sample (where each event represents a feature's value measured at a certain time).

For a set of N samples, D static features, V time-series features, and L maximum sequence length for our observations, we expect the following numpy arrays for the input data:

1. `static features`:  N x D array of static features.
2. `times`: N x L array representing the timepoints associated with each time series event.
3. `values`: N x L array representing the values associated with each time series event.
4. `features`: N x L array representing the feature observed for each time series event. Features are numbered from 1 to V.

For each row in `times`, `values`, and `features`, if there are fewer than L observed events, the remaining elements the row are filled with 0's. For each dataset, we set L as the 99th percentile of the number of observed observations across samples, and for the 1% of samples with more than L observations, we randomly select L events to keep.


***Fore_train_op.json & fore_valid_op.json***: Forecasting labels are of the shape N x 2V, where N is the number of samples and V is the number of time series features. The first V columns represent the true values of each feature in the forecasting window, and missing values are filled with 0.  The second V columns are binary mask values to indicate whether the corresponding feature was indeed observed during the forecasting window (1 if yes, 0 if no).

***train_op.json, valid_op.json & test_op.pkl***: Binary classification labels of shape (N,)

The [STraTS codebase](https://github.com/sindhura97/STraTS/tree/main/preprocess) provides examples of how to convert raw data from MIMIC III and Physionet 2012 datasets to this format.

### Caching geometric masks

We found that one of our augmentation strategies, geometric masking (first proposed by [Zerveas et al](https://github.com/gzerveas/mvts_transformer)), is slow to re-sample at each training iteration. Instead, we pre-compute many geometric masks, and then sample from these for each training batch.The following code should be run to provide a large set of masks to sample from:

```
python save_geometric_masks.py 20000000 .3 3
python save_geometric_masks.py 20000000 .5 3
python save_geometric_masks.py 20000000 .8 5
```
Note that 20M rows of masks requires ~20GB storage, so it may be necessary to run the above commands with a smaller number of masks.

### Training & evaluation

1. **Set hyperparameter choices in strategy_config.py.** We have provided a default set of hyperparameters to search. If hyperparameters relating to geometric masking are changed, be sure to re-save the appropriate geometric masks as described above.

2. Choose training iterations in bash script **run.sh** - calls to the train_models.py script specify how many hyperparameter settings are tested.

3. Run bash script for end-to-end training and evaluation:

```
sh run.sh
```
Final results from each of the test runs are saved to results/dataset/finetuning/SUPERVISED_LOGS/ folder.






