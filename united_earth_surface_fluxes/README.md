# Earth Surface Fluxes

This is not an officially supported Google product.

This package provides a framework for ERA5 Earth surface fluxes data processing
and model routines.

## Pipeline Overview

Running the complete platform requires two specific stages: Generating the
localized data, and executing the neural network models.

### 1. Data Generation

Before executing the models, you must compile your spatial grid and generate chunked `.tfrecord` variables. For our paper submission, data generation was executed across the entire year of 2024 (`--period_list="2024-01-01:2024-12-31"`). Navigate to the `data_processing/` directory and observe the local `README.md` guidelines there.

### 2. Model Training & Testing

The models leverage the localized `data/` folder to pull configuration matrices and process the `.tfrecord` iterations. For our paper submission, model training was conducted on a single NVIDIA A100 GPU using specific discrete dates sampled from the 2024 dataset:
* `--train_period_list`: `"2024-02-07,2024-04-07,2024-06-07,2024-08-07,2024-10-07,2024-12-07"`
* `--val_period_list`: `"2024-02-15,2024-04-15,2024-06-15,2024-08-15,2024-10-15,2024-12-15"`
* `--test_period_list`: `"2024-01-22,2024-02-22,2024-03-22,2024-04-22,2024-05-22,2024-06-22,2024-07-22,2024-08-22,2024-09-22,2024-10-22,2024-11-22,2024-12-22"`

To train the latent heat mixture of experts model (or any provided baseline architecture), execute `train.py` from the root of this structure. It natively targets `./data/` for mapping parameters and `.tfrecord` inputs.

```bash
python train.py \
  --model_name="UniTeD" \
  --train_period_list="2024-02-07,2024-04-07,2024-06-07,2024-08-07,2024-10-07,2024-12-07" \
  --val_period_list="2024-02-15,2024-04-15,2024-06-15,2024-08-15,2024-10-15,2024-12-15" \
  --fraction_filter_type="pure" \
  --num_epochs=500 \
  --batch_size=80 \
  --learning_rate=0.0001 \
  --trunk_num_layers=1 \
  --exp_num_layers=2 \
  --hid_dim=256 \
  --transformer_embed_dim=128 \
  --transformer_ff_dim=512 \
  --transformer_num_heads=4 \
  --budget_eq_weight=0.001
```

### Training Arguments

* `--fraction_filter_type`: Filtering strategy applied to the dataset land fractions. Options include:
  * `"pure"`: Model is trained exclusively on locations where only one land surface cover is present (fraction = 1.0). To account for severe geographical representation disparity across pure surface types, the training pipeline automatically applies a dataset balancing strategy to equalize class distributions.
  * `"mixed"`: Model is trained on locations where no single land surface has a fraction greater than 0.5.
  * `"none"`: No fraction filtering is applied; trains on all available locations.

Once executed successfully, outputs and model weights will drop into the `./output` directory. You can query testing schemas locally using:

```bash
python test.py \
  --model_name="UniTeD" \
  --model_dir=./output \
  --test_period_list="2024-01-22,2024-02-22,2024-03-22,2024-04-22,2024-05-22,2024-06-22,2024-07-22,2024-08-22,2024-09-22,2024-10-22,2024-11-22,2024-12-22" \
  --batch_size=1256 \
  --seed=1000 \
  --trunk_num_layers=1 \
  --exp_num_layers=2 \
  --hid_dim=256 \
  --transformer_embed_dim=128 \
  --transformer_ff_dim=512 \
  --transformer_num_heads=4
```

