Download the Wine dataset from
https://www.kaggle.com/datasets/dbahri/wine-ratings/data
and make an IID 70/10/20 train/val/test split.
Then run the following to train models on the Wine dataset:
The model_type flag supports linear, rtl, and dnn.
```shell
python -m quantile_regression.qr_wine_training -- \
  --epochs=1 \
  --model_type=rtl \
  --train_filename=path/to/train.csv \
  --val_filename=path/to/val.csv \
  --test_filename=path/to/test.csv
```

Download the Beijing Air Quality dataset from
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
and make an IID 60/20/20 train/val/test split.
Then run the following to train models on the Beijing Air Quality dataset.
The model_type flag supports linear, rtl, and dnn.
```shell
python -m quantile_regression.qr_beijing_training -- \
  --epochs=1 \
  --model_type=rtl \
  --train_filename=path/to/train.csv \
  --val_filename=path/to/val.csv \
  --test_filename=path/to/test.csv
```

Download the Puzzles dataset from
https://www.mayagupta.org/data/PuzzleClub_HoldTimes.csv
Split the data into a test set and a combined train/validation set with the
following script:
```shell
python -m quantile_regression.puzzles_split_data \
  --full_puzzles_data_file=path/to/HoefnagelHoldTimes_Jan29_2021.csv \
  --output_trainval_filename=path/to/trainval.csv \
  --output_test_filename=path/to/test.csv
```
Further split the combined train/validation set IID 80/20.
Then run the following to train models on the Puzzles dataset.
The model_type flag supports linear, rtl, and dnn.
```shell
python -m quantile_regression.qr_puzzles_training \
  --epochs=1 \
  --model_type=rtl \
  --train_filename=path/to/train.csv \
  --val_filename=path/to/val.csv \
  --test_filename=path/to/test.csv
```

Here are some illustrative commands to run various simulation experiments.

Predicting q99 of exponential distribution using expected pinball loss + lattice
model, cross-validating number of keypoints.
```shell
python -m quantile_regression.simulation_experiments \
  --train_steps=5000 \
  --train_size=505 \
  --batch_size=505 \
  --method=TFL \
  --simulation=EXPONENTIAL \
  --quantile_type=BATCH_RANDOM \
  --optimize_q_keypoints_type=P99
```

Computing accuracy of Harrell-Davis method on uniform distribution.
```shell
python -m quantile_regression.simulation_experiments \
  --method=HARRELL \
  --simulation=UNIFORM
```

Predicting q50 of sin skew distribution using beta pinball loss + lattice model.
```shell
python -m quantile_regression.simulation_experiments \
  --train_steps=50000 \
  --train_size=1000 \
  --batch_size=500 \
  --x_keypoints=20 \\
  --q_keypoints=10 \
  --lattice_sizes=7 \
  --q_lattice_size=5 \
  --method=TFL \
  --simulation=SIN \
  --sin_left_skew=1 \
  --sin_right_skew=7 \
  --quantile_type=BETA \
  --mode=0.5 \
  --concentration=10
```

Predicting all quantiles of Griewank distribution using Gasthaus model.
```shell
python -m quantile_regression.simulation_experiments \
  --train_steps=5000 \
  --train_size=1000 \
  --batch_size=500 \
  --num_hidden_layers=3 \
  --hidden_dim=20 \
  --gasthaus_keypoints=50 \
  --method=GASTHAUS \
  --simulation=GRIEWANK \
  --quantile_type=BATCH_RANDOM
```

Predicting all quantiles of Ackley distribution using expected pinball loss +
lattice model.
```shell
python -m quantile_regression.simulation_experiments \
  --train_steps=200000 \
  --train_size=10000 \
  --batch_size=500 \
  --learning_rate=0.01 \
  --q_keypoints=20 \
  --x_keypoints=20 \
  --lattice_sizes=2 \
  --method=TFL \
  --simulation=ACKLEY \
  --quantile_type=BATCH_RANDOM
```