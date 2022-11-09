# Revisiting Neural Scaling Laws

This is an official codebase for the implementation of [scaling law estimators](https://arxiv.org/abs/2209.06640). We also release a benchmark dataset comprising of 92 evaluations in language and vision.

```
@inproceedings{alabdulmohsin2022revisiting,
  title={Revisiting Neural Scaling Laws in Language and Vision},
  author={Alabdulmohsin, Ibrahim and Neyshabur, Behnam and Zhai, Xiaohua},
  booktitle={{NeurIPS}},
  year={2022}
}
```

# Benchmark data

To accelerate research in scaling laws, we provide a benchmark dataset in `data/` comprising of 92 evaluations in learning curve extrapolation. We split the data into vision-related tasks `data/benchmark.vision.csv` and language-related tasks `data/benchmark.lang.csv`. The columns are:

### Domain

This takes four possible values:

| Domain  | Description                          | # evals | Metric            |
|---------|--------------------------------------|---------|-------------------| 
| `IC`    | Image Classification                 | 72      | n-shot error rate |
| `NMT`   | Neural Machine Translation           | 5       | log-perplexity    |
| `LM`    | Language Modeling                    | 5       | log-loss          |
| `BB `   | Big Bench                            | 10      | preferred metrics |

To learn more about the preferred metrics in the Big-Bench evaluations, please consult (Srivastava, et al. 2022). See our [paper](https://arxiv.org/abs/2209.06640) for more details.

### Task

Available tasks are listed below:

| Domain  | Available Tasks   | Description                                    |
|---------|-------------------|------------------------------------------------|
| `IC`    | `bird_5`          | 5-shot;  Birds dataset (Welinder, et al. 2010) |
|         | `bird_10`         | 10-shot; Birds.                                |
|         | `bird_25`         | 25-shot; Birds.                                |
|         | `inet_5`          | 5-shot; ImageNet/ILSRCV2012 (Deng, et al. 2009)|
|         | `inet_10`         | 10-shot; ImageNet/ILSRCV2012                   |
|         | `inet_25`         | 25-shot; ImageNet/ILSRCV2012                   |
|         | `c_5`             | 5-shot; CIFAR100 (Krizhevsky, 2009)            |
|         | `c_10`            | 10-shot; CIFAR100                              |
|         | `c_25`            | 25-shot; CIFAR100                              |
|         | `cal_5`           | 5-shot; Caltech101 (Fei-Fei, et al. 2004).     |
|         | `cal_10`          | 10-shot; Caltech101                            |
|         | `cal_25`          | 25-shot; Caltech101                            |
|---------|-------------------|------------------------------------------------|
| `NMT`   | `log_perplexity`  | Models are trained with per-token cross-entropy|
:         :                   : loss and evaluated using log-perplexity.       :
|---------|-------------------|------------------------------------------------|
| `LM`    | `val_loss`        | Predict the next token.                        |
|---------|-------------------|------------------------------------------------|
| `BB`    |`('date','1-shot')`| 1-shot: date_understanding                     |
|         |`('date','2-shot')`| 2-shot: date_understanding                     |
|         |`('ling','1-shot')`| 1-shot: linguistic_mappings                    |
|         |`('ling','2-shot')`| 2-shot: linguistic_mappings                    |
|         |`('qa','1-shot')`  | 1-shot: qa_wikidata                            |
|         |`('qa','2-shot')`  | 2-shot: qa_wikidata.                           |
|         |`('mult','1-shot')`| 1-shot: mult_data_wrangling                    |
|         |`('mult','2-shot')`| 2-shot: mult_data_wrangling                    |
|         |`('unit','1-shot')`| 1-shot: unit_conversion                        |
|         |`('unit','1-shot')`| 1-shot: unit_conversion                        |


### Model

Available models are:

| Domain | Models     | Params | Description                                   |
|--------|------------|--------|-----------------------------------------------|
| `IC`   | `ViT/S/16` | 32M    | Vision Transformer (Dosovitskiy, et al. 2020) |
|        | `ViT/B/16` | 110M   | Vision Transformer (Dosovitskiy, et al. 2020) |
|        | `BiT/50/1` | 61M    | Big-Transfer ResNet (Kolesnikov, et al. 2020) |
|        | `BiT/101/3`| 494M   | Big-Transfer ResNet (Kolesnikov, et al. 2020) |
|        | `MiX/B/16` | 73M    | MLP Mixer (Tolstikhin, et al. 2021)           |
|        | `MiX/L/16` | 226M   | MLP Mixer (Tolstikhin, et al. 2021)           |
|--------|------------|--------|-----------------------------------------------|
| `NMT`  |`6Enc,6Dec` | 300M   | 6 Encoders - 6 Decoders (Bansal, et al. 2022) |
|        |`6Enc,28Dec`| 300M   | 6 Encoders - 28 Decoders (Bansal, et al. 2022)|
|        |`28Enc,6Dec`| 300M   | 28 Encoders - 6 Decoders (Bansal, et al. 2022)|
|        |`Dec-only`  | 300M   | Decoder-only / LM loss (Bansal, et al. 2022)  |
|        |`TEnc-LSTM` | 300M   | Transformer encoder - LSTM decoder (Bansal, et al. 2022) |
|--------|------------|--------|-----------------------------------------------|
| `LM`   |`1.68e+07`  | 17M    | Decoder-only (Thoppilan, et al, 2022)         |
|        |`1.34e+08`  | 134M   | Decoder-only (Thoppilan, et al, 2022)         |
|        |`2.62e+08`  | 262M   | Decoder-only (Thoppilan, et al, 2022)         |
|        |`4.53e+08`  | 453M   | Decoder-only (Thoppilan, et al, 2022)         |
|        |`1.07e+09`  | 1B     | Decoder-only (Thoppilan, et al, 2022)         |
|--------|------------|--------|-----------------------------------------------|
| `BB`   |`2.62e+08`  | 262M   | Decoder-only (Thoppilan, et al, 2022)         |

### Seen Examples

Number of seen examples or tokens, which equals batch_size * training_steps.


### Loss

Numeric value of the loss/metric. 

### Training

This is a binary number: 1 if used for training and 0 if used for testing (extrapolation).


# Methods

See the provided colab.

### 1. Setting up the Python environment
The code has been tested with Python 3.7.10 and Numpy 1.23.2.

### 2. Running the code
We provide below a walk-through example using the proposed scaling law estimator M4. First, we load the data:

```
np.random.seed(2021)
df_vision = pd.read_csv('benchmark.vision.csv')
df_lang = pd.read_csv('benchmark.lang.csv')
df_all = pd.concat([df_vision, df_lang])
```

Choose the task and fetch its training data:

```
model = 'ViT/B/16'
task = 'inet_5'
df_subset_train = df[(df['Model'] == model) & (df['Task'] == task) & (df['Training'] == 1)]
```

Set up the loss values in a dict:

```
x_vals = np.array(df_subset_train['Seen Examples'])
y_vals = np.array(df_subset_train['Loss'])
fit_values = {x: y for x, y in zip(x_vals, y_vals)}
```

Train the scaling law estimator:

```
M4 = m4.Estimator
slaw = M4(fit_values, err_inf=None, err_0=0.999, update_err_0=True)
slaw.estimate_scaling_params(verbose=0, max_iterations=10_000)
print('beta, c, alpha, err_inf = %.2f, %0.2f, %0.2f, %0.2f' %(
      slaw.beta, slaw.c, slaw.alpha, slaw.err_inf)
```

Fetch test data and evaluate by reporting the RMSE and standard error:

```
df_subset_test = df[(df['Model'] == model) & (df['Task'] == task) & (df['Training'] == 0)]
x_test = np.array(df_subset_test['Seen Examples'])
y_test = np.array(df_subset_test['Loss'])
y_pred = np.array([slaw.predict_loss(xi) for xi in x])
error = (np.log(y_pred) - np.log(y_test)) ** 2
err_mu = np.mean(error)
err_std = np.sqrt(err_mu + np.std(error) / (len(y_test)**0.5)) - np.sqrt(err_mu)
print(f'Mean RMSE = {}, Standard Error = {err_std}')
```
