# TSMixer for Multivariate Long-Term Forecasting

## Installation
Install the dependencies:
```
pip install -r requirements.txt
```

## Data Preparation
We use pre-processed datasets provided in [Autoformer](https://github.com/thuml/Autoformer).
```
mkdir dataset
cd dataset
# Download zip file from [Google Drive](https://drive.google.com/corp/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and put it under dataset/
unzip all_six_datasets.zip
mv all_six_datasets/*/*.csv ./
```

## Training Example
Use `run_tuned_hparam.sh` to reproduce results of 96 prediction length.
```
sh run_tuned_hparam.sh ETTm2
sh run_tuned_hparam.sh weather
sh run_tuned_hparam.sh electricity
sh run_tuned_hparam.sh traffic
```
Please check Appendix in our paper for the best hyperparameters tuned in other settings.


## Acknowledgement
We appreciate the following github repos for their valuable code base or datasets:
1. [Are Transformers Effective for Time Series Forecasting?](https://github.com/cure-lab/LTSF-Linear)
2. [Autoformer](https://github.com/thuml/Autoformer)
