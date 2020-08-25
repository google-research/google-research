# Airdialogue OPE

## Step1. Download Datasets

1. Rule based agent to Rule based customer on Airdialogue
```
cd data
./download_syn_opedata.sh
cd ..
```

2. Transformer based agent to Transformer based customer on Airdialogue
```
cd data
./download_selfplay_opedata.sh
cd ..
```

3. Transformer based agent to Human on Airdialogue
```
cd data
./download_selfplay_opedata.sh
cd ..
```

4. NN based agent to Human on convai2
```
cd data/convai2
./download_convai2_opedata.sh
```


## Step2. Run Scripts

Run single experiments:
```
./script/xxx.sh
```

- `DATA` is the name of subfolder in the ope data folder, it usually represents the target model name
- For other hyperparameters please refer to `main.py`

Run multiple experiments (one by one):
```
python keepruning_xxx.py
```





## Reference
[1] Reinforcement Learning via Fenchel-Rockafellar Duality
