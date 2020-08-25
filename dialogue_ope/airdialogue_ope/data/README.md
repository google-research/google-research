# Datasets

## AirDialogue Synthetic Data

You can either make the data by your own or download the preprocessed data.

### Option 1. Make data by your own

Step 1. Install airdialogue toolkit somewhere

See https://github.com/google/airdialogue/

Step 2. Download Airdialogue

```
sh download_air.sh
```

Step 3. Make data

```
cd orig
airdialogue sim_ope --output_dir ../syn_opedata/orig/syn_ope_data_500 --num_samples 500 --verbose
sh make_syn_opedata.sh
```

### Option 2. Download preprocessed data

```
sh download_syn_opedata.sh
```

## AirDialogue Neural Selfplay Data

```
sh download_selfplay_opedata.sh
```

## AirDialogue Human Eval Data

```
sh download_human_opedata.sh
```

## Convai2 Huamn Evaluation Data

```
cd convai2
sh download_convai2_opedata.sh
```
