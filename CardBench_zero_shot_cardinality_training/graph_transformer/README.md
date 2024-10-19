# Graph Transformer for Cardinality Prediction in CardBench

## Data Preprocessing

### Build scaling strategy 

Calculate global statistics for the scaling strategy

```bash
DATASET_PATH="training_datasets";
DATASET_TYPE="binary_join"; # binary_join or "single_table"
python graph_transformer/data/build_scaling_strategy.py \
  --dataset_names="accidents,airline,cms_synthetic_patient_data_omop,consumer,covid19_weathersource_com,crypto_bitcoin_cash,employee,ethereum_blockchain,geo_openstreetmap,github_repos,human_variant_annotation,idc_v10,movielens,open_targets_genetics,samples,stackoverflow,tpch_10G,usfs_fia,uspto_oce_claims,wikipedia" \
  --input_dataset_path=$DATASET_PATH \
  --dataset_type=$DATASET_TYPE \
  --output_path=$DATASET_PATH
```

### Preprocess dataset

Preprocess the graph datasets and store them into TF datasets for the graph transformer

```bash
DATASET_PATH="training_datasets";
DATASET_TYPE="binary_join"; # binary_join or "single_table"
for DATASET_NAME in "accidents" "airline" "cms_synthetic_patient_data_omop" \
    "consumer" "covid19_weathersource_com" "crypto_bitcoin_cash" "employee" \
    "ethereum_blockchain" "geo_openstreetmap" "github_repos" "human_variant_annotation" \
    "idc_v10" "movielens" "open_targets_genetics" "samples" "stackoverflow" "tpch_10G" \
    "usfs_fia" "uspto_oce_claims" "wikipedia"; do
  python graph_transformer/data/preprocess_dataset.py \
    --dataset_name=$DATASET_NAME \
    --input_dataset_path=$DATASET_PATH \
    --output_path=$DATASET_PATH \
    --dataset_type=$DATASET_TYPE \
    --scaling_strategy_filename="scaling_strategy.json"
done
```

## Train Graph Transformer Model

### Instance based model
```bash
DATASET_PATH="training_datasets";
MODEL_PATH="models"
DATASET_TYPE="binary_join"; # binary_join or "single_table"
TRAINING_DATASET="accidents";
TEST_DATASET="accidents";
python graph_transformer/train.py \
  --training_dataset_names=$TRAINING_DATASET \
  --test_dataset_name=$TEST_DATASET \
  --input_dataset_path=$DATASET_PATH \
  --model_path=$MODEL_PATH \
  --dataset_type=$DATASET_TYPE \
  --scaling_strategy_filename="scaling_strategy.json" \
  --label="cardinality" \
  --batch_size=128 \
  --train_val_sample_size=5000 \
  --test_sample_size=500
```

### Zeroshot model
```bash
DATASET_PATH="training_datasets";
MODEL_PATH="models";
DATASET_TYPE="binary_join"; # binary_join or "single_table"
TRAINING_DATASETS="airline,cms_synthetic_patient_data_omop,consumer,covid19_weathersource_com,crypto_bitcoin_cash,employee,ethereum_blockchain,geo_openstreetmap,github_repos,human_variant_annotation,idc_v10,movielens,open_targets_genetics,samples,stackoverflow,tpch_10G,usfs_fia,uspto_oce_claims,wikipedia";
TEST_DATASET="accidents";
python graph_transformer/train.py \
  --training_dataset_names=$TRAINING_DATASETS \
  --test_dataset_name=$TEST_DATASET \
  --input_dataset_path=$DATASET_PATH \
  --model_path=$MODEL_PATH \
  --dataset_type=$DATASET_TYPE \
  --scaling_strategy_filename="scaling_strategy.json" \
  --label="cardinality" \
  --batch_size=128 \
  --train_val_sample_size=5000 \
  --test_sample_size=500 \
```

### Finetuned model
```bash
DATASET_PATH="training_datasets";
MODEL_PATH="models";
DATASET_TYPE="binary_join"; # binary_join or "single_table"
TRAINING_DATASET="accidents";
TEST_DATASET="accidents";
BASE_MODEL_CKPT_PATH="models/graph_transformer.ckpt"
python graph_transformer/train.py \
  --training_dataset_names=$TRAINING_DATASET \
  --test_dataset_name=$TEST_DATASET \
  --input_dataset_path=$DATASET_PATH \
  --model_path=$MODEL_PATH \
  --dataset_type=$DATASET_TYPE \
  --scaling_strategy_filename="scaling_strategy.json" \
  --label="cardinality" \
  --batch_size=128 \
  --train_val_sample_size=500 \
  --test_sample_size=500 \
  --base_model_checkpoint_path=$BASE_MODEL_CKPT_PATH
```