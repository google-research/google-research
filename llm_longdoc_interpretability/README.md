# Long Document Interpretability using Large Language Models via Hierarchical Version of Shapley and Banzhaf values

Implements methods for [TextGenSHAP: Scalable Post-hoc Explanations in Text
Generation with Long Documents](https://arxiv.org/abs/2312.01279).

## Setup and run minimal experiment

Make sure you're in the `/experiments` directory before running these commands.

Dependencies are defined in `requirements.txt`. `run.sh` installs them and
runs a minimal toy version of the experiments in this repo. Start with:
```
bash run.sh
```

## Run Full Experiments

### Download the NQ dataset.
```
bash scripts/download-nq.sh
```


### Generate embeddings and retrieval results for NQ

```
bash scripts/run-nq-embedding-and-retrieval.sh psgs_w100 100
```

5. You can now use the generated outputs to run any of the evaluation experiments

### Generate embeddings and retrieval results for MirACL

```
bash scripts/run-miracl-embedding-and-retrieval.sh
```

### Download the FiD model and parameters

```
bash scripts/download-fid-model.sh
```

### Get interpretability results

1. Run the interpretability script for NQ.

```
bash run-nq-interpretations.sh shapley
```

2. Run the interpretability script for MIRACL.

```
bash run-miracl-interpretations.sh banzhaf
```

## Flash Attention

`flash_attn.py` is heavily inspired by the following two implementations: 
- https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
- https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

The four types of flash attention implemented are described below: 
- `flash_attn_v1` = vanilla FA with full bias matrix 
- `flash_attn_v2` = vanilla FA with virtual bias matrix; encoder-encoder self attention 
- `flash_attn_v6` = block-sparse (FiD) FA; encoder-encoder self attention 
- `flash_attn_v7` = block-sparse (FiD) FA; encoder-decoder cross attention
