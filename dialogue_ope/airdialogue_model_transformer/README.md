# Airdialogue


## Install

1. install parlai
```
git submodule init
git submodule update
cd ParlAI
python setup.py develop
cd ..
```

2. Patch new AirDialogue
```
cp -r task/* ParlAI/parlai/tasks/airdialogue
ln -s ~/airdialogue_model_transformer ParlAI/projects/airdialogue
```
replace `~/airdialogue_model_transformer` by your path

3. Download Data From internal
```
cd ParlAI
mkdir data
cd data
gsutil cp gs://airdialogue_share/airdialogue_data.tar.gz ./airdialogue.tar.gz
cd ../..
```

4. Try it by display samples
```
parlai display_data -t airdialogue:agent:300
```



## Intro

tasks:
`airdialogue:<agenttype>:<datasize>`

- `<agenttype>` is `agent` or `customer`
- `<datasize>` is training data size, if not specified use whole data.

## Training

```
CUDA_VISIBLE_DEVICS=0 python train_customer.py
CUDA_VISIBLE_DEVICS=0 python train_agent.py
```

## Download Pre-Train Models
```
mkdir outputs
cd outputs
gsutil cp gs://airdialogue_share/air_pretrain_model.tar.gz ./
tar -xvzf air_pretrain_model.tar.gz
```

## Inference Eval (BLEU)

```
CUDA_VISIBLE_DEVICES=0 parlai eval_model -mf outputs/customer/model -t airdialogue:customer --skip-generation false -bs 96 --beam-size 10 --inference beam
CUDA_VISIBLE_DEVICES=0 parlai eval_model -mf outputs/agent/model -t airdialogue:agent --skip-generation false -bs 96 --beam-size 10 --inference beam
```

## Human Interactive Chat

```
parlai interactive -mf outputs/agent/model -t airdialogue:agent:500 --skip-generation false
parlai interactive -mf outputs/customer/model -t airdialogue:customer:500 --skip-generation false
```

## SelfPlay Eval

```
python selfplay.py --selfchat-max-turns 10 -t airdialogue:both -mf outputs/customer/model -pmf outputs/agent/model --outfile outputs/selfchat_eval --num-self-chats 200
```

Turn off example display:
```
python selfplay.py --selfchat-max-turns 10 -t airdialogue:both -mf outputs/customer/model -pmf outputs/agent/model --outfile outputs/selfchat_eval --num-self-chats 1000 --display-examples false
```

## Command Line Human Evaluation
This provides semiauto human eval
```
python selfplay.py --selfchat-max-turns 10 -t airdialogue:both:300 -mf human -pmf outputs/agent/model --outfile outputs/human_eval --num-self-chats 50 --display-examples false
```

- `--start-cid` can start from a give conversation id

You can enter you own words, or select predefined response by '-x', 'x=0,1,2,3' represents the response id. 

## Generate OPE Data

Selfplay:
```
CUDA_VISIBLE_DEVICES=0 python generate_ope_data.py --tgt-agent full
```

Human Evaluation:
```
CUDA_VISIBLE_DEVICES=0 python generate_ope_data.py --eval-dir outputs/human_eval/ --log-file log_0-50.jsonl --save-dir outputs/human_ope_data/ --tgt-agent full
```
