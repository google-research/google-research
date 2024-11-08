# Model Swarms

Repository for [Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence](https://arxiv.org/abs/2410.11163v1).

## Quick Start

#### Initialization

Create a conda environment for Model Swarms.
```
conda env create -f swarm.yml
conda activate swarm
```

Log into huggingface (for gemma access).
```
huggingface-cli login
```

Download initial experts.
```
cd initial_experts
python initial_experts.py
cd ..
```

#### Execute your first Model Swarms search.

Let's run Model Swarms on the [NLGraph](https://arxiv.org/abs/2305.10037) dataset focusing on LLM graph reasoning. `search_nlgraph.sh` provides the starter script for this.

Before running, how many GPUs do you have (and what are the GPU ids?). Change `-g` in line 23 of `search_nlgraph.sh`: by default five GPUs with ids `0,1,2,3,4`, but you can change to `0`, `0,1`, `0,2,4,5,6` or any combination you'd like.

Run it!
```
bash search_nlgraph.sh
```

You will be prompted to log into WandB and it is highly recommended. There will be a directory in `search/` that starts with `nlgraph_...`, all the logs, models, and results will be stored there. You can check out `search/nlgraph_.../log.txt` to see current progress.

End of search: the best-found model will be in `search/nlgraph_.../global_best` and performance metrics will be in `search/nlgraph_.../utility_scratchpad.json`. The json file presents the change in utility values in the search process: for global best, global worst, and individual models. The `log.txt` will contain in the end several overall metrics. Watch for `starting_best_validation_utility`, the best initial expert's utility value on the dev set; `starting_best_single_test_accuracy`, this initial expert's utility on the test set; `ending_best_validation_utility`, the ending global best's utility on the dev set; `ending_best_single_test_accuracy`, the ending global best's utility on the test set.

## Other Objectives

For Objective 1: single task, follow `search_nlgraph.sh` and change `-e` and `-d` to combinations of `(multiple_choice, mmlu)`, `(multiple_choice, mmlu_pro)`, `(multiple_choice, knowledge_crosswords)`, `(multiple_choice, hellaswag)`, `(exact_match, gsm8k)`, `(exact_match, nlgraph)`, `(multiple_choice, truthfulqa)`, `(external_api, realtoxicityprompts)`, `(AbstainQA, mmlu)`. Make sure to change `-n` to include the dataset name in the search directory name. For RealToxicityPrompts you will be prompted to setup Perspective API though google cloud.

For Objective 2: multi-task domains, follow `search_legal.sh` and change `-d` to `legal`, `medical`, `science`, and `culture`. Make sure to change `-n` as well.

For Objective 3: reward models, follow `search_concise.sh` and change `-e` to `rm_default`, `rm_verbose`, and `rm_concise`. Make sure to change `-n` as well.

For Objective 4: human interests, follow `search_phd_application.sh` and change line 8 `domain_list` into the `human_<name>.json` in `data/eval/`. Make sure to change `-n` as well. You will be prompted to setup Vertex AI API to access Gemini APIs through google cloud.

## Changing Hyperparameters and Settings

Do `python search.py -h` to see a list of all possible hyperparameters and settings. Additionally look at the comments for hyperparameters in `search.py`. We already included the default settings in the four `search_<name>.sh` starter scripts, but feel free to play around different settings.

## Adpated Models and Reproducibility

We provide best-found model checkpoints for the objectives in `link pending: we are going through internal model release process, stay tuned!`

## Citation

If Model Swarms is helpful to you, please cite [the paper](https://www.arxiv.org/abs/2410.11163).
