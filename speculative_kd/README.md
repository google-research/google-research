# Speculative Knowledge Distillation (SKD)

## Environment Setup

We recommend creating separate environments for training and inference due to
conflicting dependencies.

### Data
Data used in our experiments can be found [here](https://drive.google.com/corp/drive/folders/1pb1itG5ITdz5efvY9RqeAbTqISxV1JPh). Please download them into
`./data`.

### Training Environment

Use conda or a virtual environment with Python 3.10:

```bash
pip3 install torch
pip3 install -r requirements.txt
```

### Inference Environment

For inference, install [VLLM](https://github.com/vllm-project/vllm). VLLM dependencies may conflict with the training environment. 

Additionally, after installing the specific version of Transformers
(`transformers==4.44.2`), manually update the following files to enable
speculative knowledge distillation:

- Replace `candidate_generator.py` and `utils.py` from the `transformers/src` directory with the provided versions from `transformers/`:

```bash
cp transformers/* /path-to-lib/python*/site-packages/transformers/generation/
```

## Supervised Fine-Tuning (SFT) Pipeline

We utilize [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main) for supervised fine-tuning (SFT).

- Follow the handbook's installation instructions, ideally in a fresh environment.
- Use the provided YAML files to reproduce paper results:
  - `config/sft_config_gsm.yaml`: GSM8k
  - `config/sft_config_math.yaml`: Math instruction following task
  - `config/sft_config_samsum.yaml`: Summarization task
  - `config/sft_config_trans.yaml`: Translation task

### Running Training (Example: Summarization)

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file config/deepspeed_zero3.yaml train/train_sft.py config/sft/sft_config_samsum.yaml
```

- Modify data and model paths by editing `dataset_mixer` and `model_name_or_path` in `config/sft/sft_config_example.yaml`. Further details are provided within this file.

## Knowledge Distillation Training Pipeline

This pipeline demonstrates speculative knowledge distillation (SKD). The current setup requires 8x NVIDIA A100 (80GB) GPUs (teacher: 7B, student: 2B models). Experiment details can be found in [this page](https://github.com/xu1998hz/efficient_kd/blob/main/experimental_setup.md).

### Configuration

Update parameters in `config/kd_train.yaml`. Minimal necessary changes include:

```yaml
task_params:
  task_type: "summ_1k"
  inp_length: 1024
  max_new_tokens: 128

kd_params:
  kd_type: "skd"
  top_k: 25

model_params:  
  checkpoint_template: "/path-to-teacher-model-checkpoint/"
  assistant_checkpoint_template: "/path-to-student-model-checkpoint/"
  tokenizer_name: /tokenizer address, for example "google/gemma-2b-it"/

resource_params:
  user: "your_username"
  wandb_key: "your_wandb_key"
  wandb_proj: "your_project_name"
```

**Note:** Avoid using `nohup` with `accelerate` as it may cause unexpected crashes.

### Execute Knowledge Distillation

Run training using:

```bash
python3 train/run_kd_train.py
```

or specify the configuration explicitly:

```bash
python train/run_kd_train.py config/kd_train.yaml
```

## Evaluation Pipeline

### GSM8k Evaluation

```bash
python3 eval/eval_gsm.py -max_tokens 512 -ckpt /path-to-gsm-checkpoint/
```

### Translation Evaluation

```bash
python3 eval/eval_mt.py -max_tokens 256 -ckpt /path-to-translation-checkpoint/
```

### Summarization Evaluation

```bash
python3 eval/eval_summ.py -max_tokens 128 -ckpt /path-to-summarization-checkpoint/
```

### Math Instruction Evaluation

For math instruction tasks, follow [EvalPlus](https://github.com/evalplus/evalplus) instructions.

## Citation

```
@misc{xu2025speculativeknowledgedistillationbridging,
      title={Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling}, 
      author={Wenda Xu and Rujun Han and Zifeng Wang and Long T. Le and Dhruv Madeka and Lei Li and William Yang Wang and Rishabh Agarwal and Chen-Yu Lee and Tomas Pfister},
      year={2025},
      eprint={2410.11325},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.11325}, 
}
```