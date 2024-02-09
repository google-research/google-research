# EV3
Code for paper: [Ever Evolving Evaluator (EV3): Towards Flexible and Reliable Meta-Optimization for Knowledge Distillation](https://arxiv.org/abs/2305.16381).

## Env Installation

We recommend the Anaconda version Anaconda3-2023.09-0. Run all commands from the
google_research folder (**not** the ev3 folder).

```bash
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash /tmp/Anaconda3-2023.09-0-Linux-x86_64.sh
conda init bash
source ~/.bashrc
```

Create a conda environment and install required modules.

```bash
conda env create -f ev3/environment.yaml
conda activate ev3
```

## Testing the EV3 optimizer end-to-end

```bash
python -m ev3.run
```

