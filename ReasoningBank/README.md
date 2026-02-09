# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

## 📜 Overview

<p align="center">
    <img src="assets/reasoningbank.png" width="80%" alt="intro_case">
</p>

We introduce ReasoningBank, a memory mechanism for agents that learns from both 
successful and failed trajectories, with reasoning stored as memory content.

<p align="center">
    <img src="assets/method.png" width="100%" alt="intro_case">
</p>

Building upon this memory formulation, we propose memory-aware test-time scaling,
which leverages the bidirectional synergy between memory and test-time scaling,
establishing experience-driven memory as another scaling dimension for agent
systems.


## 📂 Code Setup
We release code for `SWE-Bench` (software engineering) and `WebArena` (web-browsing),
as in corresponding directories.

Before we start, please install required packages by running `pip install -r requirements.txt`.


### 1. WebArena
#### Docker Configuration
Make sure to correctly install `browsergym` following the [official documentation](https://github.com/ServiceNow/BrowserGym). 

Download and config docker environment for WebArena. Get in to `./webarena_env`,
executing the scripts follow the numerical order of file names. Before executing,
make sure to config the address of each website in corresponding scripts as
instructed correspondingly.

#### Directory Structure

* `WebArena/agents/`: implementation for web agents integrating with browsergym
* `WebArena/autoeval/`: llm-as-a-judge for obtaining correctness signal for trajectories
* `WebArena/config_files/`: data processing for webarena tasks
* `WebArena/prompt/`: instructions used across the implementation

#### Data preprocessing

Download raw test files from [here](https://github.com/web-arena-x/webarena/blob/main/config_files/test.raw.json) and put it to `config_files`.

Run `generate_config_files.py` to process raw test data to config files as input.


#### Run the code

Run directly with ReasoningBank: `bash run.sh`, config `model`, `output_dir`, and
`website`, and `memory_mode` accordingly.

To run with scaling setting, please refer to
`pipeline_scaling.py` and `induce_scaling.py`.

### 2. SWE-Bench
We built upon [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). 

The script `SWE-Bench/run.sh` provides direct running command, which will generate
result files in the output directory. Before running, make sure the
configuration for VertexAI is properly configured as instructed in `run.sh`.

For evaluation, please refer to `sb-cli` command in the [official documentation](https://mini-swe-agent.com/latest/usage/swebench/). 

## Acknowledgement
We adopt code from the following code repositories. We sincerely appreciate these
great work/codebases:

- [Agent-workflow-memory](https://github.com/zorazrw/agent-workflow-memory)
- [webarena](https://github.com/web-arena-x/webarena)
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)

## 📚 Citation
If you find this work useful, please kindly cite our paper:
```
@inproceedings{
  ouyang2026reasoningbank,
  title={ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory},
  author={Ouyang, Siru and Yan, Jun and Hsu, I-Hung and Chen, Yanfei and Jiang, Ke and Wang, Zifeng and Han, Rujun and Le, Long T and Daruki, Samira and Tang, Xiangru and Tirumalashetty, Vishy and Lee,
  George and Rofouei, Mahsan and Lin, Hangfei and Han, Jiawei and Lee, Chen-Yu and Pfister, Tomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=jL7fwchScm}
}
```