# Hyperbolic Agent

This is the code for the paper
[Hyperbolic Discounting and Learning Over Multiple Horizons](https://arxiv.org/abs/1902.06865)
by William Fedus, Carles Gelada, Yoshua Bengio, Marc Bellemare and Hugo
Larochelle (2019).

You may cite us at

```
@article{fedus2019hyperbolic,
  title={Hyperbolic discounting and learning over multiple horizons},
  author={Fedus, William and Gelada, Carles and Bengio, Yoshua and Bellemare, Marc G and Larochelle, Hugo},
  journal={arXiv preprint arXiv:1902.06865},
  year={2019}
}
```

## Agent Configuration
The agent configuration is handled via gin-bindings. The subdirectory
`hyperbolic_discount/configs` holds the gin-files for the base agents we
considered in the paper:

*   Hyperbolic/Multi-DQN
*   Hyperbolic/Multi-C51
*   Hyperbolic/Multi-Rainbow

### Setup
All of the commands below are run from the parent `google_research` directory.
Start a virtualenv with these commands:

```
virtualenv -p python3 .
source ./bin/activate
```

Then install necessary packages:

```
pip install -r hyperbolic_discount/requirements.txt
```

## Running the Code
To train the agent execute,

```
python -m hyperbolic_discount.train \
  --agent_name=hyperbolic_rainbow \
  --gin_files=configs/hyperbolic_rainbow_agent.gin \
  --base_dir=/tmp/hyperbolic_base_dir \
  --gin_bindings="HyperRainbowAgent.acting_policy='largest_gamma'" \
  --gin_bindings=HyperRainbowAgent.number_of_gammas=10 \
  --gin_bindings=HyperRainbowAgent.hyp_exponent=0.01 \
  --gin_bindings=HyperRainbowAgent.gamma_max=0.99 \
  --atari_roms_path=/tmp/roms" \
  --gin_bindings="Runner.game_name='MontezumaRevenge'
```
