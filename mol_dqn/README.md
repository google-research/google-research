# MolDQN

This package contains libraries and scripts for reproducing the results
described in Zhou Z, Kearnes S, Li L, Zare RN, Riley P. Optimization of
Molecules via Deep Reinforcement Learning; http://arxiv.org/abs/1810.08678.

The main library functions, such as the MDP definition in
`chemgraph/mcts/molecules.py`, are of primary interest.

Note that this implementation of the MDP has certain limitations, including (but
not limited to):

  * No support for modification of aromatic bonds. This includes bonds that are
    perceived as aromatic during parsing of the initial state.
  * No support for multiple atom oxidation states. For example, it is not
    currently possible to generate CS(=O)C from an empty initial state, since
    the default oxidation state of sulfur is 2 and the MDP actions are based on
    the available valence without considering alternate oxidation states.

See the paper for additional details.

Here are the commands to produce the experimental results:
## Prepare

### Install the `Contrib` module of `rdkit`

```
git clone https://github.com/rdkit/rdkit
cp -R ./rdkit/Contrib/SA_Score ./chemgraph/dqn/py
```

### Choose the output directory

```
export OUTPUT_DIR="./save"
```

## Single Property Optimization

### Optimization of QED

#### Naive DQN

```
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/naive_dqn.json"
```

#### Bootstrap DQN
##### Step 1
```
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/bootstrap_dqn_step1.json"
```
##### Step 2
```
python optimize_qed.py --model_dir=${OUTPUT_DIR} --hparams="./configs/bootstrap_dqn_step2.json"
```
### Optimization of logP

#### Naive DQN

```
python optimize_logp.py --model_dir=${OUTPUT_DIR} --hparams="./configs/naive_dqn.json"
```

#### Bootstrap DQN
```
python optimize_logp.py --model_dir=${OUTPUT_DIR} --hparams="./configs/bootstrap_dqn_step1.json"
```

## Constraint Optimization

### Naive DQN
```
python optimize_logp_of_800_molecules.py --model_dir=${OUTPUT_DIR} --hparams="./configs/naive_dqn_opt_800.json" --similarity_constraint=0.0
```
### Bootstrap DQN
```
python optimize_logp_of_800_molecules.py --model_dir=${OUTPUT_DIR} --hparams="./configs/bootstrap_dqn_opt_800.json" --similarity_constraint=0.0
```

## Multi-objective Optimization

### Bootstrap DQN
```
python multi_obj_opt.py --model_dir=${OUTPUT_DIR} --hparams="./configs/multi_obj_dqn.json" --start_molecule="CCN1c2ccccc2Cc3c(O)ncnc13" --target_molecule="CCN1c2ccccc2Cc3c(O)ncnc13" --similarity_weight=0.0
```

