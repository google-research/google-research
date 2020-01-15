## NoRML: No-Reward Meta Learning

This repository contains code released for the paper
[NoRML: No-Reward Meta Learning](https://arxiv.org/pdf/1903.01063.pdf).

First, install all dependencies by
```
pip install -r norml/requirements.txt
```
The HalfCheetah environment requires Mujoco, so make sure you also followed the proper [instructions](https://github.com/openai/mujoco-py) to install mujoco and mujoco-py.

You can start training from scratch by
```
python -m norml.train_maml --config MOVE_POINT_ROTATE_MAML --logs maml_checkpoints
```
Where config should be one of the configs defined in `config_maml.py`. The config string is of the type `{ENV_NAME}_{ALG_NAME}`, where `ENV_NAME` is one of `MOVE_POINT_ROTATE`, `MOVE_POINT_ROTATE_SPARSE`, `CARTPOLE_SENSOR`, `HALFCHEETAH_MOTOR` and `ALG_NAME` is one of `DR`, `MAML`, `MAML_OFFSET`, `MAML_LAF`, `NORML` as mentioned in the paper.

`MOVE_POINT_ROTATE` are fast to train and can converge within minutes. Training `MOVE_POINT_ROTATE_SPARSE` and `CARTPOLE_SENSOR` can take as long as a day. The Halfcheetah training was done via parallelized workers on a cloud server, and can take a long time on a single machine.

We also provide a convenient script to evaluate the training performance:
```
python -m norml.eval_maml \
--model_dir norml/example_checkpoints/move_point_rotate_sparse/norml/all_weights.ckpt-991 \
--output_dir maml_eval_results \
--render=True \
--num_finetune_steps 1 \
--test_task_index 0 \
--eval_finetune=True
```
You should be able to see states/actions logs and an optional rendered video in the maml_eval_results folder.


Citing
------
If you use this code in your research, please cite the following paper:

> Yang, Y., Caluwaerts, K., Iscen, A., Tan, J. & Finn, C. (2019).
> NoRML: No-Reward Meta Learning.

    @article{yang2019norml,
      title={NoRML: No-Reward Meta Learning},
      author={Yang, Yuxiang and Caluwaerts, Ken and Iscen, Atil and Tan, Jie and Finn, Chelsea},
      journal={arXiv preprint arXiv:1903.01063},
      year={2019}
    }

---

*Disclaimer: This is not an official Google product.*
