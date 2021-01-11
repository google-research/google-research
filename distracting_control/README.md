# The Distracting Control Suite

`distracting_control` extends `dm_control` with static or dynamic visual
distractions in the form of changing colors, backgrounds, and camera poses. 
Details and experimental results can be found in our
[paper](https://arxiv.org/pdf/2101.02722.pdf).

## Requirements and Installation

* Clone this repository
* `sh run.sh`
* Follow the instructions and install
[dm_control](https://github.com/deepmind/dm_control#requirements-and-installation). Make sure you setup your MuJoCo keys correctly.
* Download the [DAVIS 2017
  dataset](https://davischallenge.org/davis2017/code.html).

## Instructions

* You can run the `distracting_control_demo` to generate sample images of the
  different tasks at different difficulties:

  ```
  python distracting_control_demo --davis_path=$HOME/DAVIS/JPEGImages/480p/
  --output_dir=/tmp/distrtacting_control_demo
  ```
* As seen from the demo to generate an instance of the environment you simply
  need to import the suite and use `suite.load` while specifying the
  `dm_control` domain and task, then choosing a difficulty and providing the
  dataset_path.

* Note the environment follows the dm_control environment APIs.

## Paper

If you use this code, please cite the accompanying [paper](https://arxiv.org/pdf/2101.02722.pdf) as:

```
@article{stone2021distracting,
      title={The Distracting Control Suite -- A Challenging Benchmark for Reinforcement Learning from Pixels}, 
      author={Austin Stone and Oscar Ramirez and Kurt Konolige and Rico Jonschkowski},
      year={2021},
      journal={arXiv preprint arXiv:2101.02722},
}
```

## Disclaimer

This is not an official Google product.
