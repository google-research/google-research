# `distracting_control_suite`: A visual distraction suite for `dm_control`

## Requirements and Installation

* Clone this repository
* `sh run.sh`
* Follow the instructions and install
[dm_control](https://github.com/deepmind/dm_control#requirements-and-installation). Make sure you setup your MuJoCo keys correctly.
* Download the [DAVIS 2017
  dataset](https://davischallenge.org/davis2017/code.html).

## Instructions

* You can run the `distracting_control_demo` to generate sample images of thet
  different tasks at different difficulties:

  ```
  python distracting_control_demo --davis_path=$HOME/DAVIS/JPEGImages/480p/
  --output_dir=/tmp/distrtacting_control_demo
  ```
* As seen from the demo to generate an instance of the environment you simply
  need to import the suitet and use `suite.load` while specifying the
  `dm_control` domain and task, then choosing a difficulty and providing the
  dataset_path.

* Note the environment follows the dm_control environment APIs.

## Paper

TODO: link to arxiv paper

If you use this code, please cite it as:

```
```

## Disclaimer

This is not an official Google product.
