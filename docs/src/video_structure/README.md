# Unsupervised Learning of Object Structure and Dynamics from Videos

This reprository contains code for the model described in
https://arxiv.org/abs/1906.07889.

Also see the [project website](https://mjlm.github.io/video_structure/).

## Usage

### Installation

This code has been tested on Linux.

1. Clone the repository.

  ```
  git clone -b master --single-branch https://github.com/google-research/google-research.git
  ```

2. Make sure `google-research` is the current directory:

  ```
  cd google-research
  ```

3. Create and activate a new virtualenv:

  ```
  virtualenv -p python3 video_structure
  source video_structure/bin/activate
  ```
  
4. Install required packages:

  ```
  pip install -r video_structure/requirements.txt
  ```
  
### Run minimal example

Run on GPU device 0:

```
CUDA_VISIBLE_DEVICES=0 python -m video_structure.train
```

Run on CPU:

```
CUDA_VISIBLE_DEVICES= python -m video_structure.train
```

## Citation

```
@inproceedings{minderer2019unsupervised,
	title = {Unsupervised Learning of Object Structure and Dynamics from Videos},
	author = {Minderer, Matthias and Sun, Chen and Villegas, Ruben and Cole, Forrester and Murphy, Kevin and Lee, Honglak},
	booktitle = {arXiv: 1906.07889},
	year = {2019},
}
```
