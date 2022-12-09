# Self-tuning follower


This repository provides some new task specifications for the self-tuning-follower project. Currently it includes  

  - `put-in-bowl`: This task test the ability to pick a block of a known or an unknown color and put in a blue bowl.
  - `put-block-in-bowl`: The task instructs the robot to pick a block of any color, and put in a bowl of any color.
  - `put-in-zone`: The task instructs the robot to pick a block of any color, and put it in zone.
  

## Setup
1. Create a conda environment using the `environment.yml` file.

  ```bash
  conda env create --prefix self-tuning-follower --file environment.yml
  ```
**Note**: You might need versions of `torch==1.7.1` and `torchvision==0.8.2` that are compatible with your CUDA and hardware. 

2. Clone and Install Cliport repo here by running `setup.sh` script

  ```bash
  bash setup.sh
  ``` 

## Sample Dataset Creation

To create a small data of size 10 for the block in blue bowl task, run the following command

```bash
python demos.py n=10 task=put-in-bowl-seen mode=val version=v0
```

## Contact: 
Harsh Agrawal (h.agrawal092@gmail.com) or Natasha Jaques (natashajaques@google.com) for queries about the project.

## Acknowledgements

This work builds on top of the CLIPort code base. See: [cliport.github.io](https://cliport.github.io)