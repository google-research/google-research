# FindIt: Generalized Localization with Natural Language Queries

This is a JAX/Flax implementation of our ECCV-2022 paper "FindIt: Generalized Localization with Natural Language Queries".

## Installation
We use the Python built-in virtual env to set up the environment. Run the following commands:

```
svn export https://github.com/google-research/google-research/trunk/findit

PATH_TO_VENV=/path/to/your/venv
python3 -m venv ${PATH_TO_VENV}
source ${PATH_TO_VENV}/bin/activate

pip install -r findit/requirements.txt
```

## Run the demo.

```
python -m findit.demo
```

## Citation
```
@inproceedings{kuo2022findit,
  title={FindIt: Generalized Localization with Natural Language Queries},
  author={Weicheng Kuo and Fred Bertsch and Wei Li and AJ Piergiovanni and Mohammad Saffar and Anelia Angelova},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```
