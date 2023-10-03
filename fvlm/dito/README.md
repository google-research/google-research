# Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection

This is a JAX/Flax implementation of DITO [Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection](https://arxiv.org/abs/2310.00161).

## Installation

Install the package from the root directory.

```
pip install -e .
```

## Download the DITO checkpoint and precomputed text embeddings.
Run the following commands from the root directory.

```
cd ./dito/checkpoints
./download.sh

cd ../embeddings
./download.sh
```

## Run the demo.

Run the following command from the root directory. This will run the DITO demo.

```
python ./dito/demo.py
```

You can set demo image and visualization options by the command line flags. Please refer to demo.py for more documentation on the flags.
We note that the demo model was pretrained on DataComp-1B and finetuned on the base categories of LVIS.

## Citation
```
@article{kim2023dito,
  title={Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection},
  author={Dahun Kim and Anelia Angelova and Weicheng Kuo},
  journal={arXiv preprint arXiv:2310.00161},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.