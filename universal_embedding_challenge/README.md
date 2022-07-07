# Universal Embedding Challenge baseline model implementation.

This folder contains the baseline model implementation for the [Kaggle universal image embedding challenge](https://www.kaggle.com/competitions/google-universal-image-embedding/) based on

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf).
- [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf).

Following the above ideas, we also add a 64 projection layer on top of the Vision Transformer base model as the final embedding. Please find more details in [image_classification.py](https://github.com/google-research/google-research/blob/master/universal_embedding_challenge/image_classification.py).


To use the code, please firstly install the prerequisites

```
pip install -r universal_embedding_challenge/requirements.txt

git clone https://github.com/tensorflow/models.git /tmp/models
export PYTHONPATH=$PYTHONPATH:/tmp/models
pip install --user -r /tmp/models/official/requirements.txt
```

The trainer for the model is implemented in [train.py](https://github.com/google-research/google-research/blob/master/universal_embedding_challenge/train.py), and the following example launches the training on CPU

```
python -m universal_embedding_challenge.train \
  --experiment=vit_with_bottleneck_imagenet_pretrain \
  --mode=train_and_eval \
  --model_dir=/tmp/imagenet1k_test
```

The trained model checkpoints could be further converted to savedModel format using [export_saved_model.py](https://github.com/google-research/google-research/blob/master/universal_embedding_challenge/export_saved_model.py) for Kaggle submission.

The code to compute metrics for Universal Embedding Challenge is implemented in [metrics.py](https://github.com/google-research/google-research/blob/master/universal_embedding_challenge/metrics.py) and the code to read the solution file is implemented in [read_retrieval_solution.py](https://github.com/google-research/google-research/blob/master/universal_embedding_challenge/read_retrieval_solution.py).