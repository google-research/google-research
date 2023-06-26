# VMSST

Published as a long paper at ACL 2023.

Contrastive learning has been successfully used for retrieval of semantically aligned sentences, but it often requires large batch sizes and carefully engineered heuristics to work well. In this paper, we instead propose a generative model for learning multilingual text embeddings which can be used to retrieve or score sentence pairs. Our model operates on parallel data in N languages and, through an approximation we introduce, efficiently encourages source separation in this multilingual setting, separating semantic information that is shared between translations from stylistic or language-specific variation. We show careful large-scale comparisons between contrastive and generation-based approaches for learning multilingual text embeddings, a comparison that has not been done to the best of our knowledge despite the popularity of these approaches. We evaluate this method on a suite of tasks including semantic similarity, bitext mining, and cross-lingual question retrieval––the last of which we introduce in this paper. Overall, our Variational Multilingual Source-Separation Transformer (VMSST) model outperforms both a strong contrastive and generative baseline on these tasks.

## Checkpoints

T5X (Jax): https://storage.googleapis.com/gresearch/vmsst/vmsst-large-2048-t5x.zip

PyTorch: https://storage.googleapis.com/gresearch/vmsst/vmsst-large-2048-pytorch.zip

## Usage

### Installation

1. Clone the repository.

  ```
  git clone -b master --single-branch https://github.com/google-research/google-research.git
  ```

2. Make sure `google-research` is the current directory:

  ```
  cd google-research/vmsst
  ```

3. Create and activate a new virtualenv:

  ```
  python -m venv vmsst
  source vmsst/bin/activate
  ```
  
4. This repository is tested on Python 3.10+. Install required packages:

  ```
  pip install -r requirements.txt
  ```

### Test

To test that the checkpoint and installation are working as intended, run:

    bash run.sh

The expected cosine similarity scores for the three sentences pairs are:

    0.20955035090446472,	0.204594686627388, and 0.2263302057981491.

### Inference

To embed a list of sentences:

    python score_sentence_pairs.py --sentence_pair_file test_data/test_sentence_pairs.tsv

To score a list of sentence pairs:

    python embed_sentences.py --sentence_file test_data/test_sentences.txt

## Citation

If you use our code or models your work please cite:

    @article{wieting2022beyond,
      title={Beyond Contrastive Learning: A Variational Generative Model for Multilingual Retrieval},
      author={Wieting, John and Clark, Jonathan H and Cohen, William W and Neubig, Graham and Berg-Kirkpatrick, Taylor},
      journal={arXiv preprint arXiv:2212.10726},
      year={2022}
    }
