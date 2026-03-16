# Agile Deliberation

This is a companion codebase and dataset associated with the paper
["Agile Deliberation: Concept Deliberation for Subjective Visual Classification"](https://arxiv.org/abs/2512.10821).

## Introduction
In our Agile Deliberation paper, we introduce a human-in-the-loop framework for classifying subjective and evolving visual concepts. By guiding users through structured concept scoping and iteratively surfacing borderline examples for reflection, our system helps users clarify their ambiguous mental models. 

Here we release a Jupyter notebook with step-by-step instructions guiding users through
the Agile Deliberation process.

## Setting up

Navigate to the workspace folder:

```
cd /path/to/agile_deliberation
```

Optionally create a virtual environment, and activate it:

```
python3 -m venv venv
source venv/bin/activate
```

### Requirements
Install the requirements in `requirements.txt`. You can do this by running:

```
pip install -r requirements.txt
```

Additionally, depending on if you want to GPU-accelerate your 
nearest-neighbors index:
```pip install faiss-gpu``` OR ```pip install faiss-cpu```


### Set up the nearest neighbor index 
Our method involves the use of a nearest neighbor index to retrive relevant
image examples from a dataset of image data.

You can create an index over your image dataset of choice. Here we provide
instructions on how to do this using the the Laion400M dataset, which is also
used in our `Demo.ipynb` notebook.

Run `setup.sh`. It will create the following file structure, and then build a 
FAISS index over the training set:

```
laion400m/
    |- metadata
        |- metadata_0.parquet
        |- ...
        |- metadata_99.parquet
    |- npy
        |- img_emb_0.npy
        |- ...
        |- img_emb_99.npy
    |- image.index
```

**Storage warning:** each shard is roughly 130 MB (embeddings + metadata), so
100 shards total ~13 GB.  By default `setup.sh` downloads only the first **2
shards** (~260 MB), which is enough to verify the pipeline end-to-end.
Re-run with a larger count (e.g. `bash setup.sh 20`) once you are ready for a
full experiment — more shards mean a richer retrieval pool and better results.

## Running the Demo

> **A note on performance.** The prototype interleaves image retrieval over a
> large vector index, image downloads from external URLs, and LLM calls — so
> individual steps can take anywhere from a few seconds to a couple of minutes.
> If you are using a single API key the LLM may also be rate-limited.  Please
> be patient when waiting for responses to appear in the notebook.  If you have
> access to multiple model endpoints you can register them with `ModelClient`
> to spread the load.
>
> **Caching.** Several stages of the pipeline support caching to disk so that
> you do not have to redo expensive work across sessions.  See the inline
> comments in `Demo.ipynb` for the exact cells and options — in particular, you
> can pre-collect the candidate image pool for the reflection stage once and
> reuse it in every subsequent run.

`Demo.ipynb` contains an end-to-end implementation of our prototype.  Run it
using [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html)
(recommended) or [Google Colab](https://colab.google/).

### How to run using Jupyter
To run it in Jupyter, first install Jupyter inside the environment and start it:

```
pip install notebook
jupyter notebook
```

This will open a browser window. Open `Demo.ipynb` and it will automatically attach to a Python 3 kernel running inside this environment.

Now you can run the notebook cell by cell.

### How to run using Colab
To run it in Google Colab:

1. Open [Google Colab](https://colab.google/) and upload `Demo.ipynb` (or open it directly from your GitHub repository).

**Note**: Since Colab runs in a hosted cloud environment, you must run the following setup steps *inside* the notebook session each time you connect, even if you have already set up the project locally.

2. Add a code cell at the very top of the notebook to download the library files and install dependencies:

```python
!git clone <YOUR_REPOSITORY_URL>
%cd <REPOSITORY_FOLDER>/third_party/google_research/google_research/agile_deliberation

# Install requirements
!pip install -r requirements.txt
!pip install faiss-cpu  # or faiss-gpu

# Setup the index (downloads 2 shards by default; see setup.sh for options)
!bash setup.sh
```

Now you can continue running the notebook cell by cell.



## Citation

If you found this codebase useful, please consider citing our paper:

```
@inproceedings{agile_modeling,
  title={Agile Deliberation: Concept Deliberation for Subjective Visual Classification},
  author={Wang, Leijie and Stretcu, Otilia and Qiao, Wei and Denby, Thomas and Viswanathan, Krishnamurthy and Luo, Enming and Lu, Chun-Ta and Dogra, Tushar and Krishna, Ranjay and Fuxman, Ariel},
  booktitle={Proceedings of the the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Denver, Colorado, USA},
  year={2026}
}
```

## Disclaimer

This is not an officially supported Google product.