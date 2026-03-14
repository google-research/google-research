# Agile Deliberation

This is a companion codebase and dataset associated with the paper
["Agile Deliberation: Concept Deliberation for Subjective Visual Classification"](https://arxiv.org/abs/2512.10821).

## Introduction
In our Agile Deliberation paper, we introduce a human-in-the-loop framework for classifying subjective and evolving visual concepts. By guiding users through structured concept scoping and iteratively surfacing borderline examples for reflection, our system helps users clarify their ambiguous mental models. 

Here we release a Colab with step-by-step instructions guiding users through
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

Warning: this downloads ~XX GB of data! If you want to test it on a smaller set,
you can change how many shard of LAION data are downloaded in `setup.sh`.

## Running the Demo
`Demo.ipynb` contains an end-to-end implementation of our prototype. You can
run it using [Jupyter notebook](https://docs.jupyter.org/en/latest/running.html) or,
for a nice interface, [Google Colab](https://colab.google/).

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

# Setup the index (Warning: downloads ~XX GB of data!)
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