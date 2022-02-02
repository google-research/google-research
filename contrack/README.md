# Contrack
[![TensorFlow 2.3](https://img.shields.io/badge/TensorFlow-2.3-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.3.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

In human-human conversations, Context Tracking deals with identifying important
entities and keeping track of their properties and relationships. This is a challenging
problem involving several subtasks such as entity recognition, attribute classification,
coreference resolution and resolving plural mentions. The Contrack tool approaches
this problem as an end-to-end modeling task where the conversational context is
represented by an entity repository containing the entities mentioned so far, their properties and relationships between them. The repository is updated incrementally turn-by-turn,
thus making it computationally efficient and capable of handling long conversations.

Contributions to the codebase are welcome and we would love to hear back from
you if you find this codebase useful. Finally if you use Contrack for a research publication, please consider citing:

* <a href='https://arxiv.org/abs/2201.12409' target='_blank'>Towards a Unified Approach to Entity-Centric Context Tracking in Conversations</a>,
<em>Ulrich Rückert, Srinivas Sunkara, Abhinav Rastogi, Sushant Prakash, Pranav Khaitan</em>

# Installation

The following instructions are for installing on Ubuntu 18.04.

1. Make sure you have python3 and bazel installed. Follow the instructions [here](https://docs.bazel.build/versions/4.0.0/install.html) to install bazel.


1. Download the contrack subdirectory:

    ```bash
    svn export https://github.com/google-research/google-research/trunk/contrack
    # Or
    git clone https://github.com/google-research/google-research.git
    ```

1.  Create and enter a virtual environment (optional but preferred):

    ```bash
    virtualenv -p python3 contrack_env
    source ./contrack_env/bin/activate
    ```

1.  Install the dependencies:

    ```bash
    cd contrack
    python3 configure.py
    ```

    If you want to use an existing installation of tensorflow and gensim, run the
    configuration tool with the no-deps flag to skip dependency installation:

    ```bash
    python3 configure.py --no-deps
    ```

1. Compile the source code:

    ```bash
    bazel build //:preprocess //:train //:predict
    ```

# Usage

Here is an example on how to preprpocess a small example data file and train
a model on it.

1. Download word2vec data used during preprocessing:

    ```bash
    mkdir /tmp/contrack_data
    export DATA_DIR=/tmp/contrack_data
    wget -c -P $DATA_DIR "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    gunzip $DATA_DIR/GoogleNews-vectors-negative300.bin.gz
    ```

2. Run the preprocess tool to convert text conversations to TFRecord format.

    ```bash
    mkdir /tmp/contrack_example
    export BASE_DIR=/tmp/contrack_example
    ./bazel-bin/preprocess --input_file=data/example_conversations.txt \
      --output_dir=$BASE_DIR \
      --tokenizer_handle="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3" \
      --bert_handle="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3" \
      --wordvec_path=$DATA_DIR/GoogleNews-vectors-negative300.bin \
      --logtostderr
      ```

3. Train a model on the TFRecord data. (A GPU is not necessary, but recommended
    for faster training.)

    ```bash
    cp data/example_config.json $BASE_DIR/config.json
    ./bazel-bin/train --train_data_glob $BASE_DIR/example_conversations.tfrecord \
      --config_path $BASE_DIR/config.json --model_path $BASE_DIR/model \
      --mode=two_steps --logtostderr
    ```

4. Apply model on some dataset. The accuracy measures on the dataset will be
    output to the logfile.

    ```bash
    ./bazel-bin/predict --input_data_glob $BASE_DIR/example_conversations.tfrecord \
      --model_path $BASE_DIR/model --logtostderr
    ```
