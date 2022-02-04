# SMU

This code is for a still in-progress paper and dataset. More details will be
added when the dataset is complete and published.

## Get the data

TODO: once the dataset has been released, include links and md5sums

## Installation instructions (user)

If you just want to use the database to access the records in SMU, there is a simpler process with fewer dependencies.

TODO: update the code path for final release

1) Install the [protobuf compiler](https://github.com/protocolbuffers/protobuf). Note that some linux installations have prebuilt packages. On Ubuntu, this is simply
    ```
    sudo apt install -y protobuf-compiler
    ````

1) Get the code
   * Option 1: If you are a familiar with / current user of GitHub, you can just clone the repo:
       ```
       git clone git@github.com:google-research/google-research.git --depth=1
       ```
   * Option 2: If you just want a simple way to grab the code, use subversion:
       ```
       svn export https://github.com/pfrstg/google-research/branches/feb_pr/smu
       ```

1) Install dependencies and setup environment
    ````
    cd smu
    ./install_user.sh
    ````
    The last line should say `SUCCESS`. If not, there was some issue installing dependencies.

1) The `install_user.sh` created a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html). For each terminal you want to run SMU code in, you have to activate the virtual environment:
    ```
    source venv_smu_user/bin/activate
    ```

1) Test the installation
    ```
    cd smu
    source venv_smu_user/bin/activate
    cd <path where you downloaded the SMU sqlite files>
    python -m smu.query_sqlite --input_sqlite 20220128_standard.sqlite --smiles NN
    ```
    should give you a long text output with many values for N2.


## Installation instructions (developer)

If you want to modify the code that produced SMU (as opposed to just reading the generated data), follow these instructions

TODO: write these instructions
