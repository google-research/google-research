# EEG Viewer

The Waveforms Viewer renders EEG recordings in a
[double banana](http://eegatlas-online.com/index.php/en/montages/bipolar/double-banana)
montage. It renders EEG lead data in the standard montage format, where a
standard consists of the element-wise subtraction of the input of two EEG leads.

## Setup

Pre requisites:

*   java - to compile js code with the closure-compiler
*   python 2.7


## Usage

Note: The `eeg_modelling/viewer.sh` script can only be called from
the `google_research/` folder or the `eeg_modelling/` folder.

1. Install python deps:  `./viewer.sh install_py`
   - This will create a virtual environment under `eeg_modelling/pyenv`,
     and install the py dependencies there.
1. Download extra dependencies:  `./viewer.sh download`
   - This will download JS and protobuffers deps
1. Compile protos: run `./viewer.sh compile_protos`
1.  Build css and js: `./viewer.sh build`
1.  Activate the python environment `source pyenv/bin/activate`
1.  Run the server: `./viewer.sh run`
1.  Go to [http://localhost:5000](http://localhost:5000)


## File format

Note that, for now, the only accepted file format is a binary file containing a
[tensorflow.Example protobuf](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/Example).
The EDF-tool provided in this repository allows transforming EDF files into the
needed TF format. See instructions [here](../edf/README.md)



## Develop

More commands:

*   Compile js, css and run: `./viewer.sh run -r`
*   Run tests: `./viewer.sh test`
*   Compile js: `./viewer.sh compile_js`
*   Compile css: `./viewer.sh compile_css`
*   Help: `./viewer.sh --help`
