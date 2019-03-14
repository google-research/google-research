# EEG modelling

Provide several tools to work with EEG data

## Tools
* [EEG Viewer](eeg_viewer/README.md)

  Display EEG data in a web interface

* More to come...
  - Example ML models demonstrating seizure detection
  - EDF format tools


## Setup
#### Pre requisites:
*   java - to compile js code with the closure-compiler
*   python 2.7


#### Download and install deps:

Note: The `eeg_modelling/viewer.sh` script can only be called from
the `google_research/` folder or the `eeg_modelling/` folder.


1. Install python deps: `./viewer.sh install_py`
   - This will create a virtual environment under `eeg_modelling/pyenv`,
     and install the py dependencies there.
1. Download extra dependencies: `./viewer.sh download`
   - This will download JS and protobuffers deps
1. Compile protos: run `./viewer.sh compile_protos`


## Viewer
To run the viewer locally:

1.  Activate the environment by running `source env-activate`
    (or `source eeg_modelling/env-activate`, depending on the current folder)

1.  Run `./viewer.sh run -r`, and go to [http://localhost:5000](http://localhost:5000).
    For more options, see the [EEG Viewer readme](eeg_viewer/README.md)
