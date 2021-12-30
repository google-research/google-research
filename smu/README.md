# SMU

This code is for a still in-progress paper and dataset. More details will be
added when the dataset is complete and published.

## Installation instructions for Linux (Ubuntu/Debian):
* Download the `smu` subdirectory from the Google Research repository on GitHub:
`svn export https://github.com/google-research/google-research/trunk/smu`

Note: The following instructions assume that your current working directory
is the one that contains the `smu` subdirectory.

Make sure you're using the Bash shell: `/bin/bash`

All dependencies will be installed in a virtual environment.

* Install virtualenv: `sudo python -m pip install virtualenv`
* Install protoc: `sudo apt install -y protobuf-compiler`
* Install BigQuery: `sudo python -m pip install google-cloud-bigquery`
* Run the script that sets up the virtual environment and runs some tests.
`smu/run.sh`

Note: To run any of the SMU scripts, make sure the virtual environment is active
by first calling (from the directory that contains `smu/`):
`bin/activate`

* Finally, run a script, e.g. query_sqlite.py, like so:
`python -m smu.query_sqlite --input_sqlite=/path/to/file.sqlite [other flags]`

