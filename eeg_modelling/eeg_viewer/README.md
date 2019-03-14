# EEG Viewer

The Waveforms Viewer renders EEG recordings in a
[double banana](http://eegatlas-online.com/index.php/en/montages/bipolar/double-banana)
montage. It renders EEG lead data in the standard montage format, where a
standard consists of the element-wise subtraction of the input of two EEG leads.

## Usage

Note: The `eeg_modelling/viewer.sh` script can only be called from
the `google_research/` folder or the `eeg_modelling/` folder.

1.  Follow the instructions in [EEG-modelling](../README.md) to install dependencies
1.  Build css and js: `./viewer.sh build`
1.  Activate the python environment `source env-activate`
1.  Run the server: `./viewer.sh run`
1.  Go to `localhost:5000`

## Developing

More commands:

*   Compile js, css and run: `./viewer.sh run -r`
*   Run tests: `./viewer.sh test`
*   Compile js: `./viewer.sh compile_js`
*   Compile css: `./viewer.sh compile_css`
*   Help: `./viewer.sh --help`
