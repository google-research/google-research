# Cochlear Implant Hackathon Code

This directory contains Python Colab notebooks for taking input audio (wav files of music, speech in noise, etc.), applying speech enhancement (primarily when music is not detected), and generating electrodograms for cochlear implants.

These were developed as a submission for [https://cihackathon.com/](https://cihackathon.com/).

To listen to the audio corresponding to the electrodograms, you will need to a vocoder simulation (e.g., the one provided by cihackathon organizers [here](https://github.com/jabeim/AB-Generic-Python-Toolbox)).

## Input data

To run the notebooks, you need:

1. A set of audio waveform files to process.

Sample audio files for cihackathon are available for at [https://cihackathon.com/](https://cihackathon.com/).


## Running the notebooks

At the moment these notebooks are for reference only and may not be runnable.

### yamnet_speech_enhancement_mixing.ipynb
Because speech enhanced audio often removes portions of music (i.e. music is often treated as noise-to-be-removed), this notebook uses [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) to determine the predicted music content and causally mix the original and speech enhanced audio. Also included is a baseline mixing strategy which mixes a fixed fraction of noise (music) with speech enhanced audio without using YAMNet.

### audio_to_electrodogram.ipynb
This notebook takes audio (any audio; it can be original audio, or speech enhanced audio, or speech enhanced audio mixed with original audio) and generates electrodograms in the format specified for the cihackathon.


This project is not an official Google product.

