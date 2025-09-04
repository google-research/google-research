# EgoSocial

**Note:** This codebase is currently under development. Some parts may be incomplete or require modification to function as expected.

## Overview

Current language models often lack the social awareness needed to function effectively as AI assistants in augmented and virtual reality (AR/VR) environments. Specifically, they struggle to determine the right moment to intervene in a social situation without disrupting natural human interaction.

To address this challenge, we developed EgoSocial, a large-scale egocentric dataset containing 13,500 video-question pairs focused on social interactions derived from Ego4D. This dataset serves as a benchmark for evaluating how well AI can perceive social dynamics and identify appropriate intervention timing. We also present a new method called EgoSoD (EgoSocial Detection), which uses a social thinking graph to integrate multimodal cues like audio and visuals. This approach models participants and their interactions to determine precisely when an AI assistant should intervene. 

## Dataset

The EgoSocial dataset is constructed by extracting video frames and audio clips from the [Ego4D](https://ego4d-data.org/) dataset based on the provided annotations.

### Data Generation

To generate the dataset, you will need to first download the Ego4D video clips. Then, use the provided script to process the videos.

1.  **Prerequisites**:
    *   `ffmpeg`: Make sure `ffmpeg` is installed and accessible in your system's PATH. It is used to extract frames and audio from videos.
    *   Downloaded Ego4D clips.

2.  **Configure the generation script**:
    Open `scripts/generate_data.sh` and set the following variables to your local paths:
    *   `ANNOTATION_FILE`: Path to the annotation file (e.g., `annotations/egosocial_annotations.json`).
    *   `EGO4D_CLIPS_DIR`: Path to the directory where you have downloaded the Ego4D video clips.
    *   `OUTPUT_DIR`: Path to the directory where the generated dataset will be saved.

3.  **Run the script**:
    Execute the script from the `scripts` directory:
    ```bash
    cd scripts
    bash generate_data.sh
    ```

### Data Structure

The generated dataset will have the following structure in your specified `OUTPUT_DIR`:

```
egosocial_eval
├── annotation.json        # Annotation file for the dataset
├── audio                  # Folder containing audio files
│   ├── video1.wav
│   ├── video2.wav
│   └── ...
├── frames                 # Folder containing video frames
│   ├── video1
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── video2
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └── video3
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
```

## Installation

1.  Install required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
2.  Install `ffmpeg`. You can usually install it via your system's package manager (e.g., `sudo apt-get install ffmpeg` on Debian/Ubuntu).

3.  For Gemini models, you will need a Gemini API key.

4.  For the Phi-4 model, follow the official website to set up the environment: [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).

5. Extract the annotation files from root folder using command `tar -xvzf  annotations/annotations.tar.gz`

## Project Structure

```
EgoSocial/
├── annotations/              # Annotation files
├── Gemini/                   # Scripts and code for Gemini models
│   ├── configs               # Configuration files for Gemini experiments
│   ├── src                   # Source code for Gemini models
│   └── runs                  # Scripts to run Gemini experiments
├── Phi4/                     # Scripts and code for Phi-4 models
│   ├── phi4_video_audio_SI_baseline.py
│   ├── phi4_video_audio_SI_baseline_audio2text_conv_all.py
│   ├── phi4_video_audio_SI_baseline_audio2text_conv_all_graph.py
│   └── run.sh
├── scripts/                  # Data preprocessing scripts
│   ├── data.py               # Script to extract frames and audio from videos
│   └── generate_data.sh      # Shell script to run data.py
├── config.json               # Basic configuration for data paths
├── evaluation_res.py         # Evaluation script for model predictions
├── requirements.txt          # Required packages
└── README.md                 # Project documentation
```

## Configuration

The application can be configured using the `configs/xxx.json` file. The following parameters can be set:

- `model`: model name
- `question`: prompt question.
- `frame_num`: The number of frames from the video.
- `dataset`: annotation file.
- `audio_folder`: audio path folder.
- `audio_text_path`: if you preprocessed audio to text.
- `save_json_path`: log path.

## Usage

1. To run Gemini, use the following command:

```
cd Gemini
python ../src/main.py --prefix gemini2.5 --signal all_rea_H --api Gemini_API_key --config ../configs/xxx.json
```

Replace the parameters with your desired values.


2. To run Phi-4, use the following command:

```
cd Phi4
# graph think with raw question
CUDA_VISIBLE_DEVICES=0 python phi4_video_audio_SI_baseline_audio2text_conv_all.py --signal all_rea_H --frame_num 10

# graph think with prediected cue answers
CUDA_VISIBLE_DEVICES=0 python phi4_video_audio_SI_baseline_audio2text_conv_all_graph.py --signal graph_100_H --frame_num 10


# baseline
CUDA_VISIBLE_DEVICES=0 python phi4_video_audio_SI_baseline.py
```

Replace the parameters with your desired values.

3. If you have the prediction log file, you can use `evaluation_res.py` to evaluate each cue's detection result:

```
python evaluation_res.py
```

Check the corresponding code block in the file for different cues.