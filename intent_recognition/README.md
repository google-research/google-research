# Intent recognition

This directory contains a tool to create sensor based ML models that can be used
to infer user activities.

## Usage

Using the tool consists of multiple stages: Getting sensor data (as
AnnotatedRecordingCollections), processing the data, converting the data into
SequenceExamples, model training, and model inference & metrics.

### Getting sensor data

Sensor data must be collected and formatted as AnnotatedRecordingCollections.
Specifically the recording_collection field needs to be filled out. As an
example, a binary that converts the ADL dataset into
AnnotatedRecordingCollections is provided. It can be built using:

```
bazel run --cxxopt='-std=c++17' --experimental_repo_remote_exec --define
MEDIAPIPE_DISABLE_GPU=1 intent_recognition:convert_adl_dataset_to_annotated_recording_collection -- <see file for flags>
```

### Processing

Processing is done using mediapipe and can be run using:

```
bazel run --cxxopt='-std=c++17' --experimental_repo_remote_exec --define MEDIAPIPE_DISABLE_GPU=1 intent_recognition/processing:process_annotated_recording_collection_main -- <see file for flags>
```

Sample config files can be found in `intent_recognition/sample_configs`

### Conversion

Conversion can be run using:

```
bazel run --cxxopt='-std=c++17' --experimental_repo_remote_exec --define MEDIAPIPE_DISABLE_GPU=1 intent_recognition/conversion:convert_annotated_recording_collection_to_sequence_example_main -- <see file for flags>
```

### Training

Training is done using Tensorflow. A colab notebook to perform training can be
found at `intent_recognition/training/intent_recognition_training.ipynb'
