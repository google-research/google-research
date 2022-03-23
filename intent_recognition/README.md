# Intent recognition

This directory contains a tool to create sensor based ML models that can be used
to infer user activities.

## How to build
### ADL Recognition dataset to AnnotatedRecordingCollections
```
bazel build --cxxopt='-std=c++17' --experimental_repo_remote_exec --define MEDIAPIPE_DISABLE_GPU=1 intent_recognition:convert_adl_dataset_to_annotated_recording_collection
```

### Processing
```
bazel build --cxxopt='-std=c++17' --experimental_repo_remote_exec --define MEDIAPIPE_DISABLE_GPU=1 intent_recognition/processing:process_annotated_recording_collection_main
```
