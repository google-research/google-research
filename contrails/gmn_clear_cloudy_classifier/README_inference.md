# GMN Clear/Cloudy Classifier - Inference

This guide details how to run inference using the trained Clear/Cloudy
classifier model.

## Setup

1.  **Environment:** It is highly recommended to use a virtual environment.

```bash
virtualenv venv source venv/bin/activate
pip install --index-url https://pypi.org/simple -r requirements.txt
```

## Usage

The `inference.py` script runs classification on a directory of images.

### 1. Generating Test Data (Optional)

If `extracted_day_test/` is not provided, you can generate it from raw GMN data:

1.  **Identify Clear Days:** `bash python find_clearsky_days.py`

1.  **Download Videos:** `bash python download_good_weather_videos.py for f in
    download_batch_*.sh; do bash "$f"; done`

1.  **Extract Frames:** `bash python extract_day_frames.py` This extracts 1,000
    frames from the downloaded videos into `extracted_day_test/`.

### 2. Run Inference

To run inference on the default test dataset (`extracted_day_test`) with the
default threshold (0.80): `bash python inference.py` Results will be saved to
`inference_results.csv`.

### Custom Options

You can customize the input directory, output file, threshold, and model path
using command-line arguments:

```bash
python inference.py --image_dir <path_to_images> \
                    --output_file <results.csv> \
                    --threshold <0.0-1.0> \
                    --model_path <path_to_model.pth>
```

**Example: High-Precision Inference** To use a strict threshold of 0.95 for very
high precision: `bash python inference.py --threshold 0.95 --output_file
precise_results.csv`

**Note:** Ensure `weather_model_cleaned.pth` (or your specified model file) is
present: it can be downloaded from https://storage.googleapis.com/contrails_data/gmn_clear_cloudy_classifier/weather_model_cleaned.pth
