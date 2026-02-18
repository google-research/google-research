# GMN Clear/Cloudy Classifier - Training & Data Reproduction

This guide details how to reproduce the training dataset, train the model, and
analyze performance to select an operating point.

## Setup

1.  **Environment:** It is highly recommended to use a virtual environment.

```bash
virtualenv venv source venv/bin/activate
pip install --index-url https://pypi.org/simple -r requirements.txt
```

1.  **Dependencies:** Ensure `ffmpeg` is installed on your system for robust
    frame extraction.

1. **Gemini API key** Ensure your Gemini API key is inserted at the top of
    `example_night_to_day.py`.

1. **SSH/SFTP** Ensure your credentials are inserted at the top of
    `download_good_weather_videos.py` and `find_clearsky_days.py`

## Expert Tips for Reproduction

*   **Data Integrity:** The training script `train_cleaned.py` includes a filter
    to handle partial datasets gracefully. If any images from
    `extracted_night_labels.csv` are missing in `converted_day/`, they will be
    skipped.
*   **Efficient Extraction:** Use the updated `extract_frames.py` which batches
    `ffmpeg` calls per video file. This is significantly faster and more stable
    than sequential extraction.
*   **Resumable Conversion:** `example_night_to_day.py` supports resuming. If
    the script is interrupted, it will skip already converted images upon
    restart.
*   **Targeted Training:** For optimal dataset balance, ensure
    `reproduce_training_data.py` is configured to target the specific dates
    listed in the original image lists. This typically results in a ~57% clear
    sky ratio.

## Workflow

### 1. Identify & Download Raw Night Data

Use the `reproduce_training_data.py` script to identify matching nighttime video
and clearsky time intervals (GMN calls them 'flux time intervals' because
potential meteor flux would be visible because they're not blocked by clouds,
which we re-use as a 'clearsky' classifier label. The GMN
codebase calculates these time intervals based on the presence/absence of
stars in the imagery.)

```bash
python3 reproduce_training_data.py
bash download_night_data.sh
```

This will download:

* Nighttime timelapses (`.mp4`) and frametime maps (`.json`)
to `raw_data/`.

* "Clear sky time intervals" aka "Flux time intervals" (`.json`) to `flux_time_intervals/`.

### 2. Extract Nighttime Frames

Run `extract_frames.py` to extract specific frames from the downloaded videos.

```bash
python3 extract_frames.py
```

Output: PNG images in `extracted_night/`.

### 3. Generate Labels

Run `label_images.py` to generate ground truth labels based on star visibility.

```bash
python3 label_images.py
```

Output: `extracted_night_labels.csv`.

### 4. Night-to-Day Conversion

Use `example_night_to_day.py` to convert nighttime images to a daytime style
using the Gemini API.

```bash
python3 example_night_to_day.py
```

Input: `extracted_night/` Output: `converted_day/`

### 5. Train Model & Select Operating Point

Run cross-validation folds to analyze performance and determine the optimal
threshold.

```bash
python3 train_cleaned.py fold 0
python3 train_cleaned.py fold 1
python3 train_cleaned.py fold 2
python3 train_cleaned.py analyze
```

The `analyze` step will print a table of thresholds with Precision/Recall
values. Select the threshold that meets your requirements (e.g., 0.80 for high
precision). Update `inference.py` (or your inference command) with this
threshold if necessary.

### 6. Label Audit (Optional)

At this stage, you may wish to perform a manual review of your training data.
`mccloskey@google.com` manually reviewed labels that had been auto-generated
based on the `flux_time_intervals.json` and removed ambiguous or incorrectly
labeled examples to improve model performance; you may like to do so also by
editing `extracted_night_labels.csv` or removing images from `converted_day/`.

### 7. Final Model Training

Once the operating point is selected and any label audits are complete, train
the final model using all available data:

```bash
python3 train_cleaned.py final
```

This saves the model weights to `weather_model_cleaned.pth`.

## Utilities

*   **`filter_cloudy_intervals.py`**: A helper script that scans the downloaded `flux_time_intervals/` and moves files with zero detected intervals (indicating likely full cloud cover) into a `all_cloudy_time_intervals/` subdirectory for easier organization.

*   **`create_label_files.py`**: A script that reads `inference_results_precise_clearsky.csv` and creates empty sidecar files (e.g., `image.clear.txt`) in the image directory. This facilitates quick manual review by allowing you to simply rename the extension to correct labels.
