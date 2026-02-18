# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


# --- Defaults ---
DEFAULT_MODEL_PATH = "weather_model_cleaned.pth"
DEFAULT_IMAGE_DIR = "extracted_day_test"
DEFAULT_OUTPUT_FILE = "inference_results.csv"
DEFAULT_THRESHOLD = 0.80

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")

# --- 1. Model Setup (ResNet18) ---
def get_model():
  weights = models.ResNet18_Weights.DEFAULT
  model = models.resnet18(weights=weights)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 2)
  return model


# --- 2. Inference ---
def run_inference(model_path, image_dir, output_file, threshold):
  print(f"Loading model from {model_path}...")
  model = get_model()
  try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
  except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    return

  model.to(DEVICE)
  model.eval()

  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  print(f"Processing images in {image_dir} with threshold {threshold}...")

  if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} does not exist.")
    return

  image_files = sorted([
      f
      for f in os.listdir(image_dir)
      if f.lower().endswith((".png", ".jpg", ".jpeg"))
  ])

  if not image_files:
    print("No images found.")
    return

  results = []
  with torch.no_grad():
    for i, filename in enumerate(image_files):
      filepath = os.path.join(image_dir, filename)
      try:
        img = Image.open(filepath).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        clear_prob = probs[0][1].item()

        if clear_prob >= threshold:
          pred_label = "clear"
          confidence = clear_prob
        else:
          pred_label = "cloudy"
          confidence = probs[0][0].item()  # Cloudy confidence

        results.append((filename, pred_label, f"{confidence:.4f}"))

        if (i + 1) % 100 == 0:
          print(f"Processed {i+1}/{len(image_files)}")

      except Exception as e:
        print(f"Error processing {filename}: {e}")
        results.append((filename, "ERROR", str(e)))

  print(f"Saving results to {output_file}...")
  with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "prediction", "confidence"])
    writer.writerows(results)

  print("Inference complete.")


def main():
  parser = argparse.ArgumentParser(
      description="Run inference with the Clear/Cloudy classifier."
  )
  parser.add_argument(
      "--image_dir",
      type=str,
      default=DEFAULT_IMAGE_DIR,
      help=f"Directory containing images (default: {DEFAULT_IMAGE_DIR})",
  )
  parser.add_argument(
      "--output_file",
      type=str,
      default=DEFAULT_OUTPUT_FILE,
      help=f"Output CSV file (default: {DEFAULT_OUTPUT_FILE})",
  )
  parser.add_argument(
      "--threshold",
      type=float,
      default=DEFAULT_THRESHOLD,
      help=(
          "Probability threshold for 'clear' classification (default:"
          f" {DEFAULT_THRESHOLD})"
      ),
  )
  parser.add_argument(
      "--model_path",
      type=str,
      default=DEFAULT_MODEL_PATH,
      help=f"Path to trained model weights (default: {DEFAULT_MODEL_PATH})",
  )

  args = parser.parse_args()

  run_inference(
      args.model_path, args.image_dir, args.output_file, args.threshold
  )


if __name__ == "__main__":
  main()
