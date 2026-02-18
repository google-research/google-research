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

import glob
import os
from google import genai
from google.genai import types
from PIL import Image


# Initialize the client
client = genai.Client(api_key="TODO_ADD_YOUR_API_KEY")

# Define directories
input_dir = "extracted_night"
output_dir = "converted_day"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all PNG images in the input directory
night_image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

print(f"Found {len(night_image_paths)} images to process.")

for image_path in night_image_paths:
  filename = os.path.basename(image_path)
  output_path = os.path.join(output_dir, filename)

  if os.path.exists(output_path):
    # print(f"Skipping {filename} (already exists)")
    continue

  print(f"Processing {filename}...")

  try:
    image_input = Image.open(image_path)

    text_input = """Using the provided image, make it instead look like it was taken at 10am in the daytime. Keep everything else in the image unchanged: any foreground should remain identical (but become a daytime lit) and any background should remain identical (but become daytime lit). In particular IF any clouds are present, THEY MUST REMAIN UNCHANGED. If no clouds are present, the background sky should remain identical (but become daytime lit, without any stars)."""

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[text_input, image_input],
    )

    # Helper to extract image from response
    def get_image_from_response(resp):
      if resp.parts:
        for part in resp.parts:
          if part.inline_data and part.inline_data.mime_type.startswith(
              "image/"
          ):
            return part.inline_data.as_image()
      return None

    generated_image = get_image_from_response(response)

    if generated_image:
      # Save intermediate result immediately to ensure we have a valid file for the next steps
      output_path = os.path.join(output_dir, filename)
      generated_image.save(output_path)
      print(f"Saved intermediate to {output_path}")

      # Re-open for verification
      # We use a with block or just open it. Image.open is lazy but load() might be needed if we want to read it.
      # However, the client handles paths/PIL images. Let's pass the PIL image loaded from disk.
      try:
        image_for_check = Image.open(output_path)

        # Verification step
        print(f"Verifying {filename} for stars...")
        check_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                (
                    "Reply HAS_STARS if you can see a star or stars in the sky"
                    " in this image; Reply NO_STARS if you cannot see any stars"
                    " in the sky of this image."
                ),
                image_for_check,
            ],
        )
        check_text = check_response.text if check_response.text else ""
        print(f"Verification result: {check_text.strip()}")

        if "HAS_STARS" in check_text:
          print(f"Stars detected in {filename}. Applying correction...")
          correction_prompt = (
              "That is great but remove those stars, it is daytime in this"
              " image."
          )
          response_correction = client.models.generate_content(
              model="gemini-2.5-flash-image",
              contents=[correction_prompt, image_for_check],
          )
          corrected_image = get_image_from_response(response_correction)
          if corrected_image:
            corrected_image.save(output_path)
            print(f"Correction applied and saved to {output_path}")
          else:
            print("Correction failed to produce an image. Keeping original.")

      except Exception as e:
        print(f"Verification failed: {e}")
        # We already saved the image, so no data loss.

    else:
      print(f"No image found in response for {filename}")
      # simple error printing, avoiding assuming prompt_feedback structure if it varies
      print(f"Response: {response}")

  except Exception as e:
    print(f"Error processing {filename}: {e}")

print("Processing complete.")
