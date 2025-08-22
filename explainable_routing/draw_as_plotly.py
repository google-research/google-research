# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

import re
import sys
import plotly.graph_objects as go


def create_seattle_route_map(file_path):
  """Parses a file with polyline data and creates an interactive Plotly map

  with a legend on the left.

  Args:
      file_path (str): The path to the input text file.
  """
  fig = go.Figure()

  try:
    with open(file_path, "r") as f:
      for line in f:
        if ":" not in line or "|" not in line:
          continue

        line_cleaned = re.sub(r"\\s*", "", line).strip()
        if not line_cleaned:
          continue

        metadata_part, coords_part = line_cleaned.split(":", 1)
        metadata = metadata_part.split("|")

        line_name = metadata[0]
        color = metadata[2]

        stroke_weight = 2
        for item in metadata:
          if "stroke_weight" in item:
            stroke_weight = int(item.split("=")[1])
            break

        lats, lons = [], []
        coord_pairs = coords_part.strip().split(",")
        for pair in coord_pairs:
          try:
            lat, lon = pair.split(":")
            lats.append(float(lat))
            lons.append(float(lon))
          except ValueError:
            continue

        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(color=color, width=stroke_weight),
                marker=dict(symbol="circle", size=stroke_weight),
                name=line_name,
                legendgroup=line_name,
                showlegend=True,
                hoverinfo="text",
                text=line_name,
            )
        )

  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    return
  except Exception as e:
    print(f"An error occurred: {e}")
    return

  # --- Configure Map Layout ---
  fig.update_layout(
      title=f'Routes from {file_path.split("/")[-1]}',
      mapbox_style="open-street-map",
      showlegend=True,
      mapbox=dict(center=dict(lat=47.62, lon=-122.22), zoom=10.5),
      margin={"r": 0, "t": 40, "l": 0, "b": 0},
      # Position the legend on the top-left of the map
      legend=dict(
          x=0.02,
          y=0.98,
          xanchor="left",
          yanchor="top",
          bgcolor=(  # Add a semi-transparent background
              "rgba(255, 255, 255, 0.75)"
          ),
      ),
  )

  fig.show()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 script.py <path_to_file>")
    sys.exit(1)

  input_file_path = sys.argv[1]
  create_seattle_route_map(input_file_path)
