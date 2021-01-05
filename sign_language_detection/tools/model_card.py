# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Generate a model card file for the detector model."""

import shutil

import model_card_toolkit

# Initialize the Model Card Toolkit with a path to store generate assets
model_card_output_path = 'model_card'
shutil.rmtree(model_card_output_path, ignore_errors=True)
mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

# Initialize the model_card_toolkit.ModelCard, which can be freely populated
model_card = mct.scaffold_assets()
model_card.model_details.name = 'Sign Language Detector'
model_card.model_details.overview = ("""
    This is a lightweight Keras model which aims to classify whether or not a person is signing in a given video frame.
    The model is trained on the DGS Corpus, which includes a diverse group of German Sign Language deaf signers.
    This model does not stand alone, and requires human pose estimation and shoulder width normalization as pre-processing.
    """)
model_card.model_details.owners = [{
    'name': 'Amit Moryossef',
    'contact': 'amitmoryossef@google.com'
}, {
    'name': 'Ioannis Tsochantaridis',
    'contact': 'ioannis@google.com'
}]

model_card.considerations.use_cases = [
    """
    Performing real-time sign language detection for video-conferencing applications.
    """, """
    Performing offline sign language detection on videos, containing bi-directional context in order to extract
    sequences containing signing.
    """
]
model_card.considerations.limitations = [
    """
    While the models are trained to detect sign language, they are not specifically trained to distinguish between
    gesturing and signing, and therefore should not be used outside the setting of signing.
    """
]
model_card.considerations.ethical_considerations = [{
    'name':
        """
        Bias against minorities
        """,
    'mitigation_strategy':
        """
        As the model uses optical flow based on pose estimation, it can not reconstruct the shape or color of a person.
        Make sure that the pose estimation you use works well for both majority and minority groups.
        """
}]

# Write the model card data to a JSON file
mct.update_model_card_json(model_card)

# Return the model card document as an HTML page
html = mct.export_format()
