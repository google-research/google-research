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

import argparse
import os
import pickle

import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def load_artifacts(model_filepath, tokenizer_filepath, params_filepath):
    """Loads the trained model, tokenizer, and model parameters.

    Args:
      model_filepath: Path to the saved model.
      tokenizer_filepath: Path to the saved tokenizer.
      params_filepath: Path to the saved model parameters.

    Returns:
      model: Loaded model.
      tokenizer: Loaded tokenizer.
      max_length: Maximum sequence length.
      label_columns: List of label columns.
    """
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(model_filepath)
    except OSError:
        print(f"Error: Model not found at {model_filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    try:
        print("Loading tokenizer...")
        with open(tokenizer_filepath, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {tokenizer_filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)

    try:
        print("Loading model parameters...")
        with open(params_filepath, 'rb') as handle:
            params = pickle.load(handle)
        max_length = params['max_length']
        label_columns = params['label_columns']
    except FileNotFoundError:
        print(f"Error: Parameters file not found at {params_filepath}")
        exit(1)
    except Exception as e:
        print(f"Error loading parameters: {e}")
        exit(1)

    return model, tokenizer, max_length, label_columns

def predict_on_dataframe(model, tokenizer, max_length, label_columns, df, data_column):
    """Predicts labels for a dataframe of abstracts.

    Args:
      model: Loaded model.
      tokenizer: Loaded tokenizer.
      max_length: Maximum sequence length.
      label_columns: List of label columns.
      df: DataFrame containing the abstracts.
      data_column: Name of the column in df containing the abstracts.

    Returns:
      df: DataFrame with predictions added.
    """

    predictions = []
    for abstract in tqdm(df[data_column], desc='Predicting'):
        sequences = tokenizer.texts_to_sequences([abstract])
        seq_padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        seq_padded = tf.cast(seq_padded, tf.float32)
        prediction = model.predict(tf.constant(seq_padded))
        predictions.append(prediction[0])

    predictions_df = pd.DataFrame(predictions, columns=label_columns)
    df = pd.concat([df, predictions_df], axis=1)
    return df

def predict_sample(model, tokenizer, max_length, label_columns, sample_abstract):
    """Predicts labels for a sample abstract.

    Args:
      model: Loaded model.
      tokenizer: Loaded tokenizer.
      max_length: Maximum sequence length.
      label_columns: List of label columns.
      sample_abstract: The abstract to predict labels for.
    """
    print('Predicting on a sample abstract...')
    sample_sequences = tokenizer.texts_to_sequences([sample_abstract])
    sample_padded = pad_sequences(
        sample_sequences, maxlen=max_length, padding='post', truncating='post'
    )

    predictions = model.predict(tf.constant(sample_padded))

    print('\nSample predictions:')
    print('Sample abstract:', sample_abstract)
    print('Predicted labels:', predictions[0])
    # To get predicted labels as a list of strings:
    predicted_labels = [label_columns[i] for i, prob in enumerate(predictions[0]) if prob > 0.5]
    print('Predicted labels (thresholded):', predicted_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels for a dataframe of abstracts.")
    parser.add_argument(
        "--model_filepath",
        type=str,
        default="sdg_multi_label_saved_model.keras",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--tokenizer_filepath",
        type=str,
        default="tokenizer.pkl",
        help="Path to the saved tokenizer",
    )
    parser.add_argument(
        "--params_filepath",
        type=str,
        default="model_params.pkl",
        help="Path to the saved model parameters",
    )
    parser.add_argument(
        "--data_filepath",
        type=str,
        required=True,
        help="Path to the dataset pickle file"
    )
    parser.add_argument(
        "--data_column",
        type=str,
        default="abstract",
        help="Name of the column in the data file containing the abstracts",
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        required=True,
        help="Path to save the dataframe with predictions",
    )
    parser.add_argument(
        "--sample_abstract",
        type=str,
        help="Optional sample abstract to predict on",
    )
    args = parser.parse_args()

    # Load artifacts
    model, tokenizer, max_length, label_columns = load_artifacts(
        args.model_filepath, args.tokenizer_filepath, args.params_filepath
    )

    # Load dataframe
    if args.data_filepath:
        try:
            print("Loading dataframe...")
            df = pd.read_pickle(args.data_filepath)
        except FileNotFoundError:
            print(f"Error: Data file not found at {args.data_filepath}")
            exit(1)
        except Exception as e:
            print(f"Error loading data file: {e}")
            exit(1)

        # Make predictions
        df = predict_on_dataframe(
            model, tokenizer, max_length, label_columns, df, args.data_column
        )

        # Save dataframe with predictions
        print("Saving dataframe with predictions...")
        if os.path.exists(args.output_filepath):
            print(f"Warning: Output file already exists at {args.output_filepath}. Overwriting.")
        df.to_pickle(args.output_filepath)

    # Make prediction on sample abstract (optional)
    if args.sample_abstract:
        predict_sample(model, tokenizer, max_length, label_columns, args.sample_abstract)
