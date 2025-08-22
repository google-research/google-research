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
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

RANDOM_STATE = 42

def load_and_preprocess_data(
        filepath,
        preprocessed_filepath,
        tokenizer_filepath,
        params_filepath
):
    """Loads and preprocesses the dataset.

    Args:
      filepath: Path to the pickle file containing the training dataset.
      preprocessed_filepath: Path to preprocessed dataset.
      tokenizer_filepath: Path to save the tokenizer.
      params_filepath: Path to save the model parameters.

    Returns:
      X_train: Training data (padded sequences).
      X_test: Testing data (padded sequences).
      y_train: Training labels.
      y_test: Testing labels.
      tokenizer: Fitted tokenizer.
      label_columns: List of label columns.
      max_length: Maximum sequence length.
    """
    # Load preprocessed data and params if they exist
    if (os.path.exists(preprocessed_filepath) and
        os.path.exists(tokenizer_filepath) and
        os.path.exists(params_filepath)):
        print('Reading preprocessed dataset and parameters...')
        try:
            with open(preprocessed_filepath, 'rb') as handle:
                X_y = pickle.load(handle)
                X = X_y[0]
                y = X_y[1]

            with open(params_filepath, 'rb') as handle:
                params = pickle.load(handle)
                label_columns = params['label_columns']
                max_length = params['max_length']

            with open(tokenizer_filepath, 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            print(f"Error loading preprocessed dataset and parameters: {e}")
            exit(1)
    # Otherwise preprocess raw data
    else:
        try:
            print('Reading dataset...')
            df = pd.read_pickle(filepath)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {filepath}")
            exit(1)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            exit(1)

        df.drop(columns=['doi'], inplace=True)
        label_columns = [col for col in df.columns if col.startswith('SDG')]

        # Extract features (abstract) and labels
        print('Extracting features and labels...')
        X = df['abstract'].values
        y = df[label_columns].fillna(0).to_numpy()

        # Tokenize the abstracts
        print('Tokenizing abstracts...')
        tokenizer = Tokenizer(oov_token='<OOV>')
        tokenizer.fit_on_texts(X)
        X = tokenizer.texts_to_sequences(X)

        # Pad sequences to ensure consistent length
        print('Padding sequences...')
        max_length = 20000  # You might want to determine this dynamically
        X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')

        # Save preprocessed artifacts
        save_preprocessed(
            X,
            y,
            tokenizer,
            label_columns,
            max_length,
            preprocessed_filepath,
            tokenizer_filepath,
            params_filepath
        )

    # Split data into training and testing sets
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, tokenizer, label_columns, max_length

def create_and_train_model(
    X_train,
    y_train,
    vocab_size,
    embedding_dim,
    max_length,
    quick_train,
    model_filepath,
    input_dropout,
    output_droupout,
    learning_rate
):
    """Creates and trains the multilabel multiclass model.

    Args:
      X_train: Training data.
      y_train: Training labels.
      vocab_size: Size of the vocabulary.
      embedding_dim: Dimension of the word embeddings.
      max_length: Maximum sequence length.
      quick_train: Boolean flag to enable quick training mode.
      model_filepath: Path to save the model.
      input_dropout: Dropout for LSTM input.
      output_dropout: Dropout for LSTM output.
      learning_rate: Learning rate for training.

    Returns:
      model: Trained model.
      history: Training history.
    """
    print('Building model...')
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=input_dropout)),
        tf.keras.layers.Dropout(output_dropout),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print('Training model...')
    epochs = 10
    batch_size = 64
    es_patience = 3

    if quick_train:
        print("Quick training mode enabled. Using a smaller subset of data.")
        X_train = X_train[:1000]  # Use only 1000 samples
        y_train = y_train[:1000]
        epochs = 5  # Reduce the number of epochs
        es_patience = 2

    # Split 80%-20% again to set aside validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=es_patience, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, tensorboard_callback],
        verbose=1
    )

    # Save the model
    print('Saving model...')
    if os.path.exists(model_filepath):
        print(f"Warning: Model file already exists at {model_filepath}")
    print(f"Saving model at {model_filepath}")
    model.save(model_filepath)

    return model, history

def evaluate_model(model, X_test, y_test, eval_report_filepath, quick_eval):
    """Evaluates the trained model.

    Args:
      model: Trained model.
      X_test: Testing data.
      y_test: Testing labels.
      eval_report_filepath: Path to save evaluation results.
      quick_eval: Boolean flag to enable quick eval mode
    """
    print('Evaluating model...')

    if quick_eval:
        print("Quick evaluation mode enabled. Using a smaller subset of data.")
        X_test = X_test[:250]  # Use only 250 samples
        y_test = y_test[:250]

    y_pred = model.predict(X_test)
    y_pred_thresholded = (y_pred > 0.5).astype(int)

    report = classification_report(y_test, y_pred_thresholded, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    if os.path.exists(eval_report_filepath):
        print(f"Warning: Overwriting evaluation report file at {eval_report_filepath}")
    with open(eval_report_filepath, 'wb') as handle:
        pickle.dump(report_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Classification Report:\n', report_df)
    print('Hamming Loss:', hamming_loss(y_test, y_pred_thresholded))

def save_preprocessed(
        X, y, tokenizer, label_columns, max_length,
        preprocessed_data_filepath, tokenizer_filepath, params_filepath
):
    """Saves preprocessed data, tokenizer, and model parameters.

    Args:
      X: Inputs.
      y: Labels.
      tokenizer: Fitted tokenizer.
      label_columns: List of label columns.
      max_length: Maximum sequence length.
      preprocessed_data_filepath: Path to save preprocessed data.
      tokenizer_filepath: Path to save the tokenizer.
      params_filepath: Path to save the model parameters.
    """
    # Save data
    print('Saving preprocessed data...')
    if os.path.exists(preprocessed_data_filepath):
        print(f"Warning: Overwriting preprocessed data file at {preprocessed_data_filepath}")
    with open(preprocessed_data_filepath, 'wb') as handle:
        pickle.dump([X, y], handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save tokenizer
    print('Saving tokenizer...')
    if os.path.exists(tokenizer_filepath):
        print(f"Warning: Overwriting tokenizer file at {tokenizer_filepath}")
    with open(tokenizer_filepath, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save parameters
    print('Saving parameters...')
    if os.path.exists(params_filepath):
        print(f"Warning: Overwriting parameters file already {params_filepath}")
    params = {'max_length': max_length, 'label_columns': label_columns}
    with open(params_filepath, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict_sample(model, tokenizer, max_length, label_columns, sample_abstract):
    """Predicts labels for a sample abstract.

    Args:
      model: Trained model.
      tokenizer: Fitted tokenizer.
      max_length: Maximum sequence length.
      label_columns: List of label columns.
      sample_abstract: The abstract to predict labels for.
    """
    print('Predicting a sample...')
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
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multilabel multiclass classification model."
    )
    parser.add_argument(
        "--data_filepath",
        type=str,
        required=True,
        help="Path to the dataset pickle file",
    )
    parser.add_argument(
        "--data_preprocessed_filepath",
        type=str,
        default="train_preprocessed.pkl",
        help="Path to the preprocessed dataset pickle file",
    )
    parser.add_argument(
        "--quick_train",
        action="store_true",
        help="Enable quick training mode with less data",
    )
    parser.add_argument(
        "--quick_eval",
        action="store_true",
        help="Enable quick eval mode with less data",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run eval",
    )
    parser.add_argument(
        "--model_filepath",
        type=str,
        default="sdg_multi_label_saved_model.keras",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--tokenizer_filepath",
        type=str,
        default="tokenizer.pkl",
        help="Path to save the tokenizer",
    )
    parser.add_argument(
        "--params_filepath",
        type=str,
        default="model_params.pkl",
        help="Path to save the model parameters",
    )
    parser.add_argument(
        "--input_dropout",
        type=float,
        default=0.2,
        help="Dropout for LSTM input",
    )
    parser.add_argument(
        "--output_dropout",
        type=float,
        default=0.4,
        help="Dropout for LSTM output",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--eval_report_filepath",
        type=str,
        default="eval_report.pkl",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Load and preprocess data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer,
        label_columns,
        max_length
    ) = load_and_preprocess_data(args.data_filepath,
                                 args.data_preprocessed_filepath,
                                 args.tokenizer_filepath,
                                 args.params_filepath)

    # Create and train the model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100

    if args.eval_only:
        print('Eval only mode. Skipping training...')
    else:
        model, history = create_and_train_model(
            X_train,
            y_train,
            vocab_size,
            embedding_dim,
            max_length,
            args.quick_train,
            args.model_filepath,
            args.input_dropout,
            args.output_dropout,
            args.learning_rate
        )

    model = tf.keras.models.load_model(args.model_filepath)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, args.eval_report_filepath, args.quick_eval)

    # Sample prediction
    sample_abstract = "This research explores the use of renewable energy sources to mitigate climate change and promote sustainable development. We investigate the potential of solar and wind power in reducing greenhouse gas emissions and achieving energy security."
    predict_sample(model, tokenizer, max_length, label_columns, sample_abstract)
    exit(0)
