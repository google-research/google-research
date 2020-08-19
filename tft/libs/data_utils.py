import os

import pandas as pd

from cg_common.libs.aws.s3 import S3Storage


def create_storage():
  return S3Storage(
    aws_access_key_id=None,
    aws_secret_access_key=None,
    bucket_name="cg-staging-eu-west-1",
    region_name=None,
  )


def load_file_from_s3_if_not_exists(dirpath, filename):
  local_filepath = os.path.join(dirpath, filename)
  if os.path.exists(local_filepath):
    pass
  storage = create_storage()
  s3_path = os.path.join("s3://", storage.bucket_path, "tft", filename)
  print(f"Copying from {s3_path} to {local_filepath}")
  storage.s3.get(s3_path, local_filepath)


def read_csv(filename, config, read_csv_kwargs=None):
  data_folder = config.data_folder
  input_filepath = os.path.join(data_folder, filename)
  if not read_csv_kwargs:
    read_csv_kwargs = {}
  if not os.path.exists(input_filepath):
    return None
  return pd.read_csv(input_filepath, **read_csv_kwargs)


def write_csv(df, filename, config, to_csv_kwargs=None):
  data_folder = config.data_folder
  input_filepath = os.path.join(data_folder, filename)
  if not to_csv_kwargs:
    to_csv_kwargs = {}
  return df.to_csv(input_filepath, **to_csv_kwargs)


def load_or_read_csv(filename, config, read_csv_kwargs=None):
  data_folder = config.data_folder
  load_file_from_s3_if_not_exists(data_folder, filename)
  return read_csv(filename, config, read_csv_kwargs=read_csv_kwargs)
