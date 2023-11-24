"""setup.py for screen2words."""

from setuptools import find_packages
from setuptools import setup

# cd ..
# pip install -r rouge/requirements.txt
# pip install rouge-score

setup(
    name="screen2words",
    version="0.1",
    packages=find_packages(),
    author="Bryan Wang",
    author_email="bryanw@dgp.toronto.edu",
    install_requires=[
        "absl"
        "numpy"
        "apache_beam[gcp]>=2.52.0"
        "rouge-score>=0.1.2",
        "nltk>=3.8.1",
        "keras>=2.12.0",
        #"protobuf<4.20.0",
        # On Windows only
        # "cython<3.0.0",
        # "pyyaml==5.4.1", # pip install pyyaml==5.4.1 --no-build-isolation
        "tensorflow>=2.12.0",
        "tensorflow-intel>=2.12.0",
        "tf-models-official>=2.12.0",
        "tensorflow-text>=2.12.1", # Need to compile from source on Windows, see https://github.com/tensorflow/text#a-note-about-different-operating-system-packages
        "tensorflow-metadata>=1.13.1"
    ],
)
